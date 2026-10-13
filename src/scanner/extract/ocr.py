"""Local OCR for images and scanned PDF pages.

Engines:
* RapidOCR (pip install, ONNX models bundled in the wheel) is the primary engine of the scan pipeline.
* Tesseract (a system package, driven through pytesseract) is the secondary engine.
pipeline.ocr_findings() runs both and merges their readings; when one engine is missing, the other runs alone.

Tesseract input is preprocessed first: grayscale, dark backgrounds inverted (dark-mode screenshots), small images
upscaled, and a median filter against JPEG speckle. get_engine() / ocr_image() are a single-engine API used by
scripts/ocr_eval.py and the tests.
"""

import os
import re
import shutil

from PIL import Image, ImageFilter, ImageOps, ImageStat

DEFAULTS = {"psm": 6, "scale": 2.0, "median": True, "max_side": 4000}
_ENGINE = None


def preprocess(img, scale=2.0, median=True, max_side=4000):
    """Prepare an image for Tesseract, which expects dark text on a light background and struggles with small
    screen text. The settings were chosen with scripts/ocr_eval.py, which reports value recall per image type."""
    if img.mode not in ("L", "RGB"):
        img = img.convert("RGB")
    g = img.convert("L")
    if ImageStat.Stat(g).mean[0] < 110:  # dark mode: light text on dark background
        g = ImageOps.invert(g)
    if scale and scale != 1.0:
        # Screenshot text (12-14 px) is small for Tesseract, and a 2x Lanczos upscale helps it. max_side caps the
        # result so a page-sized scan is not blown up to an image too big to process quickly.
        f = min(scale, max_side / max(g.size))
        if f > 1.05:
            g = g.resize((int(g.width * f), int(g.height * f)), Image.LANCZOS)
    if median:
        # a 3x3 median removes JPEG speckle and scanner dust, while strokes 2 px or wider survive
        g = g.filter(ImageFilter.MedianFilter(3))
    return g


def find_tesseract():
    """Path of the tesseract binary, or None.

    pytesseract only looks on PATH, and outside Linux that is often not enough. The Windows installer does not
    add itself to PATH, and Homebrew's /opt/homebrew/bin is missing in non-login shells and IDE terminals. So try
    $TESSERACT_CMD first, then PATH, then each platform's usual install locations.
    """
    env = os.environ.get("TESSERACT_CMD")
    if env:
        return env
    found = shutil.which("tesseract")
    if found:
        return found
    if os.name == "nt":
        bases = [os.environ.get("ProgramFiles"), os.environ.get("ProgramFiles(x86)"),
                 os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs")]
        candidates = [os.path.join(b, "Tesseract-OCR", "tesseract.exe") for b in bases if b]
    else:
        candidates = ["/opt/homebrew/bin/tesseract", "/usr/local/bin/tesseract", "/opt/local/bin/tesseract",
                      "/usr/bin/tesseract"]
    return next((c for c in candidates if os.path.isfile(c)), None)


class TesseractEngine:
    name = "tesseract"

    def __init__(self):
        import pytesseract

        cmd = find_tesseract()
        if cmd:
            pytesseract.pytesseract.tesseract_cmd = cmd
        self.pt = pytesseract
        # Raises when there is no usable binary; callers treat that as "engine not available".
        self.version = str(pytesseract.get_tesseract_version())

    def run(self, img, psm=6):
        # image_to_data rather than image_to_string: it gives word positions and confidences, so lines can be
        # rebuilt in reading order and an average confidence reported. --oem 1 = the LSTM recogniser only.
        d = self.pt.image_to_data(img, config=f"--psm {psm} --oem 1", output_type=self.pt.Output.DICT)
        lines, confs = {}, []
        for i, word in enumerate(d["text"]):
            if not word or not word.strip():
                continue
            key = (d["block_num"][i], d["par_num"][i], d["line_num"][i])
            lines.setdefault(key, []).append((d["left"][i], word))
            try:
                c = float(d["conf"][i])
                if c >= 0:  # -1 marks layout rows (blocks, paragraphs), not words
                    confs.append(c)
            except (TypeError, ValueError):
                pass
        text = "\n".join(" ".join(w for _, w in sorted(ws)) for _, ws in sorted(lines.items()))
        return text, (sum(confs) / len(confs) / 100.0 if confs else 0.0)


class RapidEngine:
    name = "rapidocr"

    def __init__(self, threads=1):
        import logging

        from rapidocr import RapidOCR

        # RapidOCR logs every call at INFO. Silence it both before and after construction, because the constructor
        # sets up its own logger.
        logging.getLogger("RapidOCR").setLevel(logging.ERROR)
        # Thread counts are set explicitly: by default ONNX Runtime starts one thread per core in every worker,
        # and three OCR workers would then oversubscribe the CPU many times over.
        self.eng = RapidOCR(params={"Global.log_level": "error",
                                    "EngineConfig.onnxruntime.intra_op_num_threads": int(threads),
                                    "EngineConfig.onnxruntime.inter_op_num_threads": 1})
        logging.getLogger("RapidOCR").setLevel(logging.ERROR)
        try:
            from importlib.metadata import version

            self.version = version("rapidocr")
        except Exception:
            self.version = "3"

    def run(self, img, psm=None):
        import numpy as np

        r = self.eng(np.array(img.convert("RGB")))
        boxes, txts, scores = r.boxes, r.txts, r.scores
        if boxes is None or txts is None:
            return "", 0.0
        # RapidOCR returns one box per text fragment, and a "Label: value" line often comes back as two fragments.
        # Rebuild lines so that labels and values end up on the same line again: sort by the top-left corner, then
        # start a new line whenever a box's top moves more than half a box height from the line's first box.
        items = sorted(zip(boxes, txts, scores), key=lambda t: (float(t[0][0][1]), float(t[0][0][0])))
        lines, cur, cur_y = [], [], None
        for box, txt, sc in items:
            y = float(box[0][1])
            h = abs(float(box[3][1]) - float(box[0][1])) or 10.0
            if cur and abs(y - cur_y) > 0.5 * h:
                lines.append(" ".join(t for _, t in sorted(cur)))
                cur = []
            cur.append((float(box[0][0]), txt))
            cur_y = y if cur_y is None or len(cur) == 1 else cur_y
        if cur:
            lines.append(" ".join(t for _, t in sorted(cur)))
        return "\n".join(lines), float(sum(scores) / len(scores)) if len(scores) else 0.0


def get_engine(prefer=None):
    """One engine for the single-engine API (scripts/ocr_eval.py, tests): $SCANNER_OCR, or whichever loads first.
    The scan pipeline does not use this; it builds its engine list in pipeline.ocr_engines()."""
    global _ENGINE
    if _ENGINE is not None:
        return _ENGINE
    prefer = prefer or os.environ.get("SCANNER_OCR", "tesseract")
    order = [prefer] + [e for e in ("tesseract", "rapidocr") if e != prefer]
    for name in order:
        try:
            _ENGINE = TesseractEngine() if name == "tesseract" else RapidEngine()
            return _ENGINE
        except Exception:
            continue
    return None


def ocr_image(img, **opts):
    """Return (text, confidence 0..1, engine name). Empty text if no engine is available."""
    o = dict(DEFAULTS, **opts)
    eng = get_engine()
    if eng is None:
        return "", 0.0, "none"
    g = preprocess(img, o["scale"], o["median"], o["max_side"]) if eng.name == "tesseract" else img
    text, conf = eng.run(g, o["psm"])
    return clean_ocr_text(text), conf, f"{eng.name}-{eng.version}"


# OCR engines "typeset" what they read: curly quotes, long dashes, and @ rendered as a similar-looking © or ®.
# Mapping them back to the ASCII characters the detectors expect costs nothing and recovers e-mails and key=value
# pairs.
_SMART = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "—": "-", "–": "-",
                        "©": "@", "®": "@"})


def clean_ocr_text(text):
    t = text.translate(_SMART)
    t = re.sub(r"[ \t]+", " ", t)
    # A ';' between a word and a space is almost always a misread ':' ("password; x"). Real semicolons in code
    # ("a; b") rarely matter for detection, while a lost colon hides a key/value pair.
    t = re.sub(r"(?<=[A-Za-z]);(?=\s)", ":", t)
    return t


# "Capitalised word, space, word starting with a lowercase l": the shape of a name whose second token's capital I
# was read as l ("Omar lyer"). Names are capitalised, so a lowercase start there is suspicious to begin with.
_I_AS_L = re.compile(r"\b([A-Z][a-z]{1,20} )l([a-z]{2,20})\b")
_EMAIL_SPACE = re.compile(r"<([A-Za-z0-9._%+\-]{1,40}) ([A-Za-z0-9._%+\-]{1,40}@[A-Za-z0-9.\-]+\.[A-Za-z]{2,})>")


def repair_ocr_text(text, names=None):
    """Targeted, lexicon-guided OCR repairs:
    * a capital I read as l at the start of a name token ("Omar lyer" -> "Omar Iyer") when the I-form is a known
      name and the l-form is neither a name nor an English word;
    * a space inside an e-mail in angle brackets is an underscore the OCR dropped ("<daniel chen@x.io>").
    """
    if names is not None:
        def fix(m):
            low = "l" + m.group(2)
            cand = "i" + m.group(2)
            # Repair only when the lexicons decide it: the I-form is a known name and the l-form is no name and
            # no English word. In sans-serif fonts the two glyphs can be identical, so the pixels cannot decide.
            if (cand in names.last or cand in names.first) and low not in names.eng and low not in names.first \
                    and low not in names.last:
                return m.group(1) + "I" + m.group(2)
            return m.group(0)

        text = _I_AS_L.sub(fix, text)
    # Engines often drop "_" (it sits on the baseline and gets lost in thin fonts). Inside <...> a space is never
    # valid in an address, so the only sensible reading is the underscore.
    text = _EMAIL_SPACE.sub(lambda m: f"<{m.group(1)}_{m.group(2)}>", text)
    return text
