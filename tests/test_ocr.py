"""OCR path: rendered images go through preprocessing, OCR and the same detectors (skipped if no engine)."""

import pytest
from PIL import Image, ImageDraw, ImageFont

from scanner.extract import ocr

# Liberation Sans on Linux; Arial (its metric twin) on macOS and Windows.
FONTS = ["/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
         "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
         "/System/Library/Fonts/Supplemental/Arial.ttf",
         "C:/Windows/Fonts/arial.ttf"]


def _font(size=22):
    for path in FONTS:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size)  # Pillow >= 10.1 has a scalable built-in font
    except TypeError:
        return ImageFont.load_default()


def render(lines, dark=False):
    fnt = _font()
    img = Image.new("RGB", (900, 60 + 34 * len(lines)), (30, 30, 30) if dark else "white")
    d = ImageDraw.Draw(img)
    for i, l in enumerate(lines):
        d.text((28, 28 + 34 * i), l, font=fnt, fill=(230, 230, 230) if dark else (20, 20, 20))
    return img


@pytest.fixture(scope="module")
def engine():
    eng = ocr.get_engine()
    if eng is None:
        pytest.skip("no OCR engine installed")
    return eng


@pytest.mark.parametrize("dark", [False, True])
def test_ocr_roundtrip(engine, det, dark):
    img = render(["From: Hannah Brown <hannah20@ardentlabs.ai>", "callback number +91 98765 43210"], dark)
    text, conf, name = ocr.ocr_image(img)
    found = [(c.entity, c.value) for c in det.detect(text, ocr=True)]
    assert ("EMAIL", "hannah20@ardentlabs.ai") in found
    assert ("PHONE", "+91 98765 43210") in found


def test_preprocess_inverts_dark_images():
    g = ocr.preprocess(render(["x"], dark=True), scale=1.0, median=False)
    from PIL import ImageStat

    assert ImageStat.Stat(g).mean[0] > 128


def test_repairs(det):
    fixed = ocr.repair_ocr_text("From: Omar lyer <omar_iyer@x.io>\nAuthor: Daniel Chen <daniel chen@north.io>",
                                det.names)
    assert "Omar Iyer" in fixed and "<daniel_chen@north.io>" in fixed
    assert ocr.clean_ocr_text("VPN password; lukas2006$") == "VPN password: lukas2006$"
