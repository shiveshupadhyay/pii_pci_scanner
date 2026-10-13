"""Synthetic labelled data that follows ANNOTATION_GUIDE.md: value generators, hard negatives, file writers.

Used by make_synthetic.py (format/generalisation test set) and make_perf_data.py (hidden-scale set).
Names come from pools that never appear in the dev labels (common unseen names and out-of-lexicon names).
"""

import hashlib
import io
import json
import os
import random
import string
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path[:0] = [os.path.join(ROOT, "starter")]
from contract import location_key, value_sha256  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
# Logical font -> candidate font files, best first. The Linux fonts come first, then the macOS and Windows
# equivalents (Liberation is metric-compatible with Arial / Times New Roman / Courier New, so line widths and
# layouts stay close). Images rendered with a stand-in font are not pixel-identical, so OCR scores on a
# regenerated set can move slightly from one OS to another.
FONT_FILES = {
    "sans": ["LiberationSans-Regular.ttf", "Arial.ttf"],
    "mono": ["LiberationMono-Regular.ttf", "Courier New.ttf", "cour.ttf"],
    "serif": ["LiberationSerif-Regular.ttf", "Times New Roman.ttf", "times.ttf"],
    "dejavu": ["DejaVuSans.ttf", "Verdana.ttf"],
    "dejavumono": ["DejaVuSansMono.ttf", "Menlo.ttc", "consola.ttf"],
    "noto": ["NotoSans-Regular.ttf", "Helvetica.ttc", "segoeui.ttf"],
}
FONT_DIRS = ["/usr/share/fonts", "/usr/local/share/fonts", os.path.expanduser("~/.local/share/fonts"),
             "/System/Library/Fonts", "/Library/Fonts", os.path.expanduser("~/Library/Fonts"),
             os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts"),
             os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "Windows", "Fonts")]
_FONT_INDEX = None
DOMAINS = ["orbitpay.in", "fjordtech.no", "riverstone.co.za", "zenithbank.ng", "hotmail.com", "icloud.com",
           "rediffmail.com", "web.de", "lumenhealth.org", "copperleaf.io", "quillsoft.dev", "tidewater.com.au",
           "saffronlabs.in", "nimbusretail.co.uk", "yahoo.com"]
COMPANIES = ["Brightwave Analytics", "Copperleaf Systems", "Saffron Labs", "Tidewater Logistics", "Nimbus Retail",
             "Orbit Payments", "Quill Software", "Lumen Health"]
PRODUCTS = ["Apache Spark", "Visual Studio", "Google Sheets", "Microsoft Teams", "Adobe Reader", "Oracle Cloud"]
PHRASES = ["Account Executive", "Grace Period", "Will Call", "Due Date", "Best Regards", "Data Platform",
           "Customer Success", "Release Notes", "Billing Team", "Head Office"]


def _lines(name):
    with open(os.path.join(DATA, name), encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]


def load_font(font, size):
    """A TrueType font for a logical name (see FONT_FILES), searched in the usual font folders of Linux, macOS
    and Windows; falls back to Pillow's built-in font."""
    global _FONT_INDEX
    from PIL import ImageFont

    if _FONT_INDEX is None:  # file name (lower case) -> first path found; built once
        _FONT_INDEX = {}
        for d in FONT_DIRS:
            for root, _, files in os.walk(d):
                for fn in files:
                    _FONT_INDEX.setdefault(fn.lower(), os.path.join(root, fn))
    for fn in FONT_FILES[font]:
        path = _FONT_INDEX.get(fn.lower())
        if path:
            return ImageFont.truetype(path, size)
    try:
        return ImageFont.load_default(size)  # Pillow >= 10.1: scalable
    except TypeError:
        return ImageFont.load_default()


# =============================================================================== values
_VD = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 0, 6, 7, 8, 9, 5], [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
       [3, 4, 0, 1, 2, 8, 9, 5, 6, 7], [4, 0, 1, 2, 3, 9, 5, 6, 7, 8], [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
       [6, 5, 9, 8, 7, 1, 0, 4, 3, 2], [7, 6, 5, 9, 8, 2, 1, 0, 4, 3], [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
       [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]]
_VP = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 5, 7, 6, 2, 8, 3, 0, 9, 4], [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
       [8, 9, 1, 6, 0, 4, 3, 5, 2, 7], [9, 4, 5, 3, 1, 2, 6, 8, 7, 0], [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
       [2, 7, 9, 3, 8, 0, 6, 4, 1, 5], [7, 0, 4, 6, 9, 1, 3, 2, 5, 8]]
_VINV = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9]


# Check-digit *generators* (the scanner only has validators): every generated Aadhaar and card number is valid
# by construction, so a miss on one is the scanner's fault, never the generator's.
def verhoeff_digit(num):
    c = 0
    for i, ch in enumerate(reversed(num)):
        c = _VD[c][_VP[(i + 1) % 8][int(ch)]]
    return str(_VINV[c])


def verhoeff_ok(num):
    c = 0
    for i, ch in enumerate(reversed(num)):
        c = _VD[c][_VP[i % 8][int(ch)]]
    return c == 0


def luhn_digit(num):
    total = 0
    for i, ch in enumerate(reversed(num)):
        d = int(ch)
        if i % 2 == 0:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return str((10 - total % 10) % 10)


TEST_CARDS = ["4111111111111111", "4242424242424242", "5555555555554444", "378282246310005", "6011111111111117",
              "4000056655665556", "5105105105105100"]


class Values:
    def __init__(self, seed=0, oov_share=0.5, pool=None):
        """pool: 'train' / 'test' use disjoint halves of every name list (name model trained on one half and
        evaluated on the other), None uses everything."""
        self.r = random.Random(seed)  # one seeded generator for everything: the same seed gives the same data set
        lists = [_lines(n) for n in ("common_first_names.txt", "common_surnames.txt", "oov_first_names.txt",
                                     "oov_surnames.txt")]
        # Even positions go to "train", odd to "test". The name model trains on one half, and the synthetic test
        # set uses only the other, so its name scores measure generalisation to unseen names.
        if pool in ("train", "test"):
            k = 0 if pool == "train" else 1
            lists = [[x for i, x in enumerate(l) if i % 2 == k] for l in lists]
        self.first_common, self.last_common, self.first_oov, self.last_oov = lists
        self.oov_share = oov_share

    # --------------------------------------------------------- PII
    def name(self, three=False):
        r = self.r
        oov = r.random() < self.oov_share
        f = r.choice(self.first_oov if oov else self.first_common)
        l = r.choice(self.last_oov if oov else self.last_common)
        if three:
            return f"{f} {r.choice(self.first_common)} {l}"
        return f"{f} {l}"

    def email(self, name=None):
        r = self.r
        name = name or self.name()
        f, l = name.lower().split()[0], name.lower().split()[-1]
        local = r.choice([f"{f}.{l}", f"{f}_{l}", f"{f[0]}{l}", f"{f}{r.randint(1, 99)}", f"{l}.{f}"])
        return f"{local}@{r.choice(DOMAINS)}"

    def phone(self, formatted=True):
        r = self.r
        mob = str(r.choice("6789")) + "".join(r.choice(string.digits) for _ in range(9))
        if not formatted:
            return mob
        k = r.randrange(7)
        if k == 0:
            return f"+91 {mob[:5]} {mob[5:]}"
        if k == 1:
            return f"+91-{mob}"
        if k == 2:
            return f"{mob[:5]}-{mob[5:]}"
        a = str(r.randint(2, 9)) + "".join(r.choice(string.digits) for _ in range(2))
        e = str(r.randint(2, 9)) + "".join(r.choice(string.digits) for _ in range(2))
        s = "".join(r.choice(string.digits) for _ in range(4))
        if k == 3:
            return f"+1 ({a}) {e}-{s}"
        if k == 4:
            return f"({a}) {e}-{s}"
        if k == 5:
            return f"+44 20 {r.randint(3000, 8999)} {r.randint(1000, 9999)}"
        return f"+49 30 {r.randint(1000000, 9999999)}"

    def pan(self):
        r = self.r
        L = string.ascii_uppercase
        return "".join(r.choice(L) for _ in range(3)) + r.choice("PCHFATBLJG") + r.choice(L) + \
            "".join(r.choice(string.digits) for _ in range(4)) + r.choice(L)

    def aadhaar(self, spaced=None):
        r = self.r
        base = str(r.randint(2, 9)) + "".join(r.choice(string.digits) for _ in range(10))
        num = base + verhoeff_digit(base)
        spaced = r.random() < 0.5 if spaced is None else spaced
        return f"{num[:4]} {num[4:8]} {num[8:]}" if spaced else num

    def ssn(self):
        r = self.r
        while True:
            a = r.randint(1, 899)
            if a != 666:
                break
        return f"{a:03d}-{r.randint(1, 99):02d}-{r.randint(1, 9999):04d}"

    # --------------------------------------------------------- PCI
    def card(self, brand=None, fmt=None):
        r = self.r
        brand = brand or r.choice(["visa", "visa", "mc", "amex", "discover", "rupay"])
        if brand == "amex":
            body = r.choice(["34", "37"]) + "".join(r.choice(string.digits) for _ in range(12))
        else:
            pre = {"visa": "4", "mc": str(r.randint(51, 55)), "discover": "6011",
                   "rupay": r.choice(["60", "65", "81", "82"])}[brand]
            body = pre + "".join(r.choice(string.digits) for _ in range(15 - len(pre)))
        num = body + luhn_digit(body)
        if num in TEST_CARDS:
            return self.card(brand, fmt)
        fmt = fmt or r.choice(["plain", "space", "dash"])
        if fmt == "plain":
            return num
        sep = " " if fmt == "space" else "-"
        if len(num) == 15:
            return sep.join([num[:4], num[4:10], num[10:]])
        return sep.join(num[i:i + 4] for i in range(0, 16, 4))

    def expiry(self):
        r = self.r
        m, y = r.randint(1, 12), r.randint(25, 33)
        return r.choice([f"{m:02d}/{y:02d}", f"{m:02d}/20{y:02d}", f"{m}/{y:02d}", f"{m:02d}-{y:02d}"])

    def cvv(self, amex=False):
        return "".join(self.r.choice(string.digits) for _ in range(4 if amex else 3))

    # --------------------------------------------------------- secrets
    def aws(self):
        r = self.r
        return r.choice(["AKIA", "ASIA"]) + "".join(r.choice(string.ascii_uppercase + string.digits) for _ in range(16))

    def private_key(self):
        r = self.r
        kind = r.choice(["RSA ", "EC ", "OPENSSH ", ""])
        alphabet = string.ascii_letters + string.digits + "+/"
        lines = ["".join(r.choice(alphabet) for _ in range(64)) for _ in range(r.randint(5, 12))]
        return f"-----BEGIN {kind}PRIVATE KEY-----\n" + "\n".join(lines) + f"\n-----END {kind}PRIVATE KEY-----"

    def token(self, kind=None):
        r = self.r
        an = string.ascii_letters + string.digits
        kind = kind or r.choice(["stripe_live", "stripe_test", "github", "gitlab", "slack", "google", "openai", "jwt",
                                 "hex", "b64"])
        rnd = lambda n, al=an: "".join(r.choice(al) for _ in range(n))  # noqa: E731
        if kind == "stripe_live":
            return "sk_live_" + rnd(24)
        if kind == "stripe_test":
            return "sk_test_" + rnd(24)
        if kind == "github":
            return "ghp_" + rnd(36)
        if kind == "gitlab":
            return "glpat-" + rnd(20, an + "_-")
        if kind == "slack":
            return f"xoxb-{rnd(12, string.digits)}-{rnd(13, string.digits)}-{rnd(24)}"
        if kind == "google":
            return "AIza" + rnd(35, an + "_-")
        if kind == "openai":
            return "sk-proj-" + rnd(40, an + "_-")
        if kind == "jwt":
            b = an + "_-"
            return f"eyJ{rnd(30, b)}.{rnd(60, b)}.{rnd(43, b)}"
        if kind == "hex":
            return rnd(r.choice([32, 40, 64]), "0123456789abcdef")
        return rnd(r.randint(24, 40))

    def password(self):
        r = self.r
        k = r.randrange(3)
        if k == 0:
            return r.choice(["Harbor", "Falcon", "Maple", "Orion", "Velvet", "Cobalt"]) + str(r.randint(10, 9999)) + \
                r.choice("!$%#")
        if k == 1:
            return r.choice(self.first_common).lower() + str(r.randint(1970, 2010)) + r.choice(["", "!", "$"])
        return "".join(r.choice(string.ascii_letters + string.digits + "!%^*$") for _ in range(r.randint(10, 16)))

    # --------------------------------------------------------- hard negatives
    # Values that look sensitive but must not be reported. Each targets one rule of the annotation guide (identifier
    # fields, invalid national ids, masks, placeholders, hashes) and is placed where a careless detector would bite.
    def order_id(self):
        body = "4" + "".join(self.r.choice(string.digits) for _ in range(14))
        return body + luhn_digit(body)  # Luhn-valid, Visa-shaped, but an order id

    def uuid(self):
        h = "".join(self.r.choice("0123456789abcdef") for _ in range(32))
        return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}"

    def sha256(self):
        return hashlib.sha256(str(self.r.random()).encode()).hexdigest()

    def md5(self):
        return hashlib.md5(str(self.r.random()).encode()).hexdigest()

    def bcrypt(self):
        return "$2b$12$" + "".join(self.r.choice(string.ascii_letters + string.digits + "./") for _ in range(53))

    def epoch_ms(self):
        return str(self.r.randint(1_700_000_000_000, 1_790_000_000_000))

    def bytes10(self):
        return str(self.r.randint(1_000_000_000, 9_999_999_999))

    def bad_ssn(self):
        r = self.r
        return r.choice([f"000-{r.randint(10, 99)}-{r.randint(1000, 9999)}", f"666-{r.randint(10, 99)}-{r.randint(1000, 9999)}",
                         f"9{r.randint(10, 99)}-{r.randint(10, 99)}-{r.randint(1000, 9999)}",
                         f"{r.randint(100, 599)}-00-{r.randint(1000, 9999)}"])

    def bad_pan(self):
        r = self.r
        L = string.ascii_uppercase
        return "".join(r.choice(L) for _ in range(3)) + r.choice("DEIKMNOQRSUVWXYZ") + r.choice(L) + \
            "".join(r.choice(string.digits) for _ in range(4)) + r.choice(L)

    def bad_aadhaar(self):
        while True:
            n = str(self.r.randint(2, 9)) + "".join(self.r.choice(string.digits) for _ in range(11))
            if not verhoeff_ok(n):
                return n

    def masked_card(self):
        return self.r.choice([f"**** **** **** {self.r.randint(1000, 9999)}", f"XXXX-XXXX-XXXX-{self.r.randint(1000, 9999)}",
                              f"ending in {self.r.randint(1000, 9999)}"])

    def placeholder(self):
        return self.r.choice(["REDACTED", "[FILTERED]", "<your-api-key>", "${DB_PASSWORD}", "{{ vault_secret }}",
                              "xxxxxxxx", "null", "changeme", "YOUR_TOKEN_HERE", "********"])


# =============================================================================== gold bookkeeping


class Gold:
    """The labels, kept while the files are written and saved in the evaluator's format (scope.json,
    findings.jsonl, columns.jsonl), so starter/evaluate.py scores a synthetic run exactly like dev.
    Every value is labelled at the moment it is written into a file, so labels and content cannot drift apart."""

    def __init__(self):
        self.findings, self.columns = [], []
        self.paths, self.column_only, self.errors = {}, [], []

    def add(self, file, entity, raw, location=None, sub=""):
        location = location or {}
        self.findings.append({"file": file, "entity": entity, "value": raw,
                              "value_sha256": value_sha256(entity, raw), "location_key": location_key(location),
                              "sub": sub, "slice": "", "path": self.paths.get(file, "text")})

    def col(self, file, column, entity, sheet=""):
        self.columns.append({"file": file, "sheet": sheet, "column": column, "entity": entity})

    def write(self, labels_dir):
        os.makedirs(labels_dir, exist_ok=True)
        for f in self.findings:
            f["path"] = self.paths.get(f["file"], "text")
        scope = {"all_files": sorted(self.paths), "file_paths": self.paths,
                 "column_only_files": sorted(self.column_only), "error_files": sorted(self.errors)}
        with open(os.path.join(labels_dir, "scope.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump(scope, f, indent=1)
        with open(os.path.join(labels_dir, "findings.jsonl"), "w", encoding="utf-8", newline="\n") as f:
            for x in self.findings:
                if x["file"] not in self.column_only:
                    f.write(json.dumps(x) + "\n")
        with open(os.path.join(labels_dir, "columns.jsonl"), "w", encoding="utf-8", newline="\n") as f:
            for x in self.columns:
                f.write(json.dumps(x) + "\n")


# =============================================================================== image rendering


def render_lines(lines, font="sans", size=22, dark=False, noise=0.0, width=None, seed=0, italic_lines=()):
    """A screenshot-like image of text lines: light or dark theme, plus optional salt noise that imitates a
    scanned page once it is saved as JPEG."""
    from PIL import Image, ImageDraw

    r = random.Random(seed)
    fnt = load_font(font, size)
    pad = 28
    lh = int(size * 1.55)
    w = width or max(600, int(max((fnt.getlength(l) for l in lines), default=100)) + 2 * pad)
    h = pad * 2 + lh * len(lines)
    bg, fg = ((30, 30, 30), (230, 230, 230)) if dark else ((255, 255, 255), (20, 20, 20))
    img = Image.new("RGB", (w, h), bg)
    d = ImageDraw.Draw(img)
    for i, l in enumerate(lines):
        d.text((pad, pad + i * lh), l, font=fnt, fill=fg)
    if noise:
        px = img.load()
        for _ in range(int(w * h * noise)):
            x, y = r.randrange(w), r.randrange(h)
            c = r.randint(0, 120)
            px[x, y] = (c, c, c)
    return img


def save_jpeg(img, path, quality=70):
    img.save(path, "JPEG", quality=quality)


def save_png(img, path):
    img.save(path, "PNG")


def scanned_pdf(images, path):
    """A PDF whose pages are only JPEG images, no text layer: what a scanner produces. The scanner must OCR it."""
    import pymupdf

    doc = pymupdf.open()
    for im in images:
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=75)
        w, h = im.size
        page = doc.new_page(width=w * 0.6, height=h * 0.6)
        page.insert_image(page.rect, stream=buf.getvalue())
    doc.save(path)
    doc.close()
