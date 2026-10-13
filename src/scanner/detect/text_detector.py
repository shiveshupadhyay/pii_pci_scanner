"""Detection over a piece of text: regex candidates -> validators -> context rules -> merge.

Used for every text-like unit: log/config/code chunks, PDF/DOCX/EML/HTML blocks, OCR lines and structured cells
(cells pass the column header as `key`, which stands in for an in-line key such as `mobile=`).

Each detector is generous with regex candidates and strict about accepting them. A candidate must pass its
validator (checksum, issuer, TLD, rules) and, for ambiguous shapes, a context rule (a phone key, an expiry keyword
next to a card). merge() then settles overlaps between entities.
"""

import re

from ..units import Candidate
from . import validators as V
from .context import (EXPIRY_LEFT, is_card_key, is_cvv_key, is_email_header_skip, is_expiry_key, is_hash_key,
                      is_identifier_key, is_name_key, is_password_key, is_phone_context, is_ssn_key, is_token_key,
                      key_words, left_key, left_text)
from .names import NameScorer

# When candidates overlap, merge() keeps the higher priority. The order reflects how much evidence each match
# carries: a PEM block or an AWS key id is unmistakable; a Luhn-and-issuer-valid card beats a phone made of the
# same digits; a token with a known prefix beats a generic high-entropy string; and a name, the weakest signal,
# never displaces anything.
PRIORITY = {"PRIVATE_KEY": 100, "AWS_ACCESS_KEY": 90, "API_TOKEN": 85, "CARD_NUMBER": 80, "NATIONAL_ID": 75,
            "EMAIL": 70, "PHONE": 60, "CARD_EXPIRY": 55, "CVV": 55, "PASSWORD": 50, "API_TOKEN_GENERIC": 45,
            "PERSON_NAME": 30}

# The END label must repeat the BEGIN label (\1). The body is lazy and capped at 20,000 characters, so a stray
# BEGIN line without an END cannot swallow the rest of the file.
PEM_RE = re.compile(r"-----BEGIN ([A-Z0-9 ]{1,40})-----(.{0,20000}?)-----END \1-----", re.S)
# scheme://user:password@host. The password may contain '#', '!' or '$' (real ones do), but not '@', '/' or
# whitespace. The lookahead requires a host character after the '@'.
URL_CRED_RE = re.compile(r"\b[a-zA-Z][a-zA-Z0-9+.\-]{1,20}://([^\s:/@?#'\"`<>]{1,64}):([^\s@/'\"`<>]{1,128})@"
                         r"(?=[\w.\-\[])")
AWS_RE = re.compile(r"(?<![A-Za-z0-9])(?:AKIA|ASIA)[A-Z0-9]{16}(?![A-Za-z0-9])")
# Token formats with a fixed, documented prefix. These count anywhere, with no key needed. The outer lookarounds
# stop a match from starting or ending inside a longer identifier.
KNOWN_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_\-])(?:"
    r"(?P<stripe>sk_(?:live|test)_[A-Za-z0-9]{8,128})|"
    r"(?P<github>(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,255}|github_pat_[A-Za-z0-9_]{20,255})|"
    r"(?P<gitlab>glpat-[A-Za-z0-9_\-]{16,64})|"
    r"(?P<slack>xox[abposr]-[A-Za-z0-9\-]{8,255})|"
    r"(?P<google>AIza[0-9A-Za-z_\-]{30,40})|"
    r"(?P<openai>sk-proj-[A-Za-z0-9_\-]{16,255})|"
    r"(?P<jwt>eyJ[A-Za-z0-9_\-]{4,}\.[A-Za-z0-9_\-]{4,}\.[A-Za-z0-9_\-]{4,})"
    r")(?![A-Za-z0-9_\-])")
# length of each prefix, so the placeholder check looks only at the random part ("sk_live_" + "XXXXXXXX...")
TOKEN_PREFIX_LEN = {"stripe": 8, "github": 4, "gitlab": 6, "slack": 5, "google": 4, "openai": 8, "jwt": 3}
# Local part up to 64 characters and domain labels up to 63 (the RFC limits). The domain needs at least one dot,
# so "deploy@prod-web-01" never matches. The TLD (group 3) is checked afterwards ("logo@2x.png" and
# "admin@db.local" match the pattern but fail that check).
EMAIL_RE = re.compile(r"(?<![A-Za-z0-9._%+\-])([A-Za-z0-9][A-Za-z0-9._%+\-]{0,63})@"
                      r"((?:[A-Za-z0-9](?:[A-Za-z0-9\-]{0,61}[A-Za-z0-9])?\.)+([A-Za-z]{2,24}))(?![A-Za-z0-9\-])")
# Maximal runs of digit groups joined by single spaces or dashes. Cards are searched *inside* these runs (see
# _cards), because a card often sits right next to other numbers ("4539 1488 0343 6467 12 29").
DIGIT_RUN_RE = re.compile(r"(?<![\w\-/.])\d+(?:[ \-]\d+)*(?![\w])")
# 4-4-4 digits with the same separator throughout (or none). It refuses to start or end next to another digit
# group, so 12 digits taken out of a card number or a longer id never match.
AADHAAR_RE = re.compile(r"(?<![\w\-/.])(?<!\d[ \-])([2-9]\d{3})([ \-]?)(\d{4})\2(\d{4})(?![\w])(?![ \-]\d)")
SSN_RE = re.compile(r"(?<![\w\-/.])(\d{3})-(\d{2})-(\d{4})(?![\w\-])")
SSN_BARE_RE = re.compile(r"(?<![\w\-/.+])(\d{3})[ ]?(\d{2})[ ]?(\d{4})(?![\w\-])")  # only used next to an SSN key
# Phones, from most to least self-evident:
#   +CC groups...            international; trimmed to the country's national length afterwards (_trim_phone)
#   (415) 555-0132           North American with area code in parentheses
#   98765 43210 / 415-555-0132   Indian 5-5 mobile grouping, North American 3-3-4
#   bare digit runs          accepted only next to a phone key or phrase (PHONE_CTX_RE)
PHONE_INTL_RE = re.compile(r"(?<![\w+])\+(\d{1,3})[ \-]?(?:\(\d{1,4}\)[ \-]?)?\d{1,10}(?:[ \-]\d{1,10}){0,4}(?![\w])")
PHONE_PAREN_RE = re.compile(r"(?<![\w)])\(\d{3}\)[ \-]?\d{3}[ \-]\d{4}(?![\w])")
PHONE_LOCAL_RE = re.compile(r"(?<![\w+\-/.])(?:[6-9]\d{4}[ \-]\d{5}|\d{3}-\d{3}-\d{4})(?![\w\-])")
PHONE_CTX_RE = re.compile(r"(?<![\w+\-/.])\+?\d{2,12}(?:[ \-]\d{2,8}){0,4}(?![\w\-])")
PAN_RE = re.compile(r"(?<![A-Za-z0-9])[A-Z]{3}[PCHFATBLJG][A-Z]\d{4}[A-Z](?![A-Za-z0-9])")
# a CVV keyword, then the 3-4 digit value, allowing "is", ":", "=", "#" and quotes in between
CVV_RE = re.compile(r"(?i)(?:\bcvv2?|\bcvc2?|\bcvn|\bsecurity[\s_\-]*code|\bcard[\s_\-]*verification"
                    r"(?:[\s_\-]*(?:value|code))?)(?:\s+(?:is|was|number|no\.?|code))?[\"']?\s*[:=#\-]?\s*[\"']?"
                    r"(\d{3,4})(?![\w])")
# M/YY, MM/YY, MM/YYYY or MM-YY. The lookarounds reject pieces of longer dates and versions (2025/10/12,
# 10/12/2025, 1.10/2) and times, but allow a sentence-final full stop ("valid thru 08/31.").
EXP_RE = re.compile(r"(?<![\w/.\-])(0?[1-9]|1[0-2])([/\-])((?:19|20)\d{2}|\d{2})(?![\w/\-]|[.:]\d)")
# Anchor words for key=value secrets. _kv() only looks around these words instead of parsing every "k: v" pair in
# the text, which keeps a 150 MB log affordable.
CRED_KEY_RE = re.compile(r"(?i)pass(?:word|wd|phrase)?|pwd|secret|token|api[_\-.]?key|apikey|auth|bearer|session|"
                         r"credential|access[_\-]?key")
KEY_CHARS = re.compile(r"[A-Za-z0-9_.\-]")
# What may sit between a key and its value: a closing quote or bracket of the key ('"password": ',
# "['token'] = "), then ":", "=", ":=", "=>", "->", or the prose "is" / "was" ("the new password is X").
SEP_RE = re.compile(r"[\"'`]?\]?\s*(?::=|=>|->|[:=]|\s+is\s+|\s+was\s+)\s*")
BEARER_RE = re.compile(r"(?i)(?:bearer|token)\s+")
# characters of base64, base64url, hex and JWT tokens; a value with spaces or other punctuation is not a token
TOKEN_CHARSET = re.compile(r"^[A-Za-z0-9_\-+/=.~]+$")

# phone national-number lengths for common country codes (domain knowledge, used to trim trailing groups)
NATIONAL_LEN = {"1": 10, "91": 10, "44": 10, "61": 9, "65": 8, "86": 11, "33": 9, "971": 9, "966": 9, "27": 9,
                "234": 10, "254": 9, "92": 10, "880": 10, "63": 10, "84": 9, "7": 10, "52": 10, "81": 10, "82": 10,
                "34": 9, "31": 9, "46": 9, "47": 8, "45": 8, "353": 9, "64": 9, "20": 10, "90": 10, "98": 10,
                "62": 10, "60": 9, "66": 9, "94": 9, "977": 10}
# documentation and placeholder domains: addresses there are examples, not people
PLACEHOLDER_DOMAINS = {"example.com", "example.org", "example.net", "example.edu", "domain.com", "test.com",
                       "yourdomain.com", "yourcompany.com", "mydomain.com", "email.example", "company.example"}
# reserved (RFC 2606 / 6761) and internal-network names: no real mailbox lives there
NON_PUBLIC_TLDS = {"test", "example", "invalid", "localhost", "local", "internal", "lan", "corp", "home", "intranet"}
# File extensions. Several are real TLDs (.py, .sh, .md, .rs, .so, .java), so asset and script names such as
# "icon@2x.py" or "deploy@v2.sh" would otherwise pass as e-mail addresses.
FILE_TLDS = {"png", "jpg", "jpeg", "gif", "svg", "css", "js", "ts", "py", "json", "html", "htm", "txt", "pdf", "log",
             "csv", "xml", "yml", "yaml", "md", "sh", "rb", "go", "rs", "java", "php", "exe", "dll", "so", "bin"}


def _load_tlds():
    import os

    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "resources", "tlds.txt")
    try:
        with open(path, encoding="utf-8") as f:
            return {l.strip() for l in f if l.strip()}
    except OSError:
        return set()


TLDS = _load_tlds()


def _strip_value(v):
    # a value read from prose ends at a space, but sentence punctuation can still stick to it ("is Winter2024!.")
    v = v.strip().strip("\"'`")
    return re.sub(r"[.,;:)\]}>]+$", "", v)


class Detector:
    """All entity detectors. One instance per worker process: building it loads the name lexicons and model."""

    def __init__(self, names=None, nlp=None):
        self.names = names if names is not None else NameScorer()
        self.nlp = nlp

    # ================================================================ public
    def detect(self, text, key=None, ocr=False, cell=False, table=None, use_ner=False):
        """Return merged Candidates for `text`. `key` = external context (column header / form label)."""
        if not text or not text.strip():
            return []
        ctx_words = key_words(key) if key else []
        table = table or {}
        cands, blocked = [], []  # blocked: (start, end, entity set or None for all)

        # Every detector sees the whole text and appends candidates; overlaps are settled once, in merge(). The PEM
        # and URL-credential detectors also register blocked regions (inside a PEM block or a URL's user:password@
        # part, other entities do not count), which merge() applies to every candidate whatever its detector.
        self._pem(text, cands, blocked)
        self._url_creds(text, cands, blocked)
        self._aws(text, cands, ocr)
        self._known_tokens(text, cands)
        self._emails(text, cands, blocked, ctx_words)
        # card spans are returned because CVV and expiry rules need to know where the cards are
        cards = self._cards(text, cands, ctx_words)
        self._aadhaar(text, cands, ctx_words)
        self._ssn(text, cands, ctx_words)
        self._phones(text, cands, ctx_words)
        self._pan(text, cands)
        self._cvv(text, cands, cards, ctx_words, table)
        self._expiry(text, cands, cards, ctx_words, table)
        self._kv(text, cands, ctx_words)
        if cell:
            self._cell_rules(text, cands, ctx_words, table)
        self._names(text, cands, ctx_words, use_ner, ocr)
        return merge(cands, blocked)

    # ================================================================ structural
    def _pem(self, text, cands, blocked):
        if "-----BEGIN" not in text:
            return
        for m in PEM_RE.finditer(text):
            label = m.group(1)
            if "PRIVATE KEY" in label:
                # A key pasted into a JSON or .env string carries literal "\n" escapes; the value reported is the
                # real multi-line block.
                value = m.group(0).replace("\\n", "\n").replace("\\r", "")
                cands.append(Candidate("PRIVATE_KEY", value, m.start(), m.end(), 0.99, "pem-block",
                                       ("block: " + label.strip().lower(),), PRIORITY["PRIVATE_KEY"]))
            # Every PEM body is blocked, certificates and public keys included: base64 is full of chance "tokens",
            # digit runs and name-like fragments.
            blocked.append((m.start(), m.end(), None))

    def _url_creds(self, text, cands, blocked):
        if "://" not in text:
            return
        for m in URL_CRED_RE.finditer(text):
            pw = m.group(2)
            # "user:S3cret@db.acme.io:5432" would also read as the e-mail "S3cret@db.acme.io", and the port and
            # path digits as numbers. Block those entities from the user name to just past the host.
            blocked.append((m.start(1), m.end(2) + 64, {"EMAIL", "PHONE", "CARD_NUMBER", "NATIONAL_ID"}))
            if V.is_placeholder(pw) or V.is_hash_like(pw):
                continue
            cands.append(Candidate("PASSWORD", pw, m.start(2), m.end(2), 0.92, "url-credentials",
                                   ("connection url",), PRIORITY["PASSWORD"]))

    def _aws(self, text, cands, ocr):
        for m in AWS_RE.finditer(text):
            v = m.group(0)
            # in OCR text one misread letter ("EXAMPIE") must not turn the documentation key into a finding
            if v.endswith("EXAMPLE") or (ocr and _fuzzy_example(v)):
                continue
            cands.append(Candidate("AWS_ACCESS_KEY", v, m.start(), m.end(), 0.97, "regex:aws-key-id", (),
                                   PRIORITY["AWS_ACCESS_KEY"]))

    def _known_tokens(self, text, cands):
        for m in KNOWN_TOKEN_RE.finditer(text):
            kind = m.lastgroup
            v = m.group(0)
            body = v[TOKEN_PREFIX_LEN[kind]:]
            if _token_placeholder(body):  # sk_live_XXXXXXXX..., ghp_your_token_here
                continue
            conf = 0.95 if kind != "jwt" else 0.9  # "eyJ" also starts harmless base64 JSON, hence a little lower
            cands.append(Candidate("API_TOKEN", v, m.start(), m.end(), conf, "regex:known-prefix:" + kind, (),
                                   PRIORITY["API_TOKEN"]))

    def _emails(self, text, cands, blocked, ctx_words):
        if "@" not in text:
            return
        for m in EMAIL_RE.finditer(text):
            local, domain, tld = m.group(1), m.group(2).lower(), m.group(3).lower()
            if tld not in TLDS or tld in NON_PUBLIC_TLDS or tld in FILE_TLDS:
                continue
            if domain in PLACEHOLDER_DOMAINS or domain.startswith("example.") or ".example." in domain:
                continue
            nxt = text[m.end():m.end() + 2]
            if nxt[:1] == ":" and nxt[1:2] and not nxt[1:2].isspace():  # git@github.com:org/repo.git
                continue
            if local.lower() == "git":  # SSH remotes use the git user; it is not anybody's mailbox
                continue
            # Message-ID / In-Reply-To / References values have the shape of addresses but identify messages
            lk = left_key(text, m.start())
            if is_email_header_skip(lk):
                continue
            if re.search(r"(?i)message-id\s*:\s*<?$", left_text(text, m.start(), 40)):
                continue
            cands.append(Candidate("EMAIL", m.group(0), m.start(), m.end(), 0.95, "regex:email+tld", (),
                                   PRIORITY["EMAIL"]))

    # ================================================================ numbers
    def _key_for(self, text, start, ctx_words):
        # An in-line key wins ("order_id=" in a log line); otherwise the unit's header or label applies.
        lk = left_key(text, start)
        return lk if lk else ctx_words

    def _cards(self, text, cands, ctx_words):
        cards = []
        for m in DIGIT_RUN_RE.finditer(text):
            run = m.group(0)
            if len(run) < 13 or len(run) > 400:
                continue
            # Within a run of digit groups, try every span of *whole* groups holding 13-19 digits, starting at
            # each group, and keep the longest valid one. Whole groups only: carving 16 digits out of the middle
            # of "4539 1488 0343 6467 12" would invent cards that were never written.
            groups = [(g.start(), g.end()) for g in re.finditer(r"\d+", run)]
            i = 0
            while i < len(groups):
                best = None
                ndig = 0
                for j in range(i, len(groups)):
                    ndig += groups[j][1] - groups[j][0]
                    if ndig > 19:
                        break
                    if ndig >= 13:
                        s, e = m.start() + groups[i][0], m.start() + groups[j][1]
                        raw = text[s:e]
                        if not _consistent_grouping(raw):
                            continue
                        digits = re.sub(r"\D", "", raw)
                        if V.card_ok(digits):
                            best = (s, e, raw, digits, j)
                if best is None:
                    i += 1
                    continue
                s, e, raw, digits, j = best
                # A Luhn-valid number under order_id / txn_ref / tracking_number is that identifier, not a card.
                # The guide's own example: such numbers are hard negatives.
                kw = self._key_for(text, s, ctx_words)
                if is_identifier_key(kw):
                    i = j + 1
                    continue
                brand = V.card_brand(digits)
                ctx = ("key: card",) if is_card_key(kw) else ()
                c = Candidate("CARD_NUMBER", raw, s, e, 0.97 if ctx else 0.92, "regex+luhn+issuer:" + brand,
                              ctx, PRIORITY["CARD_NUMBER"])
                cands.append(c)
                cards.append((s, e, brand))
                i = j + 1
        return cards

    def _aadhaar(self, text, cands, ctx_words):
        for m in AADHAAR_RE.finditer(text):
            digits = m.group(1) + m.group(3) + m.group(4)
            if not V.aadhaar_ok(digits):
                continue
            # a 12-digit number under a phone key is a phone with country code (91 98765 43210), even if it
            # happens to pass Verhoeff
            kw = self._key_for(text, m.start(), ctx_words)
            if is_phone_context(kw, left_text(text, m.start())):
                continue
            cands.append(Candidate("NATIONAL_ID", m.group(0), m.start(), m.end(), 0.93, "regex+verhoeff:aadhaar",
                                   (), PRIORITY["NATIONAL_ID"]))

    def _ssn(self, text, cands, ctx_words):
        for m in SSN_RE.finditer(text):
            if V.ssn_ok(m.group(1) + m.group(2) + m.group(3)):
                cands.append(Candidate("NATIONAL_ID", m.group(0), m.start(), m.end(), 0.9, "regex+rules:ssn", (),
                                       PRIORITY["NATIONAL_ID"]))
        # Nine bare digits are an SSN only right after an SSN key. The cheap text test first means the bare-digit
        # regex never runs over the large majority of text that never mentions SSNs.
        if ctx_words and is_ssn_key(ctx_words) or re.search(r"(?i)\bssn|social\s+security", text):
            for m in SSN_BARE_RE.finditer(text):
                raw = m.group(0)
                if "-" in raw:
                    continue
                kw = self._key_for(text, m.start(), ctx_words)
                if is_ssn_key(kw) and V.ssn_ok(m.group(1) + m.group(2) + m.group(3)):
                    cands.append(Candidate("NATIONAL_ID", raw, m.start(), m.end(), 0.85, "regex+rules:ssn+context",
                                           ("key: ssn",), PRIORITY["NATIONAL_ID"]))

    def _pan(self, text, cands):
        # The PAN layout (5 letters with a holder-type 4th letter, 4 digits, 1 letter) is distinctive enough on
        # its own, so no key is required.
        for m in PAN_RE.finditer(text):
            cands.append(Candidate("NATIONAL_ID", m.group(0), m.start(), m.end(), 0.9, "regex+rules:pan", (),
                                   PRIORITY["NATIONAL_ID"]))

    def _phones(self, text, cands, ctx_words):
        for m in PHONE_INTL_RE.finditer(text):
            # The regex happily runs on into whatever digit groups follow ("+91 98765 43210 1200"); cut the match
            # at the country's national number length.
            raw = _trim_phone(m.group(0), m.group(1))
            if raw and m.group(1) == "1" and len(re.sub(r"\D", "", raw)) == 11 and not _nanp_ok(re.sub(r"\D", "", raw)[1:]):
                raw = None
            if raw:
                cands.append(Candidate("PHONE", raw, m.start(), m.start() + len(raw), 0.9, "regex:phone-intl", (),
                                       PRIORITY["PHONE"]))
        for rx in (PHONE_PAREN_RE, PHONE_LOCAL_RE):
            for m in rx.finditer(text):
                d = re.sub(r"\D", "", m.group(0))
                # A 3-3-4 number must be a valid North American number. That rules out SSN-like or id-like
                # groupings such as "000-933-2653".
                if (rx is PHONE_PAREN_RE or "-" in m.group(0)[:4]) and len(d) == 10 and not _nanp_ok(d):
                    continue
                cands.append(Candidate("PHONE", m.group(0), m.start(), m.end(), 0.85, "regex:phone-formatted", (),
                                       PRIORITY["PHONE"]))
        # Bare or loosely formatted digits only with phone context. Without it, a 10-digit number is just as likely
        # bytes_sent, an epoch timestamp or an account id.
        for m in PHONE_CTX_RE.finditer(text):
            raw = m.group(0)
            digits = re.sub(r"\D", "", raw)
            if not 10 <= len(digits) <= 13:
                continue
            kw = self._key_for(text, m.start(), ctx_words)
            lt = left_text(text, m.start())
            if not is_phone_context(kw, lt):
                continue
            # More than 10 digits without a "+" is a phone only as 12 digits starting with a known country code
            # (919876543210) or as 11 digits starting with 1 (North America).
            if len(digits) > 10 and not raw.startswith("+") and not (len(digits) == 12 and digits[:2] in NATIONAL_LEN):
                if len(digits) != 11 or digits[0] != "1":
                    continue
            cands.append(Candidate("PHONE", raw, m.start(), m.end(), 0.82, "regex:phone+context", ("phone context",),
                                   PRIORITY["PHONE"]))

    def _cvv(self, text, cands, cards, ctx_words, table):
        # Only Amex has 4-digit security codes. Elsewhere a 4-digit number after "cvv" is more likely a year, a PIN
        # or a misparse.
        has_amex = any(b == "amex" for _, _, b in cards) or table.get("has_amex")
        for m in CVV_RE.finditer(text):
            v = m.group(1)
            if len(v) == 4 and not has_amex:
                continue
            cands.append(Candidate("CVV", v, m.start(1), m.end(1), 0.9, "regex+keyword:cvv", ("keyword: cvv",),
                                   PRIORITY["CVV"]))

    def _expiry(self, text, cands, cards, ctx_words, table):
        # Dates like 10/26 are everywhere (releases, sprints, invoices). The guide counts one as a card expiry only
        # next to a card, so two ways in:
        #   - an expiry keyword just before the date, and a card earlier in the same text (within 400 characters)
        #     or in the same table;
        #   - the date written directly after a card number ("4539 1488 0343 6467 10/26"), keyword or not.
        for m in EXP_RE.finditer(text):
            s = m.start()
            lt = left_text(text, s, 40)
            keyword = bool(EXPIRY_LEFT.search(lt)) or (not lt.strip() and ctx_words and is_expiry_key(ctx_words))
            near_card = any(ce <= s and s - ce <= 400 for cs, ce, _ in cards) or table.get("has_card")
            right_after = any(0 <= s - ce <= 3 and not text[ce:s].strip(" ,;") for _, ce, _ in cards)
            if (keyword and near_card) or right_after:
                cands.append(Candidate("CARD_EXPIRY", m.group(0), s, m.end(), 0.9 if keyword else 0.8,
                                       "regex+card-context:expiry", ("keyword: expiry",) if keyword else
                                       ("next to card",), PRIORITY["CARD_EXPIRY"]))

    # ================================================================ key = value secrets
    def _kv(self, text, cands, ctx_words):
        """Passwords and generic tokens written as key/value pairs, in any syntax: k=v, k: v, "k": "v", k := v,
        or prose ("the new password is X")."""
        seen = set()
        for m in CRED_KEY_RE.finditer(text):
            # Grow the anchor word into the whole key token ("spring.datasource.password", "x-api-key"). Spaces end
            # it, so in prose ("Your temporary password is ...") the key is just the anchor word itself.
            ks, ke = m.start(), m.end()
            while ks > 0 and KEY_CHARS.match(text[ks - 1]):
                ks -= 1
            while ke < len(text) and KEY_CHARS.match(text[ke]):
                ke += 1
            if (ks, ke) in seen:  # one key can contain several anchor words ("auth_token_secret")
                continue
            seen.add((ks, ke))
            key = text[ks:ke].strip(".-_")
            words = key_words(key)
            pw, tok = is_password_key(words), is_token_key(words)
            if not pw and not tok:  # e.g. password_policy, token_type, session_timeout
                continue
            sm = SEP_RE.match(text, ke)
            if not sm:
                continue
            # In a URL query string ("?api_key=X&limit=10") the value ends at the next "&".
            stop_amp = ks > 0 and text[ks - 1] in "?&"
            value, vstart = _read_value(text, sm.end(), stop_amp)
            if value is None:
                continue
            if tok:
                # "Authorization: Bearer <token>": the token is the word after "Bearer"
                bm = BEARER_RE.match(text, vstart)
                if bm:
                    value, vstart = _read_value(text, bm.end(), stop_amp)
                    if value is None:
                        continue
                elif words[-1] in ("authorization", "auth"):
                    continue  # Basic auth etc. are out of scope
            if pw:
                self._password_value(value, vstart, words, cands, "kv")
            elif tok:
                self._token_value(value, vstart, words, cands)

    def _password_value(self, value, start, words, cands, how):
        # A cell is the value as a whole; a value read from running text may carry sentence punctuation.
        v = _strip_value(value) if how != "cell" else value.strip()
        if not v or len(v) > 128 or V.is_placeholder(v) or V.is_hash_like(v) or V.is_hex_digest(v):
            return
        # Prose about passwords: "password is required", "password was changed", "password: reset by admin".
        if v.lower() in ("is", "was", "for", "to", "the", "and", "has", "will", "must", "should", "reset", "expired",
                         "changed", "updated", "required", "policy", "length", "hash"):
            return
        # 1-3 digits are policy numbers ("password: 8 characters minimum"), not passwords
        if re.fullmatch(r"\d{1,3}", v):
            return
        off = value.find(v) if how != "cell" else value.find(v)
        s = start + max(off, 0)
        cands.append(Candidate("PASSWORD", v, s, s + len(v), 0.88 if how == "kv" else 0.85, "key-value:password",
                               ("key: " + " ".join(words[-2:]),), PRIORITY["PASSWORD"]))

    def _token_value(self, value, start, words, cands, how="kv"):
        v = _strip_value(value)
        if not _generic_token_ok(v, words):
            return
        off = value.find(v)
        s = start + max(off, 0)
        cands.append(Candidate("API_TOKEN", v, s, s + len(v), 0.8, "key-value:high-entropy",
                               ("key: " + " ".join(words[-3:]),), PRIORITY["API_TOKEN_GENERIC"]))

    # ================================================================ structured cells
    def _cell_rules(self, text, cands, ctx_words, table):
        """Whole-cell values whose meaning comes only from the column header.

        In a table there is no "password=" in front of the value; the header plays that role. These rules only
        run for single cells, where the whole cell is the value, never for running text."""
        if not ctx_words:
            return
        v = text.strip()
        off = text.find(v)
        if is_password_key(ctx_words):
            self._password_value(v, off, ctx_words, cands, "cell")
        elif is_token_key(ctx_words) and not is_hash_key(ctx_words):
            self._token_value(v, off, ctx_words, cands, "cell")
        elif is_cvv_key(ctx_words) and re.fullmatch(r"\d{3,4}", v):
            if len(v) == 3 or table.get("has_amex"):
                cands.append(Candidate("CVV", v, off, off + len(v), 0.9, "header:cvv", ("header: cvv",),
                                       PRIORITY["CVV"]))
        elif is_expiry_key(ctx_words) and table.get("has_card"):
            m = EXP_RE.fullmatch(v)
            if m:
                cands.append(Candidate("CARD_EXPIRY", v, off, off + len(v), 0.9, "header+card-column:expiry",
                                       ("header: expiry",), PRIORITY["CARD_EXPIRY"]))
        elif is_name_key(ctx_words):
            # Under a name header, any 2-3 alphabetic tokens are taken in any casing: unseen names, all-caps
            # exports, names the lexicons don't know. The +5 priority makes this header-confirmed reading win over
            # the scorer's candidate for the same span.
            toks = v.split()
            if 2 <= len(toks) <= 3 and all(re.fullmatch(r"[A-Za-z][A-Za-z'\-]{0,24}\.?", t) for t in toks) \
                    and not any(t.lower() in ("na", "n/a", "unknown", "test", "user", "null", "none") for t in toks):
                cands.append(Candidate("PERSON_NAME", v, off, off + len(v), 0.92, "header:name", ("header: name",),
                                       PRIORITY["PERSON_NAME"] + 5))

    # ================================================================ names
    def _names(self, text, cands, ctx_words, use_ner, ocr=False):
        nlp = self.nlp if use_ner else None
        for start, end, tokens, f, p, tag in self.names.scored_candidates(text, ctx_words, nlp, ocr):
            if p >= self.names.threshold:
                cands.append(Candidate("PERSON_NAME", text[start:end], start, end, round(min(0.99, p), 3),
                                       "name-scorer", (tag,), PRIORITY["PERSON_NAME"]))


# ==================================================================== helpers
def _nanp_ok(d10):
    """North American numbers: area code and exchange start with 2-9."""
    return d10[0] in "23456789" and d10[3] in "23456789"


def _fuzzy_example(v):
    # "EXAMPLE" with at most one wrong letter, as OCR produces it (EXAMPIE, EXAMPLF)
    tail = v[-7:]
    return sum(a != b for a, b in zip(tail, "EXAMPLE")) <= 1


def _token_placeholder(body):
    b = body.lower()
    # Real token bodies are random; a body made of 3 or fewer distinct characters ("xxxxxxxx", "00000000") is filler.
    if len(set(b.replace("-", "").replace("_", ""))) <= 3:
        return True
    return bool(re.search(r"x{4,}|\*{3,}|your|example|placeholder|dummy|redacted|changeme|<|>", b))


def _consistent_grouping(raw):
    # Cards are written with one kind of separator. "4539 1488-0343 6467" is two numbers that happen to be adjacent.
    seps = set(re.findall(r"\D", raw))
    return len(seps) <= 1


def _trim_phone(raw, cc):
    """Cut an international match back to one phone number, or None if no plausible length is reached.

    For known country codes the number ends exactly where country code + national number is complete. For other
    codes the last group boundary between 10 and 15 digits is used, 15 being the E.164 maximum."""
    digits = re.sub(r"\D", "", raw)
    groups = list(re.finditer(r"\(?\d+\)?", raw))
    nat = NATIONAL_LEN.get(cc)
    # walk groups and stop at the expected national length (or 15 digits)
    total = 0
    end = None
    for g in groups:
        total += len(re.sub(r"\D", "", g.group()))
        if nat is not None and total == len(cc) + nat:
            end = g.end()
            break
        if nat is None and 10 <= total <= 15:
            end = g.end()
    if end is None:
        if nat is None and 10 <= len(digits) <= 15:
            end = len(raw)
        else:
            return None
    out = raw[:end]
    n = len(re.sub(r"\D", "", out))
    return out if 10 <= n <= 15 else None


def _read_value(text, vs, stop_amp=False):
    """Read a quoted or bare value starting at vs; returns (value, start) or (None, None)."""
    if vs >= len(text):
        return None, None
    q = text[vs]
    if q in "\"'`":
        # A quoted value must close on the same line and within 600 characters. Otherwise the quote probably opens
        # something else (a docstring, a sentence) and reading on would swallow unrelated text.
        e = text.find(q, vs + 1)
        nl = text.find("\n", vs + 1)
        if e == -1 or (nl != -1 and nl < e) or e - vs > 600:
            return None, None
        return text[vs + 1:e], vs + 1
    m = (_BARE_VALUE_AMP if stop_amp else _BARE_VALUE).match(text, vs, min(len(text), vs + 600))
    if not m:
        return None, None
    return m.group(0), vs


# A bare value runs until whitespace, a quote, or a list/statement separator. Passwords can contain almost any
# other character ("Cobalt8117#", "pa$$:word!"), so the stop set is kept small.
_BARE_VALUE = re.compile(r"[^\s\"'`,;<>]+")
_BARE_VALUE_AMP = re.compile(r"[^\s\"'`,;<>&]+")


def _generic_token_ok(v, words):
    """A token without a known prefix must look generated: 16+ token characters, two character classes, at
    least 3 bits of entropy per character, and not a UUID, hash, path or URL. The credential key it sits under
    is the other half of the evidence."""
    if not v or len(v) < 16 or len(v) > 512 or not TOKEN_CHARSET.match(v):
        return False
    if V.is_placeholder(v) or V.is_uuid(v) or V.is_hash_like(v) or is_hash_key(words):
        return False
    if v.startswith(("http", "/", "./")) or v.count("/") > 2 or "example" in v.lower():
        return False
    if V.char_classes(v) < 2:
        return False
    return V.shannon_entropy(v) >= 3.0


def _blocked(c, blocked):
    # A block with ents=None (a PEM body) suppresses everything that *starts* inside it, except the private key
    # itself. A block with an entity set suppresses those entities wherever they overlap it.
    for bs, be, ents in blocked:
        if ents is None:
            if c.entity != "PRIVATE_KEY" and bs <= c.start < be:
                return True
        elif c.entity in ents and c.start < be and bs < c.end:
            return True
    return False


def merge(cands, blocked=()):
    """Resolve overlaps: higher priority wins, then longer span, then higher confidence; drop blocked spans.

    Candidates are swept in start order and resolved per cluster of overlapping spans, so the cost is
    O(n log n) even for chunks with thousands of candidates."""
    items = sorted((c for c in cands if not (blocked and _blocked(c, blocked))), key=lambda c: (c.start, -c.end))
    # A cluster is a maximal chain of overlapping spans. Candidates in different clusters cannot conflict, so each
    # cluster is resolved on its own and the expensive pairwise check only ever sees a handful of candidates.
    out, cluster, cluster_end = [], [], -1
    for c in items:
        if cluster and c.start >= cluster_end:
            out.extend(_resolve(cluster))
            cluster = []
        cluster.append(c)
        cluster_end = max(cluster_end, c.end) if len(cluster) > 1 else c.end
    if cluster:
        out.extend(_resolve(cluster))
    out.sort(key=lambda c: c.start)
    return out


def _resolve(cluster):
    # Greedy: take candidates best-first (priority, then span length, then confidence) and keep each one that does
    # not overlap anything already kept. Identical duplicates collapse into one along the way.
    if len(cluster) == 1:
        return cluster
    cluster = sorted(cluster, key=lambda c: (-c.priority, -(c.end - c.start), -c.confidence, c.start))
    taken = []
    for c in cluster:
        if not any(c.start < t.end and t.start < c.end for t in taken):
            taken.append(c)
    return taken
