"""Checksum and shape validators. Pure functions, no context: context rules live in detect/context.py.

These carry most of the precision. A checksum rejects about 90% of random digit strings, so a 16-digit number
that passes Luhn *and* has a valid issuer prefix for its length is very likely a card. That lets the detectors
accept validated numbers almost anywhere, and spend context rules only on the ambiguous cases.
"""

import math
import re
from collections import Counter

# ---------------------------------------------------------------- cards
# Well-known test numbers from payment-provider docs. They pass Luhn but the annotation guide says not to label them.
TEST_CARDS = {
    "4111111111111111", "4242424242424242", "5555555555554444", "378282246310005",
    "6011111111111117", "4000056655665556", "5105105105105100",
}


def luhn_ok(digits):
    # Luhn: double every second digit from the right (subtracting 9 when the result has two digits); the total must
    # be divisible by 10. ord(ch) - 48 is int(ch) without the function-call overhead (this runs on every digit run).
    total, alt = 0, False
    for ch in reversed(digits):
        d = ord(ch) - 48
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        alt = not alt
    return total % 10 == 0


def card_brand(digits):
    """Issuer for a digit string, following ANNOTATION_GUIDE exactly (prefix + length).

    Deliberately narrower than real-world BIN tables: 13- and 19-digit Visas, Maestro and JCB are left out because
    the guide does not list them. Accepting them would add false positives that the scoring counts and no true
    positives it rewards."""
    n = len(digits)
    if n == 16:
        if digits[0] == "4":
            return "visa"
        if "51" <= digits[:2] <= "55":
            return "mastercard"
        if digits.startswith("6011"):
            return "discover"
        if digits[:2] in ("60", "65", "81", "82"):
            return "rupay"
    elif n == 15 and digits[:2] in ("34", "37"):
        return "amex"
    return None


def card_ok(digits):
    """Luhn + issuer prefix for its length + not a well-known test number."""
    return 13 <= len(digits) <= 19 and digits not in TEST_CARDS and card_brand(digits) is not None and luhn_ok(digits)


# ---------------------------------------------------------------- Aadhaar (Verhoeff)
# Verhoeff's dihedral-group tables: multiplication (_VD) and permutation (_VP). Unlike Luhn, the check catches
# every swap of two neighbouring digits.
_VD = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 0, 6, 7, 8, 9, 5], [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
       [3, 4, 0, 1, 2, 8, 9, 5, 6, 7], [4, 0, 1, 2, 3, 9, 5, 6, 7, 8], [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
       [6, 5, 9, 8, 7, 1, 0, 4, 3, 2], [7, 6, 5, 9, 8, 2, 1, 0, 4, 3], [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
       [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]]
_VP = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 5, 7, 6, 2, 8, 3, 0, 9, 4], [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
       [8, 9, 1, 6, 0, 4, 3, 5, 2, 7], [9, 4, 5, 3, 1, 2, 6, 8, 7, 0], [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
       [2, 7, 9, 3, 8, 0, 6, 4, 1, 5], [7, 0, 4, 6, 9, 1, 3, 2, 5, 8]]


def verhoeff_ok(digits):
    c = 0
    for i, ch in enumerate(reversed(digits)):
        c = _VD[c][_VP[i % 8][ord(ch) - 48]]
    return c == 0


def aadhaar_ok(digits):
    # Aadhaar numbers never start with 0 or 1, which also rules out zero-padded ids
    return len(digits) == 12 and digits[0] in "23456789" and verhoeff_ok(digits)


# ---------------------------------------------------------------- SSN / PAN
def ssn_ok(digits):
    # The SSA never issues area 000, 666 or 900-999, group 00 or serial 0000. Those values only appear in test
    # data, ITINs and placeholders, so they are rejected.
    if len(digits) != 9:
        return False
    area, group, serial = digits[:3], digits[3:5], digits[5:]
    return area != "000" and area != "666" and area[0] != "9" and group != "00" and serial != "0000"


# Indian PAN: 5 letters, 4 digits, 1 letter. The 4th letter is the holder type (P person, C company, H HUF,
# F firm, A association, T trust, B body of individuals, L local authority, J juridical person, G government),
# which rules out most random 10-character codes such as batch ids.
PAN_RE = re.compile(r"^[A-Z]{3}[PCHFATBLJG][A-Z]\d{4}[A-Z]$")


def pan_ok(value):
    return bool(PAN_RE.match(value))


# ---------------------------------------------------------------- AWS
def aws_ok(value):
    # AKIA = long-term access key id, ASIA = temporary (STS) key id. AWS documentation examples end in "EXAMPLE"
    # (AKIAIOSFODNN7EXAMPLE) and are copied into code and configs everywhere; they are not secrets.
    return bool(re.fullmatch(r"(?:AKIA|ASIA)[A-Z0-9]{16}", value)) and not value.endswith("EXAMPLE")


# ---------------------------------------------------------------- entropy / shapes
def shannon_entropy(s):
    """Bits per character of `s`. Random tokens over [A-Za-z0-9] average about 4 bits at 20 characters and 4.5 at
    32; repetitive values score far lower ("passwordpassword": 2.75). Generic tokens need at least 3.0 bits, along
    with two character classes (see text_detector._generic_token_ok)."""
    if not s:
        return 0.0
    n = len(s)
    return -sum(c / n * math.log2(c / n) for c in Counter(s).values())


HEX_RE = re.compile(r"^[0-9a-fA-F]+$")
# Dashes are required: 32 hex characters without dashes are a common API-key format, and treating them as UUIDs
# rejected real tokens.
UUID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
# Password-hash and digest notations: bcrypt ($2b$12$...), crypt ($1$ / $5$ / $6$), argon2, pbkdf2, and
# "sha256:..." or Subresource-Integrity style "sha512-...". A hash stored under a password key is not the password.
HASH_PREFIX_RE = re.compile(r"^(?:\$2[abxy]?\$\d\d\$|\$argon2|\$[156]\$|\$pbkdf2|pbkdf2_|sha\d*[:$=-]|md5[:$]|"
                            r"bcrypt\$|scrypt\$|sha512-|sha384-|sha256-|sha1-)", re.I)


def is_uuid(v):
    return bool(UUID_RE.match(v))


def is_hash_like(v):
    """Values that are hashes by shape regardless of key: crypt/bcrypt strings and prefixed digests."""
    return bool(HASH_PREFIX_RE.match(v))


def is_hex_digest(v):
    # the hex lengths of MD5, SHA-1, SHA-224, SHA-256, SHA-384 and SHA-512
    return bool(HEX_RE.match(v)) and len(v) in (32, 40, 56, 64, 96, 128)


def char_classes(v):
    return sum((any(c.islower() for c in v), any(c.isupper() for c in v), any(c.isdigit() for c in v)))


# ---------------------------------------------------------------- placeholders / masks
# Values that only stand in for a secret: empty/boolean words, redaction markers, template syntax ({{ x }}, ${X},
# $(cmd), %(x)s), "YOUR_..._HERE", "changeme", runs of x or *. The whole value must match, so a real password that
# merely contains "test" is kept.
_PLACEHOLDER_RE = re.compile(
    r"""^(?:null|none|nil|undefined|n/?a|true|false|yes|no|empty|redacted|\[redacted\]|\[filtered\]|filtered|
        masked|hidden|secret|password|passwd|changeme.*|change[_-]?me.*|your[_-].*|<.*>|\{\{.*\}\}|\$\{.*\}|\$\(.*\)|
        %\(.*\)s|\$[A-Z_][A-Z0-9_]*|.*_HERE|xxx+.*|.*x{6,}.*|\*+|•+|\.{3,}|-+|todo|tbd|placeholder|example|sample|dummy|
        test|testing|default|insert.*|enter.*|replace.*|<redacted>|\*\*\*.*|.*\*{4,}.*)$""",
    re.I | re.X)
# Code that *fetches* a secret (os.environ["API_KEY"], process.env.X, a vault reference) names where the secret
# lives; it is not the secret.
ENV_LOOKUP_RE = re.compile(r"(?:os\.environ|os\.getenv|process\.env|ENV\[|getenv\(|System\.getenv|vault:|secretsmanager:|"
                           r"\$\{|\{\{|\$\()", re.I)


def is_placeholder(v):
    v = v.strip().strip("'\"`")
    if not v:
        return True
    if _PLACEHOLDER_RE.match(v) or ENV_LOOKUP_RE.search(v):
        return True
    # mostly masked ("****-****-1234", "XXX-XX-6789"): what remains visible is too little to be the value
    masked = sum(len(m.group()) for m in re.finditer(r"[*•]+|[xX]{3,}", v))
    return masked >= max(3, len(v) // 2)
