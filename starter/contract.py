"""Output contract helpers. Use these exactly - the evaluator matches findings by `value_sha256`.

    from contract import value_sha256, CATEGORY
    h = value_sha256("CARD_NUMBER", "4539 1488 0343 6467")   # same hash as "4539148803436467"

Normalisation (applied before hashing):
    EMAIL            strip, lowercase
    PHONE            digits only - of the value as written, including any country code
    PERSON_NAME      casefold, all whitespace removed          ("Priya  Sharma" == "PriyaSharma")
    NATIONAL_ID      uppercase, non-alphanumerics removed
    CARD_NUMBER      digits only
    CARD_EXPIRY      MM/YY                                      ("8/2027" -> "08/27")
    CVV              digits only
    AWS_ACCESS_KEY   strip, uppercase
    PRIVATE_KEY      all whitespace removed (whole BEGIN ... END block)
    API_TOKEN        strip
    PASSWORD         strip
"""

import hashlib
import re

CATEGORY = {"EMAIL": "PII", "PHONE": "PII", "PERSON_NAME": "PII", "NATIONAL_ID": "PII",
            "CARD_NUMBER": "PCI", "CARD_EXPIRY": "PCI", "CVV": "PCI",
            "AWS_ACCESS_KEY": "SECRET", "PRIVATE_KEY": "SECRET", "API_TOKEN": "SECRET", "PASSWORD": "SECRET"}
LABELS = list(CATEGORY)


def normalize(entity, value):
    v = str(value)
    if entity == "EMAIL":
        return v.strip().lower()
    if entity in ("PHONE", "CARD_NUMBER", "CVV"):
        return re.sub(r"\D", "", v)
    if entity == "PERSON_NAME":
        return re.sub(r"\s+", "", v).casefold()
    if entity == "NATIONAL_ID":
        return re.sub(r"[^A-Za-z0-9]", "", v).upper()
    if entity == "CARD_EXPIRY":
        m = re.search(r"(\d{1,2})\D+(\d{2,4})", v)
        return f"{int(m.group(1)):02d}/{int(m.group(2)) % 100:02d}" if m else v.strip()
    if entity == "AWS_ACCESS_KEY":
        return v.strip().upper()
    if entity == "PRIVATE_KEY":
        return re.sub(r"\s+", "", v)
    return v.strip()


def value_sha256(entity, value):
    return hashlib.sha256(f"{entity}|{normalize(entity, value)}".encode()).hexdigest()


def location_key(location):
    """Canonical location used for matching: page for PDFs/images, sheet+column for structured files."""
    location = location or {}
    if "column" in location:
        return f"{location.get('sheet', '')}/{location['column']}"
    if "page" in location:
        return f"p{int(location['page'])}"
    return ""
