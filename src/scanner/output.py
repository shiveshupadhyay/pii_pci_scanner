"""Output contract: findings / columns / errors records. Raw values are hashed and masked here and nowhere else.

Hashes come from starter/contract.value_sha256, which normalises the value first (digits only for cards and phones,
casefolded names, MM/YY expiries...), so a finding matches its label however the value was written. Masks are for
a human triaging the report: enough to recognise the finding ("**** **** **** 6467", "p****@orbitpay.in"),
never enough to recover the value.
"""

import json
import re

from contract import CATEGORY, value_sha256  # starter/contract.py (put on sys.path by scan.py)

from . import DETECTOR_VERSION

# Prefixes that only say what kind of credential it is (Stripe, GitHub, Slack, AWS...). Showing them helps triage
# and reveals nothing secret.
KNOWN_PREFIX = re.compile(r"^(sk_live_|sk_test_|ghp_|gho_|ghu_|ghs_|ghr_|github_pat_|glpat-|xox[abposr]-|AIza|"
                          r"sk-proj-|eyJ|AKIA|ASIA)")


def _mask_keep_tail(value, keep):
    # Star every letter and digit except the last `keep`; separators stay, so the value keeps its familiar shape
    # ("***-**-1234").
    out, alnum_seen = [], 0
    total = sum(ch.isalnum() for ch in value)
    for ch in value:
        if ch.isalnum():
            alnum_seen += 1
            out.append(ch if alnum_seen > total - keep else "*")
        else:
            out.append(ch)
    return "".join(out)


def mask(entity, value):
    v = value.strip()
    if entity == "PRIVATE_KEY":
        # the BEGIN line says what kind of key it is; the body is never shown
        first = v.splitlines()[0] if v else ""
        return f"{first}[redacted {len(v)} chars]"
    if entity == "EMAIL" and "@" in v:
        # The domain stays visible ("whose address leaked?" is the first triage question); the mailbox does not.
        local, domain = v.split("@", 1)
        return (local[:1] + "*" * max(0, len(local) - 1) if len(local) > 1 else "*") + "@" + domain
    if entity == "PERSON_NAME":
        return " ".join(t[:1] + "*" * (len(t) - 1) for t in v.split())
    if entity == "PASSWORD":
        return "*" * 8  # fixed length: even a password's length is information
    if entity in ("API_TOKEN", "AWS_ACCESS_KEY"):
        # prefix + fixed stars + last 4, the convention of most secret scanners; the last 4 only for long tokens
        m = KNOWN_PREFIX.match(v)
        prefix = m.group(0) if m else ""
        tail = v[-4:] if len(v) - len(prefix) > 12 else ""
        return prefix + "*" * 8 + tail
    # Numbers (cards, phones, IDs): show the last 4 like a card receipt, fewer for short values.
    n = sum(ch.isalnum() for ch in v)
    keep = 4 if n >= 10 else 2 if n >= 6 else 0
    return _mask_keep_tail(v, keep)


def finding_record(f):
    value = f.value
    masked = mask(f.entity, value)
    # Defence in depth: whatever the masking rules above do (or a future edit breaks), a mask that still contains
    # the whole value is replaced by stars.
    if len(value) >= 4 and value in masked:
        masked = "*" * len(masked)
    return {
        "file": f.file,
        "entity": f.entity,
        "category": CATEGORY[f.entity],
        "value_sha256": value_sha256(f.entity, value),
        "value_masked": masked,
        "location": f.location,
        "confidence": round(float(min(max(f.confidence, 0.0), 1.0)), 3),
        "evidence": {"extractor": f.extractor, "detector": f.detector, "context": list(f.context)},
        "detector_version": DETECTOR_VERSION,
    }


def column_record(c):
    return {"file": c.file, "sheet": c.sheet or "", "column": c.column, "entity": c.entity,
            "confidence": round(float(c.confidence), 3)}


def write_jsonl(path, records):
    # ASCII-only JSON with "\n" line ends gives the same bytes on every OS. That matters because
    # starter/evaluate.py opens these files without an encoding, which means cp1252 on Windows: any
    # raw UTF-8 here (an accented name, a non-ASCII file or column name) could be misread or crash it.
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
