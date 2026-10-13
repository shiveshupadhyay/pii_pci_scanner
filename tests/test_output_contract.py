"""Output contract: required fields, hashes from contract.py, masking never leaks, errors listed, run continues."""

import json
import os

from contract import CATEGORY, value_sha256

from scanner import pipeline
from scanner.output import mask


def test_mask_never_contains_raw():
    cases = [("CARD_NUMBER", "4539 1488 0343 6467"), ("EMAIL", "kavya.singh@gmail.com"), ("PHONE", "+91 98765 43210"),
             ("NATIONAL_ID", "ABCPE1234F"), ("PASSWORD", "Winter2024!"), ("API_TOKEN", "sk_live_79XNSaNl1zkiINe3ayBBzzwW"),
             ("AWS_ACCESS_KEY", "AKIAJAJQUH5BKX7TSRNK"), ("PERSON_NAME", "Priya Sharma"), ("CVV", "7731"),
             ("CARD_EXPIRY", "10/2029"), ("PRIVATE_KEY", "-----BEGIN EC PRIVATE KEY-----\nAAAA\n-----END EC PRIVATE KEY-----")]
    for ent, raw in cases:
        assert raw not in mask(ent, raw)
    assert mask("NATIONAL_ID", "ABCPE1234F").endswith("234F")


def test_run_writes_contract_and_handles_errors(tmp_path):
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "starter"))
    from evaluate import check_findings

    inp = tmp_path / "in"
    (inp / "sub").mkdir(parents=True)
    (inp / "sub" / "notes.txt").write_text("From: Hannah Brown <hannah20@ardentlabs.ai>\n"
                                           "the new password is sarah2010$ - please change it\n", encoding="utf-8")
    (inp / "empty.csv").write_bytes(b"")
    (inp / "random.docx").write_bytes(os.urandom(2048))
    out = tmp_path / "out"
    pipeline.run(str(inp), str(out), workers=1)
    findings = [json.loads(l) for l in (out / "findings.jsonl").read_text(encoding="utf-8").splitlines()]
    errors = [json.loads(l) for l in (out / "errors.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {e["file"] for e in errors} == {"empty.csv", "random.docx"}
    assert all(f["file"] == "sub/notes.txt" for f in findings)
    raw = {value_sha256("PASSWORD", "sarah2010$"): "sarah2010$", value_sha256("EMAIL", "hannah20@ardentlabs.ai"):
           "hannah20@ardentlabs.ai"}
    scope = {"all_files": ["sub/notes.txt", "empty.csv", "random.docx"]}
    problems, valid = check_findings(findings, scope, raw)
    assert not problems and len(valid) == len(findings) == 3
    for f in findings:
        assert f["category"] == CATEGORY[f["entity"]] and 0 <= f["confidence"] <= 1 and f["evidence"]["detector"]
        assert f["location"] == {}
    assert {f["value_sha256"] for f in findings} >= set(raw)
    assert any(f["entity"] == "PERSON_NAME" for f in findings)
