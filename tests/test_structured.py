"""Structured extraction: dotted JSON paths, XLSX sheets, column classification, large-table sampling."""

import csv
import json

from scanner import pipeline
from scanner.detect.columns import classify_table
from scanner.extract import structured


def test_json_paths_use_dots_and_brackets(tmp_path):
    p = tmp_path / "a.json"
    p.write_text(json.dumps([{"customer": {"contact": {"email": "a@b.io"}}, "orders": [{"card": "1"}, {"card": "2"}]},
                             {"customer": {"contact": {"email": "c@d.io"}}, "orders": []}]), encoding="utf-8")
    (t,) = structured.json_tables(str(p))
    assert "customer.contact.email" in t.columns and "orders[].card" in t.columns
    ci = t.columns.index("orders[].card")
    assert sorted(r[ci] for r in t.rows if r[ci]) == ["1", "2"]


def test_json_top_level_object(tmp_path):
    p = tmp_path / "b.json"
    p.write_text(json.dumps({"data": [{"email": "a@b.io"}], "db": {"password": "x"}}), encoding="utf-8")
    (t,) = structured.json_tables(str(p))
    assert set(t.columns) == {"data[].email", "db.password"}


def _scan(tmp_path, name, writer):
    inp = tmp_path / "in"
    inp.mkdir(exist_ok=True)
    writer(inp / name)
    out = tmp_path / "out"
    pipeline.run(str(inp), str(out), workers=1)
    f = [json.loads(l) for l in (out / "findings.jsonl").read_text(encoding="utf-8").splitlines()]
    c = [json.loads(l) for l in (out / "columns.jsonl").read_text(encoding="utf-8").splitlines()]
    return f, c


def test_column_labels_and_lookalikes(tmp_path):
    def w(p):
        with open(p, "w", encoding="utf-8", newline="") as fh:
            x = csv.writer(fh)
            x.writerow(["mobile", "order_id", "card_last4", "support_email", "valid_thru", "col_7", "password"])
            for i in range(10):
                x.writerow([f"98765432{i:02d}", "4828894923646838", "1234", "support@example.com", "10/26",
                            "4539148803436467", "null" if i == 0 else f"Pw{i}xY!z"])

    findings, cols = _scan(tmp_path, "t.csv", w)
    labelled = {(c["column"], c["entity"]) for c in cols}
    assert labelled == {("mobile", "PHONE"), ("col_7", "CARD_NUMBER"), ("valid_thru", "CARD_EXPIRY"),
                        ("password", "PASSWORD")}
    assert all(f["location"] in ({"column": "mobile"}, {"column": "col_7"}, {"column": "valid_thru"},
                                 {"column": "password"}) for f in findings)
    assert sum(f["entity"] == "PASSWORD" for f in findings) == 9  # "null" is a placeholder


def test_expiry_column_without_card_column_is_not_labelled(tmp_path):
    def w(p):
        p.write_text("name,exp\nA B,10/26\nC D,11/27\n", encoding="utf-8")

    findings, cols = _scan(tmp_path, "t.csv", w)
    assert not any(f["entity"] == "CARD_EXPIRY" for f in findings)


def test_xlsx_sheet_in_location(tmp_path):
    import openpyxl

    def w(p):
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Customers"
        ws.append(["name", "pan"])
        ws.append(["Priya Sharma", "ABCPE1234F"])
        ws2 = wb.create_sheet("Vendors")
        ws2.append(["contact_email"])
        ws2.append(["v@vendor.io"])
        wb.save(p)

    findings, cols = _scan(tmp_path, "t.xlsx", w)
    locs = {(f["entity"], f["location"]["sheet"], f["location"]["column"]) for f in findings}
    assert ("NATIONAL_ID", "Customers", "pan") in locs and ("EMAIL", "Vendors", "contact_email") in locs
    assert ("PERSON_NAME", "Customers", "name") in locs


def test_consensus_guard_drops_coincidental_checksums(det):
    # random 12-digit account numbers: ~10% pass Verhoeff by chance and must not become Aadhaar findings
    import random

    r = random.Random(1)
    rows = [[str(r.randint(2 * 10 ** 11, 9 * 10 ** 11))] for _ in range(60)]
    t = structured.Table("", ["account_no"], rows, len(rows))
    findings, labels = classify_table(det, t)
    assert not findings and not labels


def test_large_table_is_column_only(tmp_path):
    def w(p):
        with open(p, "w", encoding="utf-8", newline="\n") as fh:
            fh.write("email,city\n")
            for i in range(structured.LARGE_ROWS + 5):
                fh.write(f"user{i}@bluepeak.com,Pune\n")

    old = structured.LARGE_ROWS
    findings, cols = _scan(tmp_path, "big.csv", w)
    assert old == structured.LARGE_ROWS
    assert findings == [] and [(c["column"], c["entity"]) for c in cols] == [("email", "EMAIL")]
