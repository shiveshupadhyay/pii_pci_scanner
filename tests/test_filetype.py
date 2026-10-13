"""Routing by content, never by extension; unreadable files become errors."""

import gzip
import json
import os

import pytest

from scanner.filetype import sniff


def write(tmp_path, name, data):
    p = tmp_path / name
    p.write_bytes(data if isinstance(data, bytes) else data.encode())
    return str(p)


def make_pdf(path, text="hello world", **save_kw):
    import pymupdf

    doc = pymupdf.open()
    doc.new_page().insert_text((50, 50), text)
    doc.save(path, **save_kw)
    doc.close()


def make_png(path):
    from PIL import Image

    Image.new("RGB", (40, 20), "white").save(path, "PNG")


def test_misleading_extensions(tmp_path):
    csv_txt = write(tmp_path, "leads.txt", "name,email,phone\nA B,a@x.io,9876543210\nC D,c@x.io,9876543211\n")
    jsonl_log = write(tmp_path, "users.log", "\n".join(json.dumps({"id": i, "email": f"u{i}@x.io"}) for i in range(5)))
    pdf_dat = str(tmp_path / "invoice.dat")
    make_pdf(pdf_dat)
    png_pdf = str(tmp_path / "screen.pdf")
    make_png(png_pdf)
    tsv = write(tmp_path, "export.dat", "name\temail\tphone\nA B\ta@x.io\t1\nC D\tc@x.io\t2\n")
    assert sniff(csv_txt).kind == "csv" and sniff(csv_txt).meta["delimiter"] == ","
    assert sniff(jsonl_log).kind == "jsonl"
    assert sniff(pdf_dat).kind == "pdf"
    assert sniff(png_pdf).kind == "image"
    assert sniff(tsv).kind == "csv" and sniff(tsv).meta["delimiter"] == "\t"


def test_text_structures(tmp_path):
    ini = write(tmp_path, "service.ini", "[default]\napi_key = x\npassword = y\n")
    html = write(tmp_path, "page.txt", "<html><body><table><tr><td>a</td></tr></table><p>x</p></body></html>")
    sql = write(tmp_path, "dump.txt", "-- PostgreSQL database dump\nINSERT INTO users (id, email) VALUES (1, 'a@b.io');\n")
    eml = write(tmp_path, "mail.txt", "From: A <a@b.io>\nTo: B <b@c.io>\nSubject: x\nMIME-Version: 1.0\n"
                                      "Content-Type: text/html\n\n<html><body><p>hi</p><div>x</div><span>y</span></body></html>")
    log = write(tmp_path, "app.log", "2025-01-01T00:00:00Z INFO [a] started, pid=1\n2025-01-01T00:00:01Z INFO [a] ok\n")
    assert sniff(ini).kind == "text"
    assert sniff(html).kind == "html"
    assert sniff(sql).kind == "sql"
    assert sniff(eml).kind == "eml"          # a MIME message with an HTML body is still an e-mail
    assert sniff(log).kind == "text"


def test_gzip_is_sniffed_inside(tmp_path):
    p = tmp_path / "rotated.log.gz"
    with gzip.open(p, "wb") as f:
        f.write(b"2025-01-01 INFO user login email=a@b.io\n")
    s = sniff(str(p))
    assert s.kind == "text" and s.meta.get("gzip")


@pytest.mark.parametrize("maker,reason", [
    (lambda d: write(d, "empty.csv", b""), "empty"),
    (lambda d: write(d, "blank.txt", b"   \n\n"), "empty"),
    (lambda d: write(d, "random.docx", os.urandom(4096)), "unrecognized binary"),
    (lambda d: write(d, "corrupt.xlsx", b"PK\x03\x04" + os.urandom(2000)), "corrupt zip"),
])
def test_unreadable(tmp_path, maker, reason):
    s = sniff(maker(tmp_path))
    assert s.error and reason in s.error


def test_truncated_and_encrypted_pdf(tmp_path):
    import pymupdf

    full = str(tmp_path / "full.pdf")
    make_pdf(full)
    data = open(full, "rb").read()
    trunc = write(tmp_path, "trunc.pdf", data[:200])
    assert "truncated" in sniff(trunc).error
    enc = str(tmp_path / "enc.pdf")
    make_pdf(enc, encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="u", owner_pw="o")
    from scanner.extract.pdf import open_pdf

    assert sniff(enc).kind == "pdf"
    doc, err = open_pdf(enc)
    assert doc is None and err == "encrypted pdf"
