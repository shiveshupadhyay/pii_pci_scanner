"""File type detection from content (magic bytes, container structure, text structure) - never from the extension.

sniff(path) -> Sniff(kind, error, meta). Kinds:
  pdf, image, parquet, xlsx, docx, pptx, csv (incl. TSV; meta['delimiter']), json, jsonl, eml, html, xml, sql, text
`error` is a short reason for unreadable files (empty, truncated, encrypted, corrupt, unrecognised binary).

Why content and not extensions: real directories (and the dev set) contain PDFs named .dat, PNGs named .pdf,
CSVs named .txt and JSONL named .log. The location of a finding has to follow the real type, so a misrouted file
costs every finding in it. The order of the checks below is part of the design: binary signatures first, then
containers, then the structure of decoded text, from the most specific format to the most generic.
"""

import csv
import gzip
import io
import json
import os
import re
import zipfile
from dataclasses import dataclass, field

HEAD = 64 * 1024  # enough to see headers, a CSV header plus ~50 rows, or the start of a JSON document


@dataclass
class Sniff:
    kind: str
    error: str = None
    meta: dict = field(default_factory=dict)


def decode_bytes(data, hint=None):
    """Decode with BOM / UTF-8 / charset-normalizer fallback. Returns (text, encoding)."""
    # A BOM is the only certain signal; Excel-exported CSVs and Windows tools often write one.
    if data.startswith(b"\xef\xbb\xbf"):
        return data[3:].decode("utf-8", "replace"), "utf-8-sig"
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16", "replace"), "utf-16"
    if hint:
        return data.decode(hint, "replace"), hint
    # Strict UTF-8 next: random non-UTF-8 bytes almost never form valid UTF-8, so success here is trustworthy.
    try:
        return data.decode("utf-8"), "utf-8"
    except UnicodeDecodeError as e:
        if e.start > len(data) - 4:  # head cut in the middle of a multi-byte char
            return data[:e.start].decode("utf-8", "replace"), "utf-8"
    # Only then guess (cp1252, Shift-JIS, ...). Statistical detection is slower and can be wrong, hence last.
    try:
        from charset_normalizer import from_bytes

        best = from_bytes(data[:HEAD]).best()
        if best is not None and best.encoding:
            return data.decode(best.encoding, "replace"), best.encoding
    except Exception:
        pass
    # latin-1 maps every byte to a character, so this never fails; at worst a few characters are wrong
    return data.decode("latin-1"), "latin-1"


def _printable_ratio(data):
    """Share of text-like bytes, with control bytes counted against it four times over. Bytes >= 128 count as
    text, so UTF-8 in any script passes, while binary formats full of NULs and control bytes fall well below
    the 0.85 cut-off used by sniff()."""
    if not data:
        return 1.0
    sample = data[:8192]
    good = sum(32 <= b < 127 or b in (9, 10, 13) or b >= 128 for b in sample)
    ctrl = sum(b < 9 or 13 < b < 32 for b in sample)
    return (good - 4 * ctrl) / len(sample)


def sniff(path):
    try:
        size = os.path.getsize(path)
    except OSError as e:
        return Sniff("error", f"unreadable: {e.strerror}")
    if size == 0:
        return Sniff("error", "empty file")
    with open(path, "rb") as f:
        head = f.read(HEAD)
        if size > HEAD:
            f.seek(max(0, size - 4096))
            tail = f.read()
        else:
            tail = head[-4096:]
    if not head.strip():
        return Sniff("error", "empty file (whitespace only)")

    # ---------------------------------------------------------------- binary containers
    # The PDF header may legally sit anywhere in the first 1 KB (some generators prepend junk).
    pdf_at = head.find(b"%PDF-", 0, 1024)
    if pdf_at != -1:
        # MuPDF silently "repairs" a truncated PDF and returns whatever pages it can, so truncation has to be
        # caught here: a complete file ends with a cross-reference pointer and %%EOF.
        if b"%%EOF" not in tail and b"startxref" not in tail:
            return Sniff("pdf", "truncated pdf (no startxref/%%EOF)")
        return Sniff("pdf")
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return Sniff("image", meta={"format": "png"})
    if head.startswith(b"\xff\xd8\xff"):
        return Sniff("image", meta={"format": "jpeg"})
    if head.startswith((b"GIF87a", b"GIF89a")):
        return Sniff("image", meta={"format": "gif"})
    if head.startswith((b"II*\x00", b"MM\x00*")):
        return Sniff("image", meta={"format": "tiff"})
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return Sniff("image", meta={"format": "webp"})
    # "BM" alone is too weak (a text file can start with it); a real BMP also stores its own size at offset 2.
    if head[:2] == b"BM" and len(head) > 26 and int.from_bytes(head[2:6], "little") in (size, 0):
        return Sniff("image", meta={"format": "bmp"})
    if head.startswith(b"PAR1"):
        # Parquet keeps its schema in a footer that ends with the magic again; without it nothing can be read.
        if not tail.endswith(b"PAR1"):
            return Sniff("parquet", "truncated parquet (missing footer)")
        return Sniff("parquet")
    if head.startswith(b"PK\x03\x04") or head.startswith(b"PK\x05\x06"):
        return _sniff_zip(path)
    if head.startswith(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"):
        # OLE2 is both the legacy .doc/.xls format and the wrapper Office uses for password-protected files
        # (their encrypted payload is stored in a stream named EncryptedPackage).
        if "EncryptedPackage".encode("utf-16-le") in head or b"EncryptedPackage" in head:
            return Sniff("error", "encrypted office document")
        return Sniff("error", "unsupported legacy office (OLE2) format")
    if head.startswith(b"\x1f\x8b"):
        # gzip (rotated logs, compressed exports): sniff what is inside and remember to decompress when reading
        try:
            with gzip.open(path, "rb") as g:
                inner = g.read(HEAD)
        except (OSError, EOFError) as e:
            return Sniff("error", f"corrupt gzip: {e}")
        k = _sniff_text(inner)
        k.meta["gzip"] = True
        return k
    if _printable_ratio(head) < 0.85:
        return Sniff("error", "unrecognized binary content")
    return _sniff_text(head)


def _sniff_zip(path):
    try:
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            # testzip() reads every member and checks its CRC: a damaged XLSX/DOCX is reported as an error here
            # instead of failing half-way through extraction
            bad = z.testzip()
            if bad is not None:
                return Sniff("error", f"corrupt zip member {bad}")
    except (zipfile.BadZipFile, OSError, EOFError, ValueError) as e:
        return Sniff("error", f"corrupt zip container ({type(e).__name__})")
    # Office Open XML files are ZIPs; the top-level folder of their members tells which application wrote them
    if any(n.startswith("word/") for n in names):
        return Sniff("docx")
    if any(n.startswith("xl/") for n in names):
        return Sniff("xlsx")
    if any(n.startswith("ppt/") for n in names):
        return Sniff("pptx")
    return Sniff("error", "unsupported zip archive")


_HEADER_LINE = re.compile(r"^[A-Za-z][A-Za-z0-9\-]{0,40}:[ \t]")
_MIME_HEADERS = ("mime-version", "content-type", "content-transfer-encoding", "message-id", "received",
                 "return-path", "delivered-to", "x-mailer", "date")
_HTML_TAGS = re.compile(r"<(?:html|head|body|div|table|tr|td|th|p|span|br|ul|li|a|h[1-6]|form|input|label)\b", re.I)
_SQL = re.compile(r"(?im)^\s*(?:INSERT\s+INTO|CREATE\s+TABLE|COPY\s+[\w.\"]+(?:\s*\([^)]*\))?\s+FROM\s+stdin|"
                  r"--\s*(?:MySQL|PostgreSQL|SQLite)\s+(?:database\s+)?dump)")
_HEADER_FIELD = re.compile(r"^[A-Za-z_\"'][\w .\-/()#\"']{0,60}$")


def _sniff_text(head):
    text, enc = decode_bytes(head)
    meta = {"encoding": enc}
    s = text.lstrip("﻿ \t\r\n")
    # JSON: an object, or "[" followed by something a JSON array can actually contain. A bare "[" is not enough:
    # INI/TOML files start with "[section]".
    if s[:1] == "{" or re.match(r"\[\s*(?:[{\[\"\d\-]|true\b|false\b|null\b|\])", s):
        lines = [l for l in s.splitlines() if l.strip()][:10]
        # JSONL when the first lines each parse as a complete object on their own. The last line read is skipped,
        # because the 64 KB head may have cut it.
        if len(lines) >= 2 and lines[0].lstrip().startswith("{") and all(_json_obj(l) for l in lines[:-1][:6]):
            return Sniff("jsonl", meta=meta)
        if len(lines) == 1 and _json_obj(lines[0]):
            return Sniff("jsonl" if s.strip().count("\n") else "json", meta=meta)
        return Sniff("json", meta=meta)
    low = s[:4096].lower()
    if _looks_eml(s):  # before HTML: a MIME message may carry an HTML part
        return Sniff("eml", meta=meta)
    if low.startswith("<?xml") and "<html" not in low:
        return Sniff("xml", meta=meta)
    # HTML fragments without <html> still count once several common tags appear
    if low.startswith(("<!doctype html", "<html")) or len(_HTML_TAGS.findall(s[:8192])) >= 4:
        return Sniff("html", meta=meta)
    if _SQL.search(s[:HEAD]):
        return Sniff("sql", meta=meta)
    # CSV/TSV is checked last among the structured formats because almost any text has some commas
    delim = _csv_delimiter(s)
    if delim:
        meta["delimiter"] = delim
        return Sniff("csv", meta=meta)
    return Sniff("text", meta=meta)


def _json_obj(line):
    try:
        return isinstance(json.loads(line), (dict, list))
    except ValueError:
        return False


def _looks_eml(s):
    """An RFC 822 header block: only "Name: value" lines (and folded continuations) up to the first blank line, at
    least three of them, including From or Received and a MIME/transport header. That last condition keeps
    "Name: value" notes and configs (which also start with header-like lines) out."""
    hdrs = []
    for line in s.splitlines()[:60]:
        if not line.strip():
            break
        if _HEADER_LINE.match(line):
            hdrs.append(line.split(":", 1)[0].lower())
        elif line[:1] in " \t" and hdrs:
            continue  # folded header
        else:
            return False
    return len(hdrs) >= 3 and ("from" in hdrs or "received" in hdrs) and any(h in _MIME_HEADERS for h in hdrs)


def _csv_delimiter(s):
    """The delimiter of a CSV/TSV file, or None for plain text.

    Two tests must both pass:
      * at least 90% of the first ~50 lines split into the same number of fields;
      * the first row looks like a header: short, identifier-like names, not numbers or dates.
    The header test is what keeps logs out: "2025-01-01 INFO a, b" can split consistently on commas, but its
    first row is not a list of column names. When several delimiters pass, the one giving the most columns wins.
    """
    lines = [l for l in s.splitlines()[:51] if l.strip()]
    if len(lines) < 2:
        return None
    best = None
    for d in (",", "\t", ";", "|"):
        if d not in lines[0]:
            continue
        try:
            # the last line is left out when possible: the 64 KB head may have cut it mid-row
            rows = list(csv.reader(io.StringIO("\n".join(lines[:-1] if len(lines) > 2 else lines)), delimiter=d))
        except csv.Error:
            continue
        n = len(rows[0])
        if n < 2:
            continue
        same = sum(len(r) == n for r in rows) / len(rows)
        header_ok = sum(bool(_HEADER_FIELD.match(f.strip())) and not re.fullmatch(r"[\d.\-+: /]+", f.strip())
                        for f in rows[0]) >= max(2, int(0.8 * n))
        if same >= 0.9 and header_ok and (best is None or n > best[1]):
            best = (d, n)
    return best[0] if best else None
