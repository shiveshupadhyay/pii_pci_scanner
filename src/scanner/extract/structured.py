"""Structured files -> Table objects (column name -> list of cell strings).

CSV/TSV (encoding + delimiter sniffed), Parquet (pyarrow, batches), XLSX (all sheets, read-only), JSON / JSONL
(flattened to dotted paths, `[]` for list elements; top-level list elements are records).
Tables with more than LARGE_ROWS rows are column-only in the scoring: they are sampled (head + random rows)
and only produce columns.jsonl entries.
"""

import csv
import io
import json
import os
import random
import zlib
from dataclasses import dataclass, field

from ..filetype import decode_bytes

# The scoring treats tables over 50k rows as column-only, so for them a sample is enough: the head plus random
# rows from the rest, so that a column whose content changes deep in the file is still seen.
LARGE_ROWS = 50_000
SAMPLE_HEAD = 5_000
SAMPLE_RANDOM = 5_000
# Allow cells up to 16 MB (embedded JSON, base64 blobs). The default limit is 128 KB. sys.maxsize would overflow
# the C long on Windows, so use a fixed size.
csv.field_size_limit(16 * 1024 * 1024)


@dataclass
class Table:
    sheet: str
    columns: list
    rows: list = field(default_factory=list)  # list of lists aligned with columns (sampled when large)
    n_rows: int = 0
    large: bool = False


def _cell(v):
    """A typed cell (XLSX, Parquet, JSON) as the string a person would have typed."""
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        # Spreadsheets store phone and account numbers as floats; 9876543210.0 must come back as "9876543210"
        # or no phone or card rule can match it.
        return str(int(v)) if v.is_integer() and abs(v) < 1e17 else repr(v)
    if isinstance(v, (bytes, bytearray)):
        return v.decode("utf-8", "replace")
    if hasattr(v, "isoformat"):
        return v.isoformat()
    return str(v)


def _rng(path):
    # Seeded from the file name with crc32, not hash(): str hashes are salted per process, which made the sampled
    # rows (and so the column confidences) change from run to run and between fork and spawn workers.
    return random.Random(zlib.crc32(os.path.basename(path).encode("utf-8")))


# ---------------------------------------------------------------- CSV / TSV


def _count_lines(path):
    # Counting newlines in 4 MB binary blocks runs at about disk speed, far cheaper than parsing the file only to
    # find out it is too big to parse.
    n = 0
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 22)
            if not b:
                break
            n += b.count(b"\n")
    return n


def csv_tables(path, delimiter=",", encoding=None, gz=False):
    size = os.path.getsize(path)
    if gz:
        import gzip

        with gzip.open(path, "rb") as f:
            text, _ = decode_bytes(f.read())
        return [_csv_from_text(text, delimiter)]
    # Files under 4 MB are cheap to parse whole, so they skip the line count; their real row count (which can still
    # exceed 50k for very short rows) comes from the parse.
    approx_rows = _count_lines(path) - 1 if size > 4 * 1024 * 1024 else None
    if approx_rows is None or approx_rows <= LARGE_ROWS:
        with open(path, "rb") as f:
            text, _ = decode_bytes(f.read(), None if encoding in (None, "utf-8") else encoding)
        t = _csv_from_text(text, delimiter)
        t.large = t.n_rows > LARGE_ROWS
        return [t]
    # large: header + head rows + random seeks
    with open(path, "rb") as f:
        head = f.read(1 << 22)
    text, enc = decode_bytes(head)
    text = text[:text.rfind("\n") + 1] if "\n" in text else text  # drop the row the 4 MB read cut in half
    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    columns = next(reader, [])
    rows = []
    for r in reader:
        rows.append(r)
        if len(rows) >= SAMPLE_HEAD:
            break
    rng = _rng(path)
    with open(path, "rb") as f:
        for _ in range(SAMPLE_RANDOM):
            # Seek to a random byte, throw away the partial line we landed in, keep the next whole one. Rows that
            # contain quoted newlines can be misparsed this way; the column-count check below discards those.
            f.seek(rng.randrange(len(head), size))
            f.readline()
            line = f.readline()
            if not line:
                continue
            try:
                r = next(csv.reader([line.decode(enc or "utf-8", "replace").rstrip("\r\n")], delimiter=delimiter))
            except (csv.Error, StopIteration):
                continue
            if len(r) == len(columns):
                rows.append(r)
    return [Table("", columns, rows, approx_rows, True)]


def _csv_from_text(text, delimiter):
    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    columns = next(reader, [])
    rows = [r for r in reader if any(c.strip() for c in r)]
    return Table("", columns, rows, len(rows), len(rows) > LARGE_ROWS)


# ---------------------------------------------------------------- Parquet


def _flatten_arrow_names(schema):
    """Column names for a Parquet schema in label format: nested structs become dotted paths and list elements
    get "[]" (customer.cards[].number), the same convention as for JSON."""
    import pyarrow as pa

    out = []

    def walk(f, prefix):
        name = f"{prefix}.{f.name}" if prefix else f.name
        t = f.type
        if pa.types.is_struct(t):
            for sub in t:
                walk(sub, name)
        elif pa.types.is_list(t) or pa.types.is_large_list(t):
            vt = t.value_type
            if pa.types.is_struct(vt):
                for sub in vt:
                    walk(sub, name + "[]")
            else:
                out.append(name + "[]")
        else:
            out.append(name)

    for f in schema:
        walk(f, "")
    return out


def _flatten_record(obj, prefix, out):
    if isinstance(obj, dict):
        for k, v in obj.items():
            _flatten_record(v, f"{prefix}.{k}" if prefix else str(k), out)
    elif isinstance(obj, list):
        for v in obj:
            _flatten_record(v, prefix + "[]", out)
    else:
        out.append((prefix, obj))


def parquet_tables(path):
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    n = pf.metadata.num_rows  # free: the row count sits in the footer, so a 1M-row file costs nothing to size
    columns = _flatten_arrow_names(pf.schema_arrow)
    large = n > LARGE_ROWS
    limit = SAMPLE_HEAD + SAMPLE_RANDOM if large else n
    rows = []
    nested = len(columns) != len(pf.schema_arrow.names) or any("[]" in c or "." in c for c in columns)
    groups = list(range(pf.num_row_groups))
    if large and len(groups) > 2:
        # Row groups are Parquet's unit of reading: sample the first group plus three random ones, rather than
        # decoding the whole file to pick rows.
        rng = _rng(path)
        groups = [groups[0]] + sorted(rng.sample(groups[1:], min(len(groups) - 1, 3)))
    per_group = max(1, limit // max(1, len(groups)))
    for g in groups:
        taken = 0
        for batch in pf.iter_batches(batch_size=min(per_group, 65536), row_groups=[g]):
            for rec in batch.to_pylist():
                if nested:
                    flat = []
                    _flatten_record(rec, "", flat)
                    d = {}
                    for k, v in flat:
                        d.setdefault(k, []).append(v)
                    rows.append({k: v for k, v in d.items()})
                else:
                    rows.append([_cell(rec.get(c)) for c in columns])
                taken += 1
                if taken >= per_group:
                    break
            if taken >= per_group:
                break
    if nested:  # list-valued paths: one row per record, cells may repeat
        return [_records_table("", rows, n, large)]
    return [Table("", columns, rows, n, large)]


def _records_table(sheet, recs, n, large):
    """recs: list of {path: [values]} -> Table with list-cells expanded into separate rows per path."""
    columns = []
    seen = set()
    for r in recs:
        for k in r:
            if k not in seen:
                seen.add(k)
                columns.append(k)
    idx = {c: i for i, c in enumerate(columns)}
    rows = []
    for r in recs:
        depth = max((len(v) for v in r.values()), default=0)
        for d in range(max(1, depth)):
            row = [""] * len(columns)
            for k, vals in r.items():
                if d < len(vals):
                    row[idx[k]] = _cell(vals[d])
            rows.append(row)
    return Table(sheet, columns, rows, n, large)


# ---------------------------------------------------------------- XLSX


def xlsx_tables(path):
    import openpyxl

    # read_only streams rows instead of building the whole workbook in memory; data_only returns the cached values
    # of formula cells rather than the formulas themselves.
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    tables = []
    try:
        for ws in wb.worksheets:
            it = ws.iter_rows(values_only=True)
            header = None
            for r in it:  # the header is the first non-empty row (sheets often start with blank rows or a title)
                if r is not None and any(c not in (None, "") for c in r):
                    header = [_cell(c).strip() for c in r]
                    break
            if not header:
                continue
            # max_row comes from the sheet's stored dimension, so it is free; it can be missing (0) or approximate,
            # which is why the real count below can override it.
            dim_rows = (ws.max_row or 0) - 1
            rows, n = [], 0
            large_hint = dim_rows > LARGE_ROWS
            for r in it:
                if r is None or not any(c not in (None, "") for c in r):
                    continue
                n += 1
                if not large_hint or len(rows) < SAMPLE_HEAD + SAMPLE_RANDOM:
                    rows.append([_cell(c) for c in r][:len(header)])
                elif large_hint:
                    break
            if large_hint:
                n = max(n, dim_rows)
            while header and not header[-1]:  # formatting often extends a sheet with empty trailing columns
                header.pop()
            tables.append(Table(ws.title, header, rows, n, n > LARGE_ROWS))
    finally:
        wb.close()
    return tables


# ---------------------------------------------------------------- JSON / JSONL


def _iter_json_array(path, encoding=None):
    """Stream the elements of a top-level JSON array without loading the whole file."""
    # raw_decode() parses one value from a position in a string, so the file can be read in 1 MB pieces: decode the
    # next element, and if it runs past the end of the buffer, read more and try again. Consumed text is dropped
    # regularly, so memory stays around the size of one buffer.
    dec = json.JSONDecoder()
    with open(path, "r", encoding=encoding or "utf-8", errors="replace") as f:
        buf = f.read(1 << 20)
        pos = buf.find("[") + 1
        while True:
            while True:
                while pos < len(buf) and buf[pos] in " \t\r\n,":
                    pos += 1
                if pos < len(buf):
                    break
                more = f.read(1 << 20)
                if not more:
                    return
                buf, pos = buf[pos:] + more, 0
            if buf[pos] == "]":
                return
            while True:
                try:
                    obj, end = dec.raw_decode(buf, pos)
                    break
                except ValueError:
                    more = f.read(1 << 20)
                    if not more:
                        return
                    buf, pos = buf[pos:] + more, 0
            yield obj
            pos = end
            if pos > (1 << 20):
                buf, pos = buf[pos:], 0


def json_tables(path, jsonl=False, encoding=None):
    size = os.path.getsize(path)
    recs, n, large = [], 0, False
    if jsonl:
        approx = _count_lines(path) if size > 4 * 1024 * 1024 else None
        if approx is not None and approx > LARGE_ROWS:
            large, n = True, approx
            rng = _rng(path)
            with open(path, "rb") as f:
                for _ in range(SAMPLE_HEAD):
                    line = f.readline()
                    if not line:
                        break
                    _add_json_line(line, recs)
                for _ in range(SAMPLE_RANDOM):
                    # one record per line, so "seek, skip the partial line, read the next" lands on a whole record
                    f.seek(rng.randrange(0, size))
                    f.readline()
                    _add_json_line(f.readline(), recs)
        else:
            with open(path, "rb") as f:
                for line in f:
                    if _add_json_line(line, recs):
                        n += 1
    else:
        if size > 32 * 1024 * 1024:
            for obj in _iter_json_array(path, encoding):
                n += 1
                if len(recs) < SAMPLE_HEAD + SAMPLE_RANDOM:
                    recs.append(_flat(obj))
            large = n > LARGE_ROWS
        else:
            with open(path, "rb") as f:
                data, _ = decode_bytes(f.read())
            obj = json.loads(data)
            if isinstance(obj, list):
                recs = [_flat(o) for o in obj]
            else:
                recs = [_flat(obj)]
            n = len(recs)
            large = n > LARGE_ROWS
    return [_records_table("", recs, n, large)]


def _flat(obj):
    """{dotted path: [values]} for one record. Several values per path happen for list elements
    (orders[].card); _records_table() spreads them over rows. A bare scalar record gets the path "value"."""
    flat = []
    _flatten_record(obj, "", flat)
    d = {}
    for k, v in flat:
        d.setdefault(k or "value", []).append(v)
    return d


def _add_json_line(line, recs):
    line = line.strip()
    if not line:
        return False
    try:
        obj = json.loads(line)
    except ValueError:
        return False
    recs.append(_flat(obj))
    return True
