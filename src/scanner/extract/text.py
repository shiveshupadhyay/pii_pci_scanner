"""Plain text (logs, configs, code, notes, key files): whole-file read or newline-aligned chunks with overlap.

Chunk ownership rule: a chunk task scans [start - OVERLAP, end + OVERLAP] but keeps only candidates whose start
offset lies in [start, end). Any value shorter than OVERLAP is therefore seen whole by exactly one owner: no misses
at boundaries and no duplicates, by construction.
"""

import gzip

from ..filetype import decode_bytes

# 8 MB chunks keep each task's text (and the regex work on it) small, while giving a 150 MB log ~19 tasks to
# spread over the workers. The 64 KB overlap must exceed the longest value a detector can match: PEM blocks are
# capped at 20 KB by PEM_RE, everything else is far shorter.
CHUNK_BYTES = 8 * 1024 * 1024
OVERLAP = 64 * 1024
MAX_GZIP_BYTES = 512 * 1024 * 1024  # a decompression cap, so a gzip bomb cannot exhaust memory


def read_text(path, encoding=None, gz=False):
    if gz:
        with gzip.open(path, "rb") as f:
            data = f.read(MAX_GZIP_BYTES)
    else:
        with open(path, "rb") as f:
            data = f.read()
    # The sniffed encoding is only a hint for non-UTF-8 files; UTF-8 (and a BOM) are re-detected from the full data.
    text, _ = decode_bytes(data, None if encoding in (None, "utf-8") else encoding)
    return text


def plan_chunks(path, size, chunk=CHUNK_BYTES):
    """Byte ranges [start, end) covering the file, each boundary just after a newline."""
    bounds = [0]
    with open(path, "rb") as f:
        pos = chunk
        while pos < size:
            # jump ~chunk bytes ahead, then move the boundary forward to just after the next newline
            f.seek(pos)
            buf = f.read(1 << 16)
            if not buf:
                break
            nl = buf.find(b"\n")
            if nl == -1:  # a line longer than 64 KB: keep looking further on
                pos += len(buf)
                continue
            b = pos + nl + 1
            if b >= size:
                break
            bounds.append(b)
            pos = b + chunk
    bounds.append(size)
    return list(zip(bounds[:-1], bounds[1:]))


def read_chunk(path, start, end, size, encoding=None, overlap=OVERLAP):
    """Return (text, own_start, own_end) where [own_start, own_end) is the owned part in character offsets."""
    lo, hi = max(0, start - overlap), min(size, end + overlap)
    with open(path, "rb") as f:
        f.seek(lo)
        data = f.read(hi - lo)
    left, own, right = data[:start - lo], data[start - lo:end - lo], data[end - lo:]
    if lo > 0:  # start the left context on a line boundary so no multi-byte char is cut
        nl = left.find(b"\n")
        left = left[nl + 1:] if nl != -1 else b""
    if hi < size:  # same on the right: end the context just after a newline
        nl = right.rfind(b"\n")
        right = right[:nl + 1] if nl != -1 else b""
    enc = None if encoding in (None, "utf-8") else encoding
    # The three parts are decoded separately so the owned range can be given in *character* offsets, which is
    # what candidate positions use. Every part starts and ends on a line boundary, so each decodes cleanly.
    lt, ot, rt = (decode_bytes(x, enc or "utf-8")[0] for x in (left, own, right))
    return lt + ot + rt, len(lt), len(lt) + len(ot)
