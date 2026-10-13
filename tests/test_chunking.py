"""Big text files are scanned in newline-aligned chunks with overlap; ownership by start offset guarantees
no missed values at chunk boundaries and no duplicates."""

import json
import os
import random

from scanner import pipeline
from scanner.extract.text import plan_chunks, read_chunk


def _log_with_values(path, n_lines=4000, seed=0):
    # About 4000 log lines with an e-mail every 37 lines, a card number, and multi-line PEM keys. With 4 KB chunks
    # many values (and keys spanning several lines) straddle a boundary.
    r = random.Random(seed)
    values = []
    pem = "-----BEGIN RSA PRIVATE KEY-----\n" + "\n".join("A" * 64 for _ in range(6)) + "\n-----END RSA PRIVATE KEY-----"
    with open(path, "w", encoding="utf-8", newline="\n") as f:  # "\n" even on Windows: offsets are bytes
        for i in range(n_lines):
            k = i % 37
            if k == 5:
                f.write(f"2025-01-01T00:00:{i % 60:02d}Z INFO [api] user login ok email=user{i}@bluepeak.com\n")
                values.append(("EMAIL", f"user{i}@bluepeak.com"))
            elif k == 11:
                f.write(f"2025-01-01T00:00:{i % 60:02d}Z INFO [pay] payment debug cc_num=4539 1488 0343 6467\n")
                values.append(("CARD_NUMBER", "4539 1488 0343 6467"))
            elif k == 23 and i % 3 == 0:
                f.write("ssh_key: |\n" + pem + "\n")
                values.append(("PRIVATE_KEY", pem))
            else:
                f.write(f"2025-01-01T00:00:{i % 60:02d}Z INFO [api] GET /api/v1/items/{r.randint(1, 99999)} 200\n")
    return values


def test_plan_chunks_are_newline_aligned(tmp_path):
    p = tmp_path / "a.log"
    _log_with_values(str(p))
    size = os.path.getsize(p)
    chunks = plan_chunks(str(p), size, chunk=4096)
    assert chunks[0][0] == 0 and chunks[-1][1] == size and len(chunks) > 10
    data = p.read_bytes()
    for (a, b), (c, _) in zip(chunks, chunks[1:]):
        assert b == c and data[b - 1:b] == b"\n"


def test_read_chunk_ownership(tmp_path):
    p = tmp_path / "a.log"
    _log_with_values(str(p))
    size = os.path.getsize(p)
    owned = []
    for a, b in plan_chunks(str(p), size, chunk=4096):
        text, s, e = read_chunk(str(p), a, b, size, overlap=1024)
        owned.append(text[s:e])
    # every character is owned by exactly one chunk: concatenating the owned parts rebuilds the file
    assert "".join(owned) == p.read_bytes().decode("utf-8")


def test_chunked_equals_unchunked_and_no_duplicates(tmp_path):
    inp = tmp_path / "in"
    inp.mkdir()
    values = _log_with_values(str(inp / "big.log"))
    out_a, out_b = tmp_path / "a", tmp_path / "b"
    pipeline.run(str(inp), str(out_a), workers=2, chunk_mb=0.004)   # ~4 KB chunks: many boundaries
    pipeline.run(str(inp), str(out_b), workers=1, chunk_mb=1024)    # one piece
    rows_a = [json.loads(l) for l in (out_a / "findings.jsonl").read_text(encoding="utf-8").splitlines()]
    rows_b = [json.loads(l) for l in (out_b / "findings.jsonl").read_text(encoding="utf-8").splitlines()]
    key = lambda r: (r["entity"], r["value_sha256"])  # noqa: E731
    assert sorted(map(key, rows_a)) == sorted(map(key, rows_b))
    assert len(rows_a) == len(values)
