"""End-to-end pipeline: walk -> sniff/route -> extract -> (chunk) -> detect -> merge -> write.

    python scan.py --input-dir <files/> --output-dir <out/> [--workers 4] [--stats-file stats.json]

The parent process walks the tree, sniffs every file, and plans tasks: one per file, or one per chunk of a big text
file. Two process pools do the work:
  Phase A: extraction and detection for everything. Pages that need OCR are only recorded.
  Phase B: OCR of those pages, in fresh workers, so OCR models never share memory with big-file buffers.
Workers return hashed and masked records (see _task_entry), so raw values never leave a worker process.
scan_file() is the in-process entry point used by the tests and the analysis scripts.
"""

import argparse
import difflib
import json
import os
import sys
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool

from .detect.columns import classify_table
from .detect.text_detector import Detector
from .extract import documents, structured
from .extract.text import CHUNK_BYTES, plan_chunks, read_chunk, read_text
from .filetype import sniff
from .output import column_record, finding_record, write_jsonl
from .units import ColumnLabel, FileResult, Finding, TextUnit

STRUCTURED = {"csv", "json", "jsonl", "parquet", "xlsx"}
# Per-process state: the configuration from _init_worker plus the detector and OCR engines, built lazily and then
# reused for every task this worker runs. Loading lexicons and OCR models once per process, not once per file, is
# most of what makes the pool cheap.
_W = {}


# ============================================================================ worker state


def _init_worker(ocr_mode="ensemble", threads=1, use_ner=False, defer_ocr=False):
    # This is the only way configuration reaches a worker. With "spawn" (macOS, Windows) a worker inherits nothing
    # from the parent, so anything not passed here would silently fall back to defaults.
    # Thread caps: Tesseract (OpenMP) and numpy would otherwise each start one thread per core in every worker,
    # and 4 workers x 8 threads just fight over the same cores.
    os.environ.setdefault("OMP_THREAD_LIMIT", str(threads))
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    _W["ocr_mode"] = ocr_mode
    _W["threads"] = threads
    _W["use_ner"] = use_ner
    _W["defer_ocr"] = defer_ocr


def detector():
    if "det" not in _W:
        nlp = None
        if _W.get("use_ner", False):  # optional spaCy feature (no measured gain on dev / synthetic, +80 MB/worker)
            try:
                import spacy

                nlp = spacy.load("en_core_web_sm", disable=["lemmatizer", "attribute_ruler"])
            except Exception:
                nlp = None
        _W["det"] = Detector(nlp=nlp)
    return _W["det"]


def ocr_engines():
    # The order of the list matters: ocr_findings() treats the first engine as primary. Any engine that fails to
    # load is skipped, so a machine without Tesseract (or with a broken onnxruntime) still gets the other engine.
    if "ocr" not in _W:
        from .extract import ocr

        engines = []
        mode = _W.get("ocr_mode", "ensemble")
        if mode in ("ensemble", "rapidocr"):
            try:
                engines.append(ocr.RapidEngine(threads=_W.get("threads", 1)))
            except Exception:
                pass
        if mode in ("ensemble", "tesseract") or not engines:
            try:
                engines.append(ocr.TesseractEngine())
            except Exception:
                pass
        if not engines and mode == "tesseract":
            try:
                engines.append(ocr.RapidEngine(threads=_W.get("threads", 1)))
            except Exception:
                pass
        _W["ocr"] = engines
    return _W["ocr"]


# ============================================================================ per-file scanning


def _findings_from_units(rel, units, det, keep=None):
    """Run the detector on each unit and place its candidates in the file. `keep` filters candidates; chunk tasks
    use it to keep only the candidates that start inside the chunk's own byte range."""
    out = []
    for u in units:
        if _W.get("collect") is not None:  # analysis hook (scripts/analyze_errors.py), never used by scan.py
            _W["collect"].append((u.text, u.key, u.use_ner, u.cell, u.ocr))
        cands = det.detect(u.text, key=u.key, ocr=u.ocr, cell=u.cell, table=u.table, use_ner=u.use_ner)
        for c in cands:
            if keep is not None and not keep(c):
                continue
            # (unit order, offset) sorts findings in reading order, so the output file is stable from run to run
            out.append(Finding(rel, c.entity, c.value, u.location, c.confidence * u.conf_scale, u.extractor,
                               c.detector, c.context, (u.order, c.start)))
    return out


def _similar(a, b):
    """Two OCR readings of the same value: equal, one contains the other (a reading cut by a misread
    separator), or near-identical."""
    a, b = "".join(a.split()).lower(), "".join(b.split()).lower()
    # containment needs 5+ characters, or a short fragment such as "571" would match almost anything
    if a == b or (min(len(a), len(b)) >= 5 and (a in b or b in a)):
        return True
    return difflib.SequenceMatcher(None, a, b).ratio() >= 0.8


def ocr_findings(rel, img, loc, det):
    """OCR with every available engine; the first engine is primary, later engines only add findings that no
    primary finding of the same entity resembles (different misreads of one value never both survive)."""
    from .extract import ocr

    merged = []
    for i, eng in enumerate(ocr_engines()):
        if eng.name == "tesseract":
            # Tesseract needs the cleaned-up grayscale image; psm 4 ("one column of text of variable sizes") read
            # screenshots and scans best in scripts/ocr_eval.py. RapidOCR does its own detection on the raw image.
            text, conf = eng.run(ocr.preprocess(img), 4)
        else:
            text, conf = eng.run(img)
        text = ocr.repair_ocr_text(ocr.clean_ocr_text(text), det.names)
        # OCR findings carry less certainty than text-layer findings, and the secondary engine less than the primary
        unit = TextUnit(text, loc, f"ocr-{eng.name}", ocr=True, conf_scale=0.9 if i == 0 else 0.8,
                        order=loc.get("page", 1) * 10 + i)
        fs = _findings_from_units(rel, [unit], det)
        if i == 0:
            merged.extend(fs)
            continue
        for f in fs:
            if not any(m.entity == f.entity and _similar(m.value, f.value) for m in merged):
                merged.append(f)
    return merged


def scan_file(path, rel, s=None, chunk=None):
    """Scan one file (or one chunk of a big text file). Returns FileResult with raw-valued Findings."""
    t0 = time.time()
    s = s or sniff(path)
    res = FileResult(rel, stats={"kind": s.kind, "bytes": os.path.getsize(path)})
    if s.error:
        res.error = s.error
        return res
    det = detector()
    kind = s.kind
    # Dispatch on the sniffed kind, never on the extension. Every branch ends in the same detector; they differ
    # only in how text is extracted and which context (header, label, page) travels with it.
    try:
        if kind in STRUCTURED:
            _scan_structured(path, rel, s, det, res)
        elif kind == "pdf":
            from .extract.pdf import pdf_pages

            # In Phase A pages without a text layer are only listed (not even rendered) and come back in Phase B.
            defer = _W.get("defer_ocr", False)
            for pkind, pno, payload in pdf_pages(path, images=not defer):
                if pkind == "ocr" and defer:
                    res.stats.setdefault("ocr_pending", []).append(pno)
                    continue
                if pkind == "ocr":
                    t = time.time()
                    res.findings.extend(ocr_findings(rel, payload, {"page": pno}, det))
                    res.stats["ocr_s"] = res.stats.get("ocr_s", 0) + time.time() - t
                    res.stats["ocr_pages"] = res.stats.get("ocr_pages", 0) + 1
                else:
                    res.findings.extend(_findings_from_units(rel, payload, det))
        elif kind == "image":
            from PIL import Image

            try:
                img = Image.open(path)
                img.load()  # Image.open() is lazy; decoding now is what reveals a truncated or corrupt image
            except Exception as e:
                res.error = f"corrupt image ({type(e).__name__})"
                return res
            if _W.get("defer_ocr", False):
                res.stats["ocr_pending"] = [1]
                return res
            t = time.time()
            res.findings.extend(ocr_findings(rel, img, {"page": 1}, det))
            res.stats["ocr_s"] = time.time() - t
            res.stats["ocr_pages"] = 1
        elif kind == "docx":
            res.findings.extend(_findings_from_units(rel, documents.docx_units(path), det))
        elif kind == "pptx":
            res.findings.extend(_findings_from_units(rel, documents.pptx_units(path), det))
        elif kind == "eml":
            res.findings.extend(_findings_from_units(rel, documents.eml_units(path), det))
        elif kind in ("html", "xml"):
            text = read_text(path, s.meta.get("encoding"), s.meta.get("gzip"))
            res.findings.extend(_findings_from_units(rel, documents.html_units(text, xml=kind == "xml"), det))
        elif kind == "sql":
            text = read_text(path, s.meta.get("encoding"), s.meta.get("gzip"))
            res.findings.extend(_findings_from_units(rel, documents.sql_units(text), det))
        else:
            if chunk is not None:
                # A chunk is decoded with up to 64 KB of context on each side, but only candidates starting inside
                # its own range are kept: every value has exactly one owner, so none is missed or reported twice.
                start, end, size = chunk
                text, own_s, own_e = read_chunk(path, start, end, size, s.meta.get("encoding"))
                unit = TextUnit(text, {}, "text-stream", order=start)
                res.findings.extend(_findings_from_units(rel, [unit], det,
                                                         keep=lambda c: own_s <= c.start < own_e))
            else:
                text = read_text(path, s.meta.get("encoding"), s.meta.get("gzip"))
                res.findings.extend(_findings_from_units(rel, [TextUnit(text, {}, "text", use_ner=True)], det))
    except ValueError as e:  # raised by extractors for unreadable content (encrypted / damaged)
        res.error = str(e)[:200]
        # an errored file must not report findings: the evaluator counts those as violations
        res.findings = []
    except Exception as e:  # never stop the run on one file
        res.error = f"extraction failed ({type(e).__name__}: {str(e)[:120]})"
        res.findings, res.columns = [], []
    res.stats["seconds"] = time.time() - t0
    return res


def _scan_structured(path, rel, s, det, res):
    kind = s.kind
    if kind == "csv":
        tables = structured.csv_tables(path, s.meta.get("delimiter", ","), s.meta.get("encoding"), s.meta.get("gzip"))
    elif kind == "parquet":
        tables = structured.parquet_tables(path)
    elif kind == "xlsx":
        tables = structured.xlsx_tables(path)
    else:
        tables = structured.json_tables(path, jsonl=kind == "jsonl", encoding=s.meta.get("encoding"))
    is_xlsx = kind == "xlsx"
    res.stats["rows"] = sum(t.n_rows for t in tables)
    res.stats["large"] = any(t.large for t in tables)
    for t in tables:
        # Tables over 50k rows were sampled and are scored on columns.jsonl only, so they get labels, no findings.
        cell_findings, labels = classify_table(det, t, sample_only=t.large)
        for c_i, r_i, c in cell_findings:
            # Only workbooks have sheets. A CSV or JSON location carries just the column (a dotted path for JSON),
            # matching contract.location_key().
            loc = {"sheet": t.sheet, "column": t.columns[c_i]} if is_xlsx else {"column": t.columns[c_i]}
            res.findings.append(Finding(rel, c.entity, c.value, loc, c.confidence, f"{kind}-cell", c.detector,
                                        c.context + ("header: " + _safe_header(t.columns[c_i]),),
                                        (t.sheet, c_i, r_i, c.start)))
        for c_i, ent, share in labels:
            res.columns.append(ColumnLabel(rel, t.sheet if is_xlsx else "", t.columns[c_i], ent, share))


def _safe_header(h):
    # Headers go into the evidence. They are labels, not values, but a pathological one (a whole JSON blob used as
    # a column name) should not bloat every finding.
    h = str(h)
    return h if len(h) <= 60 else h[:57] + "..."


def ocr_page(path, rel, pno, image_file=False):
    """Phase B: OCR one deferred page (an image file, or a PDF page without a text layer)."""
    t0 = time.time()
    res = FileResult(rel, stats={"kind": "ocr-page"})
    if image_file:
        from PIL import Image

        img = Image.open(path)
        img.load()
    else:
        from .extract.pdf import pdf_page_image

        img = pdf_page_image(path, pno)
    if img is not None:
        res.findings.extend(ocr_findings(rel, img, {"page": pno}, detector()))
    res.stats.update({"seconds": time.time() - t0, "ocr_pages": 1, "ocr_s": time.time() - t0,
                      "ocr_engines": [e.name for e in ocr_engines()]})
    return res


def _task_entry(task):
    """Runs in a worker: returns serialisable records (hashes/masks only, never raw values).

    Tasks are tuples: ("file", rel, path), ("chunk", rel, path, (start, end, size)) or
    ("ocr", rel, path, page_no, is_image_file)."""
    kind, rel, path = task[0], task[1], task[2]
    chunk = task[3] if kind == "chunk" else None
    try:
        if kind == "ocr":
            r = ocr_page(path, rel, task[3], task[4])
        else:
            r = scan_file(path, rel, chunk=chunk)
    except Exception as e:
        return {"file": rel, "error": f"internal error ({type(e).__name__})", "findings": [], "columns": [],
                "stats": {"trace": traceback.format_exc()[-500:]}}
    # Hash and mask here, inside the worker: raw values never cross the process boundary, and records pickle
    # smaller than Finding objects.
    recs = [(f.order, finding_record(f)) for f in r.findings]
    return {"file": rel, "error": r.error, "findings": recs, "columns": [column_record(c) for c in r.columns],
            "stats": r.stats, "chunk": chunk, "page": task[3] if kind == "ocr" else None}


# ============================================================================ orchestration


def walk(input_dir):
    """(relative path, absolute path) for every regular file, in a stable sorted order. Hidden files are included
    (secrets love dotfiles). Symlinks are skipped, so a link can neither loop nor lead outside the input tree."""
    out = []
    for root, dirs, files in os.walk(input_dir, followlinks=False):
        dirs.sort()
        for name in sorted(files):
            p = os.path.join(root, name)
            if os.path.islink(p) or not os.path.isfile(p):
                continue
            # labels use "/" separators whatever the OS, so Windows paths are converted here, once
            rel = os.path.relpath(p, input_dir).replace(os.sep, "/")
            out.append((rel, p))
    return out


def _cost(task):
    """Rough relative cost of a task, used to start the longest tasks first. Otherwise one big file picked up last
    keeps a single worker busy while the others sit idle. Only the ordering matters: the extension is a fine guess
    here, since a wrong guess costs a little balance, never correctness."""
    kind, rel, path = task[0], task[1], task[2]
    if kind == "chunk":
        return task[3][1] - task[3][0]
    size = os.path.getsize(path)
    ext = rel.rsplit(".", 1)[-1].lower()
    if ext in ("png", "jpg", "jpeg", "tif", "tiff", "bmp", "gif", "webp"):
        return 30 * 1024 * 1024  # an image is worth about 30 MB of log text
    if ext == "pdf":
        return size * 50  # PDF layout work per byte is far heavier than plain text
    return size


def run(input_dir, output_dir, workers=None, ocr_mode="ensemble", use_ner=False, stats_file=None, chunk_mb=None,
        ocr_workers=None):
    t0 = time.time()
    os.makedirs(output_dir, exist_ok=True)
    workers = workers or min(4, os.cpu_count() or 1)
    chunk_bytes = int((chunk_mb or CHUNK_BYTES / (1024 * 1024)) * 1024 * 1024)
    files = walk(input_dir)
    tasks, early = [], []
    # Sniffing happens here in the parent. It is cheap (a 64 KB head and a 4 KB tail; ZIP containers also get a CRC
    # check), and unreadable files go straight to errors.jsonl without costing a worker. Workers sniff again,
    # which keeps every task self-contained.
    for rel, path in files:
        s = sniff(path)
        if s.error:
            early.append({"file": rel, "error": s.error, "findings": [], "columns": [], "stats": {"kind": s.kind}})
            continue
        size = os.path.getsize(path)
        # Only plain text is chunked: structured files are sampled instead, and a gzip stream cannot be seeked into.
        if s.kind == "text" and not s.meta.get("gzip") and size > chunk_bytes:
            for a, b in plan_chunks(path, size, chunk_bytes):
                tasks.append(("chunk", rel, path, (a, b, size)))
        else:
            tasks.append(("file", rel, path))
    tasks.sort(key=_cost, reverse=True)
    # Share the cores between workers: each worker's OCR / numpy threads get cores // workers.
    threads = max(1, (os.cpu_count() or 1) // max(1, workers))
    # Phase A: everything except OCR (PDF pages without a text layer and images are only recorded).
    results = list(early)
    results += _run_pool(tasks, workers, (ocr_mode, threads, use_ner, True))
    # Phase B: OCR pages, in fresh workers so OCR models never sit next to big-file buffers.
    ocr_tasks = []
    for r in results:
        if r.get("error"):
            continue
        for pno in (r.get("stats") or {}).get("ocr_pending", []):
            path = os.path.join(input_dir, r["file"])
            ocr_tasks.append(("ocr", r["file"], path, pno, r["stats"].get("kind") == "image"))
    if ocr_tasks:
        # At most 3 OCR workers by default: each one holds the RapidOCR models and ONNX runtime buffers, so OCR
        # workers decide the memory peak. Capping them was part of bringing the hidden-scale peak under 2 GB.
        ocr_workers = max(1, min(workers, len(ocr_tasks), ocr_workers or min(3, workers)))
        results += _run_pool(ocr_tasks, ocr_workers, (ocr_mode, max(1, threads), use_ner, False))
        blind = sum(1 for r in results if (r.get("stats") or {}).get("kind") == "ocr-page"
                    and not r["stats"].get("ocr_engines"))
        if blind:  # say so loudly: otherwise a broken OCR install just looks like "no findings in images"
            print(f"warning: no OCR engine could be loaded; {blind} image/scanned pages were not read "
                  f"(run: python scripts/check_setup.py)", file=sys.stderr)
    _write(results, output_dir)
    stats = _summarise(results, time.time() - t0, workers)
    if stats_file:
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
    print(f"scanned {stats['files']} files in {stats['wall_s']:.1f}s: {stats['findings']} findings, "
          f"{stats['columns']} sensitive columns, {stats['errors']} errors", file=sys.stderr)
    return stats


def _run_pool(tasks, workers, initargs):
    """Run tasks in a process pool; a crashed worker (BrokenProcessPool) only costs the file that killed it.

    A native crash (a segfault inside a PDF or image library) kills the whole pool. Python exceptions are already
    caught per task. When the pool breaks, the unfinished tasks are retried in a fresh pool. If it breaks again, the
    first unfinished task is the likely culprit: tasks run largest first, so it was almost certainly in flight. It is
    reported as "extraction crashed" and the rest are retried once more. If even that third pool dies, the
    remaining tasks are given up; by then something is wrong with the environment rather than with one file."""
    results, pending, attempts = [], list(tasks), 0
    while pending and attempts < 3:
        attempts += 1
        done_ids = set()
        try:
            with ProcessPoolExecutor(max_workers=workers, mp_context=_mp_context(), initializer=_init_worker,
                                     initargs=initargs) as ex:
                futs = {ex.submit(_task_entry, t): i for i, t in enumerate(pending)}
                for fut in as_completed(futs):
                    results.append(fut.result())
                    done_ids.add(futs[fut])
            pending = []
        except BrokenProcessPool:
            remaining = [t for i, t in enumerate(pending) if i not in done_ids]
            if attempts >= 2 and remaining:
                bad = remaining[0]
                results.append({"file": bad[1], "error": "extraction crashed", "findings": [], "columns": [],
                                "stats": {}})
                remaining = remaining[1:]
            pending = remaining
    return results


def _mp_context():
    import multiprocessing as mp

    # SCANNER_MP_START=spawn reproduces the macOS/Windows process model on Linux (see tests/test_platform.py).
    forced = os.environ.get("SCANNER_MP_START")
    if forced:
        return mp.get_context(forced)
    # fork starts workers instantly and is safe on Linux. On macOS, forking a process that has touched system
    # frameworks can crash, and Windows has no fork at all, so both use spawn (Python's own default there).
    if sys.platform.startswith("linux"):
        return mp.get_context("fork")
    return mp.get_context("spawn")


def _write(results, output_dir):
    by_file = defaultdict(list)
    for r in results:
        by_file[r["file"]].append(r)
    findings, columns, errors = [], [], []
    # A file can come back in several parts: one per chunk, plus one per OCR page from Phase B.
    for rel in sorted(by_file):
        parts = sorted(by_file[rel], key=lambda r: (r.get("chunk") or (0,))[0])
        err = next((p["error"] for p in parts if p.get("error")), None)
        if err:
            # Any failed part makes the whole file an error, and its findings are dropped: reporting findings in a
            # file that is listed as unreadable would count as a violation.
            errors.append({"file": rel, "error": err})
            continue
        recs = []
        for p in parts:
            recs.extend(p["findings"])
        recs.sort(key=lambda x: _order_key(x[0]))
        findings.extend(r for _, r in recs)
        seen = set()
        for p in parts:
            for c in p["columns"]:
                k = (c["file"], c["sheet"], c["column"], c["entity"])
                if k not in seen:
                    seen.add(k)
                    columns.append(c)
    write_jsonl(os.path.join(output_dir, "findings.jsonl"), findings)
    write_jsonl(os.path.join(output_dir, "columns.jsonl"), columns)
    write_jsonl(os.path.join(output_dir, "errors.jsonl"), errors)


def _order_key(o):
    # Order tuples mix numbers (offsets, pages) and strings (sheet names); tagging each element keeps the
    # comparison total, so sorting never raises TypeError and the output order is deterministic.
    return tuple((0, x) if isinstance(x, (int, float)) else (1, str(x)) for x in (o if isinstance(o, tuple) else (o,)))


def _summarise(results, wall, workers):
    kinds = Counter()
    secs = Counter()
    ocr_pages = ocr_s = 0
    for r in results:
        st = r.get("stats") or {}
        k = st.get("kind", "?")
        kinds[k] += 1
        secs[k] += st.get("seconds", 0.0)
        ocr_pages += st.get("ocr_pages", 0)
        ocr_s += st.get("ocr_s", 0.0)
    files = {r["file"] for r in results}
    return {"files": len(files), "wall_s": round(wall, 2), "workers": workers,
            "findings": sum(len(r["findings"]) for r in results),
            "columns": sum(len(r["columns"]) for r in results),
            "errors": len({r["file"] for r in results if r.get("error")}),
            "tasks_by_kind": dict(kinds), "cpu_s_by_kind": {k: round(v, 2) for k, v in secs.items()},
            "ocr_pages": ocr_pages, "ocr_cpu_s": round(ocr_s, 2),
            "ocr_engines": sorted({e for r in results for e in (r.get("stats") or {}).get("ocr_engines", [])})}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Sensitive data scanner (PII / PCI / secrets)")
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--workers", type=int, default=None, help="default: min(4, cpu count)")
    ap.add_argument("--ocr", default=os.environ.get("SCANNER_OCR", "ensemble"),
                    choices=["ensemble", "rapidocr", "tesseract"])
    ap.add_argument("--ner", action="store_true", help="add the optional spaCy feature to the person-name model "
                                                       "(needs spacy + en_core_web_sm installed)")
    ap.add_argument("--ocr-workers", type=int, default=None, help="workers for the OCR phase "
                                                                  "(default: min(3, --workers))")
    ap.add_argument("--chunk-mb", type=float, default=None)
    ap.add_argument("--stats-file")
    a = ap.parse_args(argv)
    run(a.input_dir, a.output_dir, a.workers, a.ocr, a.ner, a.stats_file, a.chunk_mb, a.ocr_workers)
