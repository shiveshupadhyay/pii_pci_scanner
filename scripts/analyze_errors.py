"""Error analysis on a labelled set: false positives / negatives grouped by cause, slice and entity.

    python scripts/analyze_errors.py [--path text] [--file-filter logs/] [--entity PHONE] [--show 40]

Runs the scanner in memory (single process) so raw values are available for debugging; nothing is written
to disk. Matching is identical to starter/evaluate.py. Cause attribution:
  FN: routing (location type differs) | extraction / ocr (value absent from extracted text) |
      entity (value found as another entity) | location (same value, other page/column) | detection
  FP: duplicate | entity | ocr-misread | location | detection
"""

import argparse
import difflib
import json
import os
import sys
from collections import Counter, defaultdict

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "starter")]

from contract import location_key, normalize, value_sha256  # noqa: E402

from scanner import pipeline  # noqa: E402


def norm_text(s):
    return "".join(s.split()).lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=os.path.join(ROOT, "data/dev/labels"))
    ap.add_argument("--files", default=os.path.join(ROOT, "data/dev/files"))
    ap.add_argument("--path", choices=["structured", "text", "ocr"])
    ap.add_argument("--file-filter", default="")
    ap.add_argument("--entity")
    ap.add_argument("--show", type=int, default=60)
    ap.add_argument("--ner", action="store_true", help="mirror scan.py --ner (spaCy feature)")
    a = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):  # raw values get printed; a non-UTF-8 Windows console must not crash
        sys.stdout.reconfigure(errors="backslashreplace")
    with open(os.path.join(a.labels, "scope.json"), encoding="utf-8") as fh:
        scope = json.load(fh)
    with open(os.path.join(a.labels, "findings.jsonl"), encoding="utf-8") as fh:
        gold = [json.loads(l) for l in fh if l.strip()]
    col_only, err_files = set(scope["column_only_files"]), set(scope["error_files"])
    files = [f for f, p in sorted(scope["file_paths"].items())
             if f not in col_only and f not in err_files and (not a.path or p == a.path) and a.file_filter in f]
    # Same configuration as a scan.py worker, but in this process and without deferring OCR: scan_file() returns
    # Findings with their raw values, which never leave this process.
    pipeline._init_worker("ensemble", 1, a.ner)

    fp_rows, fn_rows = [], []
    cause_fp, cause_fn = Counter(), Counter()
    for f in files:
        # The collect hook records the text of every unit the detector saw. "Is the gold value anywhere in the
        # extracted text?" is what separates extraction/OCR misses from detection misses.
        pipeline._W["collect"] = []
        r = pipeline.scan_file(os.path.join(a.files, f), f)
        texts = pipeline._W.pop("collect")
        joined = norm_text("\n".join(t[0] for t in texts))
        gk = defaultdict(list)
        for g in gold:
            if g["file"] == f:
                gk[(g["entity"], g["value_sha256"], g["location_key"])].append(g)
        pk = defaultdict(list)
        for x in r.findings:
            pk[(x.entity, value_sha256(x.entity, x.value), location_key(x.location))].append(x)
        pred_by_hash = defaultdict(set)
        pred_loc_kinds = set()
        for (e, h, lk), xs in pk.items():
            pred_by_hash[h].add((e, lk))
            pred_loc_kinds.add(lk[:1] if lk else "")
        gold_by_value = defaultdict(set)
        for (e, h, lk), gs in gk.items():
            gold_by_value[normalize("PASSWORD", gs[0]["value"]).lower()].add((e, lk))
        path = scope["file_paths"][f]
        for key in set(gk) | set(pk):
            e, h, lk = key
            ng, np_ = len(gk.get(key, [])), len(pk.get(key, []))
            if a.entity and e != a.entity:
                continue
            for _ in range(max(0, ng - np_)):  # FN
                # Causes are tried from the most structural to the most specific:
                #   routing     the file was read as another type (its location keys have another kind)
                #   entity      the value was found, but as another entity
                #   location    the value was found under another page / column
                #   ocr / extraction   the value never appears in the extracted text
                #   detection   the text was there and the detector missed it
                g = gk[key][0]
                same_val = [(e2, lk2) for e2, lk2 in pred_by_hash.get(h, ()) if (e2, lk2) != key[0::2]]
                other_ent = [p for p in r.findings if norm_text(p.value) == norm_text(g["value"]) and p.entity != e]
                if lk and lk[:1] not in pred_loc_kinds and pred_loc_kinds:
                    cause = "routing"
                elif other_ent:
                    cause = f"entity->{other_ent[0].entity}"
                elif same_val:
                    cause = "location"
                elif norm_text(g["value"]) not in joined:
                    cause = "ocr" if path == "ocr" else "extraction"
                else:
                    cause = "detection"
                cause_fn[(path, e, cause.split("->")[0])] += 1
                fn_rows.append((path, f, e, g.get("sub", ""), g.get("slice", ""), lk, cause, g["value"]))
            for x in pk.get(key, [])[ng:] if np_ > ng else []:  # FP
                # an extra prediction on a key that exists in gold is a duplicate (e.g. chunk overlap); a value
                # that is gold under another entity or location is a mix-up; a near-miss of a gold value on an OCR
                # page is a misread; anything else is a genuine false detection
                if ng:
                    cause = "duplicate"
                elif any(norm_text(x.value) == v for v in [norm_text(gg["value"]) for gs in gk.values() for gg in gs]):
                    cause = "entity/location"
                elif path == "ocr" and any(gs[0]["entity"] == e and difflib.SequenceMatcher(
                        None, norm_text(gs[0]["value"]), norm_text(x.value)).ratio() >= 0.75 for gs in gk.values()):
                    cause = "ocr-misread"
                else:
                    cause = "detection"
                cause_fp[(path, e, cause)] += 1
                fp_rows.append((path, f, e, x.detector, lk, cause, x.value, x.context))

    print("== FALSE NEGATIVES by (path, entity, cause)")
    for k, v in sorted(cause_fn.items(), key=lambda kv: -kv[1]):
        print(f"  {v:4d}  {k}")
    print("== FALSE POSITIVES by (path, entity, cause)")
    for k, v in sorted(cause_fp.items(), key=lambda kv: -kv[1]):
        print(f"  {v:4d}  {k}")
    print(f"\n== FN examples (first {a.show})")
    for row in fn_rows[:a.show]:
        path, f, e, sub, sl, lk, cause, val = row
        print(f"  [{cause}] {f} {lk or '{}'} {e}/{sub} slice={sl}: {val[:70]!r}")
    print(f"\n== FP examples (first {a.show})")
    for row in fp_rows[:a.show]:
        path, f, e, det, lk, cause, val, ctx = row
        print(f"  [{cause}] {f} {lk or '{}'} {e} via {det} {list(ctx)}: {val[:70]!r}")


if __name__ == "__main__":
    main()
