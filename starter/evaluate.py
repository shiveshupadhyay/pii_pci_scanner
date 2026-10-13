"""Official metric for the v2 take-home. We score the hidden test set with exactly this code.

    python starter/evaluate.py --labels data/dev/labels --output out/ [--json metrics.json]

`out/` must contain findings.jsonl, columns.jsonl and errors.jsonl (see BRIEF.md).

Matching (findings): key = (file, entity, value_sha256, location_key).
  location_key = "p<page>" for PDFs and images, "<sheet>/<column>" for structured files, "" otherwise.
  Every occurrence is a separate finding: per key, TP = min(gold, predicted), FP = the excess, FN = the shortfall.
  Excess predictions on a key that does exist in gold are also reported as "duplicates" (e.g. chunk overlap).
  Findings on column-only files (very large structured files) are ignored; findings on error files are violations.

Headline: PATH-BALANCED MACRO-F1 = mean over the three paths (structured, text, ocr) of the macro-F1 over the
entity types present in that path. Also reported: column-level P/R/F1, error handling, F0.5 for PCI and SECRET.
Stdlib only.
"""

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from contract import CATEGORY, LABELS, location_key  # noqa: E402

PATHS = ["structured", "text", "ocr"]
REQUIRED = ["file", "entity", "category", "value_sha256", "value_masked", "location", "confidence", "evidence", "detector_version"]
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def prf(tp, fp, fn, beta=1.0):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    b2 = beta * beta
    return p, r, ((1 + b2) * p * r / (b2 * p + r) if p + r else 0.0)


def load_gold(labels_dir):
    scope = json.load(open(os.path.join(labels_dir, "scope.json")))
    findings = load_jsonl(os.path.join(labels_dir, "findings.jsonl"))
    columns = load_jsonl(os.path.join(labels_dir, "columns.jsonl"))
    return scope, findings, columns


def check_findings(preds, scope, raw_by_hash):
    problems, valid = Counter(), []
    files = set(scope["all_files"])
    for p in preds:
        miss = [k for k in REQUIRED if k not in p]
        if miss:
            problems["missing fields: " + ",".join(miss)] += 1
            continue
        if p["file"] not in files:
            problems["unknown file"] += 1
            continue
        if p["entity"] not in CATEGORY:
            problems["unknown entity"] += 1
            continue
        if not HEX64.match(str(p["value_sha256"])):
            problems["value_sha256 is not a sha256 hex digest"] += 1
            continue
        if p["category"] != CATEGORY[p["entity"]]:
            problems["wrong category"] += 1
        if not isinstance(p["confidence"], (int, float)) or not 0 <= p["confidence"] <= 1:
            problems["confidence not in [0,1]"] += 1
        if not p["evidence"]:
            problems["empty evidence"] += 1
        raw = raw_by_hash.get(p["value_sha256"])
        if raw and sum(ch.isalnum() for ch in raw) >= 4 and raw in str(p["value_masked"]):
            problems["value_masked leaks the raw value"] += 1
        valid.append(p)
    return problems, valid


def evaluate(labels_dir, out_dir, quiet=False):
    scope, gold, gold_cols = load_gold(labels_dir)
    path_of = scope["file_paths"]
    col_only, err_files = set(scope["column_only_files"]), set(scope["error_files"])
    raw_by_hash = {g["value_sha256"]: g["value"] for g in gold}

    preds = load_jsonl(os.path.join(out_dir, "findings.jsonl"))
    problems, valid = check_findings(preds, scope, raw_by_hash)
    on_error_files = sum(p["file"] in err_files for p in valid)
    valid = [p for p in valid if p["file"] not in col_only and p["file"] not in err_files]

    gk = Counter((g["file"], g["entity"], g["value_sha256"], g["location_key"]) for g in gold)
    pk = Counter((p["file"], p["entity"], p["value_sha256"], location_key(p["location"])) for p in valid)

    tp, fp, fn = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)  # [path][entity]
    dups = 0
    for k in set(gk) | set(pk):
        path, ent = path_of.get(k[0], "text"), k[1]
        t = min(gk[k], pk[k])
        tp[path][ent] += t
        fp[path][ent] += pk[k] - t
        fn[path][ent] += gk[k] - t
        if gk[k] and pk[k] > gk[k]:
            dups += pk[k] - gk[k]

    out = {"per_path": {}, "per_entity": {}}
    path_macros = []
    for path in PATHS:
        ents = [e for e in LABELS if tp[path][e] + fp[path][e] + fn[path][e] > 0]
        per = {e: dict(zip(("precision", "recall", "f1"), prf(tp[path][e], fp[path][e], fn[path][e])),
                       support=tp[path][e] + fn[path][e]) for e in ents}
        macro = sum(v["f1"] for v in per.values()) / len(per) if per else 0.0
        out["per_path"][path] = {"macro_f1": macro, "per_entity": per,
                                 "micro_f1": prf(sum(tp[path].values()), sum(fp[path].values()), sum(fn[path].values()))[2]}
        path_macros.append(macro)
    out["path_balanced_macro_f1"] = sum(path_macros) / len(PATHS)

    T, F, N = Counter(), Counter(), Counter()
    for path in PATHS:
        T.update(tp[path]), F.update(fp[path]), N.update(fn[path])
    for e in LABELS:
        out["per_entity"][e] = dict(zip(("precision", "recall", "f1"), prf(T[e], F[e], N[e])), support=T[e] + N[e])
    for cat in ("PII", "PCI", "SECRET"):
        ls = [l for l in LABELS if CATEGORY[l] == cat]
        out[f"{cat.lower()}_f05"] = prf(sum(T[l] for l in ls), sum(F[l] for l in ls), sum(N[l] for l in ls), 0.5)[2]
    out["duplicates"] = dups

    # columns
    gc = {(c["file"], c.get("sheet", "") or "", c["column"], c["entity"]) for c in gold_cols}
    pc = {(c["file"], c.get("sheet", "") or "", c["column"], c["entity"]) for c in load_jsonl(os.path.join(out_dir, "columns.jsonl"))
          if c.get("file") in path_of}
    ctp = len(gc & pc)
    out["columns"] = dict(zip(("precision", "recall", "f1"), prf(ctp, len(pc) - ctp, len(gc) - ctp)))
    out["columns_large_files_recall"] = (len({x for x in gc & pc if x[0] in col_only}) / max(1, len({x for x in gc if x[0] in col_only})))

    # error handling
    listed = {e.get("file") for e in load_jsonl(os.path.join(out_dir, "errors.jsonl"))}
    out["errors"] = {"error_files_reported": len(listed & err_files), "error_files_total": len(err_files),
                     "false_errors": len(listed - err_files), "findings_on_error_files": on_error_files}
    out["contract_problems"] = dict(problems)

    if not quiet:
        print(f"PATH-BALANCED MACRO-F1 {out['path_balanced_macro_f1']:.3f}   "
              + "   ".join(f"{p} {out['per_path'][p]['macro_f1']:.3f}" for p in PATHS))
        print(f"F0.5  PCI {out['pci_f05']:.3f}   SECRET {out['secret_f05']:.3f}   PII {out['pii_f05']:.3f}   "
              f"duplicate findings {dups}")
        c = out["columns"]
        print(f"columns  P {c['precision']:.3f}  R {c['recall']:.3f}  F1 {c['f1']:.3f}   "
              f"(large files recall {out['columns_large_files_recall']:.2f})")
        e = out["errors"]
        print(f"errors   reported {e['error_files_reported']}/{e['error_files_total']}   false errors {e['false_errors']}   "
              f"findings on error files {e['findings_on_error_files']}")
        print(f"\n{'entity':16s}" + "".join(f"{p:>22s}" for p in PATHS) + f"{'all':>8s}")
        for ent in LABELS:
            row = f"{ent:16s}"
            for p in PATHS:
                v = out["per_path"][p]["per_entity"].get(ent)
                row += f"{'':>4s}{v['precision']:.2f}/{v['recall']:.2f}/{v['f1']:.2f} {v['support']:4d}" if v else f"{'-':>22s}"
            row += f"{out['per_entity'][ent]['f1']:8.3f}"
            print(row)
        print("   (cells: precision/recall/F1 support)")
        if problems:
            print("\nCONTRACT PROBLEMS (invalid findings are dropped before scoring):")
            for k, v in problems.most_common():
                print(f"  {k}: {v}")
        else:
            print("\noutput contract: OK")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, help="e.g. data/dev/labels")
    ap.add_argument("--output", required=True, help="directory with findings.jsonl, columns.jsonl, errors.jsonl")
    ap.add_argument("--json")
    a = ap.parse_args()
    out = evaluate(a.labels, a.output)
    if a.json:
        json.dump(out, open(a.json, "w"), indent=2)


if __name__ == "__main__":
    main()
