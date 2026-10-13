"""OCR value recall on the dev OCR files: share of gold raw values that appear verbatim in the OCR text.

    python scripts/ocr_eval.py [--engine tesseract|rapidocr] [--psm 6] [--scale 2] [--no-median]

Reported per image type (slice) so preprocessing choices can be checked leave-one-type-out.
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "starter")]

from PIL import Image  # noqa: E402

from scanner.extract import ocr  # noqa: E402
from scanner.filetype import sniff  # noqa: E402


def page_images(path):
    s = sniff(path)
    if s.kind == "image":
        im = Image.open(path)
        im.load()
        yield 1, im
    elif s.kind == "pdf":
        from scanner.extract.pdf import pdf_pages

        for kind, pno, payload in pdf_pages(path):
            if kind == "ocr":
                yield pno, payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=os.path.join(ROOT, "data/dev/labels"))
    ap.add_argument("--files", default=os.path.join(ROOT, "data/dev/files"))
    ap.add_argument("--engine", default="tesseract")
    ap.add_argument("--psm", type=int, default=6)
    ap.add_argument("--scale", type=float, default=2.0)
    ap.add_argument("--no-median", action="store_true")
    ap.add_argument("--show-misses", action="store_true")
    a = ap.parse_args()
    os.environ["SCANNER_OCR"] = a.engine
    with open(os.path.join(a.labels, "scope.json"), encoding="utf-8") as fh:
        scope = json.load(fh)
    gold = defaultdict(list)
    with open(os.path.join(a.labels, "findings.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            g = json.loads(line)
            if g["path"] == "ocr":
                gold[(g["file"], g["location_key"])].append(g)
    hit, tot = defaultdict(int), defaultdict(int)
    t0 = time.time()
    pages = 0
    for f, p in sorted(scope["file_paths"].items()):
        if p != "ocr":
            continue
        for pno, im in page_images(os.path.join(a.files, f)):
            pages += 1
            text, conf, eng = ocr.ocr_image(im, psm=a.psm, scale=a.scale, median=not a.no_median)
            flat = " ".join(text.split())
            for g in gold.get((f, f"p{pno}"), []):
                ok = g["value"] in text or " ".join(g["value"].split()) in flat
                key = g["slice"]
                tot[key] += 1
                hit[key] += ok
                tot["ALL"] += 1
                hit["ALL"] += ok
                if not ok and a.show_misses:
                    print(f"MISS {f} p{pno} {g['entity']:13s} {g['value']!r}")
    for k in sorted(tot):
        print(f"{k:18s} {hit[k]:4d}/{tot[k]:<4d} {hit[k] / tot[k]:.3f}")
    print(f"engine={eng} pages={pages} time={time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
