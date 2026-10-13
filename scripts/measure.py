"""Measure a scan: wall-clock and peak memory summed over the whole process tree (parent + workers + OCR
subprocesses), sampled every 0.1 s with psutil.

    python scripts/measure.py --input-dir data/dev/files --output-dir out/dev [--workers 4] [-- extra scan args]

Prints and writes <output-dir>/../<name>.measure.json with wall_s, peak_rss_mb (tree sum, an upper bound because
shared pages are counted per process), peak_single_rss_mb and the pipeline's own stage statistics.
"""

import argparse
import json
import os
import sys
import time

import psutil

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def tree_rss(proc):
    """(sum of RSS over the process and all its descendants, largest single RSS).

    The sum is what the 3 GB budget is about: a scan is a parent plus pool workers, and looking at the parent
    alone would hide most of the memory. Under fork, pages shared copy-on-write are counted once per process, so
    the sum is an upper bound."""
    total, single = 0, 0
    try:
        procs = [proc] + proc.children(recursive=True)
    except psutil.NoSuchProcess:
        return 0, 0
    for p in procs:
        try:
            rss = p.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        total += rss
        single = max(single, rss)
    return total, single


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--workers", type=int, default=4)
    a, extra = ap.parse_known_args()
    stats_file = os.path.normpath(a.output_dir) + ".stats.json"
    cmd = [sys.executable, os.path.join(ROOT, "scan.py"), "--input-dir", a.input_dir, "--output-dir", a.output_dir,
           "--workers", str(a.workers), "--stats-file", stats_file] + [x for x in extra if x != "--"]
    t0 = time.time()
    proc = psutil.Popen(cmd)
    peak = peak_single = 0
    # Polling every 0.1 s can miss a spike shorter than that. Worker memory grows and shrinks over seconds (a
    # table loaded, OCR models allocated), so this resolution is enough to see the peak.
    while proc.poll() is None:
        t, s = tree_rss(proc)
        peak, peak_single = max(peak, t), max(peak_single, s)
        time.sleep(0.1)
    wall = time.time() - t0
    out = {"wall_s": round(wall, 1), "peak_rss_mb": round(peak / 2 ** 20), "peak_single_rss_mb": round(peak_single / 2 ** 20),
           "workers": a.workers, "returncode": proc.returncode}
    if os.path.exists(stats_file):
        with open(stats_file, encoding="utf-8") as f:
            out["pipeline"] = json.load(f)
    path = os.path.normpath(a.output_dir) + ".measure.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "pipeline"}))
    if "pipeline" in out:
        p = out["pipeline"]
        print("cpu seconds by kind:", p.get("cpu_s_by_kind"), "| ocr pages:", p.get("ocr_pages"),
              "| ocr cpu s:", p.get("ocr_cpu_s"))


if __name__ == "__main__":
    main()
