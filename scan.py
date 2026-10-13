"""Pipeline entrypoint:

    python scan.py --input-dir <files/> --output-dir <out/>

Writes findings.jsonl, columns.jsonl and errors.jsonl into <out/> (see README.md and DESIGN.md).
"""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
# src/ holds the scanner package. starter/ holds contract.py, which output.py imports instead of copying, so every
# hash comes from the exact code the evaluator uses.
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "starter")]

from scanner.pipeline import main  # noqa: E402

# The guard is required, not just good style: on macOS and Windows every worker process re-imports this module, and
# without it each worker would start a scan of its own.
if __name__ == "__main__":
    main()
