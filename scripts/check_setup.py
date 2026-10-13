"""Check that this machine can run the scanner, on Linux, macOS or Windows.

    python scripts/check_setup.py

One line per check. Exit code 1 if something required is missing. A missing Tesseract is only a warning:
RapidOCR alone scores almost the same.
"""

import importlib
import os
import platform
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "starter")]

REQUIRED = [("pymupdf", "PDF text and layout"), ("pyarrow", "Parquet"), ("openpyxl", "XLSX"), ("docx", "DOCX"),
            ("lxml", "HTML / XML parsing"), ("bs4", "HTML"), ("charset_normalizer", "text encodings"),
            ("PIL", "images"), ("numpy", "name model, OCR"), ("pytesseract", "Tesseract bridge"),
            ("onnxruntime", "RapidOCR runtime"), ("rapidocr", "primary OCR engine"),
            ("psutil", "scripts/measure.py")]


def report(status, what, detail=""):
    print(f"[{status:<4}] {what}" + (f"  ({detail})" if detail else ""))


def main():
    ok = True
    py_ok = sys.version_info >= (3, 10)
    report("ok" if py_ok else "FAIL", f"Python {platform.python_version()}, {platform.system()} {platform.machine()}",
           "" if py_ok else "Python 3.10 or newer is required")
    ok &= py_ok

    for mod, why in REQUIRED:
        try:
            m = importlib.import_module(mod)
            report("ok", f"{mod} {getattr(m, '__version__', '')}".strip(), why)
        except Exception as e:  # an ImportError, or a DLL that fails to load (onnxruntime on Windows)
            ok = False
            report("FAIL", mod, f"{why}; {type(e).__name__}: {e}")

    from PIL import Image

    from scanner.extract import ocr

    blank = Image.new("RGB", (120, 40), "white")
    engines = []
    try:
        eng = ocr.RapidEngine()
        eng.run(blank)
        engines.append("rapidocr")
        report("ok", f"RapidOCR {eng.version} runs", "primary OCR engine")
    except Exception as e:
        report("WARN", "RapidOCR does not run", f"{type(e).__name__}: {e}")
    try:
        eng = ocr.TesseractEngine()
        eng.run(blank, 6)  # also proves the English language data is installed
        engines.append("tesseract")
        report("ok", f"Tesseract {eng.version} runs", ocr.find_tesseract() or "on PATH")
    except Exception as e:
        report("WARN", "Tesseract not usable",
               f"optional second OCR engine; install it or set TESSERACT_CMD. {type(e).__name__}: {str(e)[:120]}")
    if not engines:
        ok = False
        report("FAIL", "no OCR engine", "images and scanned PDF pages would not be read")

    try:
        from scanner.detect.text_detector import TLDS, Detector

        det = Detector()
        found = {(c.entity, c.value) for c in det.detect("From: Priya Sharma <priya.sharma@orbitpay.in>")}
        assert ("EMAIL", "priya.sharma@orbitpay.in") in found and ("PERSON_NAME", "Priya Sharma") in found
        report("ok", "detector and vendored resources load", f"{len(TLDS)} TLDs, name model ready")
    except Exception as e:
        ok = False
        report("FAIL", "detector / resources", f"{type(e).__name__}: {e}")

    from scanner import pipeline

    report("ok", f"worker start method: {pipeline._mp_context().get_start_method()}",
           "fork on Linux, spawn on macOS / Windows; override with SCANNER_MP_START")
    print("ready" if ok else "NOT ready: fix the FAIL lines above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
