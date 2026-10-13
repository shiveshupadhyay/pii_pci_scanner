# Sensitive data scanner: PII, PCI and secrets

A local, CPU-only scanner that finds 11 kinds of sensitive values (emails, phones, person names, PAN / Aadhaar / SSN,
card numbers, expiries, CVVs, AWS key ids, private keys, API tokens, passwords) in a directory of mixed files. Supported
inputs are CSV/TSV, Parquet, XLSX, JSON/JSONL, PDF, DOCX, EML, HTML, SQL dumps, logs, configs, code, key files and
images. It reports them as SHA-256 hashes and masked strings, never as raw values.

* [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) is the guide to working on the code: design decisions, the problems solved
  along the way, a walkthrough of the codebase and the order to read it in, and detailed setup for Linux, macOS
  and Windows.
* `DESIGN.md` contains the high-level design ([HLD.png](HLD.png), source [HLD.dot](HLD.dot)) and the answers to the design questions.
* `REPORT.md` contains the results, error analysis, performance and limitations.
* `BRIEF.md` and `ANNOTATION_GUIDE.md` are the task and the labelling rules.

## Setup

Python 3.10 to 3.13; 3.12 is the recommended version, and 3.10 and 3.12 are the tested ones. Every dependency installs
from pre-built wheels on Linux, macOS (Intel and Apple silicon) and Windows x64. The test runs were on Linux; see
[DEVELOPER_GUIDE.md §2](DEVELOPER_GUIDE.md#2-setup-on-linux-macos-and-windows) for how macOS and Windows support was
checked, plus details and troubleshooting.

**Linux / macOS**

```bash
python3 -m venv .venv && . .venv/bin/activate          # or: uv venv -p 3.12 .venv
pip install -r requirements.txt
python scripts/check_setup.py                          # checks packages, OCR engines and resources
```

**Windows (PowerShell)**

```powershell
py -3.12 -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts\check_setup.py
```

**Optional system package: Tesseract 5 with English data.** It is the second engine of the OCR ensemble.

| OS | command |
|---|---|
| Debian / Ubuntu | `sudo apt-get install -y tesseract-ocr tesseract-ocr-eng` |
| Fedora | `sudo dnf install -y tesseract tesseract-langpack-eng` |
| Arch | `sudo pacman -S tesseract tesseract-data-eng` |
| macOS | `brew install tesseract` |
| Windows | `winget install -e --id UB-Mannheim.TesseractOCR` |

Tesseract is found on `PATH` or in its usual install folders (including `C:\Program Files\Tesseract-OCR`, which
the Windows installer does not add to `PATH`). For any other location, set `TESSERACT_CMD` to the executable.

The primary OCR engine is RapidOCR (PP-OCR models in ONNX). It installs with pip and ships its models inside the
wheel. If Tesseract is missing, the scanner automatically runs RapidOCR alone. On dev that costs 0.003 of OCR-path
macro-F1 (0.961 vs 0.964).

**No network access is used at scan time.** Name lexicons, the character n-gram model and the name-model weights are
vendored in `src/scanner/resources/`.

## Run

```bash
python scan.py --input-dir data/dev/files --output-dir out/dev
python starter/evaluate.py --labels data/dev/labels --output out/dev
```

The commands are the same on every OS. `out/dev` receives `findings.jsonl`, `columns.jsonl` and `errors.jsonl`.

Optional flags:

| flag | default | meaning |
|---|---|---|
| `--workers N` | `min(4, cores)` | Phase A pool size (extraction + detection) |
| `--ocr-workers N` | `min(3, workers)` | Phase B pool size (OCR pages) |
| `--ocr {ensemble,rapidocr,tesseract}` | `ensemble` | OCR engines |
| `--ner` | off | add spaCy as a feature of the person-name model (`pip install spacy` and `en_core_web_sm`) |
| `--chunk-mb X` | 8 | chunk size for large text files |
| `--stats-file F` | - | per-stage timing summary as JSON |

Environment variables:

| variable | meaning |
|---|---|
| `TESSERACT_CMD` | path of the tesseract executable |
| `SCANNER_MP_START` | worker start method; `spawn` reproduces macOS/Windows behaviour on Linux |
| `SCANNER_OCR` | default for `--ocr` |

## Dev data

`data/` is not committed. Place the provided dev set as `data/dev/files/` and `data/dev/labels/`.

## Reproduce everything

Run from the repository root (on Windows too):

```bash
python -m pytest tests -q                                    # 177 tests, ~5 s

# 1. dev
python scripts/measure.py --input-dir data/dev/files --output-dir out/final-dev        # wall clock + peak memory
python starter/evaluate.py --labels data/dev/labels --output out/final-dev
python scripts/analyze_errors.py                             # FP/FN grouped by cause, slice, entity (in memory)

# 2. hidden-like synthetic set: new formats, unseen names, hard negatives
python scripts/make_synthetic.py --out synthetic --seed 1 --pool test
python scripts/measure.py --input-dir synthetic/files --output-dir out/final-synth
python starter/evaluate.py --labels synthetic/labels --output out/final-synth

# 3. hidden-scale performance: 1M-row CSV + Parquet, 150 MB log, 50 OCR pages (labelled)
python scripts/make_perf_data.py --out perf
python scripts/measure.py --input-dir perf/files --output-dir out/final-perf
python starter/evaluate.py --labels perf/labels --output out/final-perf

# 4. OCR engine study (value recall per image type)
python scripts/ocr_eval.py --engine rapidocr
python scripts/ocr_eval.py --engine tesseract --psm 4
```

Synthetic images are drawn with Liberation fonts on Linux and with Arial / Times New Roman / Courier New on macOS and
Windows, so OCR scores on a set generated there can differ slightly from the ones in REPORT.md.

Rebuilding the vendored resources is optional. It needs network and `pip install -r requirements-dev.txt`:

```bash
python scripts/build_lexicons.py                             # name / surname / English-word / TLD lexicons
python -c "import sys; sys.path[:0]=['src','starter']; from scanner.detect.names import pack_lexicon; pack_lexicon()"
python scripts/make_synthetic.py --out synthetic_train --seed 2 --pool train
python scripts/train_name_model.py --sets data/dev           # char n-gram model + logistic regression
```

## Layout

```
scan.py                     entry point (adds src/ and starter/ to the path)
src/scanner/pipeline.py     walk -> sniff -> plan -> Phase A pool -> Phase B OCR pool -> write
src/scanner/filetype.py     content-based type detection and unreadable-file reasons
src/scanner/extract/        structured.py, pdf.py (layout), documents.py (DOCX/EML/HTML/SQL/PPTX), ocr.py, text.py (chunks)
src/scanner/detect/         validators.py, context.py, text_detector.py (+ merge), names.py, columns.py
src/scanner/output.py       hashing (starter/contract.py), masking, evidence, JSONL writers
scripts/                    setup check, evaluation, error analysis, synthetic and performance data, training, lexicons
tests/                      validators, routing, chunk boundaries, context rules, merge, structured, documents, OCR,
                            contract, cross-platform behaviour
```

## Licences of bundled resources

* PyMuPDF is AGPL-3.0.
* Lexicons are derived from: names-dataset (MIT), the US Census 2010 surname file (public domain), wordfreq (Apache-2.0, data CC-BY-SA) and the IANA TLD list.
* RapidOCR models are Apache-2.0.
