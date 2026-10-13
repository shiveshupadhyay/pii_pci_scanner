"""Cross-platform behaviour that a plain Linux run would not exercise:

* macOS and Windows start pool workers with "spawn", so workers inherit nothing from the parent process;
* output paths use "/" whatever the OS separator is;
* output files are ASCII with "\\n" line ends, because starter/evaluate.py opens them without an encoding;
* Tesseract is found even when it is not on PATH;
* no text-mode open() in the code relies on the locale encoding (cp1252 on most Windows machines).
"""

import ast
import json
import os
import pathlib
import unicodedata

from scanner import pipeline
from scanner.extract import ocr

REPO = pathlib.Path(__file__).resolve().parents[1]


def _sample_tree(root):
    deep = root / "nested" / "deeper"
    deep.mkdir(parents=True)
    (deep / "notes_é.txt").write_text("From: Hannah Brown <hannah20@ardentlabs.ai>\n"
                                      "the new password is sarah2010$ - please change it\n", encoding="utf-8")
    (root / "nested" / "people.csv").write_text("name,email\nPriya Sharma,priya@orbitpay.in\n"
                                                "Rahul Verma,rahul.v@saffronlabs.in\n", encoding="utf-8")
    (root / "empty.csv").write_bytes(b"")


def _outputs(out):
    return {n: (out / n).read_bytes() for n in ("findings.jsonl", "columns.jsonl", "errors.jsonl")}


def test_walk_returns_forward_slash_paths(tmp_path):
    _sample_tree(tmp_path)
    rels = [unicodedata.normalize("NFC", rel) for rel, _ in pipeline.walk(str(tmp_path))]
    assert "nested/deeper/notes_é.txt" in rels
    assert not any("\\" in r for r in rels)


def test_output_files_are_ascii_with_lf(tmp_path):
    inp, out = tmp_path / "in", tmp_path / "out"
    inp.mkdir()
    _sample_tree(inp)
    pipeline.run(str(inp), str(out), workers=1)
    for name, data in _outputs(out).items():
        data.decode("ascii")  # raises if any raw UTF-8 slipped through
        assert b"\r" not in data, name
    files = {unicodedata.normalize("NFC", json.loads(l)["file"])
             for l in (out / "findings.jsonl").read_text(encoding="ascii").splitlines()}
    assert "nested/deeper/notes_é.txt" in files


def test_spawn_workers_give_the_same_output(tmp_path, monkeypatch):
    """Run once with the platform's default start method (fork on Linux) and once with spawn: identical bytes."""
    inp = tmp_path / "in"
    inp.mkdir()
    _sample_tree(inp)
    results = {}
    for method in ("default", "spawn"):
        if method == "default":
            monkeypatch.delenv("SCANNER_MP_START", raising=False)
        else:
            monkeypatch.setenv("SCANNER_MP_START", method)
        out = tmp_path / method
        pipeline.run(str(inp), str(out), workers=2)
        results[method] = _outputs(out)
    assert results["default"] == results["spawn"]
    assert results["spawn"]["findings.jsonl"].count(b"\n") >= 5


def test_tesseract_cmd_env_var_wins(tmp_path, monkeypatch):
    fake = tmp_path / "tesseract.exe"
    fake.write_bytes(b"")
    monkeypatch.setenv("TESSERACT_CMD", str(fake))
    assert ocr.find_tesseract() == str(fake)


def test_tesseract_lookup_without_path(monkeypatch):
    monkeypatch.delenv("TESSERACT_CMD", raising=False)
    monkeypatch.setattr(ocr.shutil, "which", lambda name: None)
    found = ocr.find_tesseract()
    assert found is None or os.path.isfile(found)


def _text_opens_without_encoding(tree):
    """Line numbers of text-mode open() / gzip.open() / read_text() / write_text() calls without encoding=."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f, kws = node.func, {k.arg for k in node.keywords}
        name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else ""
        if name in ("read_text", "write_text") and isinstance(f, ast.Attribute):  # Path methods, not our helper
            if "encoding" not in kws:
                yield node.lineno
            continue
        if name != "open":
            continue
        if isinstance(f, ast.Name) or (isinstance(f.value, ast.Name) and f.value.id == "io"):
            default = "r"  # builtin open() is text mode by default
        elif isinstance(f.value, ast.Name) and f.value.id in ("gzip", "bz2", "lzma"):
            default = "rb"  # ...but these default to binary
        else:
            continue  # pymupdf.open, Image.open, ZipFile.open: not text files
        mode = node.args[1] if len(node.args) > 1 else next((k.value for k in node.keywords if k.arg == "mode"), None)
        mode = mode.value if isinstance(mode, ast.Constant) else default
        if "b" not in mode and "encoding" not in kws:
            yield node.lineno


def test_no_text_io_relies_on_the_locale_encoding():
    files = [REPO / "scan.py", *sorted((REPO / "src").rglob("*.py")), *sorted((REPO / "scripts").glob("*.py")),
             *sorted((REPO / "tests").glob("*.py"))]
    bad = []
    for py in files:
        tree = ast.parse(py.read_text(encoding="utf-8"))
        bad += [f"{py.relative_to(REPO)}:{line}" for line in _text_opens_without_encoding(tree)]
    assert not bad, "text-mode file I/O without encoding= (cp1252 on Windows): " + ", ".join(bad)
