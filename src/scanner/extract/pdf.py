"""PDF extraction with layout context (PyMuPDF word boxes).

page.get_text() alone loses the structure the detector needs: which header a table cell sits under, which label
a form value belongs to, and which column of a two-column page a sentence comes from. So pages are rebuilt from
word boxes (x0, y0, x1, y1, text):
  * no text layer (and an image)        -> OCR (location {"page": n})
  * tables: a line of identifier-like headers followed by rows whose words start at the header x positions.
            A word belongs to the last header starting at or left of it; rows are never split on gaps, because
            adjacent columns can be only 2-3 pt apart
  * two-column pages: split at a vertical gutter, each column is its own context unit
  * forms ("Label    value") and flowing text: rendered line by line with a tab between distant words, so
            the label sits immediately left of its value
"""

import re

from ..units import TextUnit
from .tables import looks_like_header, table_context

Y_TOL = 2.5  # points: words whose tops differ by less than this are on the same visual line
MIN_TEXT_CHARS = 12  # fewer text-layer characters than this (plus an image) means a scanned page


def open_pdf(path):
    import pymupdf

    doc = pymupdf.open(path)
    # A password-protected PDF opens without error but shows no content; it has to be reported as an error file.
    if doc.needs_pass or doc.is_encrypted:
        doc.close()
        return None, "encrypted pdf"
    if doc.page_count == 0:
        doc.close()
        return None, "pdf has no pages"
    return doc, None


def pdf_text_only(path):
    doc, err = open_pdf(path)
    if err:
        return []
    try:
        return [p.get_text() for p in doc]
    finally:
        doc.close()


def pdf_pages(path, images=True):
    """Yield ('text', page_no, units) or ('ocr', page_no, PIL.Image | None) per page.
    images=False defers OCR: the page number is yielded without rendering. Raises ValueError on bad PDFs."""
    doc, err = open_pdf(path)
    if err:
        raise ValueError(err)
    try:
        for pno, page in enumerate(doc, 1):
            words = _words(page)
            nchars = sum(len(w[4]) for w in words)
            # A page with (almost) no text but an image or vector drawings is a scan. A page with neither is just
            # blank and is skipped. A few characters are tolerated: scanners often stamp a page number.
            if nchars < MIN_TEXT_CHARS and (page.get_images() or page.get_drawings()):
                yield "ocr", pno, (_page_image(doc, page) if images else None)
                continue
            yield "text", pno, page_units(words, page.rect.width, pno)
    finally:
        doc.close()


def pdf_page_image(path, pno):
    doc, err = open_pdf(path)
    if err:
        raise ValueError(err)
    try:
        return _page_image(doc, doc[pno - 1])
    finally:
        doc.close()


def _words(page):
    import pymupdf

    # By default PyMuPDF drops text outside the page's visible box. Text printed past the edge is still in the
    # file and still sensitive (dev has a phone number at x > 595 pt), so read with an unbounded clip.
    try:
        return page.get_text("words", clip=pymupdf.INFINITE_RECT())
    except Exception:
        return page.get_text("words")


def _page_image(doc, page):
    from PIL import Image
    import io

    imgs = page.get_images(full=True)
    if not imgs and not page.get_drawings():
        return None
    # A scanned page is usually one embedded image. Taking it as stored keeps its native resolution and avoids
    # re-rasterising; tiny images (logos, icons) don't qualify, and then the page is rendered instead.
    if len(imgs) == 1:
        try:
            info = doc.extract_image(imgs[0][0])
            im = Image.open(io.BytesIO(info["image"]))
            im.load()
            if im.width >= 400:
                return im
        except Exception:
            pass
    pix = page.get_pixmap(dpi=200)  # 200 dpi: text big enough for OCR without huge images
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


# ---------------------------------------------------------------------------- layout


def _lines(words):
    """Group words into visual lines (top-to-bottom, then left-to-right). Comparing each word with the line's first
    word, within Y_TOL, keeps slightly misaligned baselines (mixed font sizes) on one line."""
    ws = sorted(words, key=lambda w: (round(w[1], 1), w[0]))
    lines = []
    for w in ws:
        if lines and abs(lines[-1]["y"] - w[1]) <= Y_TOL:
            lines[-1]["words"].append(w)
        else:
            lines.append({"y": w[1], "words": [w]})
    for ln in lines:
        ln["words"].sort(key=lambda w: w[0])
    return lines


def _render(words):
    """Join words of one visual line; distant words are separated by a tab (label/value boundary)."""
    out, prev = [], None
    for w in words:
        if prev is not None:
            gap = w[0] - prev[2]
            height = max(1.0, w[3] - w[1])
            # A gap wider than 1.5 x the text height is not a word space but a jump to another field. The tab
            # makes "Mobile<TAB>9876543210" read like "mobile: 9876543210" to the key-left-of-value rules.
            out.append("\t" if gap > 1.5 * height else " ")
        out.append(w[4])
        prev = w
    return "".join(out)


def _find_tables(lines, tol=2.0):
    """Return list of (header_idx, last_idx, header_cells). A header word starts a column when data rows
    below have words starting at the same x (cells are left-aligned under their header)."""
    tables, i = [], 0
    while i < len(lines) - 1:
        hw = lines[i]["words"]
        if len(hw) >= 3:
            # Treat line i as a possible header and grow the table downwards while the following lines have
            # enough words starting at the same x positions as header words (cells are left-aligned under their
            # header in generated reports). The first line that stops aligning ends the table.
            starts = [w[0] for w in hw]
            data, j = [], i + 1
            while j < len(lines):
                lw = lines[j]["words"]
                hits = sum(1 for w in lw if any(abs(w[0] - x) <= tol for x in starts))
                if hits >= max(2, min(len(hw), len(lw)) // 2) and lw[0][0] >= starts[0] - 3:
                    data.append(lw)
                    j += 1
                else:
                    break
            if data:
                # A header word opens a column only if at least half of the data rows have a word starting under
                # it; otherwise it is the second word of the previous header ("Card number"). With a laxer
                # threshold (30% was tried), coincidental alignment turned label/value forms into tables.
                cells = []
                for k, w in enumerate(hw):
                    aligned = sum(any(abs(d[0] - w[0]) <= tol for d in lw) for lw in data)
                    if k == 0 or aligned >= max(1, 0.5 * len(data)):
                        cells.append({"x0": w[0], "x1": w[2], "text": w[4]})
                    else:
                        cells[-1]["text"] += " " + w[4]
                        cells[-1]["x1"] = w[2]
                if len(cells) >= 3 and _is_table_header([c["text"] for c in cells]):
                    tables.append((i, j - 1, cells))
                    i = j
                    continue
        i += 1
    return tables


def _is_table_header(cells):
    """Column names, not the first line of a form: >=3 distinct identifier-like cells, mostly single tokens
    (snake_case / camelCase) or known field names."""
    from ..detect.context import key_words
    if len(cells) < 3 or not looks_like_header(cells):
        return False
    if any(re.search(r"\d{4,}|@", c) for c in cells):  # a header never contains an id or an e-mail address
        return False
    low = [c.lower() for c in cells]
    if len(set(low)) < len(low):  # repeated cells are a row of values ("Yes Yes No"), not column names
        return False
    # Real headers are mostly single tokens (snake_case, camelCase) or recognisable field names. A form's first
    # line ("Applicant details  Reference  Status") is neither.
    single = sum(" " not in c for c in cells)
    known = sum(bool(_KNOWN_HEADER.search(" ".join(key_words(c)))) for c in cells)
    return single >= 0.5 * len(cells) or known >= 0.6 * len(cells)


_KNOWN_HEADER = re.compile(r"\b(?:name|email|e mail|phone|mobile|card|number|no|id|exp|expiry|cvv|ssn|pan|aadhaar|"
                           r"password|token|key|secret|status|amount|date|city|address|code|ref|type|account|"
                           r"created|updated|user|customer|employee|order|invoice|total|balance)\b")


def _table_rows(lines, hdr):
    """Cells per row: each word goes to the last header column that starts at or left of it (2 pt of slack for
    rounding). A cell value may have several words ("Meera Khan") and still lands in one column."""
    xs = [c["x0"] for c in hdr]
    rows = []
    for ln in lines:
        cells = [[] for _ in xs]
        for w in ln["words"]:
            k = 0
            for idx, x in enumerate(xs):
                if w[0] >= x - 2.0:
                    k = idx
            cells[k].append(w[4])
        rows.append([" ".join(c) for c in cells])
    return rows


def _gutter(lines, width):
    """x of a vertical gutter splitting the page into two text columns, or None.

    Candidate bands are x-ranges in the middle third of the page that no word crosses while >= 2 lines have words
    on both sides. The band nearest the page centre wins, because a page with two forms side by side also has
    empty bands between each form's labels and values."""
    if not lines:
        return None
    # Only the middle of the page can hold a column gutter; the gaps between a form's labels and values sit
    # further left or right.
    lo, hi, step = width * 0.35, width * 0.65, 2.0
    bands, cur = [], None
    x = lo
    while x <= hi:
        # x qualifies when no word on any line crosses it, and at least two lines have words on both sides of it
        crossing = both = 0
        for ln in lines:
            ws = ln["words"]
            if any(w[0] < x < w[2] for w in ws):
                crossing += 1
                break
            if any(w[2] <= x for w in ws) and any(w[0] >= x for w in ws):
                both += 1
        ok = crossing == 0 and both >= 2
        if ok:
            cur = [x, x] if cur is None else [cur[0], x]
        elif cur is not None:
            bands.append(cur)
            cur = None
        x += step
    if cur is not None:
        bands.append(cur)
    if not bands:
        return None
    centre = width / 2
    a, b = min(bands, key=lambda bd: (abs((bd[0] + bd[1]) / 2 - centre), -(bd[1] - bd[0])))
    x0 = (a + b) / 2
    # Widen the band to the real empty space around it; a gap narrower than 15 pt is just a wide word space.
    left = max((w[2] for ln in lines for w in ln["words"] if w[2] <= x0), default=x0)
    right = min((w[0] for ln in lines for w in ln["words"] if w[0] >= x0), default=x0)
    if right - left < 15:
        return None
    g = (left + right) / 2
    # each side must hold real lines (label + value, or prose), not just the labels or just the values of a form
    for side in (lambda w: w[2] <= g, lambda w: w[0] >= g):
        counts = [sum(1 for w in ln["words"] if side(w)) for ln in lines]
        counts = [c for c in counts if c]
        if len(counts) < 2 or sum(counts) / len(counts) < 2.0:
            return None
    return g


def page_units(words, width, pno):
    """TextUnits for one page: first every table cell (keyed by its header), then the remaining lines as one text
    region, or two when the page has a column gutter."""
    loc = {"page": pno}
    lines = _lines(words)
    units = []
    tables = _find_tables(lines)
    used = set()
    for t_i, (a, b, hdr) in enumerate(tables):
        rows = _table_rows(lines[a + 1:b + 1], hdr)
        header = [c["text"] for c in hdr]
        ctx = table_context(rows)
        for r_i, row in enumerate(rows):
            for c_i, cell in enumerate(row):
                if cell:
                    units.append(TextUnit(cell, loc, "pdf-table", key=header[c_i], cell=True, table=ctx,
                                          order=a * 1000 + r_i * 50 + c_i))
        used.update(range(a, b + 1))
    # Table lines are taken out before looking for a gutter: a table's own column gaps would look like one.
    rest = [ln for i, ln in enumerate(lines) if i not in used]
    g = _gutter(rest, width)
    regions = []
    if g is not None:
        left = [{"y": ln["y"], "words": [w for w in ln["words"] if w[2] <= g]} for ln in rest]
        right = [{"y": ln["y"], "words": [w for w in ln["words"] if w[0] >= g]} for ln in rest]
        regions = [("pdf-two-column", [ln for ln in left if ln["words"]]),
                   ("pdf-two-column", [ln for ln in right if ln["words"]])]
    else:
        regions = [("pdf-text", rest)]
    for r_i, (name, lns) in enumerate(regions):
        if not lns:
            continue
        text_lines, prev_y = [], None
        for ln in lns:
            # a vertical gap of more than ~2 line heights starts a new paragraph: keep it as a blank line
            if prev_y is not None and ln["y"] - prev_y > 2.2 * max(1.0, ln["words"][0][3] - ln["words"][0][1]):
                text_lines.append("")
            text_lines.append(_render(ln["words"]))
            prev_y = ln["y"]
        text = "\n".join(text_lines)
        # Mostly "label<TAB>value" lines make a form. That is only a label for the evidence, plus no spaCy pass,
        # since forms are not prose.
        kind = "pdf-form" if sum("\t" in l for l in text_lines) >= max(2, len([l for l in text_lines if l]) // 2) \
            else name
        units.append(TextUnit(text, loc, kind, use_ner=kind != "pdf-form", order=100000 + r_i))
    return units
