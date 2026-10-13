"""Helpers shared by every extractor that sees a table (PDF, DOCX, HTML, SQL).

A document table becomes one TextUnit per cell, keyed by its column header, so a cell is judged exactly like a
value written after `header=` in a log line.
"""

import re

from ..detect.validators import card_brand, card_ok
from ..units import TextUnit

_CARDISH = re.compile(r"^\s*\d[\d \-]{11,22}\d\s*$")


def table_context(rows):
    """{'has_card', 'has_amex'} for a table given as a list of rows (lists of cell strings).

    The annotation guide counts an expiry date or CVV only next to a card number, and a 4-digit CVV only with an
    Amex card. Inside a table "next to" means "somewhere in the same table", so this is computed once per table
    and handed to every cell."""
    has_card = has_amex = False
    for row in rows:
        for cell in row:
            if cell and _CARDISH.match(cell):
                d = re.sub(r"\D", "", cell)
                if card_ok(d):
                    has_card = True
                    has_amex = has_amex or card_brand(d) == "amex"
    return {"has_card": has_card, "has_amex": has_amex}


def looks_like_header(cells):
    """Column names are short labels that start with a letter. A run of 4+ digits means data (a phone, an id),
    while `file_sha256` or `address2` are still headers."""
    cells = [c.strip() for c in cells if c and c.strip()]
    if len(cells) < 2:
        return False
    ok = sum(bool(re.fullmatch(r"[A-Za-z][A-Za-z0-9_ .\-/()#]{0,40}", c)) and not re.search(r"\d{4,}", c)
             for c in cells)
    return ok >= 0.8 * len(cells)


def table_units(rows, location, extractor, order=0):
    """Units for a table: header row gives column context; 2-column label/value tables give row context."""
    rows = [[(c or "").strip() for c in r] for r in rows if any((c or "").strip() for c in r)]
    if not rows:
        return []
    ctx = table_context(rows)
    units = []
    ncols = max(len(r) for r in rows)
    # A two-column table whose left cells are labels ("Name | Priya Sharma", "Mobile | 98...") is a form laid out
    # as a table: each value takes its own row's label as key, not a column header. (rows[0][1:2] * 2 just asks
    # whether the first *value* reads like a header word, i.e. whether row 0 is really a header row.)
    label_value = ncols == 2 and sum(bool(re.fullmatch(r"[A-Za-z][A-Za-z0-9_ .\-/()#]{0,40}:?", r[0]))
                                     for r in rows if r) >= 0.8 * len(rows) and not looks_like_header(rows[0][1:2] * 2)
    if label_value:
        for i, r in enumerate(rows):
            if len(r) == 2 and r[1]:
                units.append(TextUnit(r[1], location, extractor + "-form", key=r[0].rstrip(":"), cell=True,
                                      table=ctx, order=order + i))
        return units
    if looks_like_header(rows[0]) and len(rows) > 1:
        header, body = rows[0], rows[1:]
    else:
        # no recognisable header row: every cell is detected on its value alone
        header, body = [""] * ncols, rows
    for i, r in enumerate(body):
        for j, cell in enumerate(r):
            if cell:
                units.append(TextUnit(cell, location, extractor, key=header[j] if j < len(header) else "",
                                      cell=True, table=ctx, order=order + i * 1000 + j))
    return units
