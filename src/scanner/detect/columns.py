"""Column-level classification and per-cell findings for structured tables.

Pass 1 detects every cell with the column header as context; the table is then known to hold card numbers
(any column dominated by CARD_NUMBER) and pass 2 re-runs expiry/CVV columns with that table context.
A column is labelled with entity E when E covers >= DOMINANCE of its non-empty cells (look-alike columns of
order ids, UUIDs, last-4 digits, placeholder e-mails never reach that share).
Consensus guard: in a column where only a minority of card/Aadhaar-shaped values validate, the validating ones
are coincidences (10% of random digit strings pass Luhn/Verhoeff) and are dropped.
"""

import re
from collections import Counter

from .context import is_cvv_key, is_expiry_key, key_words

DOMINANCE = 0.5
_CARD_SHAPE = re.compile(r"^\d[\d \-]{11,22}\d$")
_AADHAAR_SHAPE = re.compile(r"^[2-9]\d{3}[ \-]?\d{4}[ \-]?\d{4}$")


def classify_table(detector, table, sample_only=False):
    """Returns (cell_findings, column_labels).

    cell_findings: list of (column_index, row_index, Candidate)
    column_labels: list of (column_index, entity, share)
    """
    cols = table.columns
    ncol = len(cols)
    col_cells = [[] for _ in range(ncol)]
    for r_i, row in enumerate(table.rows):
        for c_i in range(min(ncol, len(row))):
            v = row[c_i]
            if v is not None and str(v).strip():
                col_cells[c_i].append((r_i, str(v)))

    # Pass 1: every cell with only its own header as context.
    results = [None] * ncol
    for c_i in range(ncol):
        results[c_i] = _detect_column(detector, cols[c_i], col_cells[c_i], {})

    # Table context from pass 1. A table "holds cards" only if some column is *mostly* cards: a stray Luhn-valid
    # order number must not unlock expiry and CVV detection for the whole table.
    has_card = has_amex = False
    for c_i in range(ncol):
        n = len(col_cells[c_i])
        cards = [c for _, cands in results[c_i] for c in cands if c.entity == "CARD_NUMBER"]
        if n and len(cards) / n >= DOMINANCE:
            has_card = True
            has_amex = has_amex or any("amex" in c.detector for c in cards)
    tctx = {"has_card": has_card, "has_amex": has_amex}
    # Pass 2: only expiry / CVV columns need the table context, so only they are detected again.
    if has_card:
        for c_i in range(ncol):
            w = key_words(cols[c_i])
            if is_expiry_key(w) or is_cvv_key(w):
                results[c_i] = _detect_column(detector, cols[c_i], col_cells[c_i], tctx)

    findings, labels = [], []
    for c_i in range(ncol):
        cells = col_cells[c_i]
        n = len(cells)
        if not n:
            continue
        res = _consensus(cells, results[c_i])
        # count *cells* per entity (a cell with two e-mails still counts once), then label the column with the
        # dominant entity if it covers at least half of the non-empty cells
        counts = Counter()
        for _, cands in res:
            for e in {c.entity for c in cands}:
                counts[e] += 1
        if counts:
            ent, cnt = counts.most_common(1)[0]
            if cnt / n >= DOMINANCE:
                labels.append((c_i, ent, round(cnt / n, 3)))
        # Cell findings are reported even in columns that are not labelled (a notes column with one phone number).
        # sample_only is set for >50k-row tables, which are scored on column labels alone.
        if not sample_only:
            for (r_i, _), (_, cands) in zip(cells, res):
                for c in cands:
                    findings.append((c_i, r_i, c))
    return findings, labels


def _detect_column(detector, header, cells, tctx):
    # Columns repeat values a lot (cities, statuses, the same test e-mail). Detection depends only on (value,
    # header, table context), so each distinct value is detected once per column.
    memo = {}
    out = []
    for r_i, v in cells:
        key = v
        if key not in memo:
            memo[key] = detector.detect(v, key=header, cell=True, table=tctx)
        out.append((r_i, memo[key]))
    return out


def _consensus(cells, res):
    # Compare "how many cells look like cards / Aadhaar" with "how many actually validate". In a real card column
    # nearly all validate; in a column of random account numbers about 10% pass by chance. The 5-cell minimum
    # keeps tiny columns out of the vote.
    card_shape = sum(bool(_CARD_SHAPE.match(v.strip())) for _, v in cells)
    aad_shape = sum(bool(_AADHAAR_SHAPE.match(v.strip())) for _, v in cells)
    cards = sum(any(c.entity == "CARD_NUMBER" for c in cands) for _, cands in res)
    aad = sum(any(c.detector.endswith("aadhaar") for c in cands) for _, cands in res)
    drop_cards = card_shape >= 5 and cards < 0.5 * card_shape
    drop_aad = aad_shape >= 5 and aad < 0.5 * aad_shape
    if not (drop_cards or drop_aad):
        return res
    out = []
    for r_i, cands in res:
        keep = [c for c in cands if not ((drop_cards and c.entity == "CARD_NUMBER") or
                                         (drop_aad and c.detector.endswith("aadhaar")))]
        out.append((r_i, keep))
    return out
