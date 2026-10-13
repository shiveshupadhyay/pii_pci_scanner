"""Data passed between stages.

Extractors produce TextUnits; the detector turns a unit's text into Candidates; the pipeline places candidates in a
file as Findings; output.py turns Findings into hashed and masked records. Raw values live only in Candidate and
Finding objects, inside worker processes.
"""

from dataclasses import dataclass, field


@dataclass
class Candidate:
    """A detected value inside one piece of text. `value` is the raw value (never written to output)."""

    entity: str
    value: str
    start: int  # character offsets in the unit's text; chunk ownership and merging work on these
    end: int
    confidence: float
    detector: str  # which rule fired, e.g. "regex+luhn+issuer:visa"; goes into the evidence
    context: tuple = ()  # labels such as "key: card" (never values); also evidence
    priority: int = 0  # merge() keeps the higher priority when two candidates overlap


@dataclass
class TextUnit:
    """A piece of extracted text plus the context the detector needs.

    key   : external key context (table header, form label) for values that carry no in-line key
    cell  : True for a single table cell (header-driven whole-cell rules apply)
    table : {'has_card': bool, 'has_amex': bool} for cells of a table that holds card numbers
    """

    text: str
    location: dict  # {"page": n}, {"column": c}, {"sheet": s, "column": c} or {}
    extractor: str  # e.g. "pdf-table", "eml-body:base64"; goes into the evidence
    key: str = None
    cell: bool = False
    table: dict = None
    use_ner: bool = False  # prose worth running the optional spaCy feature on (never set for table cells)
    ocr: bool = False  # OCR text: enables the OCR-tolerant rules (fuzzy EXAMPLE, misread greetings)
    order: int = 0  # position of the unit in its file, for a stable output order
    conf_scale: float = 1.0  # multiplied into every confidence (OCR units are less certain)


@dataclass
class Finding:
    file: str
    entity: str
    value: str
    location: dict
    confidence: float
    extractor: str
    detector: str
    context: tuple = ()
    order: tuple = ()  # sort key: (page/sheet/column, offset)


@dataclass
class ColumnLabel:
    file: str
    sheet: str
    column: str
    entity: str
    confidence: float  # the share of the column's non-empty cells detected as `entity`


@dataclass
class FileResult:
    """What scanning one file, chunk or OCR page produces. `error` set means the file is reported as unreadable
    and its findings are dropped."""

    file: str
    findings: list = field(default_factory=list)
    columns: list = field(default_factory=list)
    error: str = None
    stats: dict = field(default_factory=dict)  # timings and counters; "ocr_pending" lists pages left for Phase B
