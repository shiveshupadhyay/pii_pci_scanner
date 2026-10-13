"""PERSON_NAME detection: 2-3 capitalised tokens scored with context cues and name lexicons.

Lexicons (src/scanner/resources, built by scripts/build_lexicons.py) come from public name statistics, never from the
dev labels. Weights live in resources/name_model.json (logistic regression, see scripts/train_name_model.py); the
defaults below are used when the file is missing.

Names have no checksum and no fixed format, and capitalised word pairs are everywhere ("Grace Period", "Hong Kong",
"Account Manager"). The scorer weighs three kinds of evidence:
  * the tokens themselves: how common each is as a first name, as a surname and as an English word, and how
    name-like its spelling is (character n-grams; this covers names missing from the lexicons);
  * the context: "From:", a name column header, an <email> right after it, "for X (id=...)";
  * a learned logistic regression that combines both into one probability.
"""

import gzip
import hashlib
import json
import math
import os
import re

from .context import is_name_key

RES = os.path.join(os.path.dirname(os.path.dirname(__file__)), "resources")

# One capitalised token, with an optional hyphenated or apostrophe part: "Priya", "Jean-Luc", "O'Brien".
_TOKEN = r"[A-Z][a-z]{1,20}(?:[-'][A-Z]?[a-z]{1,20})?"
# Runs of up to 5 such tokens; trim_tokens() then strips greetings and labels at either end ("Hi Priya Sharma
# Thanks") and keeps the run only if 2-3 tokens remain. The lookarounds skip tokens glued to e-mails, paths and
# identifiers.
NAME_CAND_RE = re.compile(rf"(?<![\w@./\\-])({_TOKEN}(?:[ \t]{{1,2}}{_TOKEN}){{1,4}})(?![\w@/\\-])")

# Words that are never part of a person's name in this kind of data: greetings and sign-offs, field labels, company
# suffixes, job titles, product names, months and weekdays, status words. A candidate containing one is
# discarded outright, before any scoring.
STOP = set("""
hi hello dear hey thanks thank regards best cheers sincerely warm kind from to cc bcc subject re fwd fw note notes fyi
attn attention author date sent name names full signed by customer employee applicant reporter assignee requester
contact mr mrs ms miss mx dr prof sir madam team support admin the a an and or of for with in on at via per your our my
inc ltd llc llp plc corp corporation company co group holdings labs lab capital traders trading health healthcare bank
systems technologies technology tech solutions services partners ventures industries enterprises international global
consulting analytics software networks media studio studios university college school institute foundation hospital
clinic pharma motors airlines insurance finance financial securities investments logistics retail foods energy
cloud data digital online store shop market markets agency associates department dept office division unit board
council committee ministry government authority bureau manager managers director engineer developer officer lead head
president vice chief executive analyst specialist consultant coordinator administrator assistant associate intern
representative agent owner founder supervisor accountant architect designer account accounts sales marketing
operations payroll billing payments payment product project program service success relations resources human legal
compliance security risk audit treasury google chrome apache kafka microsoft windows office excel word outlook teams
slack zoom amazon aws azure oracle linux ubuntu android iphone ipad macos safari firefox edge github gitlab jira
confluence docker kubernetes redis postgres mysql mongodb elasticsearch python java javascript node react angular
django flask rails visa mastercard amex rupay paypal stripe razorpay whatsapp telegram facebook twitter instagram
linkedin youtube netflix spotify uber airbnb january february march april june july august september october november
december monday tuesday wednesday thursday friday saturday sunday jan feb mar apr jun jul aug sep sept oct nov dec mon
tue wed thu fri sat sun period call policy request update question summary report statement invoice order details
register schedule form number card address street road avenue city state country total amount balance due time
status active pending closed failed approved rejected new old error warning info debug true false yes no none null
test demo sample example default primary secondary main general public private internal external quick access incident
escalation onboarding issue meeting minutes agenda draft final version release sprint welcome please regarding
reference ticket case issue bug feature task story epic customer client vendor vendors supplier partner user users
member members guest guests staff people person private limited north south east central street
""".split())

# Hand-set fallback weights, used only if resources/name_model.json is missing. The dict order also defines the
# feature order (FEATURES) used by the training script.
DEFAULT_WEIGHTS = {
    "bias": -3.2, "first": 2.6, "last": 2.4, "eng_first": -1.6, "eng_last": -2.2, "first_as_last": 0.3,
    "last_as_first": 0.4, "three": -0.3, "mid": 0.6, "strong": 4.0, "medium": 1.6, "addr": 3.0, "line_start": 0.4,
    "spacy_person": 1.2, "spacy_org": -1.5, "absent_both": -2.0, "charlm_first": 1.0, "charlm_last": 1.0,
    "non_eng_first": 0.5, "non_eng_last": 0.5, "spec_first": 1.0, "spec_last": 1.0,
}
FEATURES = list(DEFAULT_WEIGHTS)[1:]
THRESHOLD = 0.5

# Context cues, from strongest to weakest:
#   strong: a header or label that announces a person (From:, Author:, Customer name:, Assignee ...)
#   medium: a preposition or verb that usually precedes a person (for, with, escalated to, spoke with, Dear ...),
#           or "(id=123" right after the candidate
#   addr:   an address in brackets right after the name ("Priya Sharma <priya@x.io>")
_STRONG_LEFT = re.compile(
    r"(?:^|[\n|;,(])\s*(?:from|to|cc|bcc|reply-to|author|authored by|sender|recipient|reporter|assignee|requester|"
    r"requestor|signed by|approved by|prepared by|reviewed by|submitted by|created by|owner|name|full name|"
    r"customer|customer name|client|client name|employee|employee name|applicant|applicant name|account holder|"
    r"card ?holder|cardholder name|contact person|contact name|holder name|beneficiary|nominee|patient|attendee|"
    r"passenger|guest|member|agent)\s*[:=\-]?\s*[\"']?$", re.I)
_MEDIUM_LEFT = re.compile(r"(?:\b(?:for|with|by|to|from|cc|dear|hi|hello|thanks|thank you|looped in|loop in|"
                          r"spoke with|spoke to|talked to|met|meet|met with|assigned to|escalated to|forwarded to|"
                          r"reach out to|contact|ask|asked|cc'?d|ping|per|via|and|or|is|was|named|called|mr\.?|mrs\.?"
                          r"|ms\.?|dr\.?)\s+[\"']?)$", re.I)
_ADDR_RIGHT = re.compile(r"^\s*[<(\[][^<>\s@]+@[^<>\s]+[>)\]]")
_ID_RIGHT = re.compile(r"^\s*\((?:id|uid|user|emp|employee|customer)?\s*[:=#]?\s*\d", re.I)


def _load_tsv(name):
    path = os.path.join(RES, name)
    out = {}
    if not os.path.exists(path):
        return out
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            out[p[0]] = p[1:]
    return out


# uint8 quantisation of (first, first_log, last, last_log, eng). Scores in 0..1 get 200 steps, log-frequencies
# (0..~8) get steps of 1/30. Far finer than the model needs, and it fits each feature in one byte.
SCALE = (200.0, 30.0, 200.0, 30.0, 30.0)


def _tok_hash(tok):
    # Tokens are stored as 64-bit hashes, not strings. With ~400k tokens the chance of any collision is around 1 in
    # 10^8, and a collision would only blur one token's statistics.
    return int.from_bytes(hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest(), "little")


def pack_lexicon():
    """Build sorted 64-bit token hashes + uint8 features from the TSV lexicons (scripts/build_lexicons.py)."""
    import numpy as np

    first = _load_tsv("first_names.tsv.gz")
    last = _load_tsv("surnames.tsv.gz")
    eng = _load_tsv("english_zipf.tsv.gz")
    toks = sorted(set(first) | set(last) | set(eng))
    feats = np.zeros((len(toks), 5), dtype=np.uint8)
    for i, tok in enumerate(toks):
        fs = fl = ls = ll = 0.0
        # rank = best rank of the name in any country's top list, cnt = in how many countries' lists it appears.
        # The *_log variants are on a log-frequency scale, so they can be compared with English word frequency
        # (zipf) in the "more a name than a word?" features.
        if tok in first:
            rank, cnt = int(first[tok][0]), int(first[tok][1])
            fs = NameScorer._rank_score(rank, cnt)
            fl = 6.0 - math.log10(max(rank, 1)) + 0.1 * min(cnt, 10)
        if tok in last:
            rank, cnt, census = (int(x) for x in last[tok])
            nd = NameScorer._rank_score(rank, cnt) if cnt else 0.0
            # census = number of US people with the surname; 10^5.5 (~300k) and above counts as fully common
            cs = min(1.0, math.log10(census) / 5.5) if census > 0 else 0.0
            ls = max(nd, cs)
            ll = max(math.log10(census) if census > 0 else 0.0,
                     (6.0 - math.log10(max(rank, 1)) + 0.1 * min(cnt, 10)) if cnt else 0.0)
        e = float(eng[tok][0]) if tok in eng else 0.0
        feats[i] = [min(255, round(v * k)) for v, k in zip((fs, fl, ls, ll, e), SCALE)]
    hashes = np.array([_tok_hash(t) for t in toks], dtype=np.uint64)
    order = np.argsort(hashes)  # sorted by hash, so a lookup is one binary search (feat())
    np.savez_compressed(os.path.join(RES, "lexicon.npz"), hashes=hashes[order], feats=feats[order])
    return len(toks)


class NameScorer:
    """Name / surname / English-word statistics for ~400k tokens live in two numpy arrays (sorted 64-bit token
    hashes, uint8 features): ~5 MB per worker instead of >100 MB of Python dicts.

    The dicts were the single biggest memory cost: with four workers they alone took close to half a gigabyte."""

    def __init__(self):
        import numpy as np

        self._np = np
        path = os.path.join(RES, "lexicon.npz")
        if not os.path.exists(path):
            pack_lexicon()
        z = np.load(path)
        self._hashes, self._feats = z["hashes"], z["feats"]
        self._cache = {}
        self.first = _View(self, 0)
        self.first_log = _View(self, 1)
        self.last = _View(self, 2)
        self.last_log = _View(self, 3)
        self.eng = _View(self, 4)
        self.charlm = {}
        self.charlm_default = 0.0
        cpath = os.path.join(RES, "charlm.json.gz")
        if os.path.exists(cpath):
            with gzip.open(cpath, "rt", encoding="utf-8") as f:
                cm = json.load(f)
            self.charlm = cm["logodds"]
            self.charlm_default = cm.get("default", 0.0)
        self.weights = dict(DEFAULT_WEIGHTS)
        self.threshold = THRESHOLD
        mpath = os.path.join(RES, "name_model.json")
        if os.path.exists(mpath):
            with open(mpath, encoding="utf-8") as f:
                m = json.load(f)
            self.weights.update(m.get("weights", {}))
            self.threshold = m.get("threshold", THRESHOLD)

    def feat(self, tok):
        """(first, first_log, last, last_log, eng) for a lower-case token, zeros when unknown."""
        r = self._cache.get(tok)
        if r is not None:
            return r
        h = self._np.uint64(_tok_hash(tok))
        i = int(self._np.searchsorted(self._hashes, h))  # binary search in the sorted hash array
        if i < len(self._hashes) and self._hashes[i] == h:
            r = tuple(float(v) / k for v, k in zip(self._feats[i], SCALE))
        else:
            r = (0.0, 0.0, 0.0, 0.0, 0.0)
        # The same few thousand tokens come up again and again in a log. The cache is simply emptied when it gets
        # big, which bounds memory with no LRU bookkeeping.
        if len(self._cache) > 200_000:
            self._cache.clear()
        self._cache[tok] = r
        return r

    @staticmethod
    def _rank_score(rank, n_countries):
        # Half "how high in some country's top-2000 list" (log scale), half "in how many countries' lists". A name
        # common across many countries is safer evidence than one that is merely top-ranked in a single one.
        rank_part = max(0.0, 1.0 - math.log(max(rank, 1)) / math.log(2000))
        return 0.5 * rank_part + 0.5 * min(n_countries, 10) / 10.0

    def name_likeness(self, token):
        """Character n-gram naive Bayes: average log-odds name vs English word, squashed to 0..1.

        This is the feature that generalises to names the lexicons have never seen: "Cheehong" and "Romlin" are
        in no list, but their 2-4 letter pieces (with ^ and $ marking word start and end) look like names, not
        like English words."""
        if not self.charlm:
            return 0.5
        t = "^" + token.lower() + "$"
        grams = [t[i:i + n] for n in (2, 3, 4) for i in range(len(t) - n + 1)]
        if not grams:
            return 0.5
        lo = sum(self.charlm.get(g, self.charlm_default) for g in grams) / len(grams)
        return 1.0 / (1.0 + math.exp(-4.0 * lo))

    # ------------------------------------------------------------ features
    def features(self, tokens, strong=False, medium=False, addr=False, line_start=False, spacy_label=None):
        t = [x.lower() for x in tokens]
        first, last = t[0], t[-1]
        f_first = self.first.get(first, 0.0)
        f_last = self.last.get(last, 0.0)
        mid = 0.0
        if len(t) == 3:
            mid = max(self.first.get(t[1], 0.0), self.last.get(t[1], 0.0))
        return {
            "first": f_first, "last": f_last,
            # English-word frequency on the zipf scale (3 = once per million words, "the" is about 7.7). Only words
            # more frequent than zipf 3 count against a candidate.
            "eng_first": max(0.0, self.eng.get(first, 0.0) - 3.0) / 4.0,
            "eng_last": max(0.0, self.eng.get(last, 0.0) - 3.0) / 4.0,
            # tokens in the swapped slots ("Smith John") are weaker evidence, so they get their own weights
            "first_as_last": self.last.get(first, 0.0), "last_as_first": self.first.get(last, 0.0),
            "three": 1.0 if len(t) == 3 else 0.0, "mid": mid,
            "strong": 1.0 if strong else 0.0, "medium": 1.0 if medium and not strong else 0.0,
            "addr": 1.0 if addr else 0.0, "line_start": 1.0 if line_start else 0.0,
            "spacy_person": 1.0 if spacy_label == "PERSON" else 0.0,
            "spacy_org": 1.0 if spacy_label in ("ORG", "PRODUCT", "GPE", "FAC", "LOC", "WORK_OF_ART") else 0.0,
            "absent_both": 1.0 if f_first == 0.0 and f_last == 0.0 else 0.0,
            "charlm_first": self.name_likeness(tokens[0]), "charlm_last": self.name_likeness(tokens[-1]),
            "non_eng_first": 1.0 if self.eng.get(first, 0.0) < 2.5 else 0.0,
            "non_eng_last": 1.0 if self.eng.get(last, 0.0) < 2.5 else 0.0,
            # is the token more common as a name than as an English word? (Brown/Smith yes, Call/Period no)
            "spec_first": max(-1.0, min(1.0, (self.first_log.get(first, 0.0) - self.eng.get(first, 0.0)) / 4.0)),
            "spec_last": max(-1.0, min(1.0, (self.last_log.get(last, 0.0) - self.eng.get(last, 0.0)) / 4.0)),
        }

    def scored_candidates(self, text, ctx_key_words=None, nlp=None, ocr=False):
        """(start, end, tokens, features, prob, context_tag) for every plausible candidate in text.
        spaCy (optional) is run on the candidate's line only, and only when the context is not strong."""
        out, line_cache = [], {}
        for start, end, tokens, strong, medium, addr, line_start in name_candidates(text, ctx_key_words, ocr):
            if not self.plausible_tokens(tokens):
                continue
            label = None
            # spaCy is slow, so it runs on one line at a time, at most once per line (line_cache), and never for a
            # candidate whose context already settles the question.
            if nlp is not None and not (strong or addr):
                ls = text.rfind("\n", 0, start) + 1
                le = text.find("\n", end)
                le = len(text) if le == -1 else le
                if ls not in line_cache:
                    try:
                        doc = nlp(text[ls:le][:1000])
                        line_cache[ls] = [(e.start_char + ls, e.end_char + ls, e.label_) for e in doc.ents]
                    except Exception:
                        line_cache[ls] = []
                for s_, e_, lab in line_cache[ls]:
                    if s_ < end and start < e_:
                        label = lab
                        break
            f = self.features(tokens, strong, medium, addr, line_start, label)
            tag = "context: strong" if strong or addr else "context: medium" if medium else "context: none"
            out.append((start, end, tokens, f, self.prob(f), tag))
        return out

    def prob(self, feats):
        z = self.weights["bias"] + sum(self.weights.get(k, 0.0) * v for k, v in feats.items())
        return 1.0 / (1.0 + math.exp(-z))

    def plausible_tokens(self, tokens):
        return 2 <= len(tokens) <= 3 and not any(x.lower() in STOP for x in tokens)


_GREETINGS = ("hi", "hey", "dear", "hello", "thanks")


def _greeting_like(tok):
    """OCR turns "Hi," into "Hil" / "Hii": a short token one edit away from a greeting is a greeting."""
    t = tok.lower()
    if len(t) > 6:
        return False
    for g in _GREETINGS:
        if abs(len(t) - len(g)) <= 1 and t[:1] == g[:1]:
            diff = sum(a != b for a, b in zip(t, g)) + abs(len(t) - len(g))
            if diff <= 1:
                return True
    return False


class _View:
    """Read-only dict-like view on one field of the packed lexicon (0.0 for zero / missing)."""

    __slots__ = ("scorer", "i")

    def __init__(self, scorer, i):
        self.scorer, self.i = scorer, i

    def get(self, tok, default=0.0):
        v = self.scorer.feat(tok)[self.i]
        return v if v else default

    def __contains__(self, tok):
        return bool(self.scorer.feat(tok)[self.i])


def trim_tokens(tokens, ocr=False):
    """Drop greeting / label tokens at either end ("Hi Priya" -> ["Priya"], "From Hannah Brown" -> [...]).

    Returns the (start, end) index range of what is left. Only the edges are trimmed: a stop word in the middle
    stays, and plausible_tokens() then rejects the whole candidate."""
    i, j = 0, len(tokens)
    while i < j and (tokens[i].lower() in STOP or (ocr and i == 0 and _greeting_like(tokens[i]))):
        i += 1
    while j > i and tokens[j - 1].lower() in STOP:
        j -= 1
    return i, j


def name_candidates(text, ctx_key_words=None, ocr=False):
    """Yield (start, end, tokens, strong, medium, addr, line_start) for each candidate in `text`."""
    header_strong = bool(ctx_key_words) and is_name_key(ctx_key_words)
    for m in NAME_CAND_RE.finditer(text):
        raw = m.group(1)
        parts = list(re.finditer(r"\S+", raw))
        tokens = [p.group() for p in parts]
        i, j = trim_tokens(tokens, ocr)
        if j - i < 2:
            continue
        sub = parts[i:j]
        if len(sub) > 3:
            continue
        start = m.start(1) + sub[0].start()
        end = m.start(1) + sub[-1].end()
        ls = text.rfind("\n", 0, start) + 1
        left = text[ls:start]
        right = text[end:end + 60]
        # Context comes from a short window, at most 40 characters to the left and 60 to the right: enough for
        # "Escalated to" or " <priya.sharma@orbitpay.in>", too little to borrow cues from unrelated text.
        strong = header_strong or bool(_STRONG_LEFT.search(text[max(0, start - 40):start])) or \
            bool(re.match(r"^\s*(?:from|to|cc|bcc|reply-to|author)\s*:", left, re.I))
        addr = bool(_ADDR_RIGHT.match(right))
        medium = bool(_MEDIUM_LEFT.search(left[-40:])) or bool(_ID_RIGHT.match(right))
        # alone on its line or after a bullet: typical for signatures ("Thanks,\nPriya Sharma") and lists
        line_start = not left.strip() or bool(re.fullmatch(r"\s*[-*•>|#]*\s*", left))
        yield start, end, [p.group() for p in sub], strong, medium, addr, line_start
