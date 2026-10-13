"""Train the PERSON_NAME scorer.

    python scripts/train_name_model.py [--sets data/dev synthetic_train] [--no-ner]

1. Character n-gram naive Bayes (names vs English words), from the vendored lexicons only
   -> src/scanner/resources/charlm.json.gz
2. Candidate set: every plausible 2-3 token candidate the detector sees in non-structured units of the
   training sets, labelled 1 when it equals a gold PERSON_NAME of that file.
3. L2 logistic regression (numpy) over generic features (lexicon frequency, English frequency, name-likeness,
   context cues, optional spaCy label). Generalisation is estimated with GroupKFold grouped by first name, so
   validation names are never seen in training. Final weights -> src/scanner/resources/name_model.json
"""

import argparse
import gzip
import io
import json
import math
import os
import sys
import zlib
from collections import Counter, defaultdict

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "starter")]
RES = os.path.join(ROOT, "src", "scanner", "resources")

from scanner import pipeline  # noqa: E402
from scanner.detect.context import key_words  # noqa: E402
from scanner.detect.names import FEATURES, _load_tsv  # noqa: E402


def build_charlm(min_count=3, top=40000):
    def load(name):
        with gzip.open(os.path.join(RES, name), "rt", encoding="utf-8") as f:
            return [l.rstrip("\n").split("\t") for l in f]

    # Two classes for the n-gram model: reasonably common names (top-1500 in some country, or 500+ US bearers of a
    # surname) against reasonably common English words that are not also names. Tokens in both lists are left out of
    # the word side, so "Brown" teaches name spelling, not word spelling.
    names = set()
    for n, rank, cnt in load("first_names.tsv.gz"):
        if int(rank) <= 1500 and int(cnt) >= 1:
            names.add(n)
    for n, rank, cnt, census in load("surnames.tsv.gz"):
        if int(census) >= 500 or (int(cnt) >= 1 and int(rank) <= 1500):
            names.add(n)
    words = {w for w, z in load("english_zipf.tsv.gz") if float(z) >= 3.0} - names

    def grams(t):
        t = "^" + t + "$"
        return [t[i:i + n] for n in (2, 3, 4) for i in range(len(t) - n + 1)]

    cn, cw = Counter(), Counter()
    for t in sorted(names):
        cn.update(grams(t))
    for t in sorted(words):
        cw.update(grams(t))
    # count desc, then n-gram: set iteration order changes with every interpreter run, so most_common() alone
    # would pick a different vocabulary at the cut-off each time
    ranked = sorted((cn + cw).items(), key=lambda kv: (-kv[1], kv[0]))[:top]
    vocab = [g for g, c in ranked if c >= min_count]
    tn, tw, V = sum(cn.values()), sum(cw.values()), len(vocab) + 1
    logodds = {g: round(math.log((cn[g] + 1) / (tn + V)) - math.log((cw[g] + 1) / (tw + V)), 4) for g in vocab}
    default = math.log(1 / (tn + V)) - math.log(1 / (tw + V))
    # mtime=0: re-running the script on the same lexicons gives a byte-identical file (no spurious git diff)
    with gzip.GzipFile(os.path.join(RES, "charlm.json.gz"), "wb", mtime=0) as gz, \
            io.TextIOWrapper(gz, encoding="utf-8") as f:
        json.dump({"logodds": logodds, "default": round(default, 4), "n_names": len(names), "n_words": len(words)}, f)
    print(f"charlm: {len(names)} name tokens vs {len(words)} English words, {len(logodds)} n-grams")


def collect(set_dir, use_ner):
    labels, files = os.path.join(set_dir, "labels"), os.path.join(set_dir, "files")
    with open(os.path.join(labels, "scope.json"), encoding="utf-8") as fh:
        scope = json.load(fh)
    gold = defaultdict(Counter)
    with open(os.path.join(labels, "findings.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            g = json.loads(line)
            if g["entity"] == "PERSON_NAME":
                gold[g["file"]][" ".join(g["value"].split()).lower()] += 1
    pipeline._init_worker("ensemble", 1, use_ner)
    det = pipeline.detector()
    rows = []
    for f, path in sorted(scope["file_paths"].items()):
        if path in ("error", "structured") or f in scope["column_only_files"]:
            continue
        # Scan the file as scan.py would, and record every unit's text through the collect hook. Then score the
        # same candidates the scorer will see at run time.
        pipeline._W["collect"] = []
        pipeline.scan_file(os.path.join(files, f), f)
        for text, key, ner, cell, ocr in pipeline._W.pop("collect"):
            if cell:  # table cells are decided by the header rule (_cell_rules), not by the scorer
                continue
            kw = key_words(key) if key else []
            for start, end, tokens, feats, p, tag in det.names.scored_candidates(text, kw, det.nlp if ner else None):
                cand = " ".join(tokens).lower()
                y = 1 if gold[f].get(cand) else 0  # positive iff the candidate is a labelled name of this file
                # the first name is the CV group, so no first name is in both a training and a validation fold
                rows.append(([feats[k] for k in FEATURES], y, tokens[0].lower(), f))
    return rows


# Sentence templates for the synthetic corpus, by strength of context. {X} is the candidate phrase.
STRONG_T = ["From: {X} <{e}>", "To: {X} <{e}>", "Author: {X}", "Requester: {X}", "Full name\t{X}", "Name: {X}",
            "Assignee: {X}", "Prepared by: {X}", "Cardholder: {X}", "Reply-To: {X} <{e}>"]
MEDIUM_T = ["{ts} INFO [svc] profile updated for {X} (id={n})", "I've looped in {X} from finance.",
            "Please reach out to {X} for access.", "Spoke with {X} about the renewal.",
            "{ts} WARN [kyc] approved for {X} (customer={n})", "This was escalated to {X} yesterday.",
            "Ticket assigned to {X} by the bot.", "Thanks to {X} for the quick fix."]
WEAK_T = ["{X} approved the change request.", "{X} joined the call late.", "Meeting notes - {X}", "Regards,\n{X}",
          "Owner {X} confirmed the numbers.", "Cc {X}", "{X} | {C}", "Status update from {X}."]
PLACES = ["San Jose", "Hong Kong", "Abu Dhabi", "Los Angeles", "Cape Town", "Kuala Lumpur", "Buenos Aires",
          "Tel Aviv", "Rio Grande", "Sao Paulo", "Santa Clara", "Las Vegas", "Port Louis", "Costa Rica",
          "Puerto Rico", "Hyde Park", "Notting Hill", "Silicon Valley", "Salt Lake", "Palo Alto"]
PRODUCTS = ["Adobe Acrobat", "Tableau Server", "Splunk Cloud", "Visual Basic", "Power Automate", "Kibana Dashboard",
            "Grafana Alert", "Jenkins Pipeline", "Terraform Cloud", "Sentry Issue", "Notion Page", "Figma File"]


def corpus_rows(det, nlp, pool_first, pool_last, n=6000, seed=5):
    """Synthetic candidate corpus: person names and non-name capitalised phrases in the same contexts.

    Why it exists: the dev candidates are almost all positives (a few negatives in hundreds), far too lopsided to
    learn what is *not* a name. The corpus adds the hard cases: names made of English words ("Will Smith", "Grace
    Brown") as positives, and word pairs, places and product names as negatives."""
    import random

    r = random.Random(seed)
    N = det.names
    # the runtime lexicon is a packed hash table that cannot be iterated, so read the word list from its source
    eng_zipf = {w: float(v[0]) for w, v in _load_tsv("english_zipf.tsv.gz").items()}
    eng = [w for w, z in eng_zipf.items() if 3.2 <= z <= 5.5 and w.isalpha() and len(w) > 3]
    firsts_eng = [w for w in eng if N.first.get(w, 0) > 0.4]  # "will", "grace", "mark" ...
    # real names made of English words (Will Smith, Grace Brown): names by frequency, words by spelling
    eng_first_names = [w for w in firsts_eng if N.first_log.get(w, 0) - N.eng.get(w, 0) > -1.5]
    eng_surnames = [w for w in eng if N.last.get(w, 0) > 0.75 and N.last_log.get(w, 0) - N.eng.get(w, 0) > 0.3]
    print(f"corpus: {len(eng_first_names)} English-word first names, {len(eng_surnames)} English-word surnames")
    rows = []
    for i in range(n):
        positive = r.random() < 0.5
        if positive:
            k = r.random()
            if k < 0.25 and eng_first_names and eng_surnames:
                X = f"{r.choice(eng_first_names).title()} {r.choice(eng_surnames).title()}"
            elif k < 0.4 and eng_surnames:
                X = f"{r.choice(pool_first)} {r.choice(eng_surnames).title()}"
            else:
                X = f"{r.choice(pool_first)} {r.choice(pool_last)}"
            group = X.split()[0].lower()
        else:
            k = r.random()
            if k < 0.45:
                X = f"{r.choice(eng).title()} {r.choice(eng).title()}"
            elif k < 0.7:
                X = f"{r.choice(firsts_eng).title()} {r.choice(eng).title()}"
            elif k < 0.85:
                X = r.choice(PLACES)
            else:
                X = r.choice(PRODUCTS)
            group = "neg:" + X.split()[0].lower()
        # Real documents: names mostly sit in strong/medium contexts, other capitalised phrases mostly do not.
        # With identical context frequencies for both classes the model learned (correctly, for that data) that
        # context means nothing, and its context weights came out negative. The skew reproduces real documents.
        t = r.random()
        cut = (0.4, 0.8) if positive else (0.08, 0.35)
        tpl = r.choice(STRONG_T if t < cut[0] else MEDIUM_T if t < cut[1] else WEAK_T)
        text = tpl.format(X=X, e=f"{X.split()[0].lower()}@mail.example.org", n=r.randint(100, 99999),
                          ts="2026-03-01T10:00:00Z", C=r.choice(["Acme Corp", "Blue Labs"]))
        use = nlp if r.random() < 0.5 else None
        for start, end, tokens, feats, p, tag in det.names.scored_candidates(text, [], use):
            cand = " ".join(tokens)
            y = 1 if positive and cand == X else 0
            rows.append(([feats[k] for k in FEATURES], y, group if cand == X else "ctx:" + cand.lower(), "corpus"))
    return rows


def fit(X, y, l2=0.3, iters=3000, lr=0.5):
    # Plain batch gradient descent on L2-regularised log loss. With 22 weights and a few thousand rows, a fit takes a
    # moment, so there is no need for scikit-learn. The bias (column 0) is not regularised.
    Xb = np.hstack([np.ones((len(X), 1)), X])
    w = np.zeros(Xb.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Xb @ w))
        g = Xb.T @ (p - y) / len(y)
        g[1:] += l2 * w[1:] / len(y)
        w -= lr * g
    return w


def predict(w, X):
    return 1 / (1 + np.exp(-(np.hstack([np.ones((len(X), 1)), X]) @ w)))


def prf(y, yhat):
    tp = int(((y == 1) & (yhat == 1)).sum())
    fp = int(((y == 0) & (yhat == 1)).sum())
    fn = int(((y == 1) & (yhat == 0)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, 2 * p * r / (p + r) if p + r else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=[os.path.join(ROOT, "data/dev"), os.path.join(ROOT, "synthetic_train")])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--no-ner", action="store_true")
    a = ap.parse_args()
    build_charlm()
    rows = []
    for s in a.sets:
        r = collect(s, not a.no_ner)
        print(f"{s}: {len(r)} candidates, {sum(x[1] for x in r)} positive")
        rows += r
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from synth_lib import Values

    tv = Values(0, 0.5, "train")  # name-pool half never used by the synthetic test set
    det = pipeline.detector()
    corp = corpus_rows(det, None if a.no_ner else det.nlp, tv.first_common + tv.first_oov,
                       tv.last_common + tv.last_oov)
    print(f"corpus: {len(corp)} candidates, {sum(x[1] for x in corp)} positive")
    rows += corp
    X = np.array([r[0] for r in rows], dtype=float)
    y = np.array([r[1] for r in rows], dtype=float)
    groups = [r[2] for r in rows]
    # Group k-fold by first name: every row with a given first name lands in the same fold. Validation therefore
    # measures names the model has never seen, which is the hidden test's situation. Folds are assigned by crc32,
    # not hash(), so they are identical on every run.
    uniq = sorted(set(groups))
    fold_of = {g: i % a.folds for i, g in enumerate(sorted(uniq, key=lambda g: (zlib.crc32(g.encode("utf-8")), g)))}
    oof = np.zeros(len(y))
    for k in range(a.folds):
        tr = np.array([fold_of[g] != k for g in groups])
        w = fit(X[tr], y[tr])
        oof[~tr] = predict(w, X[~tr])
    # The threshold is picked on the out-of-fold predictions (max F1); the shipped weights are then refit on all rows.
    best = max(((t, prf(y, (oof >= t).astype(int))) for t in np.arange(0.2, 0.81, 0.05)), key=lambda x: x[1][2])
    thr = round(float(best[0]), 2)
    print(f"name-disjoint {a.folds}-fold CV: best threshold {thr}: P={best[1][0]:.3f} R={best[1][1]:.3f} "
          f"F1={best[1][2]:.3f}  (at 0.5: P/R/F1={tuple(round(x, 3) for x in prf(y, (oof >= 0.5).astype(int)))})")
    w = fit(X, y)
    weights = {"bias": round(float(w[0]), 4)}
    weights.update({k: round(float(v), 4) for k, v in zip(FEATURES, w[1:])})
    with open(os.path.join(RES, "name_model.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump({"weights": weights, "threshold": thr, "cv": {"precision": best[1][0], "recall": best[1][1],
                                                                 "f1": best[1][2], "folds": a.folds,
                                                                 "n": len(y), "positives": int(y.sum())},
                   "trained_on": [os.path.basename(os.path.normpath(s)) for s in a.sets]}, f, indent=1)
    print("weights:", json.dumps(weights))


if __name__ == "__main__":
    main()
