"""Build the vendored lexicons in src/scanner/resources/ (run once, needs network; the scanner itself is offline).

    pip install names-dataset wordfreq      # build-time only (see requirements-dev.txt)
    python scripts/build_lexicons.py [--cache-dir DIR] [--top-n 2000]

Sources (none of them derived from the dev labels):
  * first names / surnames: names-dataset (MIT, aggregated name frequencies for 105 countries), top-N per country
  * surnames: US Census 2010 surname file (public domain)
  * English word frequencies: wordfreq (Apache-2.0), used to penalise capitalised common words ("Will Call")
  * TLDs: IANA root zone list
"""

import argparse
import csv
import gzip
import io
import os
import re
import sys
import urllib.request
import zipfile

RES = os.path.join(os.path.dirname(__file__), "..", "src", "scanner", "resources")
CENSUS_URL = "https://www2.census.gov/topics/genealogy/2010surnames/names.zip"
IANA_URL = "https://data.iana.org/TLD/tlds-alpha-by-domain.txt"
UA = {"User-Agent": "Mozilla/5.0 (lexicon builder)"}
# Latin-script names only (with accents), 2-25 characters. The candidate regex in detect/names.py only matches
# Latin capitalised words, so names in other scripts would never be looked up anyway.
LATIN = re.compile(r"^[a-zà-öø-ÿ][a-zà-öø-ÿ'\-]{1,24}$")


def fetch(url, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, url.rsplit("/", 1)[-1])
    if not os.path.exists(path):
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r, open(path, "wb") as f:
            f.write(r.read())
    return path


def write_tsv(name, rows):
    path = os.path.join(RES, name)
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write("\t".join(str(x) for x in row) + "\n")
    print(f"wrote {path} ({len(rows)} rows)")


def norm(name):
    n = name.strip().lower()
    return n if LATIN.match(n) else None


def names_dataset_lists(top_n):
    """{name: (best rank in any country, number of countries listing it)} for first names and for surnames.

    Taking the top N per country (not the global top) gives the lexicon its international coverage: common names
    from every covered country make it in, not just the most common US ones."""
    from names_dataset import NameDataset

    nd = NameDataset()
    first, last = {}, {}
    for cc in nd.get_country_codes(alpha_2=True):
        cc = getattr(cc, "alpha_2", cc)
        try:
            fn = nd.get_top_names(n=top_n, use_first_names=True, country_alpha2=cc).get(cc, {})
            ln = nd.get_top_names(n=top_n, use_first_names=False, country_alpha2=cc).get(cc, [])
        except Exception as e:  # a few countries have no data
            print("skip", cc, e, file=sys.stderr)
            continue
        for gender_list in fn.values():
            for rank, name in enumerate(gender_list, 1):
                n = norm(name)
                if n:
                    best, cnt = first.get(n, (10**9, 0))
                    first[n] = (min(best, rank), cnt + 1)
        for rank, name in enumerate(ln, 1):
            n = norm(name)
            if n:
                best, cnt = last.get(n, (10**9, 0))
                last[n] = (min(best, rank), cnt + 1)
    return first, last


def census_surnames(cache_dir):
    out = {}
    with zipfile.ZipFile(fetch(CENSUS_URL, cache_dir)) as z:
        member = next(m for m in z.namelist() if m.lower().endswith(".csv"))
        reader = csv.DictReader(io.TextIOWrapper(z.open(member), encoding="latin-1"))
        for row in reader:
            n = norm(row["name"])
            if n and row["count"].isdigit():
                out[n] = int(row["count"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=os.path.expanduser("~/.cache/scanner-lexicons"))
    ap.add_argument("--top-n", type=int, default=2000)
    a = ap.parse_args()
    os.makedirs(RES, exist_ok=True)

    first, last = names_dataset_lists(a.top_n)
    census = census_surnames(a.cache_dir)
    write_tsv("first_names.tsv.gz", sorted((n, r, c) for n, (r, c) in first.items()))
    surnames = {}
    for n, (r, c) in last.items():
        surnames[n] = [r, c, 0]
    for n, cnt in census.items():
        surnames.setdefault(n, [10**9, 0, 0])[2] = cnt
    write_tsv("surnames.tsv.gz", sorted((n, r, c, k) for n, (r, c, k) in surnames.items()))

    # English word frequencies let the scorer tell "Grace Period" from "Grace Brown": a token that is far more
    # common as a word than as a name counts against a candidate.
    from wordfreq import top_n_list, zipf_frequency

    words = [w for w in top_n_list("en", 60000) if re.fullmatch(r"[a-z][a-z'\-]*", w)]
    write_tsv("english_zipf.tsv.gz", [(w, zipf_frequency(w, "en")) for w in words])

    with open(fetch(IANA_URL, a.cache_dir), encoding="utf-8") as f:
        tlds = sorted(l.strip().lower() for l in f if l.strip() and not l.startswith("#"))
    with open(os.path.join(RES, "tlds.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(tlds) + "\n")
    print(f"wrote tlds.txt ({len(tlds)} TLDs)")


if __name__ == "__main__":
    main()
