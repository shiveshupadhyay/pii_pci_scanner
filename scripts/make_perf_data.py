"""Hidden-scale data for performance measurement (and correctness at scale):

    python scripts/make_perf_data.py --out perf [--rows 1000000] [--log-mb 150] [--ocr-pages 50]

* 1M-row CSV and 1M-row Parquet (column-only, like the hidden test's large tables)
* a labelled ~150 MB application log: ~99.5% boilerplate, sensitive lines sprinkled everywhere, including across
  the scanner's chunk boundaries
* ~50 OCR pages (screenshots, JPG scans, multi-page scanned PDFs)
* the synthetic format set copied alongside
Labels are written in evaluator format so the run is also scored.
"""

import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_synthetic import Builder  # noqa: E402
from synth_lib import PHRASES  # noqa: E402


def big_table(b, rows, rel_csv, rel_parquet):
    import pyarrow as pa
    import pyarrow.parquet as pq

    v, r = b.v, b.r
    cols = ["customer_name", "email", "mobile", "col_7", "order_id", "request_id", "amount", "city"]
    ents = {"customer_name": "PERSON_NAME", "email": "EMAIL", "mobile": "PHONE", "col_7": "CARD_NUMBER"}
    cities = ["Pune", "Austin", "Leeds", "Lagos", "Hamburg", "Nairobi", "Perth", "Osaka"]
    # Written in 50k-row batches to both files at once, so a million rows never have to sit in memory. Card numbers
    # sit under the uninformative header "col_7", so the column has to be recognised by its values alone.
    p = b.path(rel_csv, "structured")
    batch = 50_000
    writer = None
    pp = b.path(rel_parquet, "structured")
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(",".join(cols) + "\n")
        for start in range(0, rows, batch):
            n = min(batch, rows - start)
            data = {c: [] for c in cols}
            for _ in range(n):
                name = v.name()
                data["customer_name"].append(name)
                data["email"].append(v.email(name))
                data["mobile"].append(v.phone(formatted=r.random() < 0.5))
                data["col_7"].append(v.card())
                data["order_id"].append(v.order_id())
                data["request_id"].append(v.uuid())
                data["amount"].append(f"{r.uniform(5, 9000):.2f}")
                data["city"].append(r.choice(cities))
            f.write("\n".join(",".join(data[c][i] for c in cols) for i in range(n)) + "\n")
            t = pa.table(data)
            if writer is None:
                writer = pq.ParquetWriter(pp, t.schema)
            writer.write_table(t, row_group_size=100_000)
    writer.close()
    for rel in (rel_csv, rel_parquet):
        b.g.column_only.append(rel)
        for c, e in ents.items():
            b.g.col(rel, c, e)


BOILER = ["GET /api/v2/items/{n} 200 {ms}ms", "GET /api/v2/cart/{n} 200 {ms}ms", "cache hit key=session:{n}",
          "GET /health 200 {ms}ms", "POST /api/v2/events 202 {ms}ms", "worker heartbeat ok seq={n}"]


def big_log(b, rel, mb):
    """A log of `mb` megabytes: about 0.6% of lines carry a sensitive value (or a hard negative), the rest is
    boilerplate, as in the dev log. Values are scattered at random, so many of them land near the scanner's
    chunk boundaries."""
    v, r, g = b.v, b.r, b.g
    p = b.path(rel, "text")
    target = mb * 1024 * 1024
    size = 0
    lines = []
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        while size < target:
            ts = f"2026-{r.randint(1, 9):02d}-{r.randint(10, 28)}T{r.randint(10, 23)}:{r.randint(10, 59)}:{r.randint(10, 59)}Z"
            svc = r.choice(["api", "billing", "auth", "notify", "search"])
            if r.random() < 0.006:
                k = r.randrange(7)
                if k == 0:
                    e = v.email()
                    g.add(rel, "EMAIL", e)
                    msg = f"login ok email={e} ip=10.{r.randint(0, 255)}.{r.randint(0, 255)}.{r.randint(1, 254)}"
                elif k == 1:
                    ph = v.phone()
                    g.add(rel, "PHONE", ph)
                    msg = f"sms queued for {ph}"
                elif k == 2:
                    c, e = v.card(), v.expiry()
                    g.add(rel, "CARD_NUMBER", c)
                    g.add(rel, "CARD_EXPIRY", e)
                    msg = f"payment debug card_number={c} exp={e}"
                elif k == 3:
                    t = v.token()
                    g.add(rel, "API_TOKEN", t)
                    msg = f"x-api-key: {t}"
                elif k == 4:
                    pw = v.password()
                    g.add(rel, "PASSWORD", pw)
                    msg = f"connecting to postgres://svc:{pw}@db-{r.randint(1, 9)}.internal:5432/app"
                elif k == 5:
                    nm = v.name()
                    g.add(rel, "PERSON_NAME", nm)
                    msg = f"profile updated for {nm} (id={r.randint(100, 99999)})"
                else:
                    msg = r.choice([f"retrying job order_id={v.order_id()}", f"ssn check {v.bad_ssn()}",
                                    f"test card 4242424242424242", f"user {r.choice(PHRASES)} updated",
                                    f"checksum={v.md5()}"])
            else:
                msg = r.choice(BOILER).format(n=r.randint(1, 99999), ms=r.randint(1, 900))
            line = f"{ts} INFO [{svc}] {msg}\n"
            lines.append(line)
            size += len(line)
            if len(lines) >= 20000:
                f.write("".join(lines))
                lines = []
        f.write("".join(lines))


def ocr_pages(b, n_pages):
    made = 0
    i = 0
    while made < n_pages:
        kind = ["config", "mail"][i % 2]
        if i % 3 == 0:
            b.screenshot(f"perf_ocr/shot_{i}.png", dark=i % 6 == 0, font="mono" if kind == "config" else "sans",
                         kind_of=kind)
            made += 1
        elif i % 3 == 1:
            b.screenshot(f"perf_ocr/scan_{i}.jpg", font="serif", fmt="jpg", kind_of=kind)
            made += 1
        else:
            b.scanned(f"perf_ocr/statement_{i}.pdf")
            made += 2
        i += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="perf")
    ap.add_argument("--rows", type=int, default=1_000_000)
    ap.add_argument("--log-mb", type=int, default=150)
    ap.add_argument("--ocr-pages", type=int, default=50)
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    if os.path.exists(a.out):
        shutil.rmtree(a.out)
    b = Builder(a.out, a.seed, 0.5, "test")
    big_table(b, a.rows, "warehouse/customers_1m.csv", "warehouse/customers_1m.parquet")
    big_log(b, "logs/gateway_150mb.log", a.log_mb)
    ocr_pages(b, a.ocr_pages)
    # the format-diverse set on top
    b.csv_like("exports/customers.csv")
    b.csv_like("exports/partners.tsv", delimiter="\t")
    b.xlsx("hr/staff.xlsx")
    b.json_records("api/customers.jsonl", jsonl=True)
    b.html("web/customers.html")
    b.sql_dump("db/dump.sql")
    b.http_log("logs/access.log", n=20000)
    b.ticket("support/ticket_1.txt")
    b.app_log("logs/app.log", n=5000)
    b.configs()
    b.emails()
    b.docx("shared/onboarding.docx")
    b.pdf("reports/payouts.pdf")
    b.errors()
    b.g.write(b.labels)
    print(f"wrote {len(b.g.paths)} files, {len(b.g.findings)} findings -> {a.out}")


if __name__ == "__main__":
    main()
