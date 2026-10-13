# Report - Shivesh Upadhyay

## 1. Summary

* **Built:** a local, CPU-only scanner. It detects file types from content, extracts structured data, documents (with layout) and OCR text, chunks large files with no duplicates by construction, and detects all 11 entities with validators, context rules and a learned person-name model.
* **Dev:** path-balanced macro-F1 **0.988** (structured 1.000, text 1.000, OCR 0.964); columns F1 1.000; errors 5/5 with no false errors; 0 duplicates.
* **Generalisation:** an independent hidden-like set (new formats, names never seen in dev, hard negatives) scores **0.992**. The hidden-scale run (1M-row tables, 150 MB log, 50 OCR pages) scores **0.985 in 57 s at 1.75 GB peak**.
* **Expected on the hidden test:** about **0.94 ± 0.03**. The rules were refined while looking at dev, the synthetic set shares an author with the scanner, and the small OCR supports make that path noisy.
* **Next:** a confidence-gated OCR cascade (better preprocessing, then a vision model) for photographed and handwritten pages.

## 2. Architecture

See the HLD at the top of `DESIGN.md`, and `DEVELOPER_GUIDE.md` for a walkthrough of the code, the design decisions behind it, and setup on Linux, macOS and Windows.

`walk → content sniffing → task planner (chunks for big text) → Phase A pool: structured / document / text extraction → detection (candidates → validators → context → name model → merge) → Phase B pool: OCR of deferred pages → same detection → writer`

The writer computes hashes with `starter/contract.py`, masks values and records evidence. Locations follow the *real* type: page for PDFs and images, `[sheet/]column` for tables, `{}` for everything else.

## 3. Key decisions

### File type detection and routing
Routing uses magic bytes and container structure:
* PDF signature within the first 1 KB; ZIP name list to tell DOCX, XLSX and PPTX apart; OLE2 with `EncryptedPackage` means an encrypted Office file; gzip is decompressed and re-sniffed.
* Decoded text is classified by structure: JSON vs JSONL, MIME headers (checked *before* HTML, because an e-mail can carry an HTML part), HTML/XML, SQL, and CSV/TSV (consistent delimiter count plus a header-like first row).

This handles every misleading extension in dev: `.dat`→PDF, `.pdf`→PNG, `.txt`→CSV, `.log`→JSONL.

Error reasons are explicit: empty, truncated PDF (no `startxref`/`%%EOF`; MuPDF would otherwise silently repair it), encrypted PDF, corrupt zip, unrecognised binary. One crashing file never stops the run: process-pool crashes are isolated per file.

### Structured data
* **Extraction:** every XLSX sheet; JSON/JSONL flattened to dotted paths with `[]` for list elements (top-level arrays are records); large JSON arrays streamed with `raw_decode`.
* **Cell detection:** each cell goes through the same detector with its **column header as the key**. A bare 10-digit number under `mobile` is a phone; under `bytes_sent` it isn't. A Luhn-valid value under `order_id` is not a card, but under `ref_a` (uninformative) it is.
* **Two passes:** expiry and CVV columns are re-detected once the table is known to hold a card column (the guide's rule).
* **Column labels:** a column is labelled when one entity covers ≥ 50% of its non-empty cells. Every dev column with gold findings has ≥ 67% coverage; look-alike columns are near 0%.
* **Consensus guard:** in a column where only a minority of card- or Aadhaar-shaped values validate, the validating ones are dropped. About 10% of random digit strings pass Verhoeff or Luhn by chance.
* **Large tables:** above 50k rows (row count from Parquet metadata, a fast newline count for CSV, or the XLSX dimension), the table is sampled (head 5k + 5k random rows/row groups) and only `columns.jsonl` is written. A 1M-row CSV or Parquet file takes about 3 s.

### Documents
* **PDF tables:** built from PyMuPDF word boxes. Table cells start exactly at their header's x position, and neighbouring columns can be only 2.6 pt apart, so cells are assigned by **header x-span**, never by splitting on gaps. A header word only opens a column if at least half of the rows below have a word starting at that x. Without this, coincidental alignment turns forms into tables.
* **PDF forms:** rendered as `label<TAB>value`, so the same "key to the left" logic used for logs gives label context.
* **Two-column pages:** split at the empty vertical band nearest the page centre, and only when both sides contain multi-word lines. Otherwise a single form would be split between its labels and its values.
* **Clipping:** text past the page edge is extracted too, because dev labels a phone number at x > 595.
* **DOCX:** tables use the header row; two-column tables are treated as label/value.
* **EML:** stdlib `email` with base64 / quoted-printable decoding (QP soft breaks split `11/=\n29`). From/To/Cc names and addresses are emitted as `From: Name <addr>`; `Message-ID` is ignored.
* **HTML:** entities decoded; `th`, `label`/`input` and `dl` give context.
* **SQL dumps:** INSERT/COPY rows carry their column names.

### OCR
* **Engines:** RapidOCR (PP-OCR in ONNX, models bundled with the pip wheel) is the primary engine and Tesseract 5 the secondary. Gold-value recall on dev images: RapidOCR 0.924, Tesseract 0.849 (best of psm 3/4/6 × scale × median), union 0.958.
* **Ensemble merge:** Tesseract findings are added only when no RapidOCR finding of the same entity is similar (≥ 0.8 similarity or containment), so two misreads of one value never both survive.
* **Preprocessing (Tesseract):** grayscale, invert dark screenshots, upscale 2×, 3×3 median.
* **Repairs:**
  * `;` → `:` after a word (`VPN password; x`);
  * a leading `l` read for capital `I` in a name, when the I-form is a known name and the l-form isn't (`Omar lyer` → `Iyer`);
  * a space inside an e-mail in `<…>` restored as `_`;
  * greetings misread as names (`Hil Amara`);
  * a fuzzy `EXAMPLE` match for AWS keys.
* **Checksums:** cards, Aadhaar and SSN are only accepted if they validate, so OCR digit errors become misses rather than false positives.
* **Fallback:** if Tesseract is missing, RapidOCR runs alone (dev OCR-path F1 0.961 vs 0.964).

### Chunking
* Text files over 8 MB are split into newline-aligned byte ranges. Each task decodes `[start-64 KB, end+64 KB]` but keeps only candidates **starting** in `[start, end)`.
* Any value shorter than the overlap (PEM blocks are capped at about 20 KB) is seen whole by exactly one owner. Newline alignment also means a multi-byte UTF-8 character is never cut.
* Tests check that chunked output equals unchunked output with 4 KB chunks. Duplicates were 0 in every run, including the 150 MB log (14.5k labelled values across 18 chunk boundaries).

### Detection
* **Validators:** Luhn plus issuer prefix by length plus the guide's 7 test numbers; Verhoeff; SSN area/group/serial; PAN 4th character; AWS `EXAMPLE`; IANA TLDs; NANP area codes; entropy and character classes.
* **Context:**
  * the key phrase immediately left of the value on the same line (`cc_num=`, `"pan": "`, `Order ID`) or the header/label;
  * vocabularies for phone, card, identifier (`order_id`, `txn_ref`, … and `Reference`, but not `ref_a`), expiry, CVV, SSN, name, password (with modifiers such as `_hash`, `_min_length`, `require_`) and token keys;
  * hash keys, which block tokens.
* **Expiry** needs an expiry keyword *and* a card within the same context (line, form column or table), or the date written directly after a card. A **4-digit CVV** needs an Amex.
* **PERSON_NAME:**
  * Candidates are 2–3 capitalised tokens, trimmed of greetings and labels.
  * Features: first-name and surname scores from vendored lexicons (names-dataset top names for 105 countries, US Census surnames); English-word frequency (wordfreq); "name vs word" specificity (Brown vs Call); a character 2–4-gram naive Bayes name-likeness score; context cues (From/To/Author/header, `<email>` after, `for … (id=`, prepositions).
  * Model: **logistic regression**, trained on dev candidates plus a 6k-sentence synthetic corpus (half the name pool, name and non-name phrases in realistic contexts). Name-disjoint 5-fold CV: **P 0.972 / R 0.979 / F1 0.976**.
  * No dev names are hard-coded. The lexicons come from public sources.
* **Merge:** priority (PRIVATE_KEY > AWS > prefixed token > card > national id > e-mail > phone > expiry/CVV > password/generic token > name), then longer span, then confidence. Blocked spans include PEM bodies and the URL authority (so `user:pass@host` never yields an e-mail).
* **Confidence:** set by the strength of the rule; OCR findings are scaled down.

### Tried and dropped
* **spaCy `en_core_web_sm` as an NER feature.** Identical results with and without it (dev 0.988, synthetic 0.992), at +80 MB per worker and an install-time download from GitHub. It is now optional (`--ner`); the name model was trained with the feature present in half of the examples, so it works either way.
* **Tesseract alone.** OCR-path F1 0.779.
* **Splitting PDF table rows on gaps, and a 30% alignment threshold.** Wrong columns and forms parsed as tables.
* **A hand-tuned name scorer, then a model trained on dev candidates only.** It had 6 negatives, and an English-word-negative corpus taught it to reject *Will Smith*. Fixed by adding name-vs-word specificity features and English-word names as positives.
* **Dict-based lexicons.** 116 MB per worker; replaced by packed NumPy arrays of token hashes and features (about 5 MB).

## 4. Evaluation

**How generalisation was estimated.** Dev is optimistic: I read many dev files while writing the rules. Three checks counter that:
1. **History:** the first complete pipeline scored 0.921 on dev. Fixes driven by the error analysis took it to 0.979, then to 0.988 with the name model. Most fixes were generic bugs (layout, regex), not dev-specific rules.
2. **Independent hidden-like set** (`scripts/make_synthetic.py`, 41 files, 2,599 findings):
   * formats absent from dev: TSV (also as `.txt`), HTML, SQL dumps (INSERT, INSERT without column list, COPY), Apache access logs with query strings, Bearer and session cookies, support tickets, `.toml`, `.properties`, Go code, multipart e-mail;
   * names never seen in dev, half of them outside the scanner's lexicons, drawn from a name-pool half the name model never trained on;
   * hard negatives: Luhn-valid order ids, test cards, invalid SSN/PAN/Aadhaar, hashes, UUIDs, placeholders, company and product phrases;
   * misleading extensions and five unreadable files.
3. **Hidden-scale set** (`scripts/make_perf_data.py`): correctness at scale. Labels cover the 150 MB log and column labels for 1M-row tables.

| set | path-balanced | structured | text | OCR | columns F1 | F0.5 PCI / SECRET / PII | dup | errors |
|---|---|---|---|---|---|---|---|---|
| dev, first complete pipeline (v0) | 0.921 | 0.994 | 0.956 | 0.812 | 0.996 | 0.989 / 0.987 / 0.998 | 0 | 5/5 |
| **dev, final** | **0.988** | 1.000 | 1.000 | 0.964 | 1.000 | 1.000 / 0.999 / 1.000 | 0 | 5/5, 0 false |
| synthetic, first run | 0.987 | 0.985 | 0.977 | 1.000 | 0.982 | 0.993 / 0.999 / 0.991 | 0 | 5/5 |
| **synthetic, final** | **0.992** | 1.000 | 1.000 | 0.977 | 1.000 | 1.000 / 1.000 / 1.000 | 0 | 5/5, 0 false |
| **hidden-scale, final** | **0.985** | 1.000 | 1.000 | 0.956 | 1.000 | 1.000 / 1.000 / 1.000 | 0 | 5/5, 0 false |

**Dev, final** (precision / recall / F1, support). Structured and text are 1.00/1.00/1.00 for every entity: support 424/451 EMAIL, 588/298 PHONE, 519/319 PERSON_NAME, 1795/205 NATIONAL_ID, 996/195 CARD_NUMBER, 436/125 CARD_EXPIRY, 60/31 CVV, 526/73 AWS, 20 PRIVATE_KEY (text), 557/204 API_TOKEN, 325/188 PASSWORD. On the OCR path every entity is 1.00 except **API_TOKEN at 0.71/0.71 (support 7)**.

**Synthetic, final:** everything 1.00 except OCR EMAIL recall 0.89 (support 9) and OCR PERSON_NAME recall 0.86 (support 7). Names never seen in dev reach F1 1.00 in text (support 87) and structured data (156). Before the name model they were at recall 0.73 and 0.81.

## 5. Error analysis

Errors are grouped by cause, using `scripts/analyze_errors.py`. It matches exactly like the evaluator and attributes causes automatically.

**Remaining errors, by cause.**

| cause | set | error | why it can't be fixed (yet) |
|---|---|---|---|
| OCR | dev | 2 FN + 2 FP, API_TOKEN: `…slCHytP` read as `…sICHytP` (DOC_3701.jpg); `…XI0llWve` read as `…XI0IIWve` (Screenshot_1313.png, dark) | Arial `I` and `l` are pixel-identical in random tokens, and there is no checksum to recover them |
| OCR | synthetic | 2 FN: an e-mail in a noisy serif JPG; `Iensen` read as `lensen` | the I/l repair needs the surname in the lexicon |

**Errors found and fixed** (examples from dev unless noted):

* **Routing**
  * An `.eml` with an HTML alternative part was sniffed as HTML, and its `<addr>` headers were swallowed as tags (synthetic). Fix: check EML before HTML.
  * An INI file starting with `[default]` was taken for a JSON array. Fix: stricter JSON test.
* **Extraction / layout**
  * PDF tables were not detected because header `file_sha256` was rejected by a "no 3-digit runs" rule. This caused 40+ FN: passwords, bare phones, expiries and Aadhaar in `pdf_table`.
  * Gap-based cell splitting merged `status` and `phone`.
  * A two-column form page was split at the label/value gap of the right-hand form, so `Order ID` values lost their label and became card FPs (`4810707030434973`). Fix: nearest-to-centre band plus the multi-word-lines check.
  * A single-column form was briefly split by the same logic after a change.
  * A phone printed beyond the page edge (`kyc/invoice_4110.pdf` p17) was clipped by PyMuPDF.
* **Detection**
  * `valid thru 08/31.` was rejected because of the sentence-final full stop.
  * 32-hex API keys were rejected as dashless UUIDs. Fix: UUIDs require dashes.
  * Cards under `ref_a` were rejected as an identifier field, while `Reference 4346…` became an FP (synthetic). Fix: bare `ref` is uninformative; `Reference` and `txn_ref` are identifiers.
  * URL passwords containing `#` were missed: 174 FN at hidden scale.
  * JSON `profile.name` was not treated as a name header (synthetic, 29 FN).
  * Unseen names in weak contexts (`kyc approved for Cheehong Romlin (customer=…)`) were missed until the name model.
* **OCR**
  * `VPN password; lukas2006$` (misread `;`).
  * `Omar lyer` (I read as l).
  * `<daniel chen@…>` (underscore dropped).
  * `Hi Amara,` read as `Hil Amara` and taken for a name (FP).
  * Tesseract fragments (`xoxb-…-571`) added next to RapidOCR's full reading.
  * `000-933-2653`, an invalid SSN in a scanned page that Tesseract rendered with phone grouping (FP). Fix: NANP rule.
* **Chunking:** no errors at any stage. 0 duplicates, and chunked output equals unchunked output.

**Hidden-test risks.** Annotation-policy cases the guide doesn't cover could cost points both ways:
* secrets inside HTML comments or `<script>` (scanned here);
* percent-encoded values in URLs (not decoded);
* names in unusual contexts or ALL CAPS;
* photographed or rotated documents (no deskew yet);
* a token format outside the guide's list (only listed prefixes count anywhere; others need a credential key).

## 6. Performance

All runs used 4 workers on an i7-1165G7 (4 cores / 8 threads). Memory is measured by `scripts/measure.py` as RSS **summed over the whole process tree**, an upper bound because shared pages are counted per process.

| run | files | wall | peak RSS (sum) | largest process | CPU-s by stage |
|---|---|---|---|---|---|
| dev | 194 | 58.0 s | 1.85 GB | 794 MB | OCR 153.7 (60 pages), text 13.2, CSV+Parquet 4.2, rest < 1 |
| synthetic | 41 | 10.8 s | 1.07 GB | 499 MB | OCR 16.0 (7 pages), CSV+Parquet 9.1 |
| hidden-scale | 63 | **56.6 s** | **1.75 GB** | 721 MB | OCR 97.0 (50 pages), text 84.9 (153 MB), CSV 2.9 + Parquet 2.7 (1M rows each) |

**Where the time goes:**
* OCR at about 2 s per page per core for the ensemble (RapidOCR 2.1 s, Tesseract 0.28 s);
* text detection at about 1.8 MB/s per core (regex candidates, key-value parsing, name candidates);
* large tables are near-constant time thanks to sampling;
* PDFs with a text layer run at about 395 pages/s.

**Extrapolation to the hidden test.** The hidden-scale set *is* the hidden test's shape: ~200 files, 1M-row tables, a 150 MB log, ~50 OCR pages. It needs ≈190 CPU-s. On a slower 4-core laptop without hyper-threading, expect 1.5–2.5× the measured 57 s, so **about 1.5–2.5 min against the 20 min budget**. Memory stays under ≈1.8 GB.

**What I optimised:**
* **Two-phase scheduling.** OCR runs in fresh workers after everything else. Together with the next two items and 3 OCR workers, this cut the hidden-scale peak from 2.83 GB to 1.75 GB.
* **Packed lexicons.** 400k tokens as sorted 64-bit hashes plus uint8 features: from 116 MB to about 5 MB per worker.
* **spaCy dropped from the default.** Saves 80 MB per worker.
* **Sampling of tables over 50k rows.**
* **Chunking of big text** with largest-first scheduling.
* **Linear-time merge** (sweep by clusters).
* **Keyword-anchored key-value parsing** instead of a generic `key: value` regex over every log line.
* **Per-column memo** of repeated cell values.
* **Thread caps:** `OMP_THREAD_LIMIT` and ONNX `intra_op_num_threads` = cores / workers.

**Safety valves:** `--workers`, `--ocr-workers` and `--ocr rapidocr|tesseract` (Tesseract-only OCR is about 9× faster per page, with lower recall).

## 7. Limitations and next steps

* **OCR:**
  * no deskewing or perspective correction for photographed documents, and no handwriting support;
  * glyph-ambiguous tokens are unrecoverable;
  * next step: a confidence-gated cascade (better preprocessing, then a private vision model, as in DESIGN §2).
* **Names:** ALL CAPS names, names after unusual cue words, and single-token surnames with titles (`Mr Okafor`) are not handled. Next: more context cues learned from data, and a small token-classification model (CPU, gated) for prose.
* **Document tables:** a two-column label/value table whose first value is a name (`Name | Priya Sharma`, then `Mobile | 98…`) is read as a header row plus data. The name is missed, and so are values that need their label (a bare phone number). This was found while documenting the code (DEVELOPER_GUIDE.md §8). The fix is to use the name lexicon to tell a value from a header.
* **Annotation-policy assumptions:** HTML comments and scripts are scanned; percent-encoded URL values are not decoded; attachments are scanned under the EML's `{}` location; a 4-digit CVV requires an Amex card. Each of these is a one-line switch once the policy is known.
* **Throughput:** pure-Python regex at 1.8 MB/s per core is fine here, but production needs a Hyperscan/RE2 prefilter (DESIGN §1).
* **Evaluation:** the synthetic set and the scanner share an author. An externally written hidden-like set would be the honest next measurement.
