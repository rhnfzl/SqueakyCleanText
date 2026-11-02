# Tool comparison protocol

How SCT is compared with other PII tools. Frozen before any test run; a change after
the first test run is logged at the end of this file with its reason. It follows the 20
fairness rules in `sct-integration-research/research/benchmark/codex-fair-comparison.md`;
section 20 maps each rule to where it is handled, or why it is not.

Code: `split.py` (dev/test), `entries.py` (the comparison units and label maps). Scoring
reuses `benchmarks/score.py` and loaders reuse `benchmarks/datasets/`. Adapters and the
runner are the next step (one venv per entry).

## 1. Datasets and splits

Every PII dataset in `benchmarks/datasets/` plus two added for this comparison (the
non-NVIDIA sets the nvidia/gliner-PII card reports):

- `gretel_en`: `gretelai/gretel-pii-masking-en-v1` @ `e06eb14`, Apache-2.0, not gated,
  train/validation/test 50,000/5,000/5,000. Gold is entity strings plus types, no offsets:
  the loader places each string at every exact occurrence. At the pinned revision every one
  of the 21,378 test entities occurs exactly once and none has two types. "Gretel PII v2"
  does not exist on the Hub (checked the gretelai org listing on 2026-11-02); the card's
  "V1/V2" cannot be followed further.
- `argilla_pii`: `argilla/textcat-tokencat-pii-per-domain` @ `683954a`. The card links only
  the Argilla org; this is the org's only PII dataset on the Hub. Its `creation_script.py`
  shows it is `ai4privacy/pii-masking-200k` train[:1%] (2,096 English records) loaded into
  Argilla, with the source `privacy_mask` as span suggestions (status "pending", not reviewed).
  No licence on the repo; the upstream Ai4Privacy licence allows use by individuals but forbids
  redistributing derivatives, so it is `scores-only`: aggregate scores, no ids, no predictions.

Split rule (`split.py`): within the scored source split (test, or validation for openpii, or
the whole set where there is no split), a record goes to `dev` when
`sha256("13:<dataset>:<group>")` falls in the first 20% of the hash range, else `test`.
`group` is the template or source record where the data has one, else the record id:

| dataset | group key | dev docs (groups) | test docs (groups) | langs | smallest language in test |
|---|---|---|---|---|---|
| gretel_en | document_type | 1,051 (202) | 3,949 (764) | 1 | en 3,949 |
| argilla_pii | id | 451 | 1,645 | 1 | en 1,645 |
| openpii | uid | 65,529 | 260,934 | 30 | 4,100 |
| gretel (finance) | document_type | 780 (6) | 4,814 (54) | 7 | 339 |
| nemotron | document_type (keeps us/intl pairs together) | 18,984 (280) | 81,016 (1,186) | 1 | en |
| redact | record_id | 175 (164) | 825 (780) | 25 | 27 |
| iahlt | id | 947 | 3,968 | 1 | ar |
| bayanat | id | 217 | 783 | 4 | 108 |
| thainer | id | 304 | 1,168 | 1 | th |
| thai_pii | id | 405 | 1,595 | 1 | th |
| naamapadam | id | 433 | 1,799 | 3 | 480 |
| multiconer | id | 11,738 | 46,785 | 3 | 14,750 |
| stockmark | id | 1,049 | 4,294 | 1 | ja |
| tab | doc id | 21 | 106 | 1 | en |

Run sizes: test runs take a seeded sample of 300 docs per language from `test`
(`sample_per_lang`, seed 13; TAB all 106; REDACT all). Tuning takes at most 100 docs per
language from `dev` (TAB all 21). openpii, ai4privacy-derived and Argilla have no template id,
so their split is by record only (template leakage between dev and test is possible there,
noted as a limit). The gretel finance split has only 60 document types, so its dev share is coarse.

PII vs NER (rule 3): iahlt, thainer, naamapadam, multiconer, stockmark are named-entity sets,
not privacy annotations (multiconer entities are public figures). They are reported in a
separate "entity recognition" table, never pooled into the privacy headline. TAB is scored
with its own DIRECT/QUASI masking rule (`score.masking_recall`).

## 2. Comparison units

An entry is a full configuration: package + version + checkpoint + revision + labels/prompts
+ threshold + mask operation. Exact values live in `entries.py`; summary:

| entry | package / version | checkpoint @ revision | GPU need |
|---|---|---|---|
| sct-default | SqueakyCleanText 0.8.1 | SCT default ONNX NER (xlm-r conll03/02, wikineural) @ `DEFAULT_NER_REVISIONS` | none (CPU) |
| sct-gliner | SqueakyCleanText[gliner] 0.8.1, `ner_mode='pii'` | knowledgator/gliner-pii-base-v1.0 ONNX @ `61726e0` | none; ~1 GB if cuda |
| presidio | presidio-analyzer 2.2.366, spaCy 3.8.18 | en_core_web_lg 3.8.0 | none (CPU) |
| gliner-nvidia | gliner 0.2.30 | nvidia/gliner-PII @ `bd23e8e` (570M) | ~3 GB fp16, batch <= 4; CPU fallback |
| gliner2-fastino | gliner2 2.1.0 | fastino/gliner2-privacy-filter-PII-multi @ `1cb4166` (205M) | ~1.5-2 GB; CPU fine |
| opf | transformers 5.22.1, torch 2.15.0 | openai/privacy-filter @ `7ffa9a0` (1.4B MoE, ~50M active, bf16) | ~3-5 GB at batch 1; CPU fallback |
| bardsai-v3 | transformers 5.22.1, onnxruntime 1.31.1 | bardsai/eu-pii-anonimization-multilang-v3 @ `091984b`, INT8 ONNX, bundled `pii_detector.py` | none (CPU) |
| datafog | datafog 4.9.0 | regex engine, no model | none |

SCT's optional GLiNER path is its own entry (it shares no checkpoint with gliner-nvidia).
Shared components are listed so a reader knows two rows are not independent: presidio and
datafog both default to regex for structured types; sct-gliner and gliner-nvidia share the
`gliner` package but not the weights.

Reserve (run if time allows, same contract): scrubadub 2.0.1 (+ scrubadub_spacy 2.0.0, own
venv), DataTrove 0.10.1 `PIIFormatter` (EMAIL/IP only), urchade/gliner_multi_pii-v1,
OpenMed/privacy-filter-multilingual-v2, Piiranha v1 (CC-BY-NC-ND weights: personal
benchmarking only, no redistribution).

Dropped: nothing in the shortlist top 8 fails the "offline, Linux, CPU or <= 6 GB GPU" bar.
scrubadub and DataTrove moved to reserve to keep 8 entries (both run, both narrow). Out
already: NeMo Anonymizer / LLMPiiModifier (need an LLM endpoint or a 70B model), cloud APIs
(not offline), LLM Guard (archived, Python < 3.13), small-LLM redactors (rewrite text, no spans).

## 3. Tracks and the configuration contract

- **default**: `pip install` the pinned packages and run the vendor quick start: vendor
  default threshold, vendor default model, no language hint beyond what the quick start
  passes. This is what a user gets.
- **tuned**: a deployment-ready configuration. Allowed changes, and only these:
  1. threshold / operating point (one scalar, or OPF's documented decoder-bias preset);
  2. language routing: the tool's own language packs for the gold language (Presidio spaCy
     `xx_core_news_lg` per language where it exists, else `xx_ent_wiki_sm`; Presidio
     `supported_regions` for phones), passed the gold language code;
  3. enabling the tool's own optional detectors/types (DataFog extra types, Presidio
     `GLiNERRecognizer` is NOT allowed: that turns Presidio into a GLiNER entry);
  4. for prompt-based models, choosing between the fixed `GLINER_PROMPTS` list and the
     model card's own label names (one choice per model, not per dataset).
  Not allowed: per-dataset settings, word lists, custom regexes, gold-label-specific prompts,
  retraining, test-split data.
- **Budget**: the same for every entry: at most 12 configurations evaluated on `dev`
  (e.g. 6 thresholds x 2 prompt lists), selected by one pre-declared objective: the
  maximum masked F2 (type-agnostic, partial, recall-weighted) macro-averaged over the
  privacy datasets' dev parts. One configuration per entry for all datasets. Logged.
- SCT's tuned track follows the same budget (`ner_confidence_threshold`,
  `gliner_threshold`). SCT's `training` and `ner-off` configs are reported only as extra
  SCT rows, not as entries.
- Tuning happens once, configs are frozen in `entries.py` (with the dev scores that chose
  them), then test runs happen once. A rerun of test needs a logged reason.

## 4. Entity coverage views (rule 4)

Scored taxonomy = the 9 SCT types (PERSON, LOCATION, ORGANISATION, EMAIL, PHONE, IP, URL,
DATE, NUMBER), with dataset-side mapping in `benchmarks/labels.py`.

1. **Tool-supported subset**: per entry, only gold of the types in its `supports`;
   unsupported types show "N/A", never 0.
2. **Shared intersection**: types every entry supports (with the current set:
   EMAIL, PHONE, DATE, NUMBER; without datafog: + PERSON, LOCATION, URL). The headline
   ranking uses this view.
3. **Full taxonomy**: all 9 types; an unsupported type counts as missed.

Type-agnostic masked recall and leak (section 7) are reported in all three views and are
the main privacy numbers, since taxonomies differ (OPF 8 classes, Nemotron 55).
DATE is shown but kept out of totals in every view (greedy for SCT and Presidio alike),
with a "with DATE" total in an appendix. NUMBER also gets a per-subtype table where gold
has fine types (IBAN, card, SSN, ID).

## 5. Label mapping per tool

Tool label -> SCT type maps are in `entries.py` (`labels`), reviewed once and frozen
before dev runs. Tool labels not in the map are out of scope: dropped before scoring
(not false positives, not masked) and counted in the result file. Key points:

- presidio: PERSON, LOCATION, ORGANIZATION, EMAIL_ADDRESS, PHONE_NUMBER, IP_ADDRESS, URL,
  DATE_TIME and ID/financial recognisers (IBAN_CODE, CREDIT_CARD, US_*, UK_*) -> NUMBER.
  NRP, MEDICAL_LICENSE, CRYPTO out of scope. ORGANIZATION is not emitted by the spaCy engine,
  so ORG is unsupported in its default row.
- GLiNER family: one fixed prompt list (`GLINER_PROMPTS`) for every GLiNER entry, so prompt
  wording is not a per-tool advantage.
- opf: private_person, private_address, private_email, private_phone, private_url,
  private_date, account_number; `secret` out of scope; no ORG and no IP class.
- bardsai-v3: its 37 labels mapped from `config.json` `id2label` at adapter time with the
  same rules (name parts -> PERSON, address parts -> LOCATION, IDs/IBAN/card/VAT -> NUMBER,
  GDPR art. 9 categories out of scope); the frozen map is written into `entries.py`
  before the dev run.
- datafog: EMAIL, PHONE, IP, CREDIT_CARD/SSN/ZIP -> NUMBER, DOB -> DATE (names checked
  against the 4.9.0 wheel when the adapter is written).
- Gold side: the `labels.py` name-part merge (now also FIRSTNAME, MIDDLENAME, LASTNAME)
  and nested-span rules apply equally to every entry.

## 6. Offsets and matching (rule 7)

- Offsets are Python code-point offsets into the original, unmodified text, half-open
  `[start, end)`. Adapters convert token/byte offsets back to code points and must pass a
  multilingual fixture test (Latin with combining marks, Arabic, Devanagari, Thai, CJK,
  emoji, CRLF) before any run.
- Every entry receives the same raw text. SCT is scored on raw offsets through reversible
  mode (as `run.py` does now).
- Prediction clean-up, the same for all: drop empty spans, merge same-type overlapping spans
  into their union, keep both when types differ.
- Strict: exact span + type, one-to-one (multiset match, `sct.evaluation.evaluate_pii`).
- Partial: overlap of at least one character, same type, one-to-one: greedy matching by
  largest overlap, then earliest start. (Note: current `score.py` partial is many-to-one;
  the one-to-one variant is added in the scoring step and both are reported.)
- Character level: per-character P/R/F1 over in-scope gold characters, type-agnostic, plus
  F2 (recall-weighted), as in PIIMB and the OPF-42 paper.

## 7. Leak definitions (rule 8)

All on the masked output of the common mask operation (section 9), per gold mention, DATE
excluded from totals:

- **exact-value visibility**: the full gold value (NFKC) still appears anywhere in the output
  (the current `leak` "full").
- **complete-mention exposure**: no character of the mention's own span was masked.
- **character survival**: share of the mention's letters/digits not masked (micro over
  mentions); also the share of mentions with any 4+ char word surviving (current "partial").
- **repeated-value exposure**: the value occurs more than once in the doc and at least one
  occurrence is unmasked (reported separately, because the current exact-value check counts
  it even when the gold mention itself was masked).
- Mask markers are excluded before checking (`<TYPE>` placeholders are never leaks).
- Document leak rate: share of docs with at least one exposed mention; denominator = docs
  with at least one in-scope gold mention.

## 8. Over-masking (rule 5)

- **non-PII characters removed**: masked characters outside any gold span (in-scope or out of
  scope) / all non-gold characters, plus masked chars / gold chars (PII-TRACE style ratio).
- **clean-document false-positive rate**: share of gold-empty docs with any prediction. Clean
  docs are the datasets' own docs with no gold span (counted per dataset). FineWeb-2 is not
  used for this: it is not annotated, so it has no clean ground truth.
- Utility: chrF of the masked output vs the gold-masked text, at matched leak levels where
  curves allow (section 9).

## 9. Masking operation (rule 9)

- **detection track** (headline): every entry's spans go through one common mask operation:
  replace each span with `<TYPE>` of its mapped SCT type, out-of-scope labels unmasked.
- **native pipeline track**: each tool's own anonymised output where it has one (SCT
  `process()`, Presidio anonymizer, DataFog redact). Scored only with leak and over-masking;
  detection scores are marked unavailable unless offsets map back.

## 10. Threshold curves and operating points (rule 6)

- Scored tools (GLiNER entries, sct-gliner, presidio scores, bardsai probabilities, OPF via
  decoder bias presets) are run once at a threshold of 0.05 (or the lowest available) and
  the curve is built by filtering: thresholds 0.05..0.95 step 0.05.
- Plots: precision-recall (partial, type-agnostic) and leak-vs-over-masking per dataset.
- Matched operating points: on `dev`, the threshold reaching masked recall 0.90 (and 0.95)
  per entry; those thresholds are then applied on `test`. Tools without scores (regex,
  sct-default) are single points.

## 11. Long documents (rule 14)

| entry | native limit | handling |
|---|---|---|
| sct-default / sct-gliner | model windows inside SCT | SCT's own chunking (native) |
| presidio | spaCy, no hard limit | whole doc (`nlp.max_length` raised if needed) |
| gliner-nvidia, gliner2-fastino | 384-512 tokens | common chunker |
| opf | 128k context | whole doc (native) and common chunker |
| bardsai-v3 | 512 | bundled detector: 512-token windows, 128 overlap (native) |
| datafog | regex | whole doc |

Common chunker: sentence-aware windows of at most the model's max tokens, 64-token overlap,
spans in overlaps merged by union, recorded per entry. Two tracks: native (each tool's
own handling) and common chunking (for the models that need chunking). TAB and the
long-doc buckets of `by_position` recall show any late-window loss. A tool that truncates
is scored on the whole doc anyway.

## 12. Language handling (rule 13)

- **gold-language track**: every tool that takes a language gets the gold code
  (Presidio `language=`, SCT `language=`). This is the main track.
- **native routing track**: SCT with its own LID (Lingua), presidio default `en`. Routing
  accuracy is reported (SCT LID vs gold), and unsupported-language behaviour is listed per
  entry (which languages the entry claims; scored languages outside the claim are flagged).
- Code-switched data: bayanat `ar-mixed_en` reported separately.

## 13. Aggregation (rule 11)

- Per dataset (and per language within it): P/R/F1 strict, partial (one-to-one), character
  level, masked recall, leak measures, over-masking, with doc, gold-span and predicted-span
  counts.
- Macro: mean over datasets of the dataset score (each dataset's own score is the macro
  over its languages). Micro: pooled counts. Both always shown.
- Synthetic (openpii, gretel, gretel_en, nemotron, argilla_pii, thai_pii, bayanat) and real or
  human-annotated (TAB, REDACT, iahlt, thainer, naamapadam, multiconer, stockmark) are
  aggregated separately; never one pooled number across both.
- Per-language quota: 300 test docs per language per dataset (fewer where the test part is
  smaller: REDACT 27-40 per language, TAB 106).
- "Out-of-domain only" averages exclude cells with any exposure in section 16.

## 14. Uncertainty (rule 12)

- Paired bootstrap over documents (over groups for gretel, gretel_en, nemotron, redact),
  1,000 resamples, seed 13: 95% CI for each score and for each difference SCT minus entry
  on the same resamples. A "win" needs the difference CI to exclude 0.
- Document-level leak: paired McNemar on per-doc "any exposed mention" for SCT vs each entry.
- Repeat runs: speed 3x (median, spread); quality runs are deterministic on CPU, GPU runs are
  repeated twice and the difference reported.

## 15. Failure handling (rule 19)

Per document: timeout 120 s (TAB 600 s), one retry, then "failed". Crashes, OOM, timeouts,
unsupported input are recorded with the reason. Reported: completion rate per entry x
dataset; quality on completed docs (conditional); and effective protection counting failed
docs as unmasked (failed doc = every mention leaked, no false positives). An entry with
completion < 99% on a dataset is flagged in every table. No silent drops.

## 16. Speed and resources (rules 17, 18)

Box: AMD Threadripper PRO 7985WX (64 cores, 2 threads each; CPUs 0-63 are one thread per
core), 754 GB RAM, 4x RTX 4090 shared with a vLLM server (16.5-19 GB in use per GPU).

- CPU: `taskset -c 0-15`, `OMP_NUM_THREADS=MKL_NUM_THREADS=16`, torch/ONNX intra-op 16,
  inter-op 1; single-thread latency run on `taskset -c 0` with all thread caps 1.
- GPU: one card, the one with the most free memory at start; `torch.cuda.set_per_process_memory_fraction`
  equal to 5 GB for torch entries; ONNX entries cap with `gpu_mem_limit` 5 GB. Free memory
  and `nvidia-smi` occupancy logged before and after; a run starts only with >= 6 GB free.
  Precision (fp32/fp16/bf16/INT8) and any CPU offload stated per entry.
- Measures, in randomised entry order: cold start (new process, cached weights, import +
  load + first prediction); warm batch-1 latency (median, p95 after 20 warmup docs, GPU
  synchronised); throughput at common batch 8 (docs/s, chars/s); best-batch throughput
  (entry's own optimum, separately labelled). Peak RSS (psutil sampler, 50 ms) and peak
  VRAM (`torch.cuda.max_memory_allocated` + `nvidia-smi` sampler for ONNX). Install size:
  fresh venv `du -sh` + model cache size.
- Speed set: FineWeb-2 speed sample (`speed` suite docs), 3 repeats, median.

## 17. Model-dataset contamination matrix (rule 2)

T = trained on that dataset's train split (or the same generator family), E = the vendor
reports evaluating on it (tuning exposure possible), D = SCT was developed or tuned against
it in earlier benchmark rounds, U = training data undisclosed, `-` = none known. None of
these prove overlap with our test records. Sources: model cards at the pinned revisions,
checked 2026-11-02.

| entry | openpii | gretel fin | gretel_en | nemotron | argilla_pii | redact | tab | others (iahlt, bayanat, thai*, naamapadam, multiconer, stockmark) |
|---|---|---|---|---|---|---|---|---|
| sct-default (regex + CoNLL/WikiNEuRal NER) | D | D | - | D | - | D | D | D |
| sct-gliner (knowledgator gliner-pii-base) | U, D | U, D | U | U, D | U | U, D | U, D | U, D |
| presidio (en_core_web_lg, OntoNotes 5) | - | - | - | - | - | - | - | - |
| gliner-nvidia | E (AI4Privacy, set unnamed) | - | E | T (Nemotron-PII train) | E | - | - | - |
| gliner2-fastino (own 4,910 synthetic docs) | - | - | - | - | - | - | - | - |
| opf | U | U | U | U | U | U | U | U |
| bardsai-v3 | T (openpii-1m train) | - | T (train split) | T (train split) | T-family (ai4privacy generator) | - | - | - |
| datafog (regex) | - | - | - | - | - | - | - | - |
| reserve: OpenMed multilingual v2 | T | - | T | T | T (pii-masking-200k is Argilla's source) | - | - | - |
| reserve: urchade gliner_multi_pii-v1 | - | - | - | - | - | - | - | - |

Notes: nvidia/gliner-PII names its base as urchade/gliner_large-v2.1 on the HF card and
knowledgator/gliner-bi-large-v1.0 on the NIM page (conflict, recorded). bardsai removed rows
that appear in the PIIMB test sets, not ours. Argilla is a slice of the ai4privacy-200k
TRAIN split, so any model trained on that split may have seen the exact records. SCT's D
marks are the datasets used in SCT's earlier benchmark rounds; `gretel_en` and
`argilla_pii` were added after SCT 0.8.1 and are held out for SCT (no tuning on them,
dev included, until this comparison is published). The headline "out-of-domain" table uses
only cells marked `-` for that entry.

## 18. Annotation audit (rule 10)

- Gold quality flags: gretel finance (LLM + GLiNER gold), gretel_en (generator-injected
  strings, exact-occurrence placement), argilla_pii (unreviewed suggestions), Nemotron
  (1.5% span-text mismatches). Human-annotated: TAB, REDACT, iahlt, thainer, naamapadam
  (partly), multiconer, stockmark.
- Audit (planned, not yet done): per dataset, 50 test docs stratified by language and by
  "false positive of any entry", reviewed blind (tool names hidden) for unlabelled PII.
  Only languages the reviewer reads fluently (en, nl, de, fr, es plus whoever helps); other
  languages are listed as not audited. Results adjust nothing; they are reported next to the
  precision numbers.
- Template families: dev/test split by template where the data has one (section 1).

## 19. Result bundle (rule 20)

Under `benchmarks/results/compare/<date>/`: per entry x dataset JSON (scores, counts,
config, versions, model revisions, device, threads, seed, sample-id hash, dataset revision),
`uv.lock` / `pip freeze` per entry venv, the frozen `entries.py` copy, dev tuning log, speed
JSON, failures, the summary tables, and this protocol's commit sha. Predictions only as
`{id, start, end, label, score}` for `scores-and-ids` datasets; none for `scores-only`
(REDACT, argilla_pii). Never any dataset text. Licences of every dataset and model in the
summary.

## 20. Rule checklist (codex-fair-comparison.md)

| # | rule | where | status |
|---|---|---|---|
| 1 | freeze tuning before testing | 1, 3 | split implemented; budget and freeze defined |
| 2 | contamination matrix | 17 | done from cards |
| 3 | PII vs NER | 1 | NER sets in a separate table |
| 4 | three coverage views | 4 | defined (`supports` in `entries.py`) |
| 5 | over-masking | 8 | defined; downstream utility only as chrF (no downstream task model) |
| 6 | threshold curves | 10 | defined |
| 7 | offsets and matching | 6 | defined; one-to-one partial to add in scoring step |
| 8 | leak definitions | 7 | defined; three new measures to add in scoring step |
| 9 | detection vs pipeline | 9 | two tracks |
| 10 | annotation audit | 18 | partial: flags done, blind audit planned, limited to languages a reviewer reads |
| 11 | aggregation and quotas | 13 | defined |
| 12 | paired uncertainty | 14 | defined |
| 13 | language detection separate | 12 | two tracks |
| 14 | long docs | 11 | defined |
| 15 | entries as configurations | 2, `entries.py` | done |
| 16 | configuration contract | 3 | done |
| 17 | CPU/GPU control | 16 | defined |
| 18 | cold/warm/batch | 16 | defined |
| 19 | failures kept | 15 | defined |
| 20 | rerunnable bundle | 19 | defined |

Not doable as written: rule 5's "downstream utility" beyond chrF (no downstream task is in
scope); rule 10's fluent review for all ~40 languages (only the languages a reviewer reads);
rule 2 cannot be complete for OPF and the knowledgator model (training data undisclosed).

## Change log

- 2026-11-02: protocol written; splits computed (seed 13, dev share 0.2).
