# Tool comparison protocol

How SCT is compared with other PII tools. Frozen before any test run; a change after
the first test run is logged at the end of this file with its reason. It follows the 20
fairness rules in `sct-integration-research/research/benchmark/codex-fair-comparison.md`;
section 20 maps each rule to where it is handled, or why it is not.

Code: `split.py` (dev/test), `entries.py` (comparison units, label maps, run registry,
contamination flags), `envs/<entry>.txt` (pinned installs, one uv venv per entry),
`adapters/` (the adapter contract in `_protocol.py`, one script per tool), `runner.py`
(venvs, offset check, supervised runs, scoring, bundle), `scorer.py` (the measures below, on
top of `benchmarks/score.py`), `speed.py` (section 16). Loaders reuse `benchmarks/datasets/`.
Adapters: one per entry in section 2 (round 1 and round 2), plus the test-only `dummy`.

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

### Round 2 datasets

Twelve open sets plus one gated set (sct-integration-research `round2-datasets/SHORTLIST.md`, pinned
2026-11-03). None was used while SCT was developed, so SCT's cells are clean (`entries.ROUND2` flags); the
shortlist's per-tool cells are in `entries.CONTAMINATION`. They score in their own groups
(`r2-privacy-synthetic`, `r2-privacy-real`, `r2-untyped`), so round-1 group numbers are unchanged, and
`tune.py` does not tune on them (round-2 default: no tuning). Same split rule (seed 13, dev share 0.2),
except that a dataset's own split is kept: the loader sets `record["part"]` and `split.py` uses it.

| dataset | source @ revision | licence, publish | split used | group key | langs | gold notes |
|---|---|---|---|---|---|---|
| newmean | `newmean-ai/pii-dataset` @ `062c374` | CC BY-NC 4.0, scores-only | hash over test (17,058) | id | 54 | SECRET, USERNAME out of scope |
| dialogpii | Zenodo 20863452, zip md5 `6b034ea` | CC BY 4.0 | hash over all 3,234 dialogs | scenario:chat_number (all languages and both versions together) | 11 + `-asr` transcripts | one record per dialog, `SPEAKER: text` lines |
| kdpii | Zenodo 10968609, md5-pinned files | CC BY 4.0 | own: `test.json` = test (4,891), `valid.json` = dev (5,011) | dialog | ko | one record per sentence; PS_ID (user name) out of scope |
| nus_lt | `VytautoDidziojoUniversitetas/NUS-LT-PII-corpus` @ `3097156` | Open RAIL-D, scores-only | hash over `validacija` (123 real docs) | id | lt | MISC split into email / url by the gold value's shape, rest out |
| redmadrobot | `redmadrobot-rnd/pii_benchmark` @ `f77ea83` | MIT | hash over test (2,841) | id | ru | BIO tokens aligned to the text; no ORG/DATE gold |
| redactionbench | `RedactionBench/RedactionBench` @ `d45e9ce` | CC BY 4.0 (49 sources all rights reserved), scores-only | hash over test (200 long docs) | id | en | untyped: `mandatory` = `UNTYPED`, `contextual` out of scope |
| tonic | `TonicAI/Privacy-Bench` @ `a79d9e8` | CC BY 4.0 | hash over the 6 human-annotated workspaces | thread / document | en | USERNAME out of scope |
| meddeid | `stighellemans/meddeid-{english,dutch}-synthetic-benchmark` @ `288c571` / `45c601b` | CC BY 4.0 | own: all test, no dev (cards: final evaluation only) | id | en-gb, en-us, nl-be | ages and Profession out of scope; birth dates DATE |
| greek_deid | `prorris/greek-deid-benchmark` @ `dd82ef6` | CC BY 4.0 | hash over test (500 long notes) | id | el | `gold_standard.json`; must-survive probes kept as `record["survive"]` |
| neonredact | `neondijital/neonredact-tr-bench` @ `72e799f` | CC BY 4.0 | own: dev 100, test 300 | id | tr | the author's 7 unscored labels out of scope |
| pl_pii | `pl-pii-bench/pl-pii-bench` @ `bef947d` | CC BY 4.0 | hash over the 7 public lanes | source document / inflected name | pl | `protection: keep` entities out of scope |
| europii | `sinceaihq/europii-bench` @ `b407a9a` | CC BY 4.0 | own: dev 415, test 2,075 | id | 12 | hard negatives are not gold |
| carmen | PhysioNet CARMEN-I v1.0.2 | PhysioNet licence + DUA, scores-only | not loaded: gated | - | es, ca | stub raises "needs PhysioNet access" |

Untyped gold (RedactionBench) is in every view and scored by the type-agnostic measures only; strict and
partial (typed) measures are reported as None for it. The Greek must-survive probes give
`result["survive"]`: the share of probe values still verbatim in the common masked output (section 8).
Dev/test sizes for the hashed sets come from `python -m benchmarks.compare.split --datasets <names>` on first
download (not computed yet: the data was not reachable from the machine that wrote the loaders).

## 2. Comparison units

An entry is a full configuration: package + version + checkpoint + revision + labels/prompts
+ threshold + mask operation. Exact values live in `entries.py`; summary:

| entry | package / version | checkpoint @ revision | GPU need |
|---|---|---|---|
| sct-default | SqueakyCleanText 0.8.2 | SCT default ONNX NER (xlm-r conll03/02, wikineural) @ `DEFAULT_NER_REVISIONS` | none (CPU) |
| sct-gliner | SqueakyCleanText[gliner] 0.8.2, `ner_mode='pii'` | knowledgator/gliner-pii-base-v1.0 @ `61726e0`, PyTorch (SCT falls back: no root `model.onnx`) | none; ~1 GB if cuda |
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

Reserve (run if time allows, same contract): DataTrove 0.10.1 `PIIFormatter` (EMAIL/IP only),
urchade/gliner_multi_pii-v1. scrubadub, OpenMed and Piiranha moved to the round-2 entries below.

Round 2 entries (`entries.ROUND2_ENTRIES`, user list in sct-integration-research `compare/ROUND2-TODO.md`).
GPU entries run in CUDA 12 venvs: torch 2.14.1+cu126 as a direct wheel from the PyTorch cu126 index (the newest
release with a cu126 build; Linux x86_64, Python 3.12), onnxruntime-gpu 1.29.0 as a direct wheel from ORT's CUDA 12
feed (PyPI's onnxruntime-gpu is the CUDA 13 build since 1.27). Round 2
has no tuning (default track only).

| entry | package / version | checkpoint @ revision | GPU need |
|---|---|---|---|
| gliner-knowledgator | gliner 0.2.30 | knowledgator/gliner-pii-base-v1.0 @ `61726e0` (the sct-gliner model, standalone), card threshold 0.3 | ~1 GB |
| piiranha | transformers 5.22.1 | iiiorg/piiranha-v1-detect-personal-information @ `255acde`, 256-token windows (CC-BY-NC-ND weights: benchmark only, never shipped) | ~1 GB |
| llm-guard | llm-guard 0.3.16 (last release, archived) | `Anonymize` defaults: Isotonic/deberta-v3-base_finetuned_ai4privacy_v2 @ `9ea9927` + Presidio regex | ~1 GB |
| scrubadub | scrubadub 2.0.1 + scrubadub_spacy 2.0.0 | default detectors + `SpacyEntityDetector` (en_core_web_trf 3.8.0) | none (CPU: spaCy needs `prefer_gpu` + cupy for GPU, neither is used) |
| ai4privacy | transformers 5.22.1 | ai4privacy/llama-ai4privacy-multilingual-categorical-anonymiser-openpii @ `83ef30d` (ModernBERT) | ~1 GB |
| openmed | transformers 5.22.1 | OpenMed/privacy-filter-multilingual-v2 @ `0d0c043` (OPF fine-tune), opf decoder at biases 0 | ~3-5 GB |
| pii-tracer | transformers 5.22.1 (remote code) | perplexity-ai/PII-Tracer @ `d25c16f`, vendor `predict`, 4096-token truncation | ~2-3 GB |
| presidio-full | presidio-analyzer 2.2.366 | every language pack, phone regions, every predefined pattern recognizer, ORGANIZATION kept | none (CPU) |
| sct-training | SqueakyCleanText 0.8.2 + onnxruntime-gpu | SCT `training` preset (extra SCT row, section 3) | ~1-2 GB |
| sct-gliner-nvidia | SqueakyCleanText[gliner] 0.8.2 | nvidia/gliner-PII @ `bd23e8e` through SCT's GLiNER backend (`gliner_model`, `ner_backend='gliner'`: PyTorch, no forced ONNX); SCT's PII prompts and threshold (tuned on knowledgator) as shipped | ~3 GB |

Shared view per round (`runner.shared_view`): a run on round-1 datasets keeps round 1's eight entries
(EMAIL, PHONE, DATE, NUMBER), so round-1 scores do not move; a run with any round-2 dataset uses every
entry: EMAIL, PHONE, NUMBER (llm-guard, scrubadub and sct-training claim no DATE; DATE is outside totals anyway).

Dropped: nothing in the shortlist top 8 fails the "offline, Linux, CPU or <= 6 GB GPU" bar.
scrubadub and DataTrove moved to reserve to keep 8 entries (both run, both narrow). Out
already: NeMo Anonymizer / LLMPiiModifier (need an LLM endpoint or a 70B model), cloud APIs
(not offline), LLM Guard (archived, Python < 3.13), small-LLM redactors (rewrite text, no spans),
NeMo Curator 1.x (no PII stage; the 0.x one was English-only Presidio), StarPII (gated
evaluation-only terms, no redistribution) and deduce (GPL-3.0; benchmark use only): neither is an
entry here. Published scores are compared only at the same unit: OPF's published 96% F1 is per
token, its per-span F1 is 0.926.

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
- Tuning happens once, configs are frozen in `configs/tuned.json` (with the dev scores that
  chose them; `runner.py --tuned configs/tuned.json` applies them), then test runs happen once.
  A rerun of test needs a logged reason.
- The tuned track chose one configuration per entry on the dev parts of all nine privacy sets
  (every entry, SCT included), so it carries tuning exposure to every privacy set's dev part. Where
  a comparison has a clean cell, the **default** track is the primary result and tuned is shown
  second.

How it ran (`tune.py`, grid in `tune.GRID`, all on the dev part, 100 docs per language, gold
language track, the 9 privacy datasets; TAB 21 docs, REDACT 175, gretel 487):

- Objective, exactly: per dataset, the language macro of masked F2 on the full taxonomy
  (DATE out), every doc counted, failed docs unmasked; then the mean over datasets. Masked
  F2 = F2 of masked recall (gold mentions overlapped by any kept prediction) and masked
  precision (kept predictions that overlap any in-view gold). Ties: higher masked precision,
  then grid order. The full view, not the shared one, so a tool is tuned on all it must mask.
- Scored tools run once per prompt list / language setting at their floor (GLiNER 0.05,
  Presidio and bardsai at the vendor 0) and the grid thresholds are filtered from that run
  (exact for Presidio and bardsai; an approximation for GLiNER flat decoding, section 10).
  Scoreless tools (SCT, DataFog, OPF presets) run once per candidate.

| entry | candidates (<= 12) |
|---|---|
| sct-default | `ner_confidence_threshold` 0.5, 0.6, 0.7, 0.8, 0.85 (default), 0.9 |
| sct-gliner | `gliner_threshold` 0.1, 0.2, 0.3 (default), 0.4, 0.5, 0.6 |
| presidio | {en model only, gold language packs + phone regions} x `score_threshold` {vendor 0, 0.1..0.5} |
| gliner-nvidia | {GLINER_PROMPTS, card labels} x threshold 0.1..0.6 |
| gliner2-fastino | {GLINER_PROMPTS, card labels} x threshold 0.1..0.6 |
| opf | bias preset b in -1, -0.5, 0 (default), 0.5, 1, 2: background_stay -b, background_to_start +b, inside_to_continue +b |
| bardsai-v3 | threshold {vendor 0, 0.3, 0.5, 0.7, 0.8, 0.9} |
| datafog | locales {none, gold}; every regex type is on by default, so no type switch is left |

- Presidio language packs: spaCy `*_core_news_lg` (zh `zh_core_web_lg`, no `nb_core_news_lg`)
  where one exists, else `xx_ent_wiki_sm` (`tune.PRESIDIO_MODELS`), plus `PhoneRecognizer`
  regions = vendor default + one region per scored language (`tune.PHONE_REGIONS`).
- OPF ships one operating point (all biases 0) and a README sentence on the direction
  (less background stay, easier span entry and continuation = more recall); the preset family
  scales that one direction. The vendor gives no named presets.
- Device: every tuning run was on CPU (`taskset` 16 cores per job). torch 2.15.0's Linux
  wheels are CUDA 13.2 builds and the box's driver is CUDA 12.6, so torch entries see no GPU
  with the pinned torch; gliner-nvidia therefore runs fp32 (its fp16 is the cuda path). Test
  runs must use the same device per entry as tuning.
- sct-gliner: SCT 0.8.2 loads knowledgator/gliner-pii-base-v1.0 with PyTorch (the repo has no
  root `model.onnx`; SCT logs the fallback), not ONNX.
- Harness fix after tuning started: `sct_adapter.sct_view` mapped view chars inserted at the end
  of the doc's last token to `len(source)`, so a placeholder there crashed span placement
  (`StopIteration`; 1 openpii dev doc of sct-gliner per run, scored as unmasked). Fixed for the
  test runs; the tuning numbers keep that one failed doc.

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
- Round 2 (frozen from each checkpoint's config at the pinned revision; each adapter's `LABELS` /
  `OUT_OF_SCOPE` cover every type and `load` refuses a checkpoint whose types differ): name parts -> PERSON,
  address parts (street, building number, city, county, state, GPS) -> LOCATION, postcodes and IDs /
  accounts / cards / device and vehicle IDs -> NUMBER, DOB and DATE -> DATE; credentials, usernames,
  crypto addresses, money, age / sex / gender / title, job and TIME out of scope. piiranha 17 types,
  ai4privacy 20, openmed 54, pii-tracer 9 (opf's classes + `other_pii`; `secret`, `other_pii` out),
  llm-guard its 13 default entity types (CRYPTO, UUID, CUSTOM out), scrubadub its filth types (credential,
  twitter, skype out). gliner-knowledgator uses `GLINER_PROMPTS`; presidio-full presidio's map;
  sct-training and sct-gliner-nvidia SCT's.
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
  largest overlap, then earliest start (`scorer.py`). The many-to-one partial of
  `score.py` is reported next to it.
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
  the curve is built by filtering: thresholds 0.05..0.95 step 0.05. Points below the
  threshold the adapter ran at are never reported (a default run at 0.5 gives 0.5..0.95; the
  full curve is a dev run with `--config <entry>='{"threshold": 0.05}'`). For GLiNER flat
  decoding a filtered point is an approximation of a run at that threshold (decoding picks
  among the candidates above the floor). OPF's span score is not its operating point: OPF
  has no filtered curve; each bias preset is its own `--config` run.
- Plots: precision-recall (partial, type-agnostic) and leak-vs-over-masking per dataset.
- Matched operating points: on `dev`, the threshold reaching masked recall 0.90 (and 0.95)
  per entry; those thresholds are then applied on `test`. Tools without scores (regex,
  sct-default) are single points.

## 11. Long documents (rule 14)

| entry | native limit | handling |
|---|---|---|
| sct-default / sct-gliner | model windows inside SCT | SCT's own chunking (native) |
| presidio | spaCy, no hard limit | whole doc (`nlp.max_length` raised if needed) |
| gliner-nvidia | 384 words / 512 subwords | vendor truncation at 384 words (native, logged) and common chunker |
| gliner2-fastino | 384 words | library `batch_extract_entities_long`: 384-word windows, 64 overlap (native) |
| opf / openmed | 128k context | vendor CPU setting on every device: 4,096-token windows, no overlap (128k does not fit ~6 GB under transformers); `flex_attention` on cuda (eager's full T x T scores hit CUDA OOM at 4,096 tokens under the 5 GB cap) |
| bardsai-v3 | 512 | bundled detector: 512-token windows, 128 overlap (native) |
| datafog | regex | whole doc |
| llm-guard | DeBERTa 512 positions | llm-guard's own 600-character chunks, 40 overlap; the adapter sets the tokenizer's 1e30 "no limit" sentinel to the model's 512 so that chunking runs (else whole docs, CUDA OOM) |

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
- **Claimed languages (permanent rule, every round).** Each entry claims a language set
  (`entries.CLAIMED_LANGUAGES`, from the model card's language tags at the pinned revision; `None`
  = no claim: regex tools and SCT's multilingual default). Round 1: nvidia/gliner-PII en;
  bardsai-v3 the 24 official EU languages; gliner2-fastino en fr es de it pt nl; opf en ("primarily
  English"); presidio en (only `en_core_web_lg` is loaded, both tracks); sct-gliner en (the
  knowledgator/gliner-pii-base-v1.0 card metadata at 61726e0 lists `language: [en]`; its
  `multilingual` tag is a topic tag, not a language tag). sct-default claims every language (SCT
  claims multilingual NER through its WikiNEural fallback), so SCT is held to every language;
  datafog (regex) makes no claim and is scored on every language too. A code is matched on its base
  (`pt_br` -> pt, `ar-mixed_en` -> ar). Every headline and pairwise score uses only the documents in
  languages the entry (both entries, for a pair) claims; scores in other languages are reported
  apart as "outside claimed languages" and are never shown or counted as plain losses
  (`runner.rule_views`, `report.own_table`, `°` in the per-language tables).
- `runner.py --track gold|native` (default gold) applies one track to every entry
  (`entries.GOLD_TRACK`; DataFog gets its locale pack in gold); a results table never mixes them.

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
- Headline and same-set averages follow section 17's rule (clean cells only) and section 12's
  claimed-language rule. Per-dataset tables also use only the docs in each entry's claimed
  languages (`report.claimed_macro`); a cell with no such doc is `°`, and the out-of-claim values
  are a separate table.

## 14. Uncertainty (rule 12)

- Paired bootstrap over documents (over groups for gretel, gretel_en, nemotron, redact),
  1,000 resamples, seed 13: 95% CI for each score and for each difference SCT minus entry
  on the same resamples. A "win" needs the difference CI to exclude 0. Each resample
  recomputes the language macro (the headline score), not pooled micro counts.
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
  (entry's own optimum, separately labelled). Peak RSS (the adapter process's own `getrusage`
  peak, exact, no sampler) and peak VRAM (`nvidia-smi` compute-apps polled every 0.2 s for the
  adapter's PID only, plus `torch.cuda.max_memory_allocated` when torch is used). Install size:
  fresh venv `du -sh` + model cache size.
- Speed set: FineWeb-2 speed sample (`speed` suite docs), 3 repeats, median.
- GPU column (round 2 on): `speed.py --device cuda` into `<bundle>/speed-gpu/`; per entry the device it ran on
  (`speed.entry_device`: cpu for a CPU-only entry), batch-8 docs/s and peak VRAM (nvidia-smi for the adapter's
  PID, `speed.vram_mb`; torch's own peak when nvidia-smi has none). `report.py` shows it in the speed table and as
  the "GPU: device, docs/s, peak VRAM" column of the headline table. Torch entries cap themselves at
  `gpu_mem_gb` with `torch.cuda.set_per_process_memory_fraction` (`_protocol.cap_gpu`).
- In a cuda run (runner and speed) a CPU-only entry gets `device` cpu and no visible GPU, so it holds no idle CUDA
  context (bardsai-v3 and scrubadub held 384 MB each in the smoke run while computing on CPU).
- A throughput batch size that raises (CUDA OOM under the cap at batch 8 for the DeBERTa GLiNER entries in the
  smoke run) is recorded on its line and in `common_batch_errors`; latency, cold start and VRAM stay. A crashed
  adapter's last stderr line goes into the speed error; its full stderr stays in the cache (`speed-err.log`).

## 17. Model-dataset contamination matrix (rule 2)

T = trained on that dataset's train split, E = the vendor reports evaluating on it (tuning
exposure possible), D = SCT was developed or tuned against it in earlier benchmark rounds,
U = training data undisclosed, `-` = none known; `X-family` = X on another dataset of the same
generator family. None of these prove overlap with our test records. Sources: model cards at
the pinned revisions, checked 2026-11-02 (families checked 2026-11-03).

**Generator families (`entries.FAMILIES`, applied to every entry, SCT included).** Gretel:
gretel finance and gretel_en (Gretel Navigator) and nemotron (NeMo Data Designer, Navigator's
successor after NVIDIA took over Gretel; the gretel_en and Nemotron-PII cards both carry the
`datadesigner` tag, and Nemotron-PII cites the Gretel PII set). ai4privacy: openpii and
argilla_pii (a slice of the ai4privacy-200k train split). thai_pii and bayanat have their own
rule-based generators; TAB, REDACT and the NER sets are not generated. `entries.contamination`
derives the family flags from the direct ones, so no family flag is set by hand.

| entry | openpii | gretel fin | gretel_en | nemotron | argilla_pii | redact | tab | others (iahlt, bayanat, thai*, naamapadam, multiconer, stockmark) |
|---|---|---|---|---|---|---|---|---|
| sct-default (regex + CoNLL/WikiNEuRal NER) | D | D | D-family | D | D-family | D | D | D |
| sct-gliner (knowledgator gliner-pii-base) | U, D | U, D | U, D-family | U, D | U, D-family | U, D | U, D | U, D |
| presidio (en_core_web_lg, OntoNotes 5) | - | - | - | - | - | - | - | - |
| gliner-nvidia | E (AI4Privacy, set unnamed) | T-family, E-family | E, T-family | T (Nemotron-PII train), E-family | E | - | - | - |
| gliner2-fastino (own 4,910 synthetic docs) | - | - | - | - | - | - | - | - |
| opf | U | U | U | U | U | U | U | U |
| bardsai-v3 | T (openpii-1m train) | T-family | T (train split) | T (train split) | T-family (ai4privacy generator) | - | - | - |
| datafog (regex) | - | - | - | - | - | - | - | - |
| reserve: urchade gliner_multi_pii-v1 | - | - | - | - | - | - | - | - |
| gliner-knowledgator | U | U | U | U | U | U | U | U |
| piiranha (pii-masking-400k) | T-family | - | - | - | T-family | - | - | - |
| llm-guard (Isotonic deberta, pii-masking-200k) | T-family | - | - | - | T (Argilla's source split) | - | - | - |
| scrubadub (rules + en_core_web_trf) | - | - | - | - | - | - | - | - |
| ai4privacy (open-pii-masking-500k train + test) | T-family, E-family | - | - | - | T-family, E-family | - | - | - |
| openmed (OPF base U + ai4privacy, Nemotron, Gretel, privy) | T | U, T-family | T | T | T | U | U | U |
| pii-tracer (in-house chats + ai4privacy; evaluates ai4privacy, Nemotron, Gretel, TAB) | T-family, E-family | E-family | E | E | T-family, E-family | - | E | - |
| presidio-full | - | - | - | - | - | - | - | - |
| sct-training | as sct-default | | | | | | | |
| sct-gliner-nvidia | E, D | D, T-family, E-family | E, T-family, D-family | T, D, E-family | E, D-family | D | D | D |

**Round-2 datasets** (round2-datasets `SHORTLIST.md` section 2 after its U-cell pass, 2026-11-03; evidence per
cell in `contamination-u-cells.md`). No entry has T or vendor E on a round-2 set, and SCT was not developed
against any of them. C = clean; E3 = the dataset author or a third party published scores for the tool (not
vendor tuning) and c* = clean by inference: both count as clean (`-` in `entries.CONTAMINATION`). An entry built
from two shortlist rows takes the worse cell. One U remains: openai/privacy-filter x KDPII (training data
undisclosed, KDPII public since 2024). openmed is fine-tuned from that OPF base, so under "any model in it" its
KDPII cell is U too (SHORTLIST's OpenMed row says C because it rates only the fine-tuning data; this is the one
deliberate difference). presidio-full takes the Presidio row (same recognizers, more spaCy packs).

| entry (SHORTLIST rows) | newmean | dialogpii | kdpii | nus_lt | redmadrobot | redactionbench | tonic | meddeid | greek_deid | neonredact | pl_pii | europii | carmen |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sct-default (SCT development (round 1); SCT ONNX (wikineural, xlm-r CoNLL)) | C | C | C | C | C | C | C | C | C | C | C | C | C |
| sct-gliner (SCT development (round 1); knowledgator gliner-pii-base-v1.0) | C | C | C | C | C | E3 | C | c* | C | C | C | C | c* |
| presidio (Presidio (en_core_web_lg)) | C | C | c* | C | C | C | E3 | C | C | C | E3 | E3 | C |
| gliner-nvidia (nvidia/gliner-PII) | C | C | C | C | C | E3 | C | C | C | C | C | C | C |
| gliner2-fastino (fastino gliner2-privacy-filter) | E3 | C | C | C | C | C | E3 | c* | C | C | C | C | C |
| opf (openai/privacy-filter) | E3 | C | U | C | C | E3 | C | E3 | C | E3 | C | C | C |
| bardsai-v3 (bardsai eu-pii v3) | C | C | C | C | C | C | C | c* | C | C | E3 | C | C |
| datafog (DataFog (regex engine)) | C | c* | c* | C | C | C | C | c* | c* | C | C | c* | C |
| gliner-knowledgator (knowledgator gliner-pii-base-v1.0) | C | C | C | C | C | E3 | C | c* | C | C | C | C | c* |
| piiranha (iiiorg/piiranha-v1) | C | C | C | c* | C | E3 | C | C | c* | C | c* | c* | c* |
| llm-guard (LLM Guard (default DeBERTa recognizer)) | C | c* | c* | C | C | C | C | c* | c* | C | C | c* | C |
| scrubadub (scrubadub) | C | c* | c* | C | C | C | C | c* | c* | C | C | c* | C |
| ai4privacy (ai4privacy multilingual model) | C | C | C | C | C | E3 | C | c* | C | C | C | c* | C |
| openmed (OpenMed multilingual v2; openai/privacy-filter) | C | C | U | C | C | C | C | E3 | C | E3 | C | c* | C |
| pii-tracer (perplexity-ai/PII-Tracer) | C | C | C | C | C | C | C | C | C | C | C | C | C |
| presidio-full (Presidio (en_core_web_lg)) | C | C | c* | C | C | C | E3 | C | C | C | E3 | E3 | C |
| sct-training (SCT development (round 1); SCT ONNX (wikineural, xlm-r CoNLL)) | C | C | C | C | C | C | C | C | C | C | C | C | C |
| sct-gliner-nvidia (SCT development (round 1); nvidia/gliner-PII) | C | C | C | C | C | E3 | C | C | C | C | C | C | C |

Training sets that are not loaded here (ai4privacy pii-masking-200k / 400k / open-pii-masking-500k / openpii-1m,
PII-Tracer's unnamed ai4privacy sample) are members of the ai4privacy family in `entries.FAMILIES`, so the family
rule flags openpii and argilla_pii.

Notes: nvidia/gliner-PII names its base as urchade/gliner_large-v2.1 on the HF card and
knowledgator/gliner-bi-large-v1.0 on the NIM page (conflict, recorded). bardsai removed rows
that appear in the PIIMB test sets, not ours. Argilla is a slice of the ai4privacy-200k
TRAIN split, so any model trained on that split may have seen the exact records. SCT's D
marks are the datasets used in SCT's earlier benchmark rounds. `gretel_en` and `argilla_pii`
were added after SCT 0.8.1, so SCT's code was not developed on them, but they are not held
out: the tuned track (section 3) chose one configuration per entry, SCT included, on the dev
parts of all nine privacy sets, gretel_en and argilla_pii among them; and under the family rule
they are D-family for SCT (gretel finance and openpii are D).

**Contamination rule (decision 21, permanent, every round, SCT included).** If a tool, or any
model inside it, was trained on a dataset or its family (T), or was developed or tuned against it
(E: the vendor evaluates on it; D: SCT developed against it), that cell is **excluded** from that
tool's comparison. `entries.status` gives `excluded` (any T, E or D), `unknown` (U only: training
data undisclosed) or `clean` (`-`). Reported:

1. Headline: each tool on its own clean cells (claimed languages, section 12), with a 95%
   bootstrap CI (`summary["own"]`).
2. Same-set: the datasets clean for every tool compared (`summary["same_set"]`). When there is
   none, the report says so and gives pairwise same-set tables: SCT vs each tool on the datasets
   clean for both, languages both claim, paired bootstrap (`summary["pairwise"][...]["clean"]`).
3. Unknown cells (U: opf and the knowledgator model in sct-gliner in round 1; opf and openmed on KDPII in
   round 2) are shown separately, labelled
   "unknown", never as clean (`own[...]["unknown"]`, `pairwise[...]["unknown"]`).
4. Excluded cells may be shown for information only, marked: in parentheses in the per-dataset
   tables, bold in the matrix; they never enter an average that is compared.

SCT is held to the same rule: its D and D-family cells are excluded, so in round 1 SCT (both
entries) has no clean cell, and round 1 cannot rank SCT; opf has none either (U everywhere).
Round 2 needs fresh datasets that no entry (SCT included) and no generator family of them has seen.

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
| 2 | contamination matrix | 17 | done from cards; decision 21 rule (T/E/D excluded, U apart) |
| 3 | PII vs NER | 1 | NER sets in a separate table |
| 4 | three coverage views | 4 | defined (`supports` in `entries.py`) |
| 5 | over-masking | 8 | defined; downstream utility only as chrF (no downstream task model) |
| 6 | threshold curves | 10 | defined |
| 7 | offsets and matching | 6 | implemented (`scorer.py`, offset fixtures in `runner.py`) |
| 8 | leak definitions | 7 | implemented (`scorer.doc_stats`) |
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

## 21. Adapter contract and runner

Each entry runs in its own venv (`uv venv --python 3.12` + `envs/<entry>.txt`; the full
`uv pip freeze` goes into the bundle) as a subprocess:

    <venv>/bin/python adapters/<tool>_adapter.py --input in.jsonl --output out.jsonl \
        [--config JSON] [--batch-size N] [--speed]

- input lines: `{"id", "text", "lang"}` (gold language code);
- output lines, one per input doc, in input order, flushed per batch:
  `{"id", "spans": [{"start", "end", "label", "score"}], "masked_text"?, "unplaced"?, "error"?, "seconds"}`;
  code-point offsets, half-open, tool's own labels, `score` float or null; `unplaced` = detections
  the adapter could not place in the source (scored as wrong predictions);
- an exception becomes `error` for that doc (a failing batch is retried doc by doc); spans out
  of range make the doc an error; `<output>.ready` is written once the model has loaded.

The runner checks offsets on seven hand-made fixtures first (an entry with a misplaced e-mail span
is not run; a missed e-mail, a span that covers it plus extra text, or a span that starts on it and stops
at its own `.` or `@` (the model tagged only the first subwords; a shifted or inclusive-end span cannot land
there) is a detection error, recorded in `offset-check.json` but not a skip),
then supervises: per-doc timeout counted from the last line or the ready marker, one retry of
the doc that was running, then an `error` line (`timeout`, `crash: exit N`); three docs in a row
failing with no output at all abort the rest (`aborted`). Raw outputs (they hold dataset text)
stay in `benchmarks/.cache/compare/`. The bundle gets, per entry x dataset part: completion,
failure kinds, the three views (each with `compare` per-language/macro/micro, `effective`
with failed docs unmasked, `with_date`, and the `score.py` report), the threshold curve, native
leaks, contamination flag, sample-id hash; `summary.json` has the group aggregates (all datasets, information
only), the per-dataset paired bootstrap of the baseline (SCT) minus each entry, and the
section 17 rule views (`own`, `same_set`, `pairwise`). Not yet built: the common chunker (section 11) and the PR/leak plots.

## Change log

- 2026-11-02: protocol written; splits computed (seed 13, dev share 0.2).
- 2026-11-02: harness built (adapter contract, runner, scorer, speed runner); no change to
  any frozen rule. Doc leak rate uses "a letter or digit of a mention survives" as the exposure rule.
- 2026-11-02: test runs (frozen configs from ec0fae9). Gold-language track only, both config
  tracks (default, tuned), all 8 entries, all 14 datasets, 300 docs per language. Every entry on
  CPU (same device as tuning; torch wheels see no GPU on the box, section 3), 4 jobs at a time on
  16 cores each within 0-63, 16 threads. Entries whose tuned config equals the default (presidio,
  bardsai-v3, datafog) reuse the default raw outputs through the cache key. One rescoring pass
  over all entries per track gives `summary.json` and the paired bootstrap. Speed: one entry at a
  time on cores 0-15 (16 threads), seeded random order, 3 repeats, FineWeb-2 speed sample cut to
  10 docs per language (100 docs; 100 per language would take hours for the CPU-bound models),
  then a single-thread run on core 0 with 3 docs per language (30 docs), one repeat (three would
  take about 4 hours for gliner-nvidia and opf alone). Speed uses each entry's default
  config (native language routing). The native language track, the common chunker, the matched
  operating points (masked recall 0.90/0.95) and the blind annotation audit were not run.
  `report.py` writes the summary tables from a bundle.
  Run notes: another job of the same user (outside this comparison) held cores 60-63 during the
  quality runs, which slowed the sct-gliner default run; its remaining datasets were rerun on
  cores 0-15 (finished datasets reused from the cache; quality does not depend on core
  placement). The sct-gliner tuned track and the speed runs used their own free 16-core slots.
  The paired bootstrap is on the shared view (EMAIL, PHONE, NUMBER: DATE out), as the runner
  computes it; predictions (spans + ids, scores-and-ids datasets only, 75 MB per track) stay
  on the server with their sha256 in the bundle manifest.
  Peak RSS: the adapter's `getrusage` peak never reads below the runner's own size on Linux
  (ru_maxrss survives fork + exec), so the first 16-thread speed run shows ~2 GB for datafog,
  presidio and bardsai-v3. `speed.py` now reads `/proc/<pid>/status` VmHWM while the adapter runs
  (`peak_rss_mb`; the getrusage value is kept as `peak_rss_mb_getrusage`). Those three entries
  were rerun at 16 threads with the fix (`speed-16t-rss`); entries above 2 GB were unaffected.
- 2026-11-02 (report): headline CIs at group level. Section 14 asks for CIs on the headline (group
  macro), but the runner only bootstrapped per dataset. `scorer.macro_bootstrap` resamples groups
  within each dataset and recomputes the macro over datasets; the runner stores it as
  `summary["paired_macro"][entry][view][group]` for the shared and full views, over the datasets that
  are in-domain (T or E, `entries.in_domain`) for neither side. D and U cells stay in (flagged).
  `masked_precision` joined `scorer.HEADLINE`. Both tracks were rescored from the same raw outputs
  (no inference): aggregates, per-dataset bootstraps and per-entry files are byte-for-byte the same
  scores. Report: `COMPARISON.md` in the bundle.
- 2026-11-03 (rules): decision 21 and the claimed-language rule applied to every table, as
  permanent rules (sections 12, 13, 17). `entries.status` replaces `entries.in_domain` (T/E only,
  D and U stayed in): T, E and D cells are now excluded, U-only cells are "unknown".
  `entries.CLAIMED_LANGUAGES` / `claims`. `runner.rule_views` writes `summary["own"]`,
  `["same_set"]` and `["pairwise"]` (replacing `paired_macro` and the ":out-of-domain" aggregate);
  `report.py` writes the own-clean headline, same-set and pairwise tables, per-language tables with
  out-of-claim cells marked, and marks excluded / unknown cells everywhere else; the pooled
  all-datasets quality table is gone.
  Scoring fix (`scorer.recover_dropped`, SCT entries, `recover_dropped` in `entries.py`): SCT spans
  come from placeholders found in its output, so a source copy whose placeholder a later cleaning
  step removed (the link text of `[a@b.se](mailto:a@b.se)`, repeated names merged into one
  placeholder) was scored as leaked though no character of it is visible. A copy of a value SCT
  masked elsewhere in the doc now counts as masked when neither the value nor any 4+ char word (`LEAK_MIN_CHARS`; this entry first said 3+, corrected below) of
  it is in SCT's output and the copy is a whole token. `runner.py --no-recover` gives the round-1
  mapping; both are reported. Both tracks rescored from the ec0fae9 raw outputs (no inference).
- 2026-11-03 (families, recovery audit): independent check of the rules pass. (1) Generator
  families (`entries.FAMILIES`): a T, E or D flag reaches every dataset of its family, for every
  entry alike; gretel_en and argilla_pii become D-family for SCT, gretel finance T-family for
  bardsai-v3 (and T-family, E-family for gliner-nvidia), nemotron E-family for gliner-nvidia.
  SCT and sct-gliner have no clean cell left. (2) The "held out, dev included" sentence on
  gretel_en / argilla_pii was wrong (the tuned track used every privacy dev part); default is now the
  primary track where a clean cell exists. (3) Per-dataset tables on claimed languages only.
  (4) Recovery rule (`scorer.dropped_copies`): the earlier text said "3+ char word"; the code used
  `leak`'s `LEAK_MIN_CHARS` = 4. Now: a value qualifies only when `leak` finds neither it nor any 4+
  char word of it, and `scorer.traces` finds nothing: no 2+ char word of it (case-, accent- and
  NFKD-folded) is a whole word of the output, and its separator-free folded form is not inside the
  separator-free folded output. The first audit (`audit_recover.py`, seeded 50 copies per set, rule
  without the folding) found such traces (separator-free form inside a longer token) in a few
  sampled copies per set, so the folded checks joined the rule. (5) Claimed languages: sct-gliner en
  from the card's language tag; sct-default every language. Both tracks rescored from the ec0fae9 raw
  outputs (no inference).
  (6) `runner.py`: the shared view is the types every comparison entry supports, whichever entries
  a run scores; the `--no-recover` run (SCT alone) had used SCT's own 9 types as its shared view, so
  the shared-view "without" numbers were not comparable; that run was redone. (7) Audit counts in the
  bundle's `audit/`; `make_comparison.py` in the bundle writes `COMPARISON.md` (prose numbers read
  from the bundle).
- 2026-11-03 (round 2 datasets): loaders for the 12 open round-2 sets and a CARMEN-I stub (section 1, "Round 2
  datasets"), pinned to the shortlist revisions. `split.py` keeps a dataset's own split (`record["part"]`:
  KDPII, NeonRedact, EuroPII; MedDeID all test). `labels.UNTYPED` for RedactionBench (typed measures None),
  `scorer.survival` for the Greek must-survive probes, three `r2-` groups, round-2 contamination flags in
  `entries.py`. No rule for round-1 data changed; round-1 scores are untouched.
- 2026-11-03 (round 2 entries): ten entries added (section 2 "Round 2 entries"), label maps (section 5),
  contamination rows (section 17), CUDA 12 venvs (torch cu126 direct wheels; onnxruntime-gpu, with
  `runner.GPU_TWINS` removing SCT's CPU onnxruntime after the install), per-entry GPU speed column (section 16),
  per-round shared view (`runner.shared_view`). `PRESIDIO_MODELS` / `PHONE_REGIONS` moved from `tune.py` to
  `entries.py` (unchanged values; `tune` re-exports them). `opf_adapter` split into `load` + `make_predict`
  (same decoding) so openmed reuses it; `gliner_nvidia_adapter` takes a `model` key and caps GPU memory. No
  frozen round-1 rule or score changed. Not yet run: no GPU, no network for venvs in the authoring session.
- 2026-11-03 (round 2 prep review): round-2 contamination cells match SHORTLIST section 2 after its U-cell
  pass (one U: opf x KDPII, carried into openmed, which is built on OPF; table in section 17). Every round-2 set
  is `scores-only` (no per-example predictions); NUS-LT and CARMEN-I set `DELETE_PREDICTIONS`, so the runner
  removes their raw outputs after scoring. Round-2 loads stop on a gold label that is neither mapped nor in
  `labels.DATASET_OUT_OF_SCOPE`, and on a span whose source surface form differs from `text[start:end]`; the
  MedDeID, Greek and NeonRedact files are checked against SHORTLIST's sha256 pins. A run that mixes round-1 and
  round-2 datasets is refused. scrubadub is a CPU entry (spaCy never moves to GPU without `prefer_gpu`);
  sct-gliner-nvidia sets `ner_backend='gliner'` (PyTorch, no forced ONNX); llm-guard and SCT's GLiNER path
  cap the GPU share (`cap_gpu`). Must-survive probes count occurrences.
- 2026-11-03: round 2 compares SCT 0.8.2 (all SCT entries). The round-1 torch entries (gliner-nvidia, gliner2-fastino, opf,
  bardsai-v3) move to torch 2.14.1+cu126 so they run on the CUDA 12.6 GPU; round 1 ran them with torch 2.15.0 (CUDA 13,
  CPU). bardsai-v3's ONNX model still runs on the CPU onnxruntime.
- 2026-11-03 (round 2 envs): every env resolves for Linux x86_64 / Python 3.12 with CUDA 12 nvidia-* wheels only.
  sct-gliner also moves to torch 2.14.1+cu126 (it resolved PyPI's CUDA 13 torch 2.15.0). sct-training takes
  onnxruntime-gpu 1.29.0 from ORT's CUDA 12 feed: PyPI's 1.27+ builds (1.31.1 included) are CUDA 13.
  `runner.gpu_twin_fixes` also matches `name @ url` pins.
