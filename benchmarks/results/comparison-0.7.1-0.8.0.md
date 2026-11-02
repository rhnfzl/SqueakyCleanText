# SCT 0.7.1 vs 0.8.0 benchmark

Run on 2026-11-01 with the harness at commit `2b7fa876` (benchmarks/ and scripts/ only, copied with
`git archive`). Both versions ran the same datasets, seeds (13) and samples (default sizes: PII 300 docs
per language, LID 100, speed and clean 100 per FineWeb-2 language, TAB all 127). No sample was reduced.
Per-dataset files are in `0.7.1/` and `0.8.0/`; each holds its command, dataset revision, package
versions and hardware. REDACT files hold aggregate scores only.

## Setup

- Hardware: AMD Threadripper PRO 7985WX (64 cores), 754 GB RAM, Linux 6.11, Python 3.12.3.
- Installs from PyPI: `squeakycleantext[fuzzy]==0.8.0` and `==0.7.1`, plus `pyarrow`.
- Quality suites (pii, long, lid, clean): both versions at the same time, 0.8.0 on cores 0-31 and
  0.7.1 on cores 32-63, ONNX threads 16, workers 2.
- Speed suite: one version at a time on cores 0-63, ONNX threads 32, workers 2. A short sweep picked
  32/2 (3.1 docs/s) over 64/1 (2.6), 32/4 (3.0) and 16/4 (2.0).
- The `docs/s` column in the PII tables of each `summary.md` comes from the 32-core quality run, not
  from the speed run.

### Differences between the two runs (read before comparing)

- **onnxruntime**: 0.8.0 ran on 1.31.1. 0.7.1 ran on 1.23.2. 0.7.1 cannot load the large
  xlm-roberta models (English, German, Dutch, Spanish) on onnxruntime 1.24 or newer: the HF cache
  symlinks `model.onnx_data`, and newer onnxruntime refuses external data outside the model folder.
  0.7.1 requires onnxruntime >= 1.24.2, so 0.7.1 as published cannot load those models on a fresh
  install. 0.8.0 fixes this.
  Speed numbers mix the SCT change with the onnxruntime change.
- **0.7.1 shim**: the harness uses 0.8.0 APIs. 0.7.1 ran through a small wrapper (kept on the
  benchmark host, not in the repo) that: adds `sct.presets` with the 0.8.0 `training` preset (minus
  `check_decode_unicode_escapes`, which 0.7.1 does not have); adds `LEGACY_TYPE_NAMES`; drops the config
  fields 0.7.1 lacks and applies `onnx_intra_op_threads` through `SessionOptions`, so thread counts match;
  defaults `ProcessResult.metadata` to `{}`; and wraps 0.7.1's `_detect_language` (it returns a name, not
  a tuple) for the LID suite. The `command` field in the 0.7.1 files says `python -m benchmarks.run`;
  it ran the same arguments through that wrapper.
- **GPU**: skipped. The shared vLLM server holds 16.5-19 GB on each RTX 4090, leaving at most 8 GB
  (GPU 3). A capped 0.8.0 run (6 GB arena cap, 1 worker) ran out of memory loading the English model,
  and peak use reached 24.1 of 24.6 GB, so it was stopped. The installed onnxruntime-gpu 1.31.1 wheel
  also needs CUDA 13; the host driver (560) supports CUDA 12.6. 0.7.1 was never tried on GPU: it
  has no GPU memory cap.
- Defaults changed between versions. 0.8.0 turns on `check_replace_dates` and
  `check_decode_unicode_escapes`, and turns off `check_to_ascii_unicode`. Scoring sets
  `check_to_ascii_unicode=False` for both, so PII scores are not affected by the ascii change. The
  clean and speed suites use the plain defaults, so they are.

## Headline

- **PII recall and precision are up on every dataset** with NER on. Partial recall, default config:
  openpii 62.5 -> 69.2, REDACT 63.2 -> 72.1, MultiCoNER 49.0 -> 73.1, Stockmark (ja) 62.7 -> 78.6,
  ThaiNER 55.0 -> 68.6, IAHLT (ar) 71.9 -> 84.5. Strict F1 rose most on openpii (34.6 -> 55.6),
  TAB (22.1 -> 49.5) and Nemotron (41.8 -> 59.7).
- **TAB long docs**: masking recall (all mentions) 31.1 -> 64.1, quasi identifiers 30.1 -> 65.9.
- **Language ID**: 0.7.1 only returns English, German, Dutch and Spanish (5.5% accuracy on WiLI's 72
  in-set languages). 0.8.0 returns 72 languages at 95.1% accuracy.
- **Speed, NER on**: default config 1.19 -> 2.65 docs/s (2.2x), training 1.68 -> 2.77 docs/s (1.65x).
- **To investigate (score drops, do not block)**:
  1. Leak rate rose on most PII datasets while recall rose. openpii default 10.3 -> 13.1 (PERSON
     7.5 -> 10.6, LOCATION 14.2 -> 16.8, NUMBER 16.2 -> 20.9). REDACT default 16.0 -> 17.9. Recall
     counts same-type overlap, leak counts any value text left in the output. Likely 0.8.0 labels more
     spans correctly but removes less text overall (precision is much higher). Check span widths and
     what 0.7.1 removed under other types.
  2. Speed with NER off halved: 211.8 -> 100.1 docs/s. Likely language detection over all Lingua
     languages instead of four. Check with a profile.
  3. LID gives a language to 96.6% of paragraphs in languages Lingua does not know (0.7.1: 66.5%,
     with only four candidates). A confidence floor would help users who route on the language.
  4. ner-off leak rose on MultiCoNER (89.9 -> 99.6) and ThaiNER (87.5 -> 96.8). Both have almost no
     regex-type PII, so the gap is text 0.7.1 changed or removed outside PII. Cause not checked. Low priority.
- **Cleaning**: 0.8.0 `default` keeps emoji (removal 100 -> 5.1). That follows from the new
  `check_to_ascii_unicode=False` default, not a regression. HTML, URL and mojibake removal is unchanged
  (99.5-100%).

## Failures

- None in the CPU runs: every suite and dataset finished for both versions.
- GPU: skipped (see above).

## PII and long docs

Partial = overlap + same type. Leak = share of gold PII values still in the output (lower is better).

| dataset | config | docs | partial R 0.7.1 | 0.8.0 | Δ | partial P 0.7.1 | 0.8.0 | Δ | strict F1 0.7.1 | 0.8.0 | Δ | leak 0.7.1 | 0.8.0 | Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bayanat | default | 845 | 64.6 | 68.2 | +3.5 | 44.5 | 48.4 | +3.9 | 42.0 | 47.1 | +5.1 | 7.7 | 7.5 | -0.2 |
| bayanat | training | 845 | 57.9 | 61.5 | +3.5 | 52.4 | 54.1 | +1.7 | 51.6 | 54.0 | +2.4 | 8.9 | 7.6 | -1.3 |
| bayanat | ner-off | 845 | 43.3 | 41.8 | -1.5 | 35.5 | 37.2 | +1.7 | 28.8 | 30.0 | +1.3 | 34.8 | 34.1 | -0.7 |
| gretel | default | 2100 | 52.4 | 57.0 | +4.6 | 19.1 | 23.1 | +4.0 | 12.9 | 19.7 | +6.7 | 35.3 | 37.8 | +2.4 |
| gretel | training | 2100 | 42.9 | 48.4 | +5.5 | 42.8 | 49.7 | +6.9 | 27.3 | 31.1 | +3.8 | 40.4 | 42.6 | +2.2 |
| gretel | ner-off | 2100 | 19.8 | 20.5 | +0.7 | 8.8 | 10.2 | +1.4 | 4.2 | 9.8 | +5.6 | 84.8 | 84.7 | -0.0 |
| iahlt | default | 300 | 71.9 | 84.5 | +12.6 | 77.0 | 88.8 | +11.9 | 63.8 | 67.0 | +3.1 | 19.5 | 20.4 | +0.8 |
| iahlt | training | 300 | 76.6 | 85.6 | +9.0 | 79.4 | 89.2 | +9.8 | 66.8 | 68.1 | +1.3 | 18.4 | 19.1 | +0.7 |
| iahlt | ner-off | 300 | 0.0 | 0.0 | +0.0 | 0.0 | 0.0 | +0.0 | 0.0 | 0.0 | +0.0 | 99.9 | 99.9 | +0.0 |
| multiconer | default | 900 | 49.0 | 73.1 | +24.1 | 53.1 | 76.9 | +23.8 | 40.4 | 55.4 | +15.0 | 16.7 | 19.0 | +2.3 |
| multiconer | training | 900 | 59.8 | 74.3 | +14.5 | 61.5 | 75.9 | +14.4 | 49.2 | 56.9 | +7.7 | 16.6 | 17.9 | +1.3 |
| multiconer | ner-off | 900 | 0.0 | 0.0 | +0.0 | 0.0 | 0.0 | +0.0 | 0.0 | 0.0 | +0.0 | 89.9 | 99.6 | +9.7 |
| naamapadam | default | 900 | 56.8 | 63.9 | +7.2 | 71.7 | 81.6 | +9.9 | 48.2 | 55.9 | +7.7 | 15.1 | 16.2 | +1.1 |
| naamapadam | training | 900 | 70.9 | 79.6 | +8.7 | 79.7 | 90.1 | +10.4 | 59.1 | 67.7 | +8.6 | 13.4 | 15.0 | +1.6 |
| naamapadam | ner-off | 900 | 0.0 | 0.0 | +0.0 | 0.0 | 0.0 | +0.0 | 0.0 | 0.0 | +0.0 | 96.7 | 96.7 | +0.0 |
| nemotron | default | 300 | 84.1 | 85.5 | +1.4 | 48.2 | 58.6 | +10.4 | 41.8 | 59.7 | +17.9 | 15.0 | 15.1 | +0.2 |
| nemotron | training | 300 | 66.5 | 67.7 | +1.2 | 82.3 | 83.4 | +1.1 | 67.3 | 67.9 | +0.7 | 19.1 | 18.9 | -0.2 |
| nemotron | ner-off | 300 | 44.1 | 44.8 | +0.7 | 34.3 | 44.6 | +10.3 | 18.9 | 37.2 | +18.3 | 64.7 | 64.4 | -0.2 |
| openpii | default | 9000 | 62.5 | 69.2 | +6.7 | 51.3 | 70.2 | +18.9 | 34.6 | 55.6 | +21.0 | 10.3 | 13.1 | +2.8 |
| openpii | training | 9000 | 40.9 | 48.9 | +8.0 | 63.3 | 73.9 | +10.6 | 43.2 | 48.2 | +5.0 | 18.3 | 21.0 | +2.7 |
| openpii | ner-off | 9000 | 40.1 | 40.0 | -0.0 | 44.0 | 59.1 | +15.1 | 20.4 | 39.3 | +18.9 | 49.9 | 50.0 | +0.0 |
| redact | default | 1000 | 63.2 | 72.1 | +8.9 | 29.9 | 41.1 | +11.2 | 23.5 | 36.2 | +12.7 | 16.0 | 17.9 | +1.9 |
| redact | training | 1000 | 63.6 | 72.5 | +8.9 | 59.3 | 69.0 | +9.7 | 48.1 | 54.0 | +5.9 | 20.8 | 21.6 | +0.7 |
| redact | ner-off | 1000 | 27.3 | 27.8 | +0.5 | 18.1 | 23.8 | +5.7 | 10.3 | 19.5 | +9.1 | 67.2 | 67.1 | -0.1 |
| stockmark | default | 300 | 62.7 | 78.6 | +16.0 | 72.0 | 86.7 | +14.7 | 59.9 | 62.3 | +2.4 | 12.5 | 13.1 | +0.6 |
| stockmark | training | 300 | 63.8 | 79.6 | +15.8 | 70.8 | 85.6 | +14.8 | 61.0 | 63.7 | +2.7 | 12.2 | 12.8 | +0.6 |
| stockmark | ner-off | 300 | 0.0 | 0.0 | +0.0 | 0.0 | 0.0 | +0.0 | 0.0 | 0.0 | +0.0 | 98.9 | 98.9 | +0.0 |
| tab | default | 127 | 77.5 | 81.4 | +3.9 | 54.2 | 67.3 | +13.1 | 22.1 | 49.5 | +27.4 | 42.3 | 42.4 | +0.1 |
| tab | training | 127 | 42.0 | 43.1 | +1.1 | 76.2 | 79.4 | +3.3 | 33.8 | 34.6 | +0.9 | 49.0 | 49.4 | +0.5 |
| tab | ner-off | 127 | 38.5 | 40.4 | +1.9 | 42.9 | 58.0 | +15.1 | 4.3 | 35.9 | +31.6 | 91.3 | 91.5 | +0.2 |
| thai_pii | default | 300 | 80.1 | 87.3 | +7.2 | 56.8 | 62.6 | +5.8 | 24.3 | 26.5 | +2.2 | 15.4 | 17.5 | +2.1 |
| thai_pii | training | 300 | 71.1 | 77.2 | +6.1 | 74.5 | 82.2 | +7.7 | 37.9 | 41.7 | +3.9 | 32.2 | 32.8 | +0.5 |
| thai_pii | ner-off | 300 | 36.0 | 36.0 | +0.0 | 46.7 | 46.9 | +0.3 | 18.2 | 17.8 | -0.3 | 59.3 | 59.5 | +0.2 |
| thainer | default | 300 | 55.0 | 68.6 | +13.6 | 53.3 | 65.6 | +12.4 | 24.9 | 34.1 | +9.1 | 23.4 | 24.6 | +1.2 |
| thainer | training | 300 | 61.7 | 69.3 | +7.6 | 82.2 | 91.8 | +9.6 | 35.4 | 40.3 | +4.9 | 23.3 | 24.2 | +0.9 |
| thainer | ner-off | 300 | 1.1 | 1.1 | +0.0 | 3.5 | 3.5 | +0.1 | 0.4 | 1.0 | +0.6 | 87.5 | 96.8 | +9.4 |

TAB masking recall (any type, macro over annotators), default config:

| | 0.7.1 | 0.8.0 |
|---|---|---|
| all | 31.1 | 64.1 |
| direct | 41.5 | 43.3 |
| quasi | 30.1 | 65.9 |

## Language ID (WiLI-2018, 100 paragraphs per language)

| | 0.7.1 | 0.8.0 |
|---|---|---|
| languages the detector returns | 4 | 72 |
| in-set docs | 7200 | 7200 |
| accuracy | 5.5 | 95.1 |
| macro accuracy | 5.5 | 95.1 |
| out-of-set docs given a language | 66.5 | 96.6 |

In-set means languages the installed Lingua knows, so the in-set count is the same for both.

## Speed, CPU (FineWeb-2, 100 docs per language, cores 0-63)

| config | docs/s 0.7.1 | 0.8.0 | ratio | chars/s 0.7.1 | 0.8.0 |
|---|---|---|---|---|---|
| default | 1.185 | 2.646 | 2.23x | 2703.0 | 6034.5 |
| training | 1.675 | 2.772 | 1.65x | 3820.1 | 6322.3 |
| ner-off | 211.805 | 100.088 | 0.47x | 483020.0 | 228250.4 |

Per script, default config (docs/s):

| script | 0.7.1 | 0.8.0 |
|---|---|---|
| arb_Arab (ARABIC) | 1.072 | 2.84 |
| cmn_Hani (CJK) | 0.895 | 2.702 |
| ell_Grek (GREEK) | 1.546 | 2.548 |
| heb_Hebr (HEBREW) | 0.942 | 2.589 |
| heb_Hebr (LATIN) | 0.437 | 0.697 |
| hin_Deva (DEVANAGARI) | 1.559 | 3.575 |
| hin_Deva (LATIN) | 1.722 | 2.341 |
| jpn_Jpan (CJK) | 1.972 | 3.936 |
| jpn_Jpan (HIRAGANA) | 1.224 | 2.652 |
| jpn_Jpan (KATAKANA) | 1.379 | 2.479 |
| jpn_Jpan (LATIN) | 0.924 | 1.472 |
| kor_Hang (HANGUL) | 1.345 | 3.128 |
| kor_Hang (LATIN) | 6.384 | 5.333 |
| nld_Latn (LATIN) | 1.083 | 1.822 |
| rus_Cyrl (CYRILLIC) | 1.424 | 2.822 |
| rus_Cyrl (LATIN) | 2.085 | 2.447 |
| tha_Thai (THAI) | 1.118 | 2.511 |

## Cleaning: injected noise removed (FineWeb-2)

| config | kind | injected | 0.7.1 | 0.8.0 |
|---|---|---|---|---|
| default | emoji | 1000 | 100.0 | 5.1 |
| default | html | 1000 | 100.0 | 100.0 |
| default | mojibake | 1000 | 100.0 | 99.6 |
| default | url | 1000 | 100.0 | 100.0 |
| ner-off | emoji | 1000 | 100.0 | 0.4 |
| ner-off | html | 1000 | 100.0 | 100.0 |
| ner-off | mojibake | 1000 | 100.0 | 99.6 |
| ner-off | url | 1000 | 100.0 | 100.0 |
| training | emoji | 1000 | 5.0 | 4.9 |
| training | html | 1000 | 0.9 | 0.9 |
| training | mojibake | 1000 | 99.5 | 99.5 |
| training | url | 1000 | 100.0 | 100.0 |
