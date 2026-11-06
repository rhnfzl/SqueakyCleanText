# SCT 0.8.0 vs unreleased (fix/benchmark-findings) benchmark

Run on 2026-11-02 on the same CPU host as the 0.7.1 vs 0.8.0 run (Python
3.12.3, onnxruntime 1.31.1, same package pins as the 0.8.0 venv). Branch commit `2078e8c4` (fixes F1-F6,
F8, F9, F11, F12 and scorer change S1), installed from a `git archive` of that commit. Same seed (13) and
default sample sizes as the 0.8.0 run (PII 300 docs per language, TAB all 127, speed and clean 100 per
FineWeb-2 language). Configs: `default` and `training` (no `ner-off` this time).

- `unreleased/`: the branch. `unreleased/thr0.7/`: the branch with `ner_confidence_threshold=0.7`
  (default config; openpii, gretel and REDACT only). It ran through a small wrapper kept on the host that
  adds that field to every config; its `command` field says `python -m benchmarks.run`.
- `0.8.0-rescored/`: `squeakycleantext==0.8.0` from PyPI, re-run with the branch harness. S1 changed the
  leak scorer (combining marks join words) and added masked recall, so the old `0.8.0/` leak numbers
  are not comparable. Strict F1 of every rescored dataset matches `0.8.0/` exactly, so only the scorer
  changed. Cleaning was not re-run for 0.8.0 (its scorer did not change); the cleaning table uses `0.8.0/`.
- Quality runs: branch on cores 0-31, 0.8.0 on cores 32-63 at the same time; ONNX threads 16, workers 2.
  Speed: one version at a time on cores 0-63, ONNX threads 32, workers 2, back to back. The host is
  shared, so absolute docs/s are lower than in the earlier run; compare the two columns here only.
- REDACT files hold aggregate scores only. `ner_merge_location_gap` stays 0 (off), so F8 is not measured.

Percentages. Recall and precision are partial (overlap + same type). Masked R = overlap, any type.
Leak = gold PII value (or a 4+ char word of it) still in the output; lower is better. d = new minus 0.8.0
in points.

## Regressions

No dataset has leak up or partial recall down by more than 0.5 points, in either config.
Other drops worth knowing:

- Precision: TAB -4.6 (default) and -12.5 (training), REDACT training -2.4. Most likely from F2
  propagation (repeated values in long documents), which also gives TAB its 30-point leak drop.
- Strict F1: thainer -1.2 (default) and -3.2 (training), TAB -1.4 and -1.0. Partial recall and leak
  still improve on both.
- Threshold 0.7 on top of the branch: leak -1.6 to -2.5 more, recall +0.2 to +0.6, precision -0.1 to
  -0.4, strict F1 -0.2 to -0.6. Small gain; the default stays 0.85.
- Speed: no measurable change overall (default 2.62 -> 2.62 docs/s, training 2.78 -> 2.78). Per-script
  changes are within run noise (largest: jpn CJK -8% default).

## default

| dataset | leak 0.8.0 | leak new | d | recall 0.8.0 | recall new | d | precision 0.8.0 | precision new | d | masked R 0.8.0 | masked R new | strict F1 0.8.0 | strict F1 new | d |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| openpii | 13.1 | 10.6 | -2.6 | 69.2 | 74.4 | +5.2 | 70.2 | 73.6 | +3.4 | 88.2 | 90.3 | 55.6 | 60.0 | +4.4 |
| gretel | 37.8 | 28.3 | -9.4 | 57.0 | 61.9 | +4.9 | 23.1 | 24.1 | +1.0 | 67.8 | 72.1 | 19.7 | 20.7 | +1.0 |
| nemotron | 15.1 | 7.7 | -7.4 | 85.5 | 89.5 | +4.0 | 58.6 | 59.6 | +1.0 | 92.3 | 96.1 | 59.7 | 63.7 | +4.1 |
| redact | 18.1 | 8.2 | -9.9 | 72.1 | 85.6 | +13.5 | 41.1 | 41.1 | -0.0 | 78.1 | 92.2 | 36.2 | 39.2 | +3.0 |
| iahlt | 20.5 | 12.7 | -7.8 | 84.5 | 90.2 | +5.6 | 88.8 | 88.8 | -0.0 | 88.1 | 94.0 | 67.0 | 69.6 | +2.6 |
| bayanat | 7.5 | 2.3 | -5.3 | 68.2 | 73.7 | +5.5 | 48.4 | 54.9 | +6.5 | 97.8 | 99.0 | 47.1 | 56.7 | +9.5 |
| thainer | 20.5 | 15.1 | -5.4 | 68.6 | 73.3 | +4.6 | 65.6 | 69.0 | +3.4 | 81.0 | 84.6 | 34.1 | 32.8 | -1.2 |
| thai_pii | 18.9 | 13.7 | -5.2 | 87.3 | 89.4 | +2.1 | 62.6 | 63.4 | +0.8 | 94.0 | 95.8 | 26.5 | 30.9 | +4.3 |
| naamapadam | 22.2 | 17.5 | -4.7 | 63.9 | 67.8 | +3.9 | 81.6 | 81.3 | -0.3 | 67.5 | 71.9 | 55.9 | 64.1 | +8.1 |
| multiconer | 25.8 | 20.1 | -5.7 | 73.1 | 77.1 | +4.0 | 76.9 | 76.2 | -0.7 | 77.3 | 82.2 | 55.4 | 57.5 | +2.0 |
| stockmark | 13.1 | 6.5 | -6.6 | 78.6 | 83.9 | +5.2 | 86.7 | 86.9 | +0.2 | 84.0 | 89.2 | 62.3 | 64.9 | +2.6 |
| tab | 42.4 | 11.7 | -30.7 | 81.4 | 85.9 | +4.5 | 67.3 | 62.8 | -4.6 | 88.1 | 92.7 | 49.5 | 48.1 | -1.4 |

## training

| dataset | leak 0.8.0 | leak new | d | recall 0.8.0 | recall new | d | precision 0.8.0 | precision new | d | masked R 0.8.0 | masked R new | strict F1 0.8.0 | strict F1 new | d |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| openpii | 21.0 | 18.6 | -2.4 | 48.9 | 51.1 | +2.2 | 73.9 | 75.1 | +1.2 | 65.7 | 67.5 | 48.2 | 50.1 | +1.9 |
| gretel | 42.6 | 33.7 | -8.9 | 48.4 | 52.5 | +4.1 | 49.7 | 49.6 | -0.1 | 55.8 | 60.0 | 31.1 | 32.3 | +1.2 |
| nemotron | 18.9 | 11.7 | -7.2 | 67.7 | 71.7 | +4.0 | 83.4 | 84.4 | +1.0 | 75.3 | 79.1 | 67.9 | 73.5 | +5.6 |
| redact | 21.6 | 12.8 | -8.9 | 72.5 | 75.3 | +2.8 | 69.0 | 66.5 | -2.4 | 78.7 | 81.7 | 54.0 | 54.0 | +0.0 |
| iahlt | 19.2 | 12.2 | -6.9 | 85.6 | 91.4 | +5.8 | 89.2 | 89.2 | +0.1 | 89.0 | 95.0 | 68.1 | 70.4 | +2.3 |
| bayanat | 7.6 | 2.7 | -4.9 | 61.5 | 70.6 | +9.2 | 54.1 | 64.1 | +10.0 | 97.9 | 98.6 | 54.0 | 63.4 | +9.4 |
| thainer | 19.9 | 15.4 | -4.5 | 69.3 | 71.7 | +2.4 | 91.8 | 91.6 | -0.2 | 71.7 | 74.7 | 40.3 | 37.1 | -3.2 |
| thai_pii | 33.3 | 31.1 | -2.3 | 77.2 | 79.3 | +2.1 | 82.2 | 82.8 | +0.5 | 83.3 | 85.3 | 41.7 | 47.0 | +5.3 |
| naamapadam | 20.4 | 14.6 | -5.8 | 79.6 | 84.3 | +4.7 | 90.1 | 89.4 | -0.7 | 83.5 | 89.0 | 67.7 | 75.3 | +7.6 |
| multiconer | 24.4 | 18.9 | -5.5 | 74.3 | 78.7 | +4.4 | 75.9 | 75.3 | -0.6 | 78.8 | 83.5 | 56.9 | 58.8 | +1.9 |
| stockmark | 12.8 | 6.7 | -6.2 | 79.6 | 84.5 | +4.9 | 85.6 | 85.8 | +0.2 | 84.3 | 89.6 | 63.7 | 66.6 | +2.8 |
| tab | 49.4 | 19.0 | -30.5 | 43.1 | 46.6 | +3.5 | 79.4 | 66.9 | -12.5 | 47.9 | 51.6 | 34.6 | 33.7 | -1.0 |

## Threshold 0.7 (default config, branch)

| dataset | leak @0.85 | leak @0.7 | recall @0.85 | recall @0.7 | precision @0.85 | precision @0.7 | strict F1 @0.85 | strict F1 @0.7 |
|---|---|---|---|---|---|---|---|---|
| openpii | 10.6 | 9.0 | 74.4 | 75.0 | 73.6 | 73.5 | 60.0 | 59.8 |
| gretel | 28.3 | 25.8 | 61.9 | 62.1 | 24.1 | 23.9 | 20.7 | 20.4 |
| redact | 8.2 | 6.2 | 85.6 | 85.8 | 41.1 | 40.7 | 39.2 | 38.6 |

## TAB masking recall (any type, macro over annotators)

| config | 0.8.0 all | new all | 0.8.0 direct | new direct | 0.8.0 quasi | new quasi |
|---|---|---|---|---|---|---|
| default | 64.1 | 66.1 | 43.3 | 43.6 | 65.9 | 68.0 |
| training | 18.3 | 19.9 | 6.0 | 6.2 | 19.3 | 21.0 |

## Speed (FineWeb-2, cores 0-63, ONNX threads 32, workers 2)

| config | script | 0.8.0 docs/s | new docs/s | change | 0.8.0 chars/s | new chars/s |
|---|---|---|---|---|---|---|
| default | all | 2.62 | 2.618 | -0% | 5975.7 | 5969.6 |
| default | arb_Arab (ARABIC) | 2.795 | 2.728 | -2% | 7595.7 | 7412.7 |
| default | cmn_Hani (CJK) | 2.658 | 2.677 | +1% | 3209.9 | 3233.3 |
| default | ell_Grek (GREEK) | 2.506 | 2.538 | +1% | 7581.5 | 7678.2 |
| default | heb_Hebr (HEBREW) | 2.55 | 2.552 | +0% | 6853.5 | 6859.0 |
| default | heb_Hebr (LATIN) | 0.7 | 0.702 | +0% | 5955.6 | 5980.2 |
| default | hin_Deva (DEVANAGARI) | 3.538 | 3.56 | +1% | 7422.2 | 7468.3 |
| default | hin_Deva (LATIN) | 2.282 | 2.215 | -3% | 5916.6 | 5741.4 |
| default | jpn_Jpan (CJK) | 3.915 | 3.611 | -8% | 3431.2 | 3164.8 |
| default | jpn_Jpan (HIRAGANA) | 2.632 | 2.632 | +0% | 3844.9 | 3844.1 |
| default | jpn_Jpan (KATAKANA) | 2.489 | 2.469 | -1% | 3535.0 | 3507.0 |
| default | jpn_Jpan (LATIN) | 1.42 | 1.4 | -1% | 2851.2 | 2811.9 |
| default | kor_Hang (HANGUL) | 3.099 | 3.097 | -0% | 4190.2 | 4186.6 |
| default | kor_Hang (LATIN) | 6.693 | 6.92 | +3% | 3661.1 | 3785.5 |
| default | nld_Latn (LATIN) | 1.822 | 1.824 | +0% | 5112.6 | 5118.5 |
| default | rus_Cyrl (CYRILLIC) | 2.805 | 2.82 | +1% | 8438.2 | 8483.9 |
| default | rus_Cyrl (LATIN) | 2.356 | 2.464 | +5% | 6637.7 | 6939.9 |
| default | tha_Thai (THAI) | 2.496 | 2.473 | -1% | 6146.9 | 6090.0 |
| training | all | 2.779 | 2.778 | -0% | 6337.3 | 6336.3 |
| training | arb_Arab (ARABIC) | 2.873 | 2.863 | -0% | 7805.6 | 7779.8 |
| training | cmn_Hani (CJK) | 2.818 | 2.826 | +0% | 3403.2 | 3413.4 |
| training | ell_Grek (GREEK) | 2.604 | 2.632 | +1% | 7878.3 | 7961.8 |
| training | heb_Hebr (HEBREW) | 2.686 | 2.673 | -0% | 7219.4 | 7185.6 |
| training | heb_Hebr (LATIN) | 0.79 | 0.76 | -4% | 6723.7 | 6467.0 |
| training | hin_Deva (DEVANAGARI) | 3.794 | 3.784 | -0% | 7957.5 | 7937.4 |
| training | hin_Deva (LATIN) | 2.457 | 2.422 | -1% | 6370.4 | 6279.9 |
| training | jpn_Jpan (CJK) | 3.939 | 3.831 | -3% | 3451.9 | 3357.7 |
| training | jpn_Jpan (HIRAGANA) | 2.758 | 2.776 | +1% | 4029.0 | 4054.2 |
| training | jpn_Jpan (KATAKANA) | 2.39 | 2.429 | +2% | 3395.0 | 3450.2 |
| training | jpn_Jpan (LATIN) | 1.726 | 1.77 | +3% | 3466.1 | 3553.6 |
| training | kor_Hang (HANGUL) | 3.289 | 3.258 | -1% | 4446.3 | 4404.6 |
| training | kor_Hang (LATIN) | 7.249 | 7.488 | +3% | 3965.4 | 4095.9 |
| training | nld_Latn (LATIN) | 1.952 | 1.96 | +0% | 5479.6 | 5499.6 |
| training | rus_Cyrl (CYRILLIC) | 2.972 | 2.974 | +0% | 8939.1 | 8947.2 |
| training | rus_Cyrl (LATIN) | 2.382 | 2.429 | +2% | 6709.0 | 6843.8 |
| training | tha_Thai (THAI) | 2.79 | 2.772 | -1% | 6870.8 | 6826.3 |

## Cleaning: injected noise removed (FineWeb-2)

| config | kind | 0.8.0 | new |
|---|---|---|---|
| default | emoji | 5.1 | 4.9 |
| default | html | 100.0 | 100.0 |
| default | mojibake | 99.6 | 99.6 |
| default | url | 100.0 | 100.0 |
| training | emoji | 4.9 | 5.1 |
| training | html | 0.9 | 0.9 |
| training | mojibake | 99.5 | 99.5 |
| training | url | 100.0 | 100.0 |

