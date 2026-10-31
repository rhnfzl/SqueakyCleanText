# Multilingual PII eval

`evaluation/multilingual_pii.jsonl`: 220 synthetic records, 20 per language for
en, nl, de, es, zh, ja, th, ar, hi, bn, ta (660 entities). Same schema as
`evaluation/release_benchmark.jsonl`. Labels: PERSON, LOCATION, ORGANISATION, EMAIL,
PHONE_NUMBER, IP (the names SCT emits). Names and places are in native script; 30
records put a Latin-script name inside non-Latin text (slice `code-switch`).

Rebuild (deterministic, byte-identical; a test checks the committed file):

```sh
python -m scripts.build_multilingual_eval
```

Measure (downloads several GB, needs `pip install gliner` for the GLiNER rows; never CI):

```sh
python -m scripts.run_multilingual_eval            # all three models
python -m scripts.run_multilingual_eval --models sct-default
```

Scoring is exact label + character span (`sct.evaluation.evaluate_pii`). Only the six
dataset labels count; other predicted labels (NUMBER, DATE, URL) are dropped.

## Results (2026-10-31)

Commit `39a5ca3`, macOS arm64, Python 3.13, CPU. Raw report and predictions:
`evaluation/results/multilingual-2026-10-31.json`. Cells are recall / precision.

- `sct-default`: the default ONNX ensemble with language detection, run like the release
  benchmark (`replacement_mode="reversible"`, `check_to_ascii_unicode=False`). 41 s.
- `gliner:urchade/gliner_multi-v2.1` (Apache-2.0), threshold 0.5. 215 s.
- `gliner:urchade/gliner_multi_pii-v1` (Apache-2.0), threshold 0.5. 136 s.

| language | gliner:urchade/gliner_multi-v2.1 R / P | gliner:urchade/gliner_multi_pii-v1 R / P | sct-default R / P |
|---|---|---|---|
| ar | 0.77 / 0.84 | 0.93 / 1.00 | 0.77 / 0.78 |
| bn | 0.17 / 0.37 | 0.35 / 0.91 | 0.82 / 0.96 |
| de | 0.82 / 0.73 | 0.92 / 0.98 | 0.98 / 1.00 |
| en | 0.77 / 0.73 | 0.97 / 0.94 | 1.00 / 1.00 |
| es | 0.82 / 0.89 | 0.93 / 0.93 | 0.87 / 0.98 |
| hi | 0.27 / 0.62 | 0.27 / 0.84 | 0.82 / 0.88 |
| ja | 0.07 / 0.22 | 0.07 / 0.21 | 0.62 / 0.67 |
| nl | 0.77 / 0.73 | 0.95 / 0.93 | 0.93 / 0.93 |
| ta | 0.20 / 0.44 | 0.33 / 0.91 | 0.78 / 0.85 |
| th | 0.20 / 0.60 | 0.33 / 1.00 | 0.90 / 0.93 |
| zh | 0.12 / 0.41 | 0.23 / 0.78 | 0.77 / 0.82 |
| overall | 0.45 / 0.68 | 0.57 / 0.91 | 0.84 / 0.89 |

| label | gliner:urchade/gliner_multi-v2.1 R / P | gliner:urchade/gliner_multi_pii-v1 R / P | sct-default R / P |
|---|---|---|---|
| PERSON | 0.54 / 0.73 | 0.55 / 0.88 | 0.95 / 0.95 |
| LOCATION | 0.49 / 0.93 | 0.42 / 0.90 | 0.97 / 0.92 |
| ORGANISATION | 0.47 / 0.76 | 0.48 / 0.83 | 0.68 / 0.88 |
| EMAIL | 0.16 / 0.37 | 0.82 / 0.99 | 0.82 / 0.82 |
| PHONE_NUMBER | 0.67 / 0.55 | 0.88 / 0.97 | 0.65 / 0.73 |
| IP | 0.00 / 0.00 | 0.18 / 0.73 | 0.93 / 1.00 |
Code-switch slice, sct-default: recall 0.76, precision 0.82.

## What the misses are (sct-default)

- **EMAIL in zh / ja**: the email regex takes the CJK text glued before the address
  (`请发送至wangfang@example.com` becomes one EMAIL). In ar it takes the Arabic comma `،`
  after the address. Regex boundary bugs, not model gaps.
- **PHONE_NUMBER**: numbers with a space between groups are cut (`010-6552 1234` ->
  `010-6552`, `06-12345678` -> `12345678`, `699 123 456` missed). Affects every language.
- **ORGANISATION**: weakest label (0.68 recall); in ar, bn, ja, th, zh, ta the multilingual
  model misses or splits organisation names.
- **Span edges in bn / ta**: names cut a few characters short (`সুমাইয়া আক্ত`) or run into the
  next word (`கோயம்புத்தூர் நக`), likely subword-to-character offset mapping.

## Recommendation

Keep today's default ensemble: it wins overall (0.84 / 0.89) and in every language.
GLiNER multi v2.1 is far behind outside Latin scripts. GLiNER multi PII v1 has the best
EMAIL / PHONE precision and the best Arabic row, but recall in hi, bn, ta, ja, zh is
0.07 to 0.35, so it is not a default candidate either. **The default NER model is NOT
switched in 0.8.0.** The cheaper wins are the regex bugs above (email boundary with
CJK and Arabic punctuation, grouped phone numbers); they are core changes, not part of
this branch. GlotLID / ICU / SaT extras: no gap traced to language ID or segmentation
here, so they stay out.

## Limits

Synthetic template text is easier than real text: short sentences, one entity per slot,
a small fixed set of names. Treat the numbers as a floor check and a regression signal,
not as real-world recall. Thai, Japanese and Chinese spans also depend on the no-space
chunking fixed in 0.7.1. One run on one machine; GLiNER results move with the threshold.
