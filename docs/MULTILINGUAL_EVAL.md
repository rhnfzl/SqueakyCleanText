# Multilingual PII eval

`evaluation/multilingual_pii.jsonl`: 220 synthetic records, 20 per language for
en, nl, de, es, zh, ja, th, ar, hi, bn, ta (660 entities). Same schema as
`evaluation/release_benchmark.jsonl`. Labels: PERSON, LOCATION, ORGANISATION, EMAIL,
PHONE, IP (the names SCT emits). Names and places are in native script; 30
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
| PHONE | 0.67 / 0.55 | 0.88 / 0.97 | 0.65 / 0.73 |
| IP | 0.00 / 0.00 | 0.18 / 0.73 | 0.93 / 1.00 |
Code-switch slice, sct-default: recall 0.76, precision 0.82.

## Final 0.8.0 re-run (sct-default)

Five runs on the same machine and settings; GLiNER rows were not re-run. Cells are recall / precision.

- `371a873`: 0.8.0 code before the last review fixes (grouped national phone numbers, RFC
  email local parts, URL and source-marker fixes).
- `28ed3a2`: after the review fixes and the new defaults, before the eval fixes. Two changes
  cost recall here. (1) WordPiece widening (`a934690`): the default MULTILINGUAL model is
  mBERT (WordPiece), and Thai and kana runs have no spaces, so one "word" is a whole clause:
  `นครราชสีมา` became `ย้ายไปนครราชสีมาเมื่อปีที่แล้ว...` as one LOCATION. (2)
  `fix_character_width=False` kept `（` `：` in `seikyu@example.com（担当：鈴木花子）`, and
  the email domain took them, hiding the name inside.
- `3a01635`: email domains stop at
  general, CJK and full-width punctuation; isolated-symbol removal keeps a hyphen between
  letters of any script (`ইসলাম-এর` stayed `ইসলামএর` before, and the name widened over the
  suffix); widening is bounded by `ner_subword_widen` (default `1.0`: the added text may be
  at most as long as the tagged span).
- `4b7e9f9`: two NER changes.
  (1) `ner_subword_widen_same_script` (default `True`): widening stops where the letters'
  Unicode script changes, so a Latin name glued to kana or a Thai prefix (`David Millerは`,
  `ถึงคุณSarah Lee`) keeps its own span: 573/56/87 to 578/51/82, 5 documents (ja, th) whose
  PERSON span lost the kana/Thai. (2) `ner_sees_tags` (default `True`): in reversible and
  synthetic mode NER reads `<EMAIL>` / `<PHONE>` like placeholder mode instead of `<EMAIL_0>`;
  this eval runs reversible mode, so it now scores what placeholder mode finds: 578/51/82 to
  584/50/76, 13 documents (ar, bn, ja, ta): +6 TP (4 ORGANISATION, 2 LOCATION), 5 FP fewer,
  4 FP more (ORGANISATION fragments, 2 of them the same Arabic company name cut short).
  `fold_styled_characters` and `synthetic_locale` do not change this eval.
- `678a59b`: an email local part ends
  where glued CJK prose meets a Latin letter (fuzz round 3, `d6e047c`), so
  `请发送至wangfang@example.com` and `...お問い合わせはseikyu@example.com` no longer give one EMAIL
  over the prose: 584/50/76 to 608/34/52, 12 documents (ja, zh): EMAIL 72 to 84 TP, and NER
  now sees the organisation and person names the email span hid (ORGANISATION +8, PERSON +4 TP).
  No other language moved. Later fuzz and Codex rounds leave the predictions identical.
- 0.8.0 security review leftovers (final, `evaluation/results/multilingual-0.8.0.json`): a mail
  domain or URL host ends at punctuation of any script (Unicode category P), so the Arabic comma
  after an address stays out of the EMAIL: 608/34/52 to 612/29/48, 4 documents (ar): EMAIL
  84/4/4 to 88/0/0, and NER finds one Arabic organisation fragment fewer (ORGANISATION FP 14 to 13).

| language | 371a873 | 28ed3a2 | 3a01635 | 4b7e9f9 | 0.8.0 final |
|---|---|---|---|---|---|
| ar | 0.77 / 0.78 | 0.75 / 0.82 | 0.75 / 0.80 | 0.75 / 0.79 | 0.82 / 0.88 |
| bn | 0.82 / 0.96 | 0.72 / 0.90 | 0.78 / 0.94 | 0.83 / 0.94 | 0.83 / 0.94 |
| de | 0.98 / 1.00 | 0.98 / 1.00 | 0.98 / 1.00 | 0.98 / 1.00 | 0.98 / 1.00 |
| en | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| es | 0.98 / 0.98 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| hi | 0.88 / 0.93 | 0.93 / 1.00 | 0.95 / 1.00 | 0.95 / 1.00 | 0.95 / 1.00 |
| ja | 0.70 / 0.78 | 0.57 / 0.71 | 0.63 / 0.68 | 0.73 / 0.77 | 1.00 / 0.98 |
| nl | 0.93 / 0.93 | 0.93 / 0.93 | 0.93 / 0.93 | 0.93 / 0.93 | 0.93 / 0.93 |
| ta | 0.85 / 0.94 | 0.83 / 0.88 | 0.87 / 0.88 | 0.88 / 0.90 | 0.88 / 0.90 |
| th | 0.90 / 0.93 | 0.65 / 0.76 | 0.85 / 0.89 | 0.87 / 0.91 | 0.87 / 0.91 |
| zh | 0.80 / 0.86 | 0.67 / 0.77 | 0.80 / 0.87 | 0.80 / 0.87 | 0.93 / 0.95 |
| overall | 0.87 / 0.92 | 0.82 / 0.89 | 0.87 / 0.91 | 0.88 / 0.92 | 0.93 / 0.95 |

| label | 371a873 | 28ed3a2 | 3a01635 | 4b7e9f9 | 0.8.0 final |
|---|---|---|---|---|---|
| PERSON | 0.96 / 0.95 | 0.87 / 0.92 | 0.95 / 0.94 | 0.97 / 0.97 | 0.99 / 0.99 |
| LOCATION | 0.95 / 0.92 | 0.89 / 0.86 | 0.97 / 0.90 | 0.99 / 0.93 | 0.99 / 0.93 |
| ORGANISATION | 0.68 / 0.88 | 0.64 / 0.91 | 0.66 / 0.89 | 0.69 / 0.87 | 0.75 / 0.88 |
| EMAIL | 0.82 / 0.82 | 0.77 / 0.77 | 0.82 / 0.82 | 0.82 / 0.82 | 1.00 / 1.00 |
| PHONE | 0.86 / 0.93 | 0.86 / 0.93 | 0.86 / 0.93 | 0.86 / 0.93 | 0.86 / 0.93 |
| IP | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |

Overall TP/FP/FN: 577/51/83 at `371a873`, 542/64/118 at `28ed3a2`, 573/56/87 at `3a01635`,
584/50/76 at `4b7e9f9`, 608/34/52 at `678a59b`, 612/29/48 final (recall 0.927, precision 0.955).
Code-switch slice: 0.84 / 0.90, 0.76 / 0.83, 0.78 / 0.82, 0.84 / 0.87, 0.92 / 0.93.

What still differs from `371a873` (WordPiece widening, which hides a name's last subword and
so is kept on by default):

- Some organisation names in ar, bn and ta are no longer masked, and 1 Thai name. Widening makes
  the mBERT span equal to the XLM-RoBERTa span, so the two scores are averaged; a weak mBERT
  score (0.63) pulls an agreed span below the 0.85 threshold, where the XLM-RoBERTa span
  alone passed before. Taking the best score per span instead of the mean gives 581/62/79
  (recall 0.88, precision 0.90); not changed in 0.8.0 (precision trade-off, needs its own
  measurement on the release benchmark).
- `ner_subword_widen=0` turns widening off (0.7 spans).

## What the misses are (sct-default)

- **EMAIL**: none left. The Arabic comma `،` after an address (4 ar records) and CJK text glued
  before it (zh / ja, fuzz round 3) are both fixed in 0.8.0.
- **PHONE**: grouped national numbers (`010-6552 1234`, `699 123 456`,
  `06 12 34 56 78`) are matched whole since 0.8.0; `06-12345678` still loses its `06-`.
- **ORGANISATION**: weakest label (0.75 recall); in ar, bn, ja, th, zh, ta the multilingual
  model misses or splits organisation names.
- **Span edges in bn / ta**: names cut a few characters short (`সুমাইয়া আক্ত`) or run into the
  next word (`கோயம்புத்தூர் நக`), likely subword-to-character offset mapping.

## Recommendation

Keep today's default ensemble: it wins overall (0.84 / 0.89 at `39a5ca3`, 0.93 / 0.95 in
0.8.0) and in every language.
GLiNER multi v2.1 is far behind outside Latin scripts. GLiNER multi PII v1 has the best
EMAIL / PHONE precision and the best Arabic row, but recall in hi, bn, ta, ja, zh is
0.07 to 0.35, so it is not a default candidate either. **The default NER model is NOT
switched in 0.8.0.** The cheaper wins were the regex bugs above: grouped phone numbers
are fixed in 0.8.0, the email boundary with glued CJK text (fuzz round 3) and the Arabic comma are fixed too. GlotLID / ICU / SaT extras: no gap traced to language ID or segmentation
here, so they stay out.

## Limits

Synthetic template text is easier than real text: short sentences, one entity per slot,
a small fixed set of names. Treat the numbers as a floor check and a regression signal,
not as real-world recall. Thai, Japanese and Chinese spans also depend on the no-space
chunking fixed in 0.7.1. One run on one machine; GLiNER results move with the threshold.
