# SCT vs other PII tools: comparison, 2026-11-02 (rules 2026-11-03, generator families 2026-11-03)

SqueakyCleanText 0.8.1 (`sct-default`, and its optional GLiNER path `sct-gliner`) against six
other offline PII tools, under the frozen protocol in `benchmarks/compare/PROTOCOL.md`
(Codex's 20 fairness rules; section 20 maps each rule). Numbers come from this bundle
(`default/`, `tuned/`, `norecover/`, `audit/`, `speed-*/`); every table can be rebuilt with
`python -m benchmarks.compare.report benchmarks/results/compare-2026-11-02` (see `TABLES.md`), and
this file with `make_comparison.py` in the bundle (prose numbers are read from the bundle).

- Entries (exact pins in `PROTOCOL.md` section 2): sct-default, sct-gliner, presidio 2.2.366
  (en_core_web_lg), gliner-nvidia (nvidia/gliner-PII), gliner2-fastino
  (fastino/gliner2-privacy-filter-PII-multi), opf (openai/privacy-filter), bardsai-v3
  (bardsai/eu-pii-anonimization-multilang-v3, INT8 ONNX), datafog 4.9.0 (regex).
- Data: test part (hash split by template/record, seed 13), 300 docs per language
  (TAB 106, REDACT 825), 14 datasets, 16,661 docs per entry. Privacy-synthetic: openpii, gretel
  (finance), gretel_en, nemotron, argilla_pii, thai_pii, bayanat. Privacy-real: TAB, REDACT.
  NER (separate, not privacy): iahlt, thainer, naamapadam, multiconer, stockmark.
- Tracks: **default** = vendor quick start (the primary track); **tuned** = one configuration per
  entry picked from at most 12 candidates on the dev parts of all nine privacy sets, frozen in
  `benchmarks/compare/configs/tuned.json` before the test run. Because the tuned choice saw the dev
  part of every privacy set, tuned results are shown second. Gold-language track. All quality runs
  on CPU. One mask operation for every entry.

## What changed on 2026-11-03

Two passes, both rescoring the same raw outputs (no new model runs).

1. **Rules pass.** Contamination (decision 21): if a tool, or any model inside it, was trained on a
   dataset (T), is evaluated on it by its vendor (E), or SCT was developed against it (D), that cell
   is excluded for that tool; unknown training data (U) is shown apart, never as clean. Claimed
   languages: each tool is scored only on the languages its model card claims.
2. **Families and recovery pass** (an independent check of pass 1):
   - The rule covers a dataset **and its generator family**, for every entry, SCT included.
     Families (`entries.FAMILIES`): Gretel (gretel finance, gretel_en, nemotron: Gretel Navigator
     and its successor NeMo Data Designer) and ai4privacy (openpii, argilla_pii). So gretel_en and
     argilla_pii are now D-family for SCT, gretel finance is T-family for bardsai-v3 (and
     T-family, E-family for gliner-nvidia), nemotron E-family for gliner-nvidia.
   - Pass 1 said gretel_en and argilla_pii were "held out for SCT, dev included". That was
     wrong: the tuned track picked SCT's configuration (like every entry's) on the dev parts of
     all privacy sets, these two included. The default track is now primary.
   - Per-dataset tables use only each entry's claimed languages (`°` = no such doc; the rest
     is a separate "outside claimed languages" table).
   - The recovery rule for SCT's dropped placeholders was described as "3+ char words"; the code
     used 4 (`LEAK_MIN_CHARS`). It is now stricter (any visible 2+ char word, case- and
     accent-insensitive, or the value's separator-free form, blocks it) and was audited; every SCT
     number it changes is shown with and without it.
   - Pass 1 said "0 recovered copies" on gretel_en and argilla_pii: sct-gliner had 1
     on gretel_en (tuned).
   - sct-gliner's claim (en) is now taken from the model card's language tag; sct-default claims
     every language and is held to every language. Licence notes corrected (end of report).

## Bottom line

- **SCT has no clean cell in round 1.** sct-default is D or D-family on all 14 datasets, sct-gliner
  U plus D or D-family on all 14. **Round 1 cannot rank SCT** (either entry), on any group. Every SCT
  number below is information only. Round 2 needs fresh datasets that no tool and no SCT
  development has seen, and that belong to no generator family any of them has seen.
- **opf has no clean cell either** (U everywhere: training data undisclosed).
- The other five tools are compared only with each other, on their own clean cells, in their
  claimed languages. What each tool has left (default track, full view):

| entry | privacy-synthetic | privacy-real | NER | claimed languages | round 1 status |
|---|---|---|---|---|---|
| sct-default | none | none | none | all (multilingual claim) | **no clean cell: round 1 cannot rank it** |
| sct-gliner | none | none | none | en | **no clean cell: round 1 cannot rank it** |
| presidio | argilla_pii, gretel, gretel_en, nemotron, openpii (+ outside its languages only: thai_pii, bayanat) | redact, tab | none (+ outside its languages only: iahlt, thainer, naamapadam, multiconer, stockmark) | en | has clean cells |
| gliner-nvidia | none (+ outside its languages only: thai_pii, bayanat) | redact, tab | none (+ outside its languages only: iahlt, thainer, naamapadam, multiconer, stockmark) | en | has clean cells |
| gliner2-fastino | argilla_pii, gretel, gretel_en, nemotron, openpii (+ outside its languages only: thai_pii, bayanat) | redact, tab | none (+ outside its languages only: iahlt, thainer, naamapadam, multiconer, stockmark) | en fr es de it pt nl | has clean cells |
| opf | none | none | none | en | **no clean cell: round 1 cannot rank it** (U on every set) |
| bardsai-v3 | none (+ outside its languages only: thai_pii, bayanat) | redact, tab | none (+ outside its languages only: iahlt, thainer, naamapadam, multiconer, stockmark) | 24 EU | has clean cells |
| datafog | argilla_pii, bayanat, gretel, gretel_en, nemotron, openpii, thai_pii | redact, tab | iahlt, multiconer, naamapadam, stockmark, thainer | all (regex, no language claim) | has clean cells |

There is no SCT pairwise or same-set table in this report: no dataset is clean for SCT.

How to read the tables:

- Scores are the group macro: mean over datasets of each dataset's language macro, failed docs
  counted unmasked, DATE out. Synthetic and real data are never pooled.
- mR = type-agnostic masked recall; mP = masked precision; strict F1 = exact span and type;
  doc leak = share of docs with a gold mention that keeps any letter or digit unmasked;
  over-mask = masked characters outside every gold span / all non-gold characters.
- Own tables: the entry on its own clean datasets, only docs in its claimed languages, with a 95%
  bootstrap CI (1,000 resamples of templates/docs within each dataset, seed 13). Beside it, for
  information: mR on clean sets outside the claimed languages, on unknown (U) sets, and (in
  brackets) on excluded sets, each with the number of datasets. SCT's excluded numbers carry
  "w/o recovery": the same number without `scorer.recover_dropped`.
- Full view = all 9 SCT types (a type the tool does not support counts as missed). Shared view =
  EMAIL, PHONE, NUMBER. Speed: CPU, 16 threads, batch 8, 100 FineWeb-2 docs, median of 3; peak
  RAM = adapter VmHWM. No GPU numbers (pinned torch wheels need CUDA 13.2; the box's driver is 12.6).
- Own-table rows are not a ranking (each row is on its own datasets and languages); compare tools
  through the same-set tables.

## Default track (primary)

### Full view (all 9 types), privacy-synthetic

| entry | clean sets with claimed-language docs (all clean sets) | mR [95% CI] | mP [95% CI] | strict F1 [95% CI] | doc leak [95% CI] | over-mask [95% CI] | mR clean, outside claimed langs | mR unknown (U) sets | (mR excluded sets, info only) | docs/s CPU | peak RAM | claimed languages | licence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gliner2-fastino | argilla_pii, gretel, gretel_en, nemotron, openpii (7) | 0.872 [0.861, 0.882] | 0.746 [0.727, 0.765] | 0.702 [0.688, 0.717] | 0.335 [0.315, 0.355] | 0.036 [0.033, 0.040] | 0.849 (4) | - | - | 0.52 | 6,832 MB | en fr es de it pt nl | Apache-2.0 |
| presidio | argilla_pii, gretel, gretel_en, nemotron, openpii (7) | 0.762 [0.742, 0.781] | 0.803 [0.773, 0.838] | 0.540 [0.521, 0.559] | 0.608 [0.584, 0.633] | 0.023 [0.019, 0.029] | 0.605 (4) | - | - | 27 | 826 MB | en | MIT |
| datafog | argilla_pii, bayanat, gretel, gretel_en, nemotron, openpii, thai_pii (7) | 0.337 [0.325, 0.348] | 0.854 [0.833, 0.878] | 0.342 [0.328, 0.355] | 0.919 [0.908, 0.929] | 0.009 [0.007, 0.010] | - | - | - | 1,787 | 30 MB | all (regex, no language claim) | MIT |
| sct-gliner | none (0) | - | - | - | - | - | - | - | (0.930 (5); w/o recovery 0.914 (5)) | 0.53 | 6,414 MB | en | MIT + Apache-2.0 model |
| sct-default | none (0) | - | - | - | - | - | - | - | (0.934 (7); w/o recovery 0.926 (7)) | 1.50 | 7,692 MB | all (multilingual claim) | MIT |
| opf | none (0) | - | - | - | - | - | - | 0.722 (5) | - | 0.29 | 26,113 MB | en | Apache-2.0 |
| gliner-nvidia | none (2) | - | - | - | - | - | 0.563 (2) | - | (0.889 (5)) | 0.15 | 25,732 MB | en | Apache-2.0 + NVIDIA Open Model |
| bardsai-v3 | none (2) | - | - | - | - | - | 0.895 (2) | - | (0.919 (5)) | 2.52 | 1,738 MB | 24 EU | Apache-2.0 |

### Full view (all 9 types), privacy-real (TAB, REDACT)

| entry | clean sets with claimed-language docs (all clean sets) | mR [95% CI] | mP [95% CI] | strict F1 [95% CI] | doc leak [95% CI] | over-mask [95% CI] | mR clean, outside claimed langs | mR unknown (U) sets | (mR excluded sets, info only) | docs/s CPU | peak RAM | claimed languages | licence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gliner2-fastino | redact, tab (2) | 0.880 [0.858, 0.900] | 0.590 [0.570, 0.611] | 0.544 [0.526, 0.562] | 0.865 [0.839, 0.890] | 0.056 [0.053, 0.059] | 0.849 (1) | - | - | 0.52 | 6,832 MB | en fr es de it pt nl | Apache-2.0 |
| bardsai-v3 | redact, tab (2) | 0.739 [0.714, 0.764] | 0.817 [0.804, 0.831] | 0.512 [0.495, 0.529] | 0.830 [0.806, 0.852] | 0.022 [0.020, 0.024] | 0.804 (1) | - | - | 2.52 | 1,738 MB | 24 EU | Apache-2.0 |
| presidio | redact, tab (2) | 0.612 [0.576, 0.648] | 0.651 [0.605, 0.697] | 0.319 [0.285, 0.349] | 0.922 [0.850, 0.984] | 0.045 [0.034, 0.057] | 0.620 (1) | - | - | 27 | 826 MB | en | MIT |
| gliner-nvidia | redact, tab (2) | 0.391 [0.335, 0.462] | 0.793 [0.752, 0.828] | 0.288 [0.245, 0.333] | 0.953 [0.891, 1.000] | 0.016 [0.012, 0.022] | 0.544 (1) | - | - | 0.15 | 25,732 MB | en | Apache-2.0 + NVIDIA Open Model |
| datafog | redact, tab (2) | 0.161 [0.144, 0.182] | 0.765 [0.751, 0.790] | 0.155 [0.147, 0.162] | 0.997 [0.994, 1.000] | 0.008 [0.007, 0.009] | - | - | - | 1,787 | 30 MB | all (regex, no language claim) | MIT |
| sct-gliner | none (0) | - | - | - | - | - | - | - | (0.770 (2); w/o recovery 0.737 (2)) | 0.53 | 6,414 MB | en | MIT + Apache-2.0 model |
| sct-default | none (0) | - | - | - | - | - | - | - | (0.956 (2); w/o recovery 0.932 (2)) | 1.50 | 7,692 MB | all (multilingual claim) | MIT |
| opf | none (0) | - | - | - | - | - | - | 0.458 (2) | - | 0.29 | 26,113 MB | en | Apache-2.0 |

### Shared view (EMAIL, PHONE, NUMBER), privacy-synthetic

| entry | clean sets with claimed-language docs (all clean sets) | mR [95% CI] | mP [95% CI] | strict F1 [95% CI] | doc leak [95% CI] | over-mask [95% CI] | mR clean, outside claimed langs | mR unknown (U) sets | (mR excluded sets, info only) | docs/s CPU | peak RAM | claimed languages | licence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gliner2-fastino | argilla_pii, gretel, gretel_en, nemotron, openpii (7) | 0.917 [0.902, 0.930] | 0.528 [0.504, 0.553] | 0.827 [0.793, 0.858] | 0.147 [0.129, 0.166] | 0.036 [0.033, 0.040] | 0.904 (4) | - | - | 0.52 | 6,832 MB | en fr es de it pt nl | Apache-2.0 |
| presidio | argilla_pii, gretel, gretel_en, nemotron, openpii (7) | 0.854 [0.831, 0.873] | 0.682 [0.650, 0.721] | 0.698 [0.656, 0.752] | 0.318 [0.289, 0.349] | 0.023 [0.019, 0.029] | 0.821 (4) | - | - | 27 | 826 MB | en | MIT |
| datafog | argilla_pii, bayanat, gretel, gretel_en, nemotron, openpii, thai_pii (7) | 0.678 [0.656, 0.700] | 0.828 [0.808, 0.852] | 0.597 [0.572, 0.623] | 0.507 [0.482, 0.532] | 0.009 [0.007, 0.010] | - | - | - | 1,787 | 30 MB | all (regex, no language claim) | MIT |
| sct-gliner | none (0) | - | - | - | - | - | - | - | (0.946 (5); w/o recovery 0.928 (5)) | 0.53 | 6,414 MB | en | MIT + Apache-2.0 model |
| sct-default | none (0) | - | - | - | - | - | - | - | (0.961 (7); w/o recovery 0.949 (7)) | 1.50 | 7,692 MB | all (multilingual claim) | MIT |
| opf | none (0) | - | - | - | - | - | - | 0.877 (5) | - | 0.29 | 26,113 MB | en | Apache-2.0 |
| gliner-nvidia | none (2) | - | - | - | - | - | 0.722 (2) | - | (0.901 (5)) | 0.15 | 25,732 MB | en | Apache-2.0 + NVIDIA Open Model |
| bardsai-v3 | none (2) | - | - | - | - | - | 0.985 (2) | - | (0.978 (5)) | 2.52 | 1,738 MB | 24 EU | Apache-2.0 |

### Shared view (EMAIL, PHONE, NUMBER), privacy-real (TAB, REDACT)

| entry | clean sets with claimed-language docs (all clean sets) | mR [95% CI] | mP [95% CI] | strict F1 [95% CI] | doc leak [95% CI] | over-mask [95% CI] | mR clean, outside claimed langs | mR unknown (U) sets | (mR excluded sets, info only) | docs/s CPU | peak RAM | claimed languages | licence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| datafog | redact, tab (2) | 0.768 [0.648, 0.846] | 0.750 [0.736, 0.775] | 0.351 [0.341, 0.360] | 0.812 [0.795, 0.830] | 0.008 [0.007, 0.009] | - | - | - | 1,787 | 30 MB | all (regex, no language claim) | MIT |
| gliner2-fastino | redact, tab (2) | 0.495 [0.467, 0.524] | 0.166 [0.149, 0.186] | 0.452 [0.429, 0.475] | 0.600 [0.574, 0.625] | 0.056 [0.053, 0.059] | 0.916 (1) | - | - | 0.52 | 6,832 MB | en fr es de it pt nl | Apache-2.0 |
| bardsai-v3 | redact, tab (2) | 0.472 [0.464, 0.479] | 0.198 [0.180, 0.220] | 0.433 [0.421, 0.447] | 0.621 [0.601, 0.643] | 0.022 [0.020, 0.024] | 0.864 (1) | - | - | 2.52 | 1,738 MB | 24 EU | Apache-2.0 |
| presidio | redact, tab (2) | 0.466 [0.429, 0.498] | 0.155 [0.112, 0.204] | 0.290 [0.234, 0.354] | 0.714 [0.625, 0.803] | 0.045 [0.034, 0.057] | 0.852 (1) | - | - | 27 | 826 MB | en | MIT |
| gliner-nvidia | redact, tab (2) | 0.398 [0.311, 0.520] | 0.457 [0.397, 0.516] | 0.515 [0.423, 0.642] | 0.480 [0.380, 0.587] | 0.016 [0.012, 0.022] | 0.570 (1) | - | - | 0.15 | 25,732 MB | en | Apache-2.0 + NVIDIA Open Model |
| sct-gliner | none (0) | - | - | - | - | - | - | - | (0.986 (2); w/o recovery 0.986 (2)) | 0.53 | 6,414 MB | en | MIT + Apache-2.0 model |
| sct-default | none (0) | - | - | - | - | - | - | - | (0.985 (2); w/o recovery 0.982 (2)) | 1.50 | 7,692 MB | all (multilingual claim) | MIT |
| opf | none (0) | - | - | - | - | - | - | 0.452 (2) | - | 0.29 | 26,113 MB | en | Apache-2.0 |

The NER group is in `TABLES.md`: no model entry claims any NER language (ar, th, bn, hi, ta, zh,
ja), so only datafog (no claim) has an in-claim NER score there; SCT's NER cells are D.

### Same-set, default, full view (the non-SCT entries with clean sets)

The only comparison between tools in this report. Privacy-synthetic:

No dataset in this group is clean for every one of: presidio, gliner-nvidia, gliner2-fastino, bardsai-v3, datafog (in languages all of them claim).

Privacy-real:

Datasets: redact, tab (clean for all: tab, redact); only docs in languages every entry claims; point values.

| entry | mR | mP | strict F1 | doc leak | over-mask |
|---|---|---|---|---|---|
| gliner2-fastino | 0.889 | 0.546 | 0.501 | 0.828 | 0.061 |
| bardsai-v3 | 0.713 | 0.758 | 0.479 | 0.859 | 0.028 |
| presidio | 0.612 | 0.651 | 0.319 | 0.922 | 0.045 |
| gliner-nvidia | 0.391 | 0.793 | 0.288 | 0.953 | 0.016 |
| datafog | 0.163 | 0.756 | 0.143 | 1.000 | 0.007 |

Shared view, privacy-real:

Datasets: redact, tab (clean for all: tab, redact); only docs in languages every entry claims; point values.

| entry | mR | mP | strict F1 | doc leak | over-mask |
|---|---|---|---|---|---|
| datafog | 0.758 | 0.724 | 0.342 | 0.822 | 0.007 |
| gliner2-fastino | 0.507 | 0.127 | 0.434 | 0.572 | 0.061 |
| bardsai-v3 | 0.468 | 0.145 | 0.418 | 0.625 | 0.028 |
| presidio | 0.466 | 0.155 | 0.290 | 0.714 | 0.045 |
| gliner-nvidia | 0.398 | 0.457 | 0.515 | 0.480 | 0.016 |

## Tuned track (secondary)

Same tables on the tuned configurations. Read with the caveat above: each tuned configuration
was chosen on the dev parts of all nine privacy sets.

### Full view, privacy-synthetic

| entry | clean sets with claimed-language docs (all clean sets) | mR [95% CI] | mP [95% CI] | strict F1 [95% CI] | doc leak [95% CI] | over-mask [95% CI] | mR clean, outside claimed langs | mR unknown (U) sets | (mR excluded sets, info only) | docs/s CPU | peak RAM | claimed languages | licence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gliner2-fastino | argilla_pii, gretel, gretel_en, nemotron, openpii (7) | 0.897 [0.886, 0.907] | 0.702 [0.682, 0.721] | 0.680 [0.665, 0.696] | 0.285 [0.265, 0.304] | 0.046 [0.042, 0.050] | 0.880 (4) | - | - | 0.52 | 6,832 MB | en fr es de it pt nl | Apache-2.0 |
| presidio | argilla_pii, gretel, gretel_en, nemotron, openpii (7) | 0.762 [0.742, 0.781] | 0.803 [0.773, 0.838] | 0.540 [0.521, 0.559] | 0.608 [0.584, 0.633] | 0.023 [0.019, 0.029] | 0.605 (4) | - | - | 27 | 826 MB | en | MIT |
| datafog | argilla_pii, bayanat, gretel, gretel_en, nemotron, openpii, thai_pii (7) | 0.337 [0.325, 0.348] | 0.854 [0.833, 0.878] | 0.342 [0.328, 0.355] | 0.919 [0.908, 0.929] | 0.009 [0.007, 0.010] | - | - | - | 1,787 | 30 MB | all (regex, no language claim) | MIT |
| sct-gliner | none (0) | - | - | - | - | - | - | - | (0.944 (5); w/o recovery 0.923 (5)) | 0.53 | 6,414 MB | en | MIT + Apache-2.0 model |
| sct-default | none (0) | - | - | - | - | - | - | - | (0.935 (7); w/o recovery 0.927 (7)) | 1.50 | 7,692 MB | all (multilingual claim) | MIT |
| opf | none (0) | - | - | - | - | - | - | 0.761 (5) | - | 0.29 | 26,113 MB | en | Apache-2.0 |
| gliner-nvidia | none (2) | - | - | - | - | - | 0.779 (2) | - | (0.931 (5)) | 0.15 | 25,732 MB | en | Apache-2.0 + NVIDIA Open Model |
| bardsai-v3 | none (2) | - | - | - | - | - | 0.895 (2) | - | (0.919 (5)) | 2.52 | 1,738 MB | 24 EU | Apache-2.0 |

### Full view, privacy-real

| entry | clean sets with claimed-language docs (all clean sets) | mR [95% CI] | mP [95% CI] | strict F1 [95% CI] | doc leak [95% CI] | over-mask [95% CI] | mR clean, outside claimed langs | mR unknown (U) sets | (mR excluded sets, info only) | docs/s CPU | peak RAM | claimed languages | licence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gliner2-fastino | redact, tab (2) | 0.904 [0.884, 0.922] | 0.542 [0.522, 0.562] | 0.516 [0.499, 0.534] | 0.833 [0.807, 0.858] | 0.071 [0.068, 0.075] | 0.884 (1) | - | - | 0.52 | 6,832 MB | en fr es de it pt nl | Apache-2.0 |
| bardsai-v3 | redact, tab (2) | 0.739 [0.714, 0.764] | 0.817 [0.804, 0.831] | 0.512 [0.495, 0.529] | 0.830 [0.806, 0.852] | 0.022 [0.020, 0.024] | 0.804 (1) | - | - | 2.52 | 1,738 MB | 24 EU | Apache-2.0 |
| presidio | redact, tab (2) | 0.612 [0.576, 0.648] | 0.651 [0.605, 0.697] | 0.319 [0.285, 0.349] | 0.922 [0.850, 0.984] | 0.045 [0.034, 0.057] | 0.620 (1) | - | - | 27 | 826 MB | en | MIT |
| gliner-nvidia | redact, tab (2) | 0.439 [0.382, 0.506] | 0.821 [0.783, 0.856] | 0.164 [0.134, 0.198] | 0.969 [0.922, 1.000] | 0.017 [0.012, 0.023] | 0.600 (1) | - | - | 0.15 | 25,732 MB | en | Apache-2.0 + NVIDIA Open Model |
| datafog | redact, tab (2) | 0.161 [0.144, 0.182] | 0.765 [0.751, 0.790] | 0.155 [0.147, 0.162] | 0.997 [0.994, 1.000] | 0.008 [0.007, 0.009] | - | - | - | 1,787 | 30 MB | all (regex, no language claim) | MIT |
| sct-gliner | none (0) | - | - | - | - | - | - | - | (0.839 (2); w/o recovery 0.799 (2)) | 0.53 | 6,414 MB | en | MIT + Apache-2.0 model |
| sct-default | none (0) | - | - | - | - | - | - | - | (0.958 (2); w/o recovery 0.933 (2)) | 1.50 | 7,692 MB | all (multilingual claim) | MIT |
| opf | none (0) | - | - | - | - | - | - | 0.523 (2) | - | 0.29 | 26,113 MB | en | Apache-2.0 |

### Same-set, tuned, full view, privacy-real

Datasets: redact, tab (clean for all: tab, redact); only docs in languages every entry claims; point values.

| entry | mR | mP | strict F1 | doc leak | over-mask |
|---|---|---|---|---|---|
| gliner2-fastino | 0.914 | 0.490 | 0.461 | 0.812 | 0.080 |
| bardsai-v3 | 0.713 | 0.758 | 0.479 | 0.859 | 0.028 |
| presidio | 0.612 | 0.651 | 0.319 | 0.922 | 0.045 |
| gliner-nvidia | 0.439 | 0.821 | 0.164 | 0.969 | 0.017 |
| datafog | 0.163 | 0.756 | 0.143 | 1.000 | 0.007 |

Privacy-synthetic: No dataset in this group is clean for every one of: presidio, gliner-nvidia, gliner2-fastino, bardsai-v3, datafog (in languages all of them claim).

## SCT: information only, with and without the recovery rule

Every SCT cell is excluded (D or D-family; sct-gliner also U), so nothing here is a comparison.
Per SCT entry and dataset, full view, claimed languages (sct-gliner: en; `°` = no en doc), masked
recall and doc leak with and without `scorer.recover_dropped`, and the copies it recovered.

### default track

| entry | dataset | masked_recall with | without | doc leak with | without | copies recovered |
|---|---|---|---|---|---|---|
| sct-default | openpii | (0.903) | (0.903) | 0.473 | 0.474 | 14 |
| sct-default | gretel | (0.744) | (0.706) | 0.658 | 0.698 | 873 |
| sct-default | gretel_en | (0.980) | (0.980) | 0.270 | 0.270 | 0 |
| sct-default | nemotron | (0.973) | (0.956) | 0.201 | 0.239 | 34 |
| sct-default | argilla_pii | (0.967) | (0.967) | 0.057 | 0.057 | 0 |
| sct-default | thai_pii | (0.975) | (0.975) | 0.782 | 0.782 | 0 |
| sct-default | bayanat | (0.995) | (0.995) | 0.050 | 0.050 | 0 |
| sct-default | tab | (0.946) | (0.935) | 1.000 | 1.000 | 189 |
| sct-default | redact | (0.966) | (0.928) | 0.612 | 0.656 | 2350 |
| sct-default | iahlt | (0.937) | (0.937) | 0.295 | 0.295 | 0 |
| sct-default | thainer | (0.877) | (0.877) | 0.498 | 0.498 | 0 |
| sct-default | naamapadam | (0.701) | (0.700) | 0.432 | 0.432 | 2 |
| sct-default | multiconer | (0.819) | (0.819) | 0.310 | 0.310 | 0 |
| sct-default | stockmark | (0.908) | (0.908) | 0.282 | 0.282 | 0 |
| sct-gliner | openpii | (0.840) | (0.840) | 0.599 | 0.599 | 64 |
| sct-gliner | gretel | (0.886) | (0.818) | 0.712 | 0.763 | 1833 |
| sct-gliner | gretel_en | (0.983) | (0.983) | 0.274 | 0.274 | 0 |
| sct-gliner | nemotron | (0.967) | (0.953) | 0.285 | 0.306 | 50 |
| sct-gliner | argilla_pii | (0.975) | (0.975) | 0.131 | 0.131 | 0 |
| sct-gliner | thai_pii | ° | ° | ° | ° | 0 |
| sct-gliner | bayanat | ° | ° | ° | ° | 0 |
| sct-gliner | tab | (0.579) | (0.565) | 1.000 | 1.000 | 164 |
| sct-gliner | redact | (0.961) | (0.909) | 0.906 | 0.938 | 2855 |
| sct-gliner | iahlt | ° | ° | ° | ° | 13 |
| sct-gliner | thainer | ° | ° | ° | ° | 0 |
| sct-gliner | naamapadam | ° | ° | ° | ° | 0 |
| sct-gliner | multiconer | ° | ° | ° | ° | 0 |
| sct-gliner | stockmark | ° | ° | ° | ° | 0 |

### tuned track

| entry | dataset | masked_recall with | without | doc leak with | without | copies recovered |
|---|---|---|---|---|---|---|
| sct-default | openpii | (0.907) | (0.907) | 0.462 | 0.463 | 18 |
| sct-default | gretel | (0.745) | (0.707) | 0.647 | 0.690 | 899 |
| sct-default | gretel_en | (0.978) | (0.978) | 0.266 | 0.266 | 0 |
| sct-default | nemotron | (0.973) | (0.956) | 0.197 | 0.236 | 33 |
| sct-default | argilla_pii | (0.967) | (0.967) | 0.057 | 0.057 | 0 |
| sct-default | thai_pii | (0.977) | (0.977) | 0.782 | 0.782 | 0 |
| sct-default | bayanat | (0.995) | (0.995) | 0.050 | 0.050 | 0 |
| sct-default | tab | (0.947) | (0.935) | 1.000 | 1.000 | 198 |
| sct-default | redact | (0.968) | (0.930) | 0.597 | 0.647 | 2379 |
| sct-default | iahlt | (0.937) | (0.937) | 0.291 | 0.291 | 0 |
| sct-default | thainer | (0.882) | (0.882) | 0.490 | 0.490 | 0 |
| sct-default | naamapadam | (0.712) | (0.711) | 0.411 | 0.411 | 2 |
| sct-default | multiconer | (0.829) | (0.829) | 0.296 | 0.296 | 0 |
| sct-default | stockmark | (0.917) | (0.917) | 0.267 | 0.267 | 0 |
| sct-gliner | openpii | (0.857) | (0.857) | 0.566 | 0.566 | 153 |
| sct-gliner | gretel | (0.912) | (0.834) | 0.588 | 0.685 | 3055 |
| sct-gliner | gretel_en | (0.983) | (0.983) | 0.239 | 0.239 | 1 |
| sct-gliner | nemotron | (0.981) | (0.958) | 0.222 | 0.271 | 105 |
| sct-gliner | argilla_pii | (0.985) | (0.985) | 0.102 | 0.102 | 0 |
| sct-gliner | thai_pii | ° | ° | ° | ° | 0 |
| sct-gliner | bayanat | ° | ° | ° | ° | 0 |
| sct-gliner | tab | (0.706) | (0.684) | 1.000 | 1.000 | 786 |
| sct-gliner | redact | (0.973) | (0.915) | 0.875 | 0.906 | 3725 |
| sct-gliner | iahlt | ° | ° | ° | ° | 33 |
| sct-gliner | thainer | ° | ° | ° | ° | 0 |
| sct-gliner | naamapadam | ° | ° | ° | ° | 0 |
| sct-gliner | multiconer | ° | ° | ° | ° | 0 |
| sct-gliner | stockmark | ° | ° | ° | ° | 0 |

## Recovery rule and its audit

Why: SCT's spans are placed from the reversible placeholders found in its output. A placeholder a
later cleaning step removed (the link text of `[a@b.se](mailto:a@b.se)`, repeated names) leaves
its source copy without a span, so it was scored as leaked although no character of it is
visible. `scorer.recover_dropped` credits such a copy of a value SCT masked elsewhere in the doc
only when all of these hold (`scorer.dropped_copies`, tested in `tests/test_benchmarks_compare.py`):

- the value has 4+ non-space chars and `leak` finds neither it nor any of its words of 4+ chars
  (`LEAK_MIN_CHARS` = 4) in SCT's output;
- no word of it of 2+ chars is a whole word of the output, case- and accent-insensitive (NFKD,
  marks dropped, casefolded);
- its separator-free folded form is not inside the separator-free folded output;
- the copy is a whole token (no letter or digit glued on).

Recovered copies also count as predictions, so they can lower SCT's precision. `runner.py
--no-recover` gives the mapping without them (`norecover/`).

Audit (`benchmarks/compare/audit_recover.py`, counts in `audit/`): a seeded sample of 50 recovered
copies per SCT entry, dataset and track (all copies when fewer), each checked for any visible
trace with the folded checks above. First pass, with the rule before the folded checks (2+ char
words, casefolded only): 11 of 950 sampled copies had a trace (at most
3 of 50 in one set, in 7 entry-dataset-track cells), all of them the
value's separator-free form inside a longer token of the output (none was a visible word). Such a
copy may still be readable, so crediting it could be a wrong credit (the gretel ones were extra
predictions, not on gold; REDACT/TAB/openpii ones were not classified, no text). The rule was
tightened, not dropped: the folded checks are now part of it. Final rule: 950 of 950 sampled copies pass;
457 of them overlap a gold span (credited recall), the rest are extra predictions. REDACT and
TAB were audited by counts and pass/fail only; no dataset text was written for them. The gretel
sample was also read by hand (private notes): the credited copies are mostly names, company
names and markdown link text, and none of the values shows in SCT's output in any form.

### Audit counts, default track

| entry | dataset | first pass: recovered | sampled | traces found | final rule: recovered | sampled | passed | sampled on gold |
|---|---|---|---|---|---|---|---|---|
| sct-default | openpii | 14 | 14 | 0 | 14 | 14 | 14 | 6 |
| sct-default | gretel | 884 | 50 | 0 | 873 | 50 | 50 | 30 |
| sct-default | nemotron | 34 | 34 | 0 | 34 | 34 | 34 | 31 |
| sct-default | tab | 192 | 50 | 0 | 189 | 50 | 50 | 25 |
| sct-default | redact | 2361 | 50 | 2 | 2350 | 50 | 50 | 15 |
| sct-default | naamapadam | 2 | 2 | 0 | 2 | 2 | 2 | 2 |
| sct-gliner | openpii | 65 | 50 | 1 | 64 | 50 | 50 | 14 |
| sct-gliner | gretel | 1871 | 50 | 2 | 1833 | 50 | 50 | 17 |
| sct-gliner | nemotron | 50 | 50 | 0 | 50 | 50 | 50 | 44 |
| sct-gliner | tab | 170 | 50 | 3 | 164 | 50 | 50 | 36 |
| sct-gliner | redact | 2881 | 50 | 1 | 2855 | 50 | 50 | 11 |
| sct-gliner | iahlt | 13 | 13 | 0 | 13 | 13 | 13 | 9 |

### Audit counts, tuned track

| entry | dataset | first pass: recovered | sampled | traces found | final rule: recovered | sampled | passed | sampled on gold |
|---|---|---|---|---|---|---|---|---|
| sct-default | openpii | 18 | 18 | 0 | 18 | 18 | 18 | 8 |
| sct-default | gretel | 910 | 50 | 1 | 899 | 50 | 50 | 26 |
| sct-default | nemotron | 33 | 33 | 0 | 33 | 33 | 33 | 31 |
| sct-default | tab | 201 | 50 | 0 | 198 | 50 | 50 | 25 |
| sct-default | redact | 2390 | 50 | 0 | 2379 | 50 | 50 | 16 |
| sct-default | naamapadam | 2 | 2 | 0 | 2 | 2 | 2 | 2 |
| sct-gliner | openpii | 158 | 50 | 0 | 153 | 50 | 50 | 16 |
| sct-gliner | gretel | 3092 | 50 | 1 | 3055 | 50 | 50 | 18 |
| sct-gliner | gretel_en | 1 | 1 | 0 | 1 | 1 | 1 | 0 |
| sct-gliner | nemotron | 105 | 50 | 0 | 105 | 50 | 50 | 35 |
| sct-gliner | tab | 797 | 50 | 0 | 786 | 50 | 50 | 9 |
| sct-gliner | redact | 3745 | 50 | 0 | 3725 | 50 | 50 | 13 |
| sct-gliner | iahlt | 33 | 33 | 0 | 33 | 33 | 33 | 18 |

## Per dataset (default, full view, masked F2, claimed languages only, completed docs)

`(x)` = excluded cell (T/E/D, family included), information only; `x?` = unknown training data;
`°` = no doc of the dataset in the entry's claimed languages. SCT cells show the value without
the recovery rule beside it.

| dataset | bardsai-v3 | datafog | gliner-nvidia | gliner2-fastino | opf | presidio | sct-default | sct-gliner |
|---|---|---|---|---|---|---|---|---|
| openpii | (0.992) | 0.338 | (0.893) | 0.869 | 0.890? | 0.771 | (0.905; w/o recovery 0.905) | (0.848; w/o recovery 0.848) |
| gretel | (0.660) | 0.150 | (0.800) | 0.695 | 0.561? | 0.546 | (0.599; w/o recovery 0.576) | (0.638; w/o recovery 0.608) |
| gretel_en | (0.994) | 0.487 | (0.933) | 0.908 | 0.903? | 0.865 | (0.933; w/o recovery 0.933) | (0.929; w/o recovery 0.929) |
| nemotron | (0.979) | 0.357 | (0.867) | 0.879 | 0.649? | 0.780 | (0.882; w/o recovery 0.870) | (0.853; w/o recovery 0.844) |
| argilla_pii | (0.983) | 0.233 | (0.895) | 0.856 | 0.747? | 0.880 | (0.964; w/o recovery 0.964) | (0.935; w/o recovery 0.935) |
| thai_pii | ° | 0.561 | ° | ° | ° | ° | (0.967; w/o recovery 0.967) | ° |
| bayanat | ° | 0.527 | ° | ° | ° | ° | (0.927; w/o recovery 0.927) | ° |
| tab | 0.638 | 0.093 | 0.308 | 0.783 | 0.229? | 0.486 | (0.836; w/o recovery 0.831) | (0.568; w/o recovery 0.558) |
| redact | 0.834 | 0.274 | 0.539 | 0.817 | 0.685? | 0.682 | (0.786; w/o recovery 0.777) | (0.726; w/o recovery 0.714) |
| iahlt | ° | 0.000 | ° | ° | ° | ° | (0.899; w/o recovery 0.899) | ° |
| thainer | ° | 0.004 | ° | ° | ° | ° | (0.860; w/o recovery 0.860) | ° |
| naamapadam | ° | 0.000 | ° | ° | ° | ° | (0.708; w/o recovery 0.707) | ° |
| multiconer | ° | 0.000 | ° | ° | ° | ° | (0.740; w/o recovery 0.740) | ° |
| stockmark | ° | 0.000 | ° | ° | ° | ° | (0.832; w/o recovery 0.832) | ° |

### Outside claimed languages (default; information only, not a loss; `-` = every doc in a claimed language)

| dataset | bardsai-v3 | datafog | gliner-nvidia | gliner2-fastino | opf | presidio | sct-default | sct-gliner |
|---|---|---|---|---|---|---|---|---|
| openpii | (0.905) | - | (0.834) | 0.849 | 0.837? | 0.642 | - | (0.860; w/o recovery 0.860) |
| gretel | - | - | (0.698) | 0.657 | 0.537? | 0.460 | - | (0.580; w/o recovery 0.554) |
| gretel_en | - | - | - | - | - | - | - | - |
| nemotron | - | - | - | - | - | - | - | - |
| argilla_pii | - | - | - | - | - | - | - | - |
| thai_pii | 0.889 | - | 0.439 | 0.930 | 0.844? | 0.486 | - | (0.687; w/o recovery 0.687) |
| bayanat | 0.893 | - | 0.722 | 0.914 | 0.887? | 0.751 | - | (0.772; w/o recovery 0.772) |
| tab | - | - | - | - | - | - | - | - |
| redact | 0.758 | - | 0.563 | 0.794 | 0.709? | 0.543 | - | (0.688; w/o recovery 0.676) |
| iahlt | 0.433 | - | 0.598 | 0.864 | 0.118? | 0.188 | - | (0.455; w/o recovery 0.447) |
| thainer | 0.394 | - | 0.077 | 0.568 | 0.325? | 0.107 | - | (0.139; w/o recovery 0.139) |
| naamapadam | 0.391 | - | 0.030 | 0.883 | 0.233? | 0.102 | - | (0.051; w/o recovery 0.051) |
| multiconer | 0.225 | - | 0.211 | 0.628 | 0.185? | 0.076 | - | (0.194; w/o recovery 0.194) |
| stockmark | 0.024 | - | 0.608 | 0.687 | 0.056? | 0.051 | - | (0.550; w/o recovery 0.550) |

## Per dataset (tuned, full view, masked F2, claimed languages only, completed docs)

`(x)` = excluded cell (T/E/D, family included), information only; `x?` = unknown training data;
`°` = no doc of the dataset in the entry's claimed languages. SCT cells show the value without
the recovery rule beside it.

| dataset | bardsai-v3 | datafog | gliner-nvidia | gliner2-fastino | opf | presidio | sct-default | sct-gliner |
|---|---|---|---|---|---|---|---|---|
| openpii | (0.992) | 0.338 | (0.975) | 0.901 | 0.920? | 0.771 | (0.908; w/o recovery 0.908) | (0.852; w/o recovery 0.852) |
| gretel | (0.660) | 0.150 | (0.800) | 0.694 | 0.579? | 0.546 | (0.599; w/o recovery 0.576) | (0.628; w/o recovery 0.599) |
| gretel_en | (0.994) | 0.487 | (0.957) | 0.919 | 0.912? | 0.865 | (0.932; w/o recovery 0.932) | (0.917; w/o recovery 0.917) |
| nemotron | (0.979) | 0.357 | (0.916) | 0.873 | 0.681? | 0.780 | (0.882; w/o recovery 0.870) | (0.831; w/o recovery 0.818) |
| argilla_pii | (0.983) | 0.233 | (0.934) | 0.845 | 0.799? | 0.880 | (0.963; w/o recovery 0.963) | (0.901; w/o recovery 0.901) |
| thai_pii | ° | 0.561 | ° | ° | ° | ° | (0.969; w/o recovery 0.969) | ° |
| bayanat | ° | 0.527 | ° | ° | ° | ° | (0.926; w/o recovery 0.926) | ° |
| tab | 0.638 | 0.093 | 0.363 | 0.776 | 0.284? | 0.486 | (0.835; w/o recovery 0.830) | (0.647; w/o recovery 0.647) |
| redact | 0.834 | 0.274 | 0.580 | 0.815 | 0.711? | 0.682 | (0.787; w/o recovery 0.778) | (0.718; w/o recovery 0.706) |
| iahlt | ° | 0.000 | ° | ° | ° | ° | (0.898; w/o recovery 0.898) | ° |
| thainer | ° | 0.004 | ° | ° | ° | ° | (0.864; w/o recovery 0.864) | ° |
| naamapadam | ° | 0.000 | ° | ° | ° | ° | (0.717; w/o recovery 0.715) | ° |
| multiconer | ° | 0.000 | ° | ° | ° | ° | (0.746; w/o recovery 0.746) | ° |
| stockmark | ° | 0.000 | ° | ° | ° | ° | (0.837; w/o recovery 0.837) | ° |

### Outside claimed languages (tuned; information only, not a loss; `-` = every doc in a claimed language)

| dataset | bardsai-v3 | datafog | gliner-nvidia | gliner2-fastino | opf | presidio | sct-default | sct-gliner |
|---|---|---|---|---|---|---|---|---|
| openpii | (0.905) | - | (0.956) | 0.888 | 0.882? | 0.642 | - | (0.858; w/o recovery 0.859) |
| gretel | - | - | (0.704) | 0.649 | 0.568? | 0.460 | - | (0.592; w/o recovery 0.563) |
| gretel_en | - | - | - | - | - | - | - | - |
| nemotron | - | - | - | - | - | - | - | - |
| argilla_pii | - | - | - | - | - | - | - | - |
| thai_pii | 0.889 | - | 0.782 | 0.952 | 0.898? | 0.486 | - | (0.708; w/o recovery 0.708) |
| bayanat | 0.893 | - | 0.731 | 0.920 | 0.898? | 0.751 | - | (0.742; w/o recovery 0.742) |
| tab | - | - | - | - | - | - | - | - |
| redact | 0.758 | - | 0.612 | 0.804 | 0.745? | 0.543 | - | (0.696; w/o recovery 0.677) |
| iahlt | 0.433 | - | 0.748 | 0.869 | 0.181? | 0.188 | - | (0.502; w/o recovery 0.490) |
| thainer | 0.394 | - | 0.215 | 0.588 | 0.502? | 0.107 | - | (0.216; w/o recovery 0.216) |
| naamapadam | 0.391 | - | 0.120 | 0.884 | 0.369? | 0.102 | - | (0.090; w/o recovery 0.090) |
| multiconer | 0.225 | - | 0.313 | 0.642 | 0.269? | 0.076 | - | (0.229; w/o recovery 0.229) |
| stockmark | 0.024 | - | 0.648 | 0.750 | 0.085? | 0.051 | - | (0.620; w/o recovery 0.620) |

## Per language (default, full view, masked recall, clean sets only)

Each cell: mean over the entry's clean datasets of the group that have the language. `°` = a
language outside the entry's claimed languages: shown apart from the claim, not a loss. `-` = the
entry has no clean dataset with that language. Tuned, unknown-set and NER tables are in `TABLES.md`.

### privacy-synthetic

| lang | datasets | bardsai-v3 | datafog | gliner-nvidia | gliner2-fastino | presidio |
|---|---|---|---|---|---|---|
| ar-egyptian | bayanat | 0.846° | 0.423 | 0.667° | 0.949° | 0.673° |
| ar-gulf | bayanat | 0.906° | 0.592 | 0.650° | 0.892° | 0.758° |
| ar-mixed_en | bayanat | 0.953° | 0.470 | 0.886° | 0.966° | 0.933° |
| ar-msa | bayanat | 0.914° | 0.441 | 0.741° | 0.918° | 0.664° |
| bg | openpii | - | 0.408 | - | 0.839° | 0.677° |
| cs | openpii | - | 0.298 | - | 0.866° | 0.632° |
| da | openpii | - | 0.383 | - | 0.893° | 0.704° |
| de | gretel, openpii | - | 0.189 | - | 0.814 | 0.640° |
| el | openpii | - | 0.262 | - | 0.864° | 0.659° |
| en | argilla_pii, gretel, gretel_en, nemotron, openpii | - | 0.294 | - | 0.906 | 0.762 |
| es | gretel, openpii | - | 0.218 | - | 0.778 | 0.552° |
| et | openpii | - | 0.352 | - | 0.887° | 0.752° |
| fi | openpii | - | 0.340 | - | 0.881° | 0.729° |
| fr | gretel, openpii | - | 0.226 | - | 0.868 | 0.566° |
| hr | openpii | - | 0.368 | - | 0.864° | 0.703° |
| hu | openpii | - | 0.251 | - | 0.863° | 0.588° |
| id | openpii | - | 0.322 | - | 0.833° | 0.708° |
| it | gretel, openpii | - | 0.216 | - | 0.821 | 0.622° |
| ja | openpii | - | 0.157 | - | 0.621° | 0.430° |
| ko | openpii | - | 0.217 | - | 0.779° | 0.409° |
| lt | openpii | - | 0.338 | - | 0.868° | 0.753° |
| lv | openpii | - | 0.306 | - | 0.857° | 0.715° |
| ms | openpii | - | 0.319 | - | 0.844° | 0.749° |
| nl | gretel, openpii | - | 0.196 | - | 0.774 | 0.607° |
| pl | openpii | - | 0.212 | - | 0.851° | 0.651° |
| pt | openpii | - | 0.297 | - | 0.833 | 0.757° |
| ro | openpii | - | 0.233 | - | 0.831° | 0.727° |
| sk | openpii | - | 0.247 | - | 0.845° | 0.683° |
| sl | openpii | - | 0.213 | - | 0.872° | 0.712° |
| sr | openpii | - | 0.376 | - | 0.861° | 0.721° |
| sv | gretel, openpii | - | 0.207 | - | 0.811° | 0.608° |
| th | thai_pii | 0.884° | 0.512 | 0.389° | 0.922° | 0.440° |
| tl | openpii | - | 0.255 | - | 0.876° | 0.733° |
| vi | openpii | - | 0.295 | - | 0.796° | 0.695° |
| zh | openpii | - | 0.150 | - | 0.628° | 0.399° |

### privacy-real

| lang | datasets | bardsai-v3 | datafog | gliner-nvidia | gliner2-fastino | presidio |
|---|---|---|---|---|---|---|
| ar | redact | 0.895° | 0.351 | 0.573° | 0.952° | 0.530° |
| cs | redact | 0.917 | 0.211 | 0.581° | 0.865° | 0.723° |
| da | redact | 0.831 | 0.191 | 0.535° | 0.809° | 0.690° |
| de | redact | 0.864 | 0.242 | 0.631° | 0.841 | 0.754° |
| en | redact, tab | 0.713 | 0.163 | 0.391 | 0.889 | 0.612 |
| es | redact | 0.929 | 0.231 | 0.540° | 0.931 | 0.776° |
| fi | redact | 0.905 | 0.330 | 0.627° | 0.891° | 0.776° |
| fr | redact | 0.893 | 0.202 | 0.665° | 0.907 | 0.650° |
| fr_ca | redact | 0.918 | 0.309 | 0.628° | 0.916 | 0.740° |
| he | redact | 0.881° | 0.310 | 0.510° | 0.940° | 0.555° |
| hi | redact | 0.887° | 0.240 | 0.335° | 0.887° | 0.542° |
| hu | redact | 0.906 | 0.212 | 0.630° | 0.871° | 0.643° |
| it | redact | 0.879 | 0.283 | 0.571° | 0.838 | 0.687° |
| ja | redact | 0.615° | 0.151 | 0.521° | 0.707° | 0.348° |
| ko | redact | 0.885° | 0.159 | 0.671° | 0.868° | 0.312° |
| nl | redact | 0.897 | 0.239 | 0.518° | 0.880 | 0.816° |
| no | redact | 0.919° | 0.249 | 0.577° | 0.919° | 0.779° |
| pt_br | redact | 0.917 | 0.220 | 0.557° | 0.886 | 0.807° |
| pt_eu | redact | 0.854 | 0.230 | 0.571° | 0.879 | 0.767° |
| ru | redact | 0.875° | 0.170 | 0.613° | 0.882° | 0.585° |
| sv | redact | 0.910 | 0.313 | 0.568° | 0.910° | 0.827° |
| th | redact | 0.884° | 0.421 | 0.269° | 0.889° | 0.370° |
| tr | redact | 0.938° | 0.208 | 0.558° | 0.918° | 0.694° |
| zh_cn | redact | 0.546° | 0.206 | 0.374° | 0.606° | 0.259° |
| zh_tw | redact | 0.519° | 0.201 | 0.426° | 0.667° | 0.258° |

## Contamination matrix

T = trained on it, E = vendor evaluates on it, D = SCT developed against it in earlier rounds, U =
training data undisclosed, `X-family` = X on another dataset of the same generator family, - =
none known (model cards at the pinned revisions, checked 2026-11-02; families 2026-11-03). Bold =
excluded, italic = unknown, plain = clean.

| entry | openpii | gretel | gretel_en | nemotron | argilla_pii | thai_pii | bayanat | tab | redact | iahlt | thainer | naamapadam | multiconer | stockmark |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sct-default | **D** | **D** | **D-family** | **D** | **D-family** | **D** | **D** | **D** | **D** | **D** | **D** | **D** | **D** | **D** |
| sct-gliner | **U, D** | **U, D** | **U, D-family** | **U, D** | **U, D-family** | **U, D** | **U, D** | **U, D** | **U, D** | **U, D** | **U, D** | **U, D** | **U, D** | **U, D** |
| presidio | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| gliner-nvidia | **E** | **T-family, E-family** | **E, T-family** | **T, E-family** | **E** | - | - | - | - | - | - | - | - | - |
| gliner2-fastino | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| opf | _U_ | _U_ | _U_ | _U_ | _U_ | _U_ | _U_ | _U_ | _U_ | _U_ | _U_ | _U_ | _U_ | _U_ |
| bardsai-v3 | **T** | **T-family** | **T** | **T** | **T-family** | - | - | - | - | - | - | - | - | - |
| datafog | - | - | - | - | - | - | - | - | - | - | - | - | - | - |

## Feature matrix (not scored)

| entry | languages (claimed, as scored) | types mapped to SCT's 9 | char spans | reversible | fake values | text cleaning | language ID | batch API | GPU | offline | licence (code / model) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| sct-default | all (SCT claims multilingual NER through its WikiNEural fallback; NER models for en, nl, de, es; LID 75 languages) | 9 | via reversible mode | yes (`TokenStore`) | yes (`synthetic` extra) | yes | yes (Lingua) | `process_batch` | optional (CUDA ONNX) | yes | MIT / per NER model card |
| sct-gliner | en (knowledgator/gliner-pii-base-v1.0 card: `language: [en]`) | 9 | via reversible mode | yes | yes | yes | yes | `process_batch` | optional | yes | MIT / Apache-2.0 |
| presidio | en (only en_core_web_lg loaded, both tracks); spaCy packs possible | 8 (no ORG with spaCy) | yes | encrypt / decrypt operator | custom operator | no | no | `BatchAnalyzerEngine` | optional (transformers engine) | yes | MIT / MIT (spaCy model) |
| gliner-nvidia | en | 9 (prompts) | yes | no | no | no | no | library batch | optional | yes | Apache-2.0 / NVIDIA Open Model License |
| gliner2-fastino | en fr es de it pt nl | 9 (prompts) | yes | no | no | no | no | library batch | optional | yes | Apache-2.0 / Apache-2.0 |
| opf | en ("primarily English") | 7 (no ORG, no IP) | yes | no | no | no | no | yes | optional | yes | Apache-2.0 / Apache-2.0 |
| bardsai-v3 | 24 EU | 9 | yes (bundled detector) | no | no | no | no | yes | optional | yes | - / Apache-2.0 |
| datafog | all (regex, no language claim; US shapes, German pack opt-in) | 5 (no PERSON, LOCATION, ORG, URL) | yes | no (hash, redact) | no | no | no | yes | no | yes | MIT / - |

## Failures (rule 19)

Failed docs are scored as unmasked (every mention leaked); the cause for gliner2-fastino is a
`ValueError` from the library's long-doc remap (spans past the text end). No dataset is under 99%
complete. Default track:

| entry | docs | failed | flagged datasets (< 99% complete) |
|---|---|---|---|
| bardsai-v3 | 16661 | 0 | - |
| datafog | 16661 | 0 | - |
| gliner-nvidia | 16661 | 0 | - |
| gliner2-fastino | 16661 | 9 | - |
| opf | 16661 | 0 | - |
| presidio | 16661 | 0 | - |
| sct-default | 16661 | 0 | - |
| sct-gliner | 16661 | 0 | - |

Tuned track:

| entry | docs | failed | flagged datasets (< 99% complete) |
|---|---|---|---|
| bardsai-v3 | 16661 | 0 | - |
| datafog | 16661 | 0 | - |
| gliner-nvidia | 16661 | 0 | - |
| gliner2-fastino | 16661 | 13 | - |
| opf | 16661 | 0 | - |
| presidio | 16661 | 0 | - |
| sct-default | 16661 | 0 | - |
| sct-gliner | 16661 | 0 | - |

Unplaced spans (the entry detected something the harness could not place in the source; scored as a
wrong prediction and the gold as missed): only the two SCT entries have them, because SCT returns
rewritten text and its spans are recovered through reversible placeholders (`unplaced_spans` in each
per-dataset file). The recovery rule above is a separate case: copies with no span at all.

## Where SCT stands

There is no clean evidence for or against SCT in round 1, so no win or loss is claimed. The excluded
cells (in brackets in the per-dataset tables) are development signals only, to be measured again on
fresh data: exact spans and types (strict F1) look weak where long digit IDs are typed PHONE
instead of NUMBER; NUMBER/PHONE precision is low (the regexes mask most digit runs, recall-first);
ORGANISATION recall on gretel finance and NER recall on naamapadam (no Indic NER model in SCT) trail
gliner2-fastino. Speed and memory are not a contamination question (FineWeb-2 speed sample): see the
docs/s and peak RAM columns; SCT runs Lingua LID, cleaning and an ensemble of fp32 ONNX NER models
with a session per language.

## Limits of this comparison

- **Round 1 cannot rank SCT.** Its 0.8.1 fixes were developed on 12 of the 14 datasets (D) and the
  other two belong to those datasets' generator families (D-family). Round 2 needs fresh datasets
  that no entry, SCT included, and no generator family of them has seen.
- **The tuned track saw every privacy dev part.** One configuration per entry, chosen on the dev
  parts of all nine privacy sets; the default track is primary.
- **No GPU numbers.** The pinned torch 2.15.0 wheels need CUDA 13.2; the box's driver is 12.6, so
  every entry ran on CPU (gliner-nvidia fp32).
- **Contamination is from cards, not data.** T/E come from model cards; opf and the knowledgator
  model do not disclose training data (U). The Gretel family includes nemotron (NeMo Data Designer,
  Navigator's successor); this changes no cell's status (every entry flagged on one of them is
  already flagged on the others).
- **Claimed languages from cards.** "Primarily English" (opf) is scored as en only. sct-default
  and datafog are held to every language.
- **SCT is scored through its reversible view.** Its spans come from placeholders; unplaced spans
  still count against SCT, and the recovery rule is conservative (any visible trace blocks it).
- **Narrow shared view.** EMAIL, PHONE, NUMBER only; read it with the full view.
- **Gold quality.** gretel finance (LLM + GLiNER), gretel_en (generator strings placed at every
  exact occurrence), argilla_pii (unreviewed suggestions), nemotron (1.5% span mismatches). The
  blind annotation audit (rule 10) was not done, so precision is a lower bound on these sets.
- **Template leakage.** openpii, argilla_pii and ai4privacy-derived sets have no template id: the
  dev/test split is by record.
- **Not run**: the native language track, the common chunker, matched operating points, PR and
  leak-vs-over-masking plots, GPU repeats, downstream utility beyond chrF.
- **One box, one seed.** Bootstrap CIs cover document sampling only.

## Published numbers and scope notes

- OPF's published 96% F1 is per token; its per-span F1 is 0.926 (0.942 on the corrected set).
  Our scores are span or mention level; compare against the span number only.
- Presidio 2.2.366 (latest at the time of the run) is the version scored here.
- NeMo Curator 1.x has no PII stage (the 0.x one was English-only Presidio); it is not an entry.
- Not entries here: **deduce** is GPL-3.0 (checked in its LICENSE file): benchmark use only, no
  code reuse. **StarPII** is gated under custom terms: evaluation only.

