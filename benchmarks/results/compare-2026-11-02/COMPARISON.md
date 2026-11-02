# SCT vs other PII tools: comparison, 2026-11-02 (rules applied 2026-11-03)

SqueakyCleanText 0.8.1 (`sct-default`, and its optional GLiNER path `sct-gliner`) against six
other offline PII tools, under the frozen protocol in `benchmarks/compare/PROTOCOL.md`
(Codex's 20 fairness rules; section 20 maps each rule). Numbers come from this bundle
(`default/`, `tuned/`, `norecover/`, `speed-*/`); every table can be rebuilt with
`python -m benchmarks.compare.report benchmarks/results/compare-2026-11-02` (see `TABLES.md`).

- Entries (exact pins in `PROTOCOL.md` section 2): sct-default, sct-gliner, presidio 2.2.366
  (en_core_web_lg), gliner-nvidia (nvidia/gliner-PII), gliner2-fastino
  (fastino/gliner2-privacy-filter-PII-multi), opf (openai/privacy-filter), bardsai-v3
  (bardsai/eu-pii-anonimization-multilang-v3, INT8 ONNX), datafog 4.9.0 (regex).
- Data: test part (hash split by template/record, seed 13), 300 docs per language
  (TAB 106, REDACT 825), 14 datasets, 16,661 docs per entry. Privacy-synthetic: openpii, gretel
  (finance), gretel_en, nemotron, argilla_pii, thai_pii, bayanat. Privacy-real: TAB, REDACT.
  NER (separate, not privacy): iahlt, thainer, naamapadam, multiconer, stockmark.
- Tracks: **default** = vendor quick start; **tuned** = one configuration per entry picked on the
  dev part from at most 12 candidates, frozen in `benchmarks/compare/configs/tuned.json` before the
  test run. Gold-language track. All quality runs on CPU. One mask operation for every entry.

## What changed on 2026-11-03

The round-1 numbers were rescored from the same raw outputs (no new model runs) under two rules
that are now permanent for every round (PROTOCOL.md sections 12 and 17):

1. **Contamination (decision 21).** If a tool, or any model inside it, was trained on a dataset
   or its family (T), or was developed or tuned against it (E: the vendor evaluates on it; D: SCT
   developed against it), that cell is **excluded** from that tool's comparison. SCT is held to
   the same rule. Unknown training data (U: opf, the knowledgator model in sct-gliner) is shown
   separately as "unknown", never as clean. The round-1 report left only T/E out and kept D and U.
2. **Claimed languages.** Each tool is scored on the languages it claims (model card tags:
   gliner-nvidia en, bardsai-v3 the 24 EU languages, gliner2-fastino en fr es de it pt nl, opf en,
   presidio en (only `en_core_web_lg` is loaded), sct-gliner en; SCT and datafog make no language
   claim). Scores in other languages are shown apart, as "outside claimed languages", never as
   plain losses.

Also: a scoring artefact on SCT was fixed (markdown `mailto:` links, see "Gretel e-mail
artefact"); it changes only cells that are now excluded for SCT, and SCT is shown with and
without it. Published-number notes are at the end.

**Bottom line.** Under the strict rule SCT has clean cells on only two datasets, gretel_en and
argilla_pii, both English and synthetic (600 docs). It has no clean cell on real data (TAB,
REDACT) or on any NER set, because its 0.8.1 fixes were developed against them (D). The
round-1 claims that SCT leads on real-data recall and on most NER sets rest only on D cells and
are withdrawn. **Round 2 needs fresh datasets that no entry, SCT included, has seen.**

How to read the tables:

- Scores are the group macro: mean over datasets of each dataset's language macro, failed docs
  counted unmasked, DATE out. Synthetic and real data are never pooled.
- mR = type-agnostic masked recall; mP = masked precision; strict F1 = exact span and type;
  doc leak = share of docs with a gold mention that keeps any letter or digit unmasked;
  over-mask = masked characters outside every gold span / all non-gold characters.
- Own tables: the entry on its own clean datasets, only docs in its claimed languages, with a 95%
  bootstrap CI (1,000 resamples of templates/docs within each dataset, seed 13). The first
  column lists the clean datasets that have docs in a claimed language (in brackets: all clean
  datasets). Next to it, for information: masked recall on clean datasets outside the claimed
  languages, on unknown (U) datasets, and (in brackets) on excluded datasets, each with the
  number of datasets.
- Pairwise tables: SCT vs one entry on the datasets clean for both, in languages both claim;
  each cell is the entry's score, then the 95% paired bootstrap CI of (entry - SCT); `*` = the CI
  excludes 0. Higher is better for mR, mP, strict F1; lower for doc leak and over-mask.
- Full view = all 9 SCT types (a type the tool does not support counts as missed). Shared view =
  EMAIL, PHONE, NUMBER (types every entry supports; narrow). Speed: CPU, 16 threads, batch 8,
  100 FineWeb-2 docs, median of 3; peak RAM = adapter VmHWM. No GPU numbers (pinned torch wheels
  need CUDA 13.2; the box's driver is 12.6).

## Headline, tuned track: each tool on its own clean sets, claimed languages

### Full view (all 9 types), privacy-synthetic

| entry | clean sets with claimed-language docs (all clean sets) | mR [95% CI] | mP [95% CI] | strict F1 [95% CI] | doc leak [95% CI] | over-mask [95% CI] | mR clean, outside claimed langs | mR unknown (U) sets | (mR excluded sets, info only) | docs/s CPU | peak RAM | claimed languages | licence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sct-default | argilla_pii, gretel_en (2) | 0.973 [0.962, 0.982] | 0.867 [0.846, 0.889] | 0.570 [0.541, 0.597] | 0.162 [0.133, 0.194] | 0.021 [0.017, 0.026] | - | - | (0.921 (5)) | 1.50 | 7,692 MB | no claim (all) | MIT |
| gliner2-fastino | argilla_pii, gretel, gretel_en, nemotron, openpii (7) | 0.897 [0.886, 0.907] | 0.702 [0.682, 0.721] | 0.680 [0.665, 0.696] | 0.285 [0.265, 0.304] | 0.046 [0.042, 0.050] | 0.880 (4) | - | - | 0.52 | 6,832 MB | en fr es de it pt nl | Apache-2.0 |
| gliner-nvidia | gretel (3) | 0.853 [0.779, 0.910] | 0.641 [0.579, 0.721] | 0.317 [0.269, 0.372] | 0.436 [0.350, 0.511] | 0.025 [0.018, 0.034] | 0.759 (3) | - | (0.951 (4)) | 0.15 | 25,732 MB | en | Apache-2.0 + NVIDIA Open Model |
| presidio | argilla_pii, gretel, gretel_en, nemotron, openpii (7) | 0.762 [0.742, 0.781] | 0.803 [0.773, 0.838] | 0.540 [0.521, 0.559] | 0.608 [0.584, 0.633] | 0.023 [0.019, 0.029] | 0.605 (4) | - | - | 27 | 826 MB | en | MIT |
| bardsai-v3 | gretel (3) | 0.649 [0.597, 0.697] | 0.714 [0.651, 0.776] | 0.559 [0.512, 0.599] | 0.638 [0.578, 0.693] | 0.015 [0.011, 0.021] | 0.895 (2) | - | (0.987 (4)) | 2.52 | 1,738 MB | 24 EU | Apache-2.0 |
| datafog | argilla_pii, bayanat, gretel, gretel_en, nemotron, openpii, thai_pii (7) | 0.337 [0.325, 0.348] | 0.854 [0.833, 0.878] | 0.342 [0.328, 0.355] | 0.919 [0.908, 0.929] | 0.009 [0.007, 0.010] | - | - | - | 1,787 | 30 MB | no claim (all) | MIT |
| sct-gliner | none (0) | - | - | - | - | - | - | 0.984 (2) | (0.919 (3)) | 0.53 | 6,414 MB | en | MIT + Apache-2.0 model |
| opf | none (0) | - | - | - | - | - | - | 0.761 (5) | - | 0.29 | 26,113 MB | en | Apache-2.0 |

### Full view, privacy-real (TAB, REDACT)

| entry | clean sets with claimed-language docs (all clean sets) | mR [95% CI] | mP [95% CI] | strict F1 [95% CI] | doc leak [95% CI] | over-mask [95% CI] | mR clean, outside claimed langs | mR unknown (U) sets | (mR excluded sets, info only) | docs/s CPU | peak RAM | claimed languages | licence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gliner2-fastino | redact, tab (2) | 0.904 [0.884, 0.922] | 0.542 [0.522, 0.562] | 0.516 [0.499, 0.534] | 0.833 [0.807, 0.858] | 0.071 [0.068, 0.075] | 0.884 (1) | - | - | 0.52 | 6,832 MB | en fr es de it pt nl | Apache-2.0 |
| bardsai-v3 | redact, tab (2) | 0.739 [0.714, 0.764] | 0.817 [0.804, 0.831] | 0.512 [0.495, 0.529] | 0.830 [0.806, 0.852] | 0.022 [0.020, 0.024] | 0.804 (1) | - | - | 2.52 | 1,738 MB | 24 EU | Apache-2.0 |
| presidio | redact, tab (2) | 0.612 [0.576, 0.648] | 0.651 [0.605, 0.697] | 0.319 [0.285, 0.349] | 0.922 [0.850, 0.984] | 0.045 [0.034, 0.057] | 0.620 (1) | - | - | 27 | 826 MB | en | MIT |
| gliner-nvidia | redact, tab (2) | 0.439 [0.382, 0.506] | 0.821 [0.783, 0.856] | 0.164 [0.134, 0.198] | 0.969 [0.922, 1.000] | 0.017 [0.012, 0.023] | 0.600 (1) | - | - | 0.15 | 25,732 MB | en | Apache-2.0 + NVIDIA Open Model |
| datafog | redact, tab (2) | 0.161 [0.144, 0.182] | 0.765 [0.751, 0.790] | 0.155 [0.147, 0.162] | 0.997 [0.994, 1.000] | 0.008 [0.007, 0.009] | - | - | - | 1,787 | 30 MB | no claim (all) | MIT |
| sct-gliner | none (0) | - | - | - | - | - | - | - | (0.840 (2)) | 0.53 | 6,414 MB | en | MIT + Apache-2.0 model |
| sct-default | none (0) | - | - | - | - | - | - | - | (0.962 (2)) | 1.50 | 7,692 MB | no claim (all) | MIT |
| opf | none (0) | - | - | - | - | - | - | 0.523 (2) | - | 0.29 | 26,113 MB | en | Apache-2.0 |

### Shared view (EMAIL, PHONE, NUMBER), privacy-synthetic

| entry | clean sets with claimed-language docs (all clean sets) | mR [95% CI] | mP [95% CI] | strict F1 [95% CI] | doc leak [95% CI] | over-mask [95% CI] | mR clean, outside claimed langs | mR unknown (U) sets | (mR excluded sets, info only) | docs/s CPU | peak RAM | claimed languages | licence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sct-default | argilla_pii, gretel_en (2) | 0.983 [0.973, 0.993] | 0.738 [0.694, 0.778] | 0.410 [0.356, 0.463] | 0.084 [0.057, 0.111] | 0.021 [0.017, 0.026] | - | - | (0.956 (5)) | 1.50 | 7,692 MB | no claim (all) | MIT |
| gliner-nvidia | gretel (3) | 0.949 [0.919, 0.979] | 0.206 [0.146, 0.257] | 0.595 [0.515, 0.656] | 0.237 [0.138, 0.349] | 0.025 [0.018, 0.034] | 0.887 (3) | - | (0.979 (4)) | 0.15 | 25,732 MB | en | Apache-2.0 + NVIDIA Open Model |
| gliner2-fastino | argilla_pii, gretel, gretel_en, nemotron, openpii (7) | 0.939 [0.925, 0.951] | 0.477 [0.455, 0.500] | 0.817 [0.781, 0.852] | 0.110 [0.094, 0.129] | 0.046 [0.042, 0.050] | 0.937 (4) | - | - | 0.52 | 6,832 MB | en fr es de it pt nl | Apache-2.0 |
| bardsai-v3 | gretel (3) | 0.918 [0.892, 0.940] | 0.292 [0.226, 0.379] | 0.780 [0.720, 0.827] | 0.141 [0.108, 0.175] | 0.015 [0.011, 0.021] | 0.985 (2) | - | (0.993 (4)) | 2.52 | 1,738 MB | 24 EU | Apache-2.0 |
| presidio | argilla_pii, gretel, gretel_en, nemotron, openpii (7) | 0.854 [0.831, 0.873] | 0.682 [0.650, 0.721] | 0.698 [0.656, 0.752] | 0.318 [0.289, 0.349] | 0.023 [0.019, 0.029] | 0.821 (4) | - | - | 27 | 826 MB | en | MIT |
| datafog | argilla_pii, bayanat, gretel, gretel_en, nemotron, openpii, thai_pii (7) | 0.678 [0.656, 0.700] | 0.828 [0.808, 0.852] | 0.597 [0.572, 0.623] | 0.507 [0.482, 0.532] | 0.009 [0.007, 0.010] | - | - | - | 1,787 | 30 MB | no claim (all) | MIT |
| sct-gliner | none (0) | - | - | - | - | - | - | 0.983 (2) | (0.927 (3)) | 0.53 | 6,414 MB | en | MIT + Apache-2.0 model |
| opf | none (0) | - | - | - | - | - | - | 0.900 (5) | - | 0.29 | 26,113 MB | en | Apache-2.0 |

### Shared view, privacy-real

| entry | clean sets with claimed-language docs (all clean sets) | mR [95% CI] | mP [95% CI] | strict F1 [95% CI] | doc leak [95% CI] | over-mask [95% CI] | mR clean, outside claimed langs | mR unknown (U) sets | (mR excluded sets, info only) | docs/s CPU | peak RAM | claimed languages | licence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| datafog | redact, tab (2) | 0.768 [0.648, 0.846] | 0.750 [0.736, 0.775] | 0.351 [0.341, 0.360] | 0.812 [0.795, 0.830] | 0.008 [0.007, 0.009] | - | - | - | 1,787 | 30 MB | no claim (all) | MIT |
| gliner2-fastino | redact, tab (2) | 0.533 [0.482, 0.581] | 0.147 [0.129, 0.166] | 0.469 [0.421, 0.528] | 0.587 [0.564, 0.611] | 0.071 [0.068, 0.075] | 0.933 (1) | - | - | 0.52 | 6,832 MB | en fr es de it pt nl | Apache-2.0 |
| bardsai-v3 | redact, tab (2) | 0.472 [0.464, 0.479] | 0.198 [0.180, 0.220] | 0.433 [0.421, 0.447] | 0.621 [0.601, 0.643] | 0.022 [0.020, 0.024] | 0.864 (1) | - | - | 2.52 | 1,738 MB | 24 EU | Apache-2.0 |
| presidio | redact, tab (2) | 0.466 [0.429, 0.498] | 0.155 [0.112, 0.204] | 0.290 [0.234, 0.354] | 0.714 [0.625, 0.803] | 0.045 [0.034, 0.057] | 0.852 (1) | - | - | 27 | 826 MB | en | MIT |
| gliner-nvidia | redact, tab (2) | 0.275 [0.194, 0.365] | 0.336 [0.250, 0.441] | 0.272 [0.192, 0.350] | 0.882 [0.805, 0.948] | 0.017 [0.012, 0.023] | 0.575 (1) | - | - | 0.15 | 25,732 MB | en | Apache-2.0 + NVIDIA Open Model |
| sct-gliner | none (0) | - | - | - | - | - | - | - | (0.983 (2)) | 0.53 | 6,414 MB | en | MIT + Apache-2.0 model |
| sct-default | none (0) | - | - | - | - | - | - | - | (0.985 (2)) | 1.50 | 7,692 MB | no claim (all) | MIT |
| opf | none (0) | - | - | - | - | - | - | 0.541 (2) | - | 0.29 | 26,113 MB | en | Apache-2.0 |

The NER group (iahlt, thainer, naamapadam, multiconer, stockmark) is in `TABLES.md`: SCT has no
clean NER cell, and no model entry claims any NER language (ar, th, bn, hi, ta, zh, ja), so for
them the NER tables only show "outside claimed languages" values.

Own-table numbers are not a ranking: each row is on its own datasets and languages (bardsai-v3
and gliner-nvidia rest on gretel alone in their claimed languages; SCT on gretel_en and
argilla_pii). Compare entries only through the same-set and pairwise tables below.

## Same-set tables

### Every entry

Full view, privacy-synthetic: No dataset in this group is clean for every one of: sct-default, sct-gliner, presidio, gliner-nvidia, gliner2-fastino, opf, bardsai-v3, datafog (in languages all of them claim).

Full view, privacy-real: No dataset in this group is clean for every one of: sct-default, sct-gliner, presidio, gliner-nvidia, gliner2-fastino, opf, bardsai-v3, datafog (in languages all of them claim).

There is no dataset clean for all eight entries: opf and sct-gliner have no clean cell (U), SCT's
only clean sets (gretel_en, argilla_pii) are excluded for gliner-nvidia (E) and bardsai-v3 (T),
and every other set is D for SCT. So the comparison with SCT is pairwise (next section).

### The other entries with clean sets (no SCT entry), tuned, full view

For information about the other tools only; SCT cannot be on these sets.

Privacy-synthetic:

Datasets: gretel (clean for all: gretel, thai_pii, bayanat); only docs in languages every entry claims; point values.

| entry | mR | mP | strict F1 | doc leak | over-mask |
|---|---|---|---|---|---|
| gliner2-fastino | 0.943 | 0.357 | 0.476 | 0.237 | 0.062 |
| gliner-nvidia | 0.853 | 0.641 | 0.317 | 0.436 | 0.025 |
| bardsai-v3 | 0.675 | 0.661 | 0.547 | 0.611 | 0.012 |
| presidio | 0.561 | 0.493 | 0.301 | 0.926 | 0.031 |
| datafog | 0.218 | 0.486 | 0.207 | 0.996 | 0.006 |

Privacy-real:

Datasets: redact, tab (clean for all: tab, redact); only docs in languages every entry claims; point values.

| entry | mR | mP | strict F1 | doc leak | over-mask |
|---|---|---|---|---|---|
| gliner2-fastino | 0.914 | 0.490 | 0.461 | 0.812 | 0.080 |
| bardsai-v3 | 0.713 | 0.758 | 0.479 | 0.859 | 0.028 |
| presidio | 0.612 | 0.651 | 0.319 | 0.922 | 0.045 |
| gliner-nvidia | 0.439 | 0.821 | 0.164 | 0.969 | 0.017 |
| datafog | 0.163 | 0.756 | 0.143 | 1.000 | 0.007 |

## SCT vs each entry (pairwise same-set), tuned track

### Full view, privacy-synthetic, datasets clean for both

| entry | sets | mR (entry - SCT, 95% CI) | mP (entry - SCT, 95% CI) | strict F1 (entry - SCT, 95% CI) | doc leak (entry - SCT, 95% CI) | over-mask (entry - SCT, 95% CI) | SCT mR, same docs | SCT mP, same docs |
|---|---|---|---|---|---|---|---|---|
| gliner2-fastino | gretel_en, argilla_pii | 0.929 (-0.062, -0.021)* | 0.733 (-0.166, -0.100)* | 0.696 (+0.092, +0.163)* | 0.167 (-0.039, +0.046) | 0.053 (+0.025, +0.039)* | 0.973 | 0.867 |
| presidio | gretel_en, argilla_pii | 0.870 (-0.124, -0.082)* | 0.886 (-0.005, +0.041) | 0.628 (+0.023, +0.094)* | 0.375 (+0.175, +0.255)* | 0.022 (-0.003, +0.005) | 0.973 | 0.867 |
| datafog | gretel_en, argilla_pii | 0.318 (-0.682, -0.628)* | 0.928 (+0.034, +0.086)* | 0.430 (-0.178, -0.102)* | 0.852 (+0.653, +0.729)* | 0.004 (-0.021, -0.013)* | 0.973 | 0.867 |

No clean dataset in common with SCT: sct-gliner, opf, gliner-nvidia, bardsai-v3.

### Shared view, privacy-synthetic, datasets clean for both

| entry | sets | mR (entry - SCT, 95% CI) | mP (entry - SCT, 95% CI) | strict F1 (entry - SCT, 95% CI) | doc leak (entry - SCT, 95% CI) | over-mask (entry - SCT, 95% CI) | SCT mR, same docs | SCT mP, same docs |
|---|---|---|---|---|---|---|---|---|
| gliner2-fastino | gretel_en, argilla_pii | 0.948 (-0.062, -0.007)* | 0.498 (-0.285, -0.187)* | 0.868 (+0.401, +0.515)* | 0.066 (-0.061, +0.025) | 0.053 (+0.025, +0.039)* | 0.983 | 0.738 |
| presidio | gretel_en, argilla_pii | 0.911 (-0.104, -0.045)* | 0.778 (-0.006, +0.089) | 0.818 (+0.350, +0.466)* | 0.192 (+0.068, +0.154)* | 0.022 (-0.003, +0.005) | 0.983 | 0.738 |
| datafog | gretel_en, argilla_pii | 0.567 (-0.475, -0.358)* | 0.907 (+0.126, +0.211)* | 0.683 (+0.203, +0.342)* | 0.507 (+0.360, +0.487)* | 0.004 (-0.021, -0.013)* | 0.983 | 0.738 |

No clean dataset in common with SCT: sct-gliner, opf, gliner-nvidia, bardsai-v3.

### Unknown training data (shown apart, not clean): full view, privacy-synthetic

| entry | sets | mR (entry - SCT, 95% CI) | mP (entry - SCT, 95% CI) | strict F1 (entry - SCT, 95% CI) | doc leak (entry - SCT, 95% CI) | over-mask (entry - SCT, 95% CI) | SCT mR, same docs | SCT mP, same docs |
|---|---|---|---|---|---|---|---|---|
| sct-gliner | gretel_en, argilla_pii | 0.984 (+0.001, +0.023)* | 0.697 (-0.193, -0.146)* | 0.396 (-0.196, -0.152)* | 0.171 (-0.025, +0.041) | 0.049 (+0.022, +0.032)* | 0.973 | 0.867 |
| opf | gretel_en, argilla_pii | 0.846 (-0.154, -0.098)* | 0.908 (+0.018, +0.064)* | 0.597 (-0.018, +0.070) | 0.231 (+0.023, +0.114)* | 0.019 (-0.006, +0.001) | 0.973 | 0.867 |

No unknown dataset in common with SCT: presidio, gliner2-fastino, gliner-nvidia, datafog, bardsai-v3.

### Unknown training data: shared view, privacy-synthetic

| entry | sets | mR (entry - SCT, 95% CI) | mP (entry - SCT, 95% CI) | strict F1 (entry - SCT, 95% CI) | doc leak (entry - SCT, 95% CI) | over-mask (entry - SCT, 95% CI) | SCT mR, same docs | SCT mP, same docs |
|---|---|---|---|---|---|---|---|---|
| sct-gliner | gretel_en, argilla_pii | 0.983 (-0.000, -0.000) | 0.426 (-0.348, -0.276)* | 0.401 (-0.020, +0.001) | 0.079 (-0.013, -0.000) | 0.049 (+0.022, +0.032)* | 0.983 | 0.738 |
| opf | gretel_en, argilla_pii | 0.930 (-0.088, -0.019)* | 0.806 (+0.015, +0.119)* | 0.820 (+0.348, +0.468)* | 0.083 (-0.047, +0.043) | 0.019 (-0.006, +0.001) | 0.983 | 0.738 |

No unknown dataset in common with SCT: presidio, gliner2-fastino, gliner-nvidia, datafog, bardsai-v3.

There is no pairwise table on real data or NER: no dataset there is clean for SCT.

## Default track (vendor quick start)

### Own clean sets, full view, privacy-synthetic

| entry | clean sets with claimed-language docs (all clean sets) | mR [95% CI] | mP [95% CI] | strict F1 [95% CI] | doc leak [95% CI] | over-mask [95% CI] | mR clean, outside claimed langs | mR unknown (U) sets | (mR excluded sets, info only) | docs/s CPU | peak RAM | claimed languages | licence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sct-default | argilla_pii, gretel_en (2) | 0.973 [0.963, 0.983] | 0.869 [0.847, 0.891] | 0.573 [0.544, 0.599] | 0.164 [0.134, 0.197] | 0.021 [0.017, 0.026] | - | - | (0.920 (5)) | 1.50 | 7,692 MB | no claim (all) | MIT |
| gliner2-fastino | argilla_pii, gretel, gretel_en, nemotron, openpii (7) | 0.872 [0.861, 0.882] | 0.746 [0.727, 0.765] | 0.702 [0.688, 0.717] | 0.335 [0.315, 0.355] | 0.036 [0.033, 0.040] | 0.849 (4) | - | - | 0.52 | 6,832 MB | en fr es de it pt nl | Apache-2.0 |
| gliner-nvidia | gretel (3) | 0.850 [0.789, 0.903] | 0.648 [0.586, 0.720] | 0.521 [0.470, 0.568] | 0.564 [0.477, 0.644] | 0.020 [0.015, 0.028] | 0.614 (3) | - | (0.899 (4)) | 0.15 | 25,732 MB | en | Apache-2.0 + NVIDIA Open Model |
| presidio | argilla_pii, gretel, gretel_en, nemotron, openpii (7) | 0.762 [0.742, 0.781] | 0.803 [0.773, 0.838] | 0.540 [0.521, 0.559] | 0.608 [0.584, 0.633] | 0.023 [0.019, 0.029] | 0.605 (4) | - | - | 27 | 826 MB | en | MIT |
| bardsai-v3 | gretel (3) | 0.649 [0.597, 0.697] | 0.714 [0.651, 0.776] | 0.559 [0.512, 0.599] | 0.638 [0.578, 0.693] | 0.015 [0.011, 0.021] | 0.895 (2) | - | (0.987 (4)) | 2.52 | 1,738 MB | 24 EU | Apache-2.0 |
| datafog | argilla_pii, bayanat, gretel, gretel_en, nemotron, openpii, thai_pii (7) | 0.337 [0.325, 0.348] | 0.854 [0.833, 0.878] | 0.342 [0.328, 0.355] | 0.919 [0.908, 0.929] | 0.009 [0.007, 0.010] | - | - | - | 1,787 | 30 MB | no claim (all) | MIT |
| sct-gliner | none (0) | - | - | - | - | - | - | 0.979 (2) | (0.900 (3)) | 0.53 | 6,414 MB | en | MIT + Apache-2.0 model |
| opf | none (0) | - | - | - | - | - | - | 0.722 (5) | - | 0.29 | 26,113 MB | en | Apache-2.0 |

### Own clean sets, full view, privacy-real

| entry | clean sets with claimed-language docs (all clean sets) | mR [95% CI] | mP [95% CI] | strict F1 [95% CI] | doc leak [95% CI] | over-mask [95% CI] | mR clean, outside claimed langs | mR unknown (U) sets | (mR excluded sets, info only) | docs/s CPU | peak RAM | claimed languages | licence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gliner2-fastino | redact, tab (2) | 0.880 [0.858, 0.900] | 0.590 [0.570, 0.611] | 0.544 [0.526, 0.562] | 0.865 [0.839, 0.890] | 0.056 [0.053, 0.059] | 0.849 (1) | - | - | 0.52 | 6,832 MB | en fr es de it pt nl | Apache-2.0 |
| bardsai-v3 | redact, tab (2) | 0.739 [0.714, 0.764] | 0.817 [0.804, 0.831] | 0.512 [0.495, 0.529] | 0.830 [0.806, 0.852] | 0.022 [0.020, 0.024] | 0.804 (1) | - | - | 2.52 | 1,738 MB | 24 EU | Apache-2.0 |
| presidio | redact, tab (2) | 0.612 [0.576, 0.648] | 0.651 [0.605, 0.697] | 0.319 [0.285, 0.349] | 0.922 [0.850, 0.984] | 0.045 [0.034, 0.057] | 0.620 (1) | - | - | 27 | 826 MB | en | MIT |
| gliner-nvidia | redact, tab (2) | 0.391 [0.335, 0.462] | 0.793 [0.752, 0.828] | 0.288 [0.245, 0.333] | 0.953 [0.891, 1.000] | 0.016 [0.012, 0.022] | 0.544 (1) | - | - | 0.15 | 25,732 MB | en | Apache-2.0 + NVIDIA Open Model |
| datafog | redact, tab (2) | 0.161 [0.144, 0.182] | 0.765 [0.751, 0.790] | 0.155 [0.147, 0.162] | 0.997 [0.994, 1.000] | 0.008 [0.007, 0.009] | - | - | - | 1,787 | 30 MB | no claim (all) | MIT |
| sct-gliner | none (0) | - | - | - | - | - | - | - | (0.770 (2)) | 0.53 | 6,414 MB | en | MIT + Apache-2.0 model |
| sct-default | none (0) | - | - | - | - | - | - | - | (0.960 (2)) | 1.50 | 7,692 MB | no claim (all) | MIT |
| opf | none (0) | - | - | - | - | - | - | 0.458 (2) | - | 0.29 | 26,113 MB | en | Apache-2.0 |

### SCT vs each entry, full view, privacy-synthetic, clean for both

| entry | sets | mR (entry - SCT, 95% CI) | mP (entry - SCT, 95% CI) | strict F1 (entry - SCT, 95% CI) | doc leak (entry - SCT, 95% CI) | over-mask (entry - SCT, 95% CI) | SCT mR, same docs | SCT mP, same docs |
|---|---|---|---|---|---|---|---|---|
| gliner2-fastino | gretel_en, argilla_pii | 0.914 (-0.079, -0.036)* | 0.772 (-0.127, -0.064)* | 0.717 (+0.108, +0.179)* | 0.198 (-0.013, +0.077) | 0.044 (+0.016, +0.029)* | 0.973 | 0.869 |
| presidio | gretel_en, argilla_pii | 0.870 (-0.124, -0.083)* | 0.886 (-0.006, +0.040) | 0.628 (+0.021, +0.090)* | 0.375 (+0.173, +0.253)* | 0.022 (-0.004, +0.005) | 0.973 | 0.869 |
| datafog | gretel_en, argilla_pii | 0.318 (-0.683, -0.630)* | 0.928 (+0.033, +0.085)* | 0.430 (-0.181, -0.105)* | 0.852 (+0.652, +0.727)* | 0.004 (-0.021, -0.013)* | 0.973 | 0.869 |

No clean dataset in common with SCT: sct-gliner, opf, gliner-nvidia, bardsai-v3.

### Unknown training data, full view, privacy-synthetic

| entry | sets | mR (entry - SCT, 95% CI) | mP (entry - SCT, 95% CI) | strict F1 (entry - SCT, 95% CI) | doc leak (entry - SCT, 95% CI) | over-mask (entry - SCT, 95% CI) | SCT mR, same docs | SCT mP, same docs |
|---|---|---|---|---|---|---|---|---|
| sct-gliner | gretel_en, argilla_pii | 0.979 (-0.006, +0.018) | 0.782 (-0.107, -0.067)* | 0.432 (-0.163, -0.119)* | 0.202 (+0.006, +0.070)* | 0.032 (+0.007, +0.014)* | 0.973 | 0.869 |
| opf | gretel_en, argilla_pii | 0.807 (-0.198, -0.136)* | 0.928 (+0.038, +0.082)* | 0.592 (-0.028, +0.063) | 0.275 (+0.064, +0.160)* | 0.015 (-0.010, -0.002)* | 0.973 | 0.869 |

No unknown dataset in common with SCT: presidio, gliner2-fastino, gliner-nvidia, datafog, bardsai-v3.

## What changed for SCT under the strict rule

- **Datasets.** Round-1 SCT headline (T/E rule): 7 synthetic sets (tuned full-view mR 0.927),
  2 real sets (0.933) and all 5 NER sets. Now: 2 synthetic sets, English only (gretel_en,
  argilla_pii: tuned full-view mR 0.973 [0.962, 0.982]); real data and NER: none. sct-gliner: no
  clean cell at all (U and D); it appears only under "unknown" (gretel_en, argilla_pii).
- **Claims withdrawn.** "SCT leads full-view masked recall on both privacy groups", "real-data
  recall 0.933 vs next best 0.895" and "SCT leads every NER set except naamapadam" rest on D cells
  and are no longer made. The SCT D cells stay in the per-dataset table in brackets, for
  information.
- **What holds on clean cells** (gretel_en + argilla_pii, English, tuned, full view): SCT's masked
  recall is higher than gliner2-fastino's (-0.062 to -0.021 for fastino), presidio's, datafog's and
  opf's (unknown, -0.154 to -0.098), all CIs excluding 0; sct-gliner (unknown) is slightly
  higher in the full view (+0.001 to +0.023) and not different in the shared view. SCT's strict F1 is lower than gliner2-fastino's (+0.092 to +0.163) and
  presidio's (+0.023 to +0.094). Doc leak: not different from gliner2-fastino, lower than presidio
  and opf. Masked precision: higher than gliner2-fastino's, not different from presidio's, lower
  than datafog's and opf's. Over-masking: lower than gliner2-fastino and sct-gliner, higher than datafog.
- **No clean comparison with gliner-nvidia or bardsai-v3.** The two sets clean for SCT are the
  sets those two were evaluated or trained on; the sets clean for them are D for SCT.
- **Thin evidence.** Two English synthetic sets, 300 docs each; argilla_pii is a slice of the
  ai4privacy-200k train split. This is not enough to rank SCT. Round 2 must use fresh datasets,
  checked against every entry's training and evaluation data and against SCT's own development
  sets, with real (not synthetic) documents and non-English languages.

### SCT with and without the e-mail scoring fix, tuned, full view, privacy-synthetic

| entry | clean sets with claimed-language docs (all clean sets) | mR [95% CI] | mP [95% CI] | strict F1 [95% CI] | doc leak [95% CI] | over-mask [95% CI] | mR clean, outside claimed langs | mR unknown (U) sets | (mR excluded sets, info only) | docs/s CPU | peak RAM | claimed languages | licence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| with recovery (this report): sct-default | argilla_pii, gretel_en (2) | 0.973 [0.962, 0.982] | 0.867 [0.846, 0.889] | 0.570 [0.541, 0.597] | 0.162 [0.133, 0.194] | 0.021 [0.017, 0.026] | - | - | (0.921 (5)) | 1.50 | 7,692 MB | no claim (all) | MIT |
| with recovery (this report): sct-gliner | none (0) | - | - | - | - | - | - | 0.984 (2) | (0.919 (3)) | 0.53 | 6,414 MB | en | MIT + Apache-2.0 model |
| without (round-1 mapping): sct-default | argilla_pii, gretel_en (2) | 0.973 [0.962, 0.982] | 0.867 [0.846, 0.889] | 0.570 [0.541, 0.597] | 0.162 [0.133, 0.194] | 0.021 [0.017, 0.026] | - | - | (0.908 (5)) | 1.50 | 7,692 MB | no claim (all) | MIT |
| without (round-1 mapping): sct-gliner | none (0) | - | - | - | - | - | - | 0.984 (2) | (0.883 (3)) | 0.53 | 6,414 MB | en | MIT + Apache-2.0 model |

## Gretel e-mail artefact (scoring fix)

Cause, checked on the server and reproduced locally: for `E-post: [a@b.se](mailto:a@b.se)` SCT's
reversible map holds `<EMAIL_0>` (link text) and `<EMAIL_1>` (the URL copy), but SCT's output is
`E-post: (mailto:<EMAIL_1>)`: a later cleaning step removes the bracketed link text with its
placeholder. The harness places spans from the placeholders found in the output, so the link-text
copy got no span and was scored as leaked although no character of it is visible. This was a
harness bug (offset mapping), not an SCT miss.

Fix: `scorer.recover_dropped` (tested in `tests/test_benchmarks_compare.py`), applied to the two
SCT entries: a source copy of a value SCT masked elsewhere in the doc counts as masked when
neither the value nor any 3+ char word of it appears in SCT's output (so no copy can be visible,
even in part) and the copy is a whole token. Recovered copies also count as predictions, so
precision can drop. The fix also credits other removed copies of detected values (e.g. repeated
names in REDACT tickets); these were not classified one by one. `runner.py --no-recover` gives the
round-1 mapping (`norecover/` in this bundle; full and supported views identical to round 1).

Effect (tuned sct-default): gretel e-mail masked recall by type (score.py) 0.673 -> 0.877;
full-view mR gretel 0.707 -> 0.750, nemotron 0.956 -> 0.977, TAB 0.935 -> 0.953, REDACT
0.930 -> 0.971 (mP 0.474 -> 0.458). All these cells are D (excluded) for SCT, so no headline or
pairwise number changes: gretel_en and argilla_pii had 0 recovered copies.

| entry | dataset | masked_recall with | without | doc leak with | without | copies recovered |
|---|---|---|---|---|---|---|
| sct-default | openpii | (0.907) | (0.907) | 0.462 | 0.463 | 22 |
| sct-default | gretel | (0.750) | (0.707) | 0.639 | 0.690 | 1040 |
| sct-default | gretel_en | 0.978 | 0.978 | 0.266 | 0.266 | 0 |
| sct-default | nemotron | (0.977) | (0.956) | 0.183 | 0.236 | 42 |
| sct-default | argilla_pii | 0.967 | 0.967 | 0.057 | 0.057 | 0 |
| sct-default | thai_pii | (0.977) | (0.977) | 0.782 | 0.782 | 0 |
| sct-default | bayanat | (0.995) | (0.995) | 0.050 | 0.050 | 0 |
| sct-default | tab | (0.953) | (0.935) | 1.000 | 1.000 | 321 |
| sct-default | redact | (0.971) | (0.930) | 0.591 | 0.647 | 2477 |
| sct-default | iahlt | (0.937) | (0.937) | 0.291 | 0.291 | 0 |
| sct-default | thainer | (0.882) | (0.882) | 0.490 | 0.490 | 0 |
| sct-default | naamapadam | (0.712) | (0.711) | 0.411 | 0.411 | 2 |
| sct-default | multiconer | (0.829) | (0.829) | 0.296 | 0.296 | 0 |
| sct-default | stockmark | (0.917) | (0.917) | 0.267 | 0.267 | 0 |
| sct-gliner | openpii | (0.888) | (0.888) | 0.534 | 0.534 | 173 |
| sct-gliner | gretel | (0.792) | (0.724) | 0.669 | 0.734 | 3347 |
| sct-gliner | gretel_en | 0.983? | 0.983? | 0.239 | 0.239 | 1 |
| sct-gliner | nemotron | (0.982) | (0.958) | 0.218 | 0.271 | 112 |
| sct-gliner | argilla_pii | 0.985? | 0.985? | 0.102 | 0.102 | 0 |
| sct-gliner | thai_pii | (0.674) | (0.674) | 0.996 | 0.996 | 0 |
| sct-gliner | bayanat | (0.930) | (0.930) | 0.274 | 0.274 | 0 |
| sct-gliner | tab | (0.708) | (0.684) | 1.000 | 1.000 | 815 |
| sct-gliner | redact | (0.875) | (0.816) | 0.768 | 0.821 | 3888 |
| sct-gliner | iahlt | (0.507) | (0.491) | 0.854 | 0.854 | 33 |
| sct-gliner | thainer | (0.190) | (0.190) | 0.855 | 0.855 | 0 |
| sct-gliner | naamapadam | (0.079) | (0.079) | 0.982 | 0.982 | 0 |
| sct-gliner | multiconer | (0.245) | (0.245) | 0.857 | 0.857 | 0 |
| sct-gliner | stockmark | (0.667) | (0.667) | 0.519 | 0.519 | 0 |

## Per dataset (tuned, full view, masked F2, completed docs)

`(x)` = excluded cell (T/E/D), information only; `x?` = unknown training data. Masked F2 = F2 of
mR and mP (the tuning objective). All languages of the dataset (not limited to claims).

| dataset | bardsai-v3 | datafog | gliner-nvidia | gliner2-fastino | opf | presidio | sct-default | sct-gliner |
|---|---|---|---|---|---|---|---|---|
| openpii | (0.969) | 0.338 | (0.957) | 0.891 | 0.883? | 0.647 | (0.908) | (0.858) |
| gretel | 0.660 | 0.150 | 0.717 | 0.688 | 0.570? | 0.472 | (0.603) | (0.598) |
| gretel_en | (0.994) | 0.487 | (0.957) | 0.919 | 0.912? | 0.865 | 0.932 | 0.917? |
| nemotron | (0.979) | 0.357 | (0.916) | 0.873 | 0.681? | 0.780 | (0.886) | (0.832) |
| argilla_pii | (0.983) | 0.233 | (0.934) | 0.845 | 0.799? | 0.880 | 0.963 | 0.901? |
| thai_pii | 0.889 | 0.561 | 0.782 | 0.952 | 0.898? | 0.486 | (0.969) | (0.708) |
| bayanat | 0.893 | 0.527 | 0.731 | 0.920 | 0.898? | 0.751 | (0.926) | (0.742) |
| tab | 0.638 | 0.093 | 0.363 | 0.776 | 0.284? | 0.486 | (0.845) | (0.649) |
| redact | 0.801 | 0.274 | 0.611 | 0.808 | 0.743? | 0.549 | (0.790) | (0.698) |
| iahlt | 0.433 | 0.000 | 0.748 | 0.869 | 0.181? | 0.188 | (0.899) | (0.503) |
| thainer | 0.394 | 0.004 | 0.215 | 0.588 | 0.502? | 0.107 | (0.864) | (0.215) |
| naamapadam | 0.391 | 0.000 | 0.120 | 0.884 | 0.369? | 0.102 | (0.717) | (0.090) |
| multiconer | 0.225 | 0.000 | 0.313 | 0.642 | 0.269? | 0.076 | (0.746) | (0.229) |
| stockmark | 0.024 | 0.000 | 0.648 | 0.750 | 0.085? | 0.051 | (0.837) | (0.620) |

## Per language (tuned, full view, masked recall, clean sets only)

Each cell: mean over the entry's clean datasets of the group that have the language. `°` = a
language outside the entry's claimed languages: shown apart from the claim, not a loss. `-` = the
entry has no clean dataset with that language. Unknown-set and NER tables are in `TABLES.md`.

### Privacy-synthetic

| lang | datasets | bardsai-v3 | datafog | gliner-nvidia | gliner2-fastino | presidio | sct-default |
|---|---|---|---|---|---|---|---|
| ar-egyptian | bayanat | 0.846° | 0.423 | 0.750° | 0.949° | 0.673° | - |
| ar-gulf | bayanat | 0.906° | 0.592 | 0.718° | 0.919° | 0.758° | - |
| ar-mixed_en | bayanat | 0.953° | 0.470 | 0.926° | 0.993° | 0.933° | - |
| ar-msa | bayanat | 0.914° | 0.441 | 0.775° | 0.945° | 0.664° | - |
| bg | openpii | - | 0.408 | - | 0.872° | 0.677° | - |
| cs | openpii | - | 0.298 | - | 0.914° | 0.632° | - |
| da | openpii | - | 0.383 | - | 0.953° | 0.704° | - |
| de | gretel, openpii | 0.644 | 0.189 | 0.684° | 0.862 | 0.640° | - |
| el | openpii | - | 0.262 | - | 0.911° | 0.659° | - |
| en | argilla_pii, gretel, gretel_en, nemotron, openpii | 0.675 | 0.294 | 0.853 | 0.928 | 0.762 | 0.973 |
| es | gretel, openpii | 0.634 | 0.218 | 0.704° | 0.825 | 0.552° | - |
| et | openpii | - | 0.352 | - | 0.932° | 0.752° | - |
| fi | openpii | - | 0.340 | - | 0.940° | 0.729° | - |
| fr | gretel, openpii | 0.696 | 0.226 | 0.816° | 0.917 | 0.566° | - |
| hr | openpii | - | 0.368 | - | 0.908° | 0.703° | - |
| hu | openpii | - | 0.251 | - | 0.919° | 0.588° | - |
| id | openpii | - | 0.322 | - | 0.883° | 0.708° | - |
| it | gretel, openpii | 0.723 | 0.216 | 0.784° | 0.865 | 0.622° | - |
| ja | openpii | - | 0.157 | - | 0.697° | 0.430° | - |
| ko | openpii | - | 0.217 | - | 0.875° | 0.409° | - |
| lt | openpii | - | 0.338 | - | 0.910° | 0.753° | - |
| lv | openpii | - | 0.306 | - | 0.911° | 0.715° | - |
| ms | openpii | - | 0.319 | - | 0.894° | 0.749° | - |
| nl | gretel, openpii | 0.544 | 0.196 | 0.645° | 0.810 | 0.607° | - |
| pl | openpii | - | 0.212 | - | 0.903° | 0.651° | - |
| pt | openpii | - | 0.297 | - | 0.879 | 0.757° | - |
| ro | openpii | - | 0.233 | - | 0.884° | 0.727° | - |
| sk | openpii | - | 0.247 | - | 0.895° | 0.683° | - |
| sl | openpii | - | 0.213 | - | 0.918° | 0.712° | - |
| sr | openpii | - | 0.376 | - | 0.905° | 0.721° | - |
| sv | gretel, openpii | 0.627 | 0.207 | 0.683° | 0.846° | 0.608° | - |
| th | thai_pii | 0.884° | 0.512 | 0.766° | 0.953° | 0.440° | - |
| tl | openpii | - | 0.255 | - | 0.917° | 0.733° | - |
| vi | openpii | - | 0.295 | - | 0.863° | 0.695° | - |
| zh | openpii | - | 0.150 | - | 0.696° | 0.399° | - |

### Privacy-real

| lang | datasets | bardsai-v3 | datafog | gliner-nvidia | gliner2-fastino | presidio |
|---|---|---|---|---|---|---|
| ar | redact | 0.895° | 0.351 | 0.687° | 0.967° | 0.530° |
| cs | redact | 0.917 | 0.211 | 0.601° | 0.902° | 0.723° |
| da | redact | 0.831 | 0.191 | 0.567° | 0.857° | 0.690° |
| de | redact | 0.864 | 0.242 | 0.705° | 0.877 | 0.754° |
| en | redact, tab | 0.713 | 0.163 | 0.439 | 0.914 | 0.612 |
| es | redact | 0.929 | 0.231 | 0.559° | 0.948 | 0.776° |
| fi | redact | 0.905 | 0.330 | 0.699° | 0.920° | 0.776° |
| fr | redact | 0.893 | 0.202 | 0.673° | 0.928 | 0.650° |
| fr_ca | redact | 0.918 | 0.309 | 0.676° | 0.934 | 0.740° |
| he | redact | 0.881° | 0.310 | 0.603° | 0.955° | 0.555° |
| hi | redact | 0.887° | 0.240 | 0.345° | 0.905° | 0.542° |
| hu | redact | 0.906 | 0.212 | 0.686° | 0.920° | 0.643° |
| it | redact | 0.879 | 0.283 | 0.593° | 0.871 | 0.687° |
| ja | redact | 0.615° | 0.151 | 0.595° | 0.744° | 0.348° |
| ko | redact | 0.885° | 0.159 | 0.718° | 0.906° | 0.312° |
| nl | redact | 0.897 | 0.239 | 0.539° | 0.901 | 0.816° |
| no | redact | 0.919° | 0.249 | 0.608° | 0.924° | 0.779° |
| pt_br | redact | 0.917 | 0.220 | 0.610° | 0.915 | 0.807° |
| pt_eu | redact | 0.854 | 0.230 | 0.575° | 0.894 | 0.767° |
| ru | redact | 0.875° | 0.170 | 0.706° | 0.920° | 0.585° |
| sv | redact | 0.910 | 0.313 | 0.586° | 0.935° | 0.827° |
| th | redact | 0.884° | 0.421 | 0.454° | 0.926° | 0.370° |
| tr | redact | 0.938° | 0.208 | 0.591° | 0.941° | 0.694° |
| zh_cn | redact | 0.546° | 0.206 | 0.465° | 0.673° | 0.259° |
| zh_tw | redact | 0.519° | 0.201 | 0.550° | 0.744° | 0.258° |

## Contamination matrix

T = trained on it (T-family: same generator), E = vendor evaluates on it, D = SCT developed
against it in earlier rounds, U = training data undisclosed, - = none known (model cards at the
pinned revisions, checked 2026-11-02). Bold = excluded, italic = unknown, plain = clean.

| entry | openpii | gretel | gretel_en | nemotron | argilla_pii | thai_pii | bayanat | tab | redact | iahlt | thainer | naamapadam | multiconer | stockmark |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sct-default | **D** | **D** | - | **D** | - | **D** | **D** | **D** | **D** | **D** | **D** | **D** | **D** | **D** |
| sct-gliner | **U, D** | **U, D** | _U_ | **U, D** | _U_ | **U, D** | **U, D** | **U, D** | **U, D** | **U, D** | **U, D** | **U, D** | **U, D** | **U, D** |
| presidio | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| gliner-nvidia | **E** | - | **E** | **T** | **E** | - | - | - | - | - | - | - | - | - |
| gliner2-fastino | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| opf | _U_ | _U_ | _U_ | _U_ | _U_ | _U_ | _U_ | _U_ | _U_ | _U_ | _U_ | _U_ | _U_ | _U_ |
| bardsai-v3 | **T** | - | **T** | **T** | **T-family** | - | - | - | - | - | - | - | - | - |
| datafog | - | - | - | - | - | - | - | - | - | - | - | - | - | - |

## Feature matrix (not scored)

| entry | languages (claimed) | types mapped to SCT's 9 | char spans | reversible | fake values | text cleaning | language ID | batch API | GPU | offline | licence (code / model) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| sct-default | 75 (LID); NER en, nl, de, es + multilingual WikiNEural | 9 | via reversible mode | yes (`TokenStore`) | yes (`synthetic` extra) | yes | yes (Lingua) | `process_batch` | optional (CUDA ONNX) | yes | MIT / per NER model card |
| sct-gliner | as SCT; GLiNER model tagged en (claim used here: en) | 9 | via reversible mode | yes | yes | yes | yes | `process_batch` | optional | yes | MIT / Apache-2.0 |
| presidio | en (only en_core_web_lg loaded, both tracks); spaCy packs possible | 8 (no ORG with spaCy) | yes | encrypt / decrypt operator | custom operator | no | no | `BatchAnalyzerEngine` | optional (transformers engine) | yes | MIT / MIT (spaCy model) |
| gliner-nvidia | en | 9 (prompts) | yes | no | no | no | no | library batch | optional | yes | Apache-2.0 / NVIDIA Open Model License |
| gliner2-fastino | 7 EU | 9 (prompts) | yes | no | no | no | no | library batch | optional | yes | Apache-2.0 / Apache-2.0 |
| opf | primarily en | 7 (no ORG, no IP) | yes | no | no | no | no | yes | optional | yes | Apache-2.0 / Apache-2.0 |
| bardsai-v3 | 24 EU | 9 | yes (bundled detector) | no | no | no | no | yes | optional | yes | - / Apache-2.0 |
| datafog | regex (US shapes; German pack opt-in) | 5 (no PERSON, LOCATION, ORG, URL) | yes | no (hash, redact) | no | no | no | yes | no | yes | MIT / - |

## Failures (rule 19)

| entry | track | failed docs | where | kind | handling |
|---|---|---|---|---|---|
| gliner2-fastino | default | 9 of 16,661 | gretel 6 (99.7%), nemotron 3 (99.0%) | `ValueError`: the library's long-doc remap returns spans past the text end | scored as unmasked (every mention leaked); no dataset under 99% |
| gliner2-fastino | tuned | 13 of 16,661 | gretel 10 (99.5%), nemotron 3 (99.0%) | same | same |
| every other entry | both | 0 | - | - | - |

Unplaced spans (the entry detected something the harness could not place in the source; scored
as a wrong prediction and the gold as missed): only the two SCT entries have them, because SCT
returns rewritten text and its spans are recovered through reversible placeholders. Tuned
sct-default: gretel 455, openpii 247, redact 159, naamapadam 120, nemotron 47, others <= 12;
sct-gliner: gretel 658, openpii 250, redact 206. Separately, a source copy whose placeholder
SCT removed after replacing it (markdown link text; other removed copies not classified one by one)
had no span at all and counted as leaked; `scorer.recover_dropped` now credits such copies (see
"Gretel e-mail artefact" above), and both numbers are reported.

## Where SCT loses, and the likely cause

Only clean cells count as comparison evidence; under the strict rule that is gretel_en and
argilla_pii (English, synthetic). No code was changed for these findings.

1. **Exact spans and types (strict F1), clean cells.** gliner2-fastino beats SCT by +0.09 to
   +0.16 (full view, tuned) and +0.40 to +0.52 (shared view); presidio by +0.02 to +0.09 (full)
   and +0.35 to +0.47 (shared); opf (unknown) by +0.35 to +0.47 in the shared view. Likely cause
   (seen on excluded sets too): PHONE vs NUMBER typing of long digit IDs (on openpii SCT puts
   PHONE on 1,884 credit-card, 1,384 tax, 1,343 social-security and 1,100 passport numbers),
   plus span edges from recovering offsets through placeholders. Privacy is not hurt (masked
   recall is type-agnostic); users who need typed spans are.
2. **Precision on numbers, clean cells.** Shared view mP: SCT 0.738 vs datafog 0.907 and opf
   0.806 (both CIs exclude 0). Likely cause: the NUMBER/PHONE regexes mask most digit runs
   (recall-first by design).
3. **Speed and memory** (not a contamination question: FineWeb-2 speed sample). CPU, 16 threads,
   batch 8: SCT 1.50 docs/s vs bardsai-v3 2.52, presidio 27, datafog 1,787 (faster than
   gliner2-fastino 0.52, sct-gliner 0.53, opf 0.29, gliner-nvidia 0.15). Peak RSS 7.7 GB vs
   bardsai-v3 1.7 GB, presidio 0.8 GB. Likely cause: native routing runs Lingua LID, cleaning and
   an ensemble of fp32 ONNX NER models with a session per language.

Development signals from excluded cells (information only, not a comparison): on gretel finance
(D for SCT) ORGANISATION masked recall 0.81 vs gliner2-fastino 0.95 (with the fix; 0.74 without),
NUMBER / PHONE precision 0.016 / 0.10 vs 0.15 / 0.66; on naamapadam (D) SCT mR 0.712 vs
gliner2-fastino 0.961 (no Indic NER model in SCT); on REDACT (D) SCT has the higher recall (0.971
vs 0.894) and the lower precision (0.458 vs 0.596). These are worth fixing but cannot be
claimed as wins or losses against other tools until they are measured on fresh data.

## Limits of this comparison

- **Round 1 cannot rank SCT.** Its 0.8.1 fixes were developed on 12 of the 14 datasets (D), so
  under decision 21 only gretel_en and argilla_pii remain. Round 2 needs fresh datasets.
- **No GPU numbers.** The pinned torch 2.15.0 wheels need CUDA 13.2; the box's driver is 12.6, so
  every entry ran on CPU (gliner-nvidia fp32).
- **Contamination is from cards, not data.** T/E come from model cards; opf and the knowledgator
  model do not disclose training data (U), so those entries have no clean cell at all.
- **Claimed languages from cards.** "Primarily English" (opf) is scored as en only. SCT and
  datafog make no language claim, so every language counts for them.
- **SCT is scored through its reversible view.** Its spans come from placeholders; unplaced
  spans still count against SCT (gretel 455 tuned), and the recovery rule above is conservative
  (any visible word of a value blocks it).
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
- StarPII (gated, evaluation-only terms, no redistribution) and deduce (GPL-3.0, benchmark use
  only) are not entries here.
