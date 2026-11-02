# SCT vs other PII tools: comparison, 2026-11-02

SqueakyCleanText 0.8.1 (`sct-default`, and its optional GLiNER path `sct-gliner`) against six
other offline PII tools, under the frozen protocol in `benchmarks/compare/PROTOCOL.md`
(Codex's 20 fairness rules; section 20 maps each rule). Numbers come from this bundle
(`default/`, `tuned/`, `speed-*/`); every table can be rebuilt with
`python -m benchmarks.compare.report benchmarks/results/compare-2026-11-02` (see `TABLES.md`).

- Entries (exact pins in `PROTOCOL.md` section 2): sct-default, sct-gliner, presidio 2.2.366
  (en_core_web_lg), gliner-nvidia (nvidia/gliner-PII), gliner2-fastino
  (fastino/gliner2-privacy-filter-PII-multi), opf (openai/privacy-filter), bardsai-v3
  (bardsai/eu-pii-anonimization-multilang-v3, INT8 ONNX), datafog 4.9.0 (regex).
- Data: test part (hash split by template/record, seed 13), 300 docs per language
  (TAB 106, REDACT 825), 14 datasets, 16,661 docs per entry. Privacy-synthetic: openpii, gretel
  (finance), gretel_en, nemotron, argilla_pii, thai_pii, bayanat. Privacy-real: TAB, REDACT.
  NER (separate, not privacy): iahlt, thainer, naamapadam, multiconer, stockmark.
  gretel_en (`gretelai/gretel-pii-masking-en-v1`) and argilla_pii
  (`argilla/textcat-tokencat-pii-per-domain`) are the two non-NVIDIA sets the nvidia/gliner-PII
  card reports; there is no Gretel PII v2 on the Hub.
- Tracks: **default** = vendor quick start; **tuned** = one configuration per entry picked on the
  dev part from at most 12 candidates (threshold, language packs, card labels), frozen in
  `benchmarks/compare/configs/tuned.json` before the test run. Gold-language track (every tool
  that takes a language gets the gold code). All quality runs on CPU.
- Detection track: every entry's spans go through one mask operation, then are scored.

How to read the headline tables:

- Scores are the group macro: mean over datasets of each dataset's language macro, failed docs
  counted unmasked, DATE out. Synthetic and real data are never pooled.
- mR = type-agnostic masked recall (gold mentions with any overlapping prediction);
  mP = masked precision (predictions overlapping any in-view gold); strict F1 = exact span and
  type; doc leak = share of docs with a gold mention that keeps any letter or digit unmasked;
  over-mask = masked characters outside every gold span / all non-gold characters.
- Each cell: the entry's score, then the 95% paired bootstrap CI of (entry - SCT) on the same
  documents (1,000 resamples over templates/docs within each dataset, seed 13). `*` = the CI
  excludes 0. Higher is better for mR, mP, strict F1; lower is better for doc leak and over-mask.
- In-domain cells (the entry was trained on that dataset or its generator family, T, or the
  vendor evaluates on it, E) are left out of the pair; "sets" is how many datasets remain, and
  "SCT mR on same sets" is SCT on exactly those. SCT has no T/E cell; its D cells (SCT was
  developed against the dataset in earlier rounds) and U cells (training data undisclosed) stay
  in and are flagged in the matrix below.
- Full view = all 9 SCT types, a type the tool does not support counts as missed (what a user
  must mask). Shared view = EMAIL, PHONE, NUMBER only (types every entry supports; the
  protocol's ranking view, narrow). Speed: CPU, 16 threads on cores 0-15, batch 8, 100 FineWeb-2
  docs, median of 3; peak RAM = adapter process VmHWM. **No GPU numbers**: the pinned torch
  2.15.0 wheel needs CUDA 13.2 and the box's driver is 12.6, so no GPU run happened.

## Headline, tuned track, in-domain sets out

### Full view (all 9 types), privacy-synthetic

| entry | sets | mR (entry - SCT, 95% CI) | mP (entry - SCT, 95% CI) | strict F1 (entry - SCT, 95% CI) | doc leak (entry - SCT, 95% CI) | over-mask (entry - SCT, 95% CI) | SCT mR on same sets | docs/s CPU | docs/s GPU | peak RAM / VRAM | languages | licence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sct-default | 7 | 0.927 [0.919, 0.934] | 0.756 [0.741, 0.772] | 0.481 [0.469, 0.493] | 0.363 [0.347, 0.379] | 0.038 [0.035, 0.042] | - | 1.50 | not run | 7,692 MB / n/a | 75 LID; NER en nl de es + multilingual | MIT |
| gliner2-fastino | 7 | 0.909 (-0.027, -0.009)* | 0.756 (-0.015, +0.016) | 0.683 (+0.188, +0.216)* | 0.335 (-0.045, -0.009)* | 0.049 (+0.007, +0.014)* | 0.927 | 0.52 | not run | 6,832 MB / n/a | 7 EU | Apache-2.0 |
| sct-gliner | 7 | 0.878 (-0.055, -0.043)* | 0.613 (-0.153, -0.134)* | 0.296 (-0.194, -0.176)* | 0.450 (+0.072, +0.100)* | 0.087 (+0.046, +0.051)* | 0.927 | 0.53 | not run | 6,414 MB / n/a | 75 LID; GLiNER model tagged en | MIT + Apache-2.0 model |
| bardsai-v3 | 3 | 0.813 (-0.095, -0.066)* | 0.824 (+0.134, +0.180)* | 0.638 (+0.256, +0.296)* | 0.419 (-0.121, -0.054)* | 0.043 (-0.032, -0.018)* | 0.893 | 2.52 | not run | 1,738 MB / n/a | 24 EU | Apache-2.0 |
| opf | 7 | 0.803 (-0.137, -0.111)* | 0.842 (+0.069, +0.104)* | 0.557 (+0.057, +0.094)* | 0.391 (+0.006, +0.051)* | 0.069 (+0.026, +0.035)* | 0.927 | 0.29 | not run | 26,113 MB / n/a | primarily en | Apache-2.0 |
| gliner-nvidia | 3 | 0.765 (-0.144, -0.111)* | 0.690 (+0.005, +0.040)* | 0.320 (-0.059, -0.021)* | 0.613 (+0.080, +0.131)* | 0.124 (+0.050, +0.065)* | 0.893 | 0.15 | not run | 25,732 MB / n/a | en | Apache-2.0 + NVIDIA Open Model |
| presidio | 7 | 0.705 (-0.234, -0.211)* | 0.730 (-0.040, -0.014)* | 0.461 (-0.034, -0.006)* | 0.645 (+0.265, +0.301)* | 0.081 (+0.038, +0.048)* | 0.927 | 27 | not run | 826 MB / n/a | en (default); spaCy packs when tuned | MIT |
| datafog | 7 | 0.337 (-0.603, -0.577)* | 0.854 (+0.079, +0.120)* | 0.342 (-0.156, -0.122)* | 0.919 (+0.538, +0.574)* | 0.009 (-0.033, -0.026)* | 0.927 | 1,787 | not run | 30 MB / n/a | regex (US shapes; DE pack) | MIT |

### Full view, privacy-real (TAB, REDACT)

| entry | sets | mR (entry - SCT, 95% CI) | mP (entry - SCT, 95% CI) | strict F1 (entry - SCT, 95% CI) | doc leak (entry - SCT, 95% CI) | over-mask (entry - SCT, 95% CI) | SCT mR on same sets | docs/s CPU | docs/s GPU | peak RAM / VRAM | languages | licence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sct-default | 2 | 0.933 [0.922, 0.943] | 0.523 [0.505, 0.547] | 0.369 [0.353, 0.389] | 0.823 [0.807, 0.840] | 0.045 [0.043, 0.048] | - | 1.50 | not run | 7,692 MB / n/a | 75 LID; NER en nl de es + multilingual | MIT |
| gliner2-fastino | 2 | 0.895 (-0.061, -0.016)* | 0.550 (+0.008, +0.045)* | 0.514 (+0.131, +0.158)* | 0.821 (-0.018, +0.015) | 0.070 (+0.022, +0.028)* | 0.933 | 0.52 | not run | 6,832 MB / n/a | 7 EU | Apache-2.0 |
| sct-gliner | 2 | 0.750 (-0.201, -0.166)* | 0.472 (-0.065, -0.039)* | 0.204 (-0.181, -0.151)* | 0.911 (+0.072, +0.103)* | 0.054 (+0.006, +0.011)* | 0.933 | 0.53 | not run | 6,414 MB / n/a | 75 LID; GLiNER model tagged en | MIT + Apache-2.0 model |
| bardsai-v3 | 2 | 0.720 (-0.240, -0.189)* | 0.809 (+0.264, +0.305)* | 0.491 (+0.099, +0.141)* | 0.833 (-0.010, +0.030) | 0.027 (-0.021, -0.015)* | 0.933 | 2.52 | not run | 1,738 MB / n/a | 24 EU | Apache-2.0 |
| presidio | 2 | 0.532 (-0.424, -0.377)* | 0.631 (+0.086, +0.128)* | 0.266 (-0.129, -0.080)* | 0.951 (+0.112, +0.144)* | 0.075 (+0.026, +0.034)* | 0.933 | 27 | not run | 826 MB / n/a | en (default); spaCy packs when tuned | MIT |
| opf | 2 | 0.520 (-0.437, -0.388)* | 0.775 (+0.228, +0.273)* | 0.447 (+0.037, +0.114)* | 0.867 (+0.025, +0.062)* | 0.043 (-0.006, +0.002) | 0.933 | 0.29 | not run | 26,113 MB / n/a | primarily en | Apache-2.0 |
| gliner-nvidia | 2 | 0.456 (-0.495, -0.454)* | 0.831 (+0.286, +0.327)* | 0.192 (-0.196, -0.159)* | 0.904 (+0.064, +0.096)* | 0.021 (-0.026, -0.020)* | 0.933 | 0.15 | not run | 25,732 MB / n/a | en | Apache-2.0 + NVIDIA Open Model |
| datafog | 2 | 0.161 (-0.792, -0.751)* | 0.765 (+0.220, +0.271)* | 0.155 (-0.236, -0.196)* | 0.997 (+0.158, +0.191)* | 0.008 (-0.040, -0.034)* | 0.933 | 1,787 | not run | 30 MB / n/a | regex (US shapes; DE pack) | MIT |

### Shared view (EMAIL, PHONE, NUMBER), privacy-synthetic

| entry | sets | mR (entry - SCT, 95% CI) | mP (entry - SCT, 95% CI) | strict F1 (entry - SCT, 95% CI) | doc leak (entry - SCT, 95% CI) | over-mask (entry - SCT, 95% CI) | SCT mR on same sets | docs/s CPU | docs/s GPU | peak RAM / VRAM | languages | licence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sct-default | 7 | 0.949 [0.942, 0.955] | 0.611 [0.593, 0.628] | 0.357 [0.340, 0.378] | 0.171 [0.158, 0.183] | 0.038 [0.035, 0.042] | - | 1.50 | not run | 7,692 MB / n/a | 75 LID; NER en nl de es + multilingual | MIT |
| bardsai-v3 | 3 | 0.963 (+0.005, +0.029)* | 0.635 (+0.099, +0.149)* | 0.794 (+0.462, +0.515)* | 0.074 (-0.108, -0.069)* | 0.043 (-0.032, -0.018)* | 0.944 | 2.52 | not run | 1,738 MB / n/a | 24 EU | Apache-2.0 |
| sct-gliner | 7 | 0.951 (-0.001, +0.005) | 0.411 (-0.212, -0.188)* | 0.347 (-0.014, -0.006)* | 0.163 (-0.013, -0.002)* | 0.087 (+0.046, +0.051)* | 0.949 | 0.53 | not run | 6,414 MB / n/a | 75 LID; GLiNER model tagged en | MIT + Apache-2.0 model |
| opf | 7 | 0.944 (-0.018, +0.007) | 0.720 (+0.088, +0.130)* | 0.729 (+0.343, +0.403)* | 0.086 (-0.102, -0.067)* | 0.069 (+0.026, +0.035)* | 0.949 | 0.29 | not run | 26,113 MB / n/a | primarily en | Apache-2.0 |
| gliner2-fastino | 7 | 0.944 (-0.019, +0.006) | 0.573 (-0.058, -0.020)* | 0.825 (+0.439, +0.494)* | 0.097 (-0.092, -0.055)* | 0.049 (+0.007, +0.014)* | 0.949 | 0.52 | not run | 6,832 MB / n/a | 7 EU | Apache-2.0 |
| gliner-nvidia | 3 | 0.886 (-0.079, -0.037)* | 0.415 (-0.118, -0.075)* | 0.642 (+0.303, +0.370)* | 0.184 (-0.016, +0.058) | 0.124 (+0.050, +0.065)* | 0.944 | 0.15 | not run | 25,732 MB / n/a | en | Apache-2.0 + NVIDIA Open Model |
| presidio | 7 | 0.861 (-0.102, -0.074)* | 0.636 (+0.008, +0.042)* | 0.664 (+0.279, +0.336)* | 0.279 (+0.088, +0.128)* | 0.081 (+0.038, +0.048)* | 0.949 | 27 | not run | 826 MB / n/a | en (default); spaCy packs when tuned | MIT |
| datafog | 7 | 0.678 (-0.295, -0.247)* | 0.828 (+0.194, +0.244)* | 0.597 (+0.211, +0.266)* | 0.507 (+0.309, +0.362)* | 0.009 (-0.033, -0.026)* | 0.949 | 1,787 | not run | 30 MB / n/a | regex (US shapes; DE pack) | MIT |

### Shared view, privacy-real

| entry | sets | mR (entry - SCT, 95% CI) | mP (entry - SCT, 95% CI) | strict F1 (entry - SCT, 95% CI) | doc leak (entry - SCT, 95% CI) | over-mask (entry - SCT, 95% CI) | SCT mR on same sets | docs/s CPU | docs/s GPU | peak RAM / VRAM | languages | licence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sct-default | 2 | 0.983 [0.978, 0.988] | 0.193 [0.158, 0.237] | 0.155 [0.141, 0.177] | 0.180 [0.160, 0.204] | 0.045 [0.043, 0.048] | - | 1.50 | not run | 7,692 MB / n/a | 75 LID; NER en nl de es + multilingual | MIT |
| sct-gliner | 2 | 0.976 (-0.014, -0.002)* | 0.177 (-0.025, -0.008)* | 0.150 (-0.006, -0.004)* | 0.188 (-0.003, +0.021) | 0.054 (+0.006, +0.011)* | 0.983 | 0.53 | not run | 6,414 MB / n/a | 75 LID; GLiNER model tagged en | MIT + Apache-2.0 model |
| datafog | 2 | 0.768 (-0.332, -0.139)* | 0.750 (+0.516, +0.601)* | 0.351 (+0.174, +0.210)* | 0.812 (+0.610, +0.654)* | 0.008 (-0.040, -0.034)* | 0.983 | 1,787 | not run | 30 MB / n/a | regex (US shapes; DE pack) | MIT |
| opf | 2 | 0.547 (-0.499, -0.352)* | 0.462 (+0.014, +0.372)* | 0.460 (+0.209, +0.399)* | 0.569 (+0.362, +0.414)* | 0.043 (-0.006, +0.002) | 0.983 | 0.29 | not run | 26,113 MB / n/a | primarily en | Apache-2.0 |
| gliner2-fastino | 2 | 0.527 (-0.501, -0.412)* | 0.161 (-0.068, -0.002)* | 0.467 (+0.258, +0.367)* | 0.599 (+0.395, +0.443)* | 0.070 (+0.022, +0.028)* | 0.983 | 0.52 | not run | 6,832 MB / n/a | 7 EU | Apache-2.0 |
| presidio | 2 | 0.456 (-0.550, -0.509)* | 0.172 (-0.046, +0.002) | 0.325 (+0.149, +0.186)* | 0.721 (+0.516, +0.564)* | 0.075 (+0.026, +0.034)* | 0.983 | 27 | not run | 826 MB / n/a | en (default); spaCy packs when tuned | MIT |
| bardsai-v3 | 2 | 0.455 (-0.537, -0.519)* | 0.193 (-0.041, +0.038) | 0.415 (+0.237, +0.277)* | 0.646 (+0.441, +0.492)* | 0.027 (-0.021, -0.015)* | 0.983 | 2.52 | not run | 1,738 MB / n/a | 24 EU | Apache-2.0 |
| gliner-nvidia | 2 | 0.308 (-0.690, -0.642)* | 0.367 (+0.086, +0.278)* | 0.297 (+0.114, +0.177)* | 0.837 (+0.622, +0.688)* | 0.021 (-0.026, -0.020)* | 0.983 | 0.15 | not run | 25,732 MB / n/a | en | Apache-2.0 + NVIDIA Open Model |

## Headline, default track, in-domain sets out

### Full view, privacy-synthetic

| entry | sets | mR (entry - SCT, 95% CI) | mP (entry - SCT, 95% CI) | strict F1 (entry - SCT, 95% CI) | doc leak (entry - SCT, 95% CI) | over-mask (entry - SCT, 95% CI) | SCT mR on same sets | docs/s CPU | docs/s GPU | peak RAM / VRAM | languages | licence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sct-default | 7 | 0.926 [0.918, 0.934] | 0.757 [0.742, 0.772] | 0.483 [0.471, 0.495] | 0.367 [0.351, 0.382] | 0.038 [0.034, 0.042] | - | 1.50 | not run | 7,692 MB / n/a | 75 LID; NER en nl de es + multilingual | MIT |
| gliner2-fastino | 7 | 0.883 (-0.053, -0.034)* | 0.795 (+0.024, +0.053)* | 0.705 (+0.208, +0.237)* | 0.382 (-0.003, +0.032) | 0.037 (-0.004, +0.003) | 0.926 | 0.52 | not run | 6,832 MB / n/a | 7 EU | Apache-2.0 |
| sct-gliner | 7 | 0.863 (-0.069, -0.056)* | 0.667 (-0.097, -0.082)* | 0.319 (-0.172, -0.156)* | 0.483 (+0.101, +0.130)* | 0.068 (+0.028, +0.032)* | 0.926 | 0.53 | not run | 6,414 MB / n/a | 75 LID; GLiNER model tagged en | MIT + Apache-2.0 model |
| bardsai-v3 | 3 | 0.813 (-0.094, -0.065)* | 0.824 (+0.134, +0.180)* | 0.638 (+0.254, +0.294)* | 0.419 (-0.124, -0.057)* | 0.043 (-0.031, -0.017)* | 0.892 | 2.52 | not run | 1,738 MB / n/a | 24 EU | Apache-2.0 |
| opf | 7 | 0.755 (-0.186, -0.158)* | 0.880 (+0.108, +0.139)* | 0.567 (+0.065, +0.101)* | 0.462 (+0.073, +0.117)* | 0.047 (+0.006, +0.013)* | 0.926 | 0.29 | not run | 26,113 MB / n/a | primarily en | Apache-2.0 |
| presidio | 7 | 0.705 (-0.233, -0.210)* | 0.730 (-0.040, -0.015)* | 0.461 (-0.036, -0.008)* | 0.645 (+0.262, +0.297)* | 0.081 (+0.039, +0.048)* | 0.926 | 27 | not run | 826 MB / n/a | en (default); spaCy packs when tuned | MIT |
| gliner-nvidia | 3 | 0.620 (-0.288, -0.253)* | 0.735 (+0.048, +0.085)* | 0.455 (+0.071, +0.116)* | 0.704 (+0.168, +0.216)* | 0.075 (+0.001, +0.017)* | 0.892 | 0.15 | not run | 25,732 MB / n/a | en | Apache-2.0 + NVIDIA Open Model |
| datafog | 7 | 0.337 (-0.602, -0.576)* | 0.854 (+0.079, +0.119)* | 0.342 (-0.159, -0.124)* | 0.919 (+0.535, +0.569)* | 0.009 (-0.033, -0.026)* | 0.926 | 1,787 | not run | 30 MB / n/a | regex (US shapes; DE pack) | MIT |

### Full view, privacy-real

| entry | sets | mR (entry - SCT, 95% CI) | mP (entry - SCT, 95% CI) | strict F1 (entry - SCT, 95% CI) | doc leak (entry - SCT, 95% CI) | over-mask (entry - SCT, 95% CI) | SCT mR on same sets | docs/s CPU | docs/s GPU | peak RAM / VRAM | languages | licence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sct-default | 2 | 0.932 [0.921, 0.943] | 0.525 [0.507, 0.548] | 0.370 [0.355, 0.391] | 0.828 [0.812, 0.845] | 0.044 [0.042, 0.047] | - | 1.50 | not run | 7,692 MB / n/a | 75 LID; NER en nl de es + multilingual | MIT |
| gliner2-fastino | 2 | 0.868 (-0.089, -0.042)* | 0.598 (+0.051, +0.091)* | 0.541 (+0.156, +0.183)* | 0.849 (+0.004, +0.037)* | 0.055 (+0.008, +0.013)* | 0.932 | 0.52 | not run | 6,832 MB / n/a | 7 EU | Apache-2.0 |
| bardsai-v3 | 2 | 0.720 (-0.240, -0.189)* | 0.809 (+0.263, +0.302)* | 0.491 (+0.096, +0.139)* | 0.833 (-0.015, +0.025) | 0.027 (-0.020, -0.014)* | 0.932 | 2.52 | not run | 1,738 MB / n/a | 24 EU | Apache-2.0 |
| sct-gliner | 2 | 0.678 (-0.278, -0.232)* | 0.485 (-0.055, -0.028)* | 0.203 (-0.184, -0.153)* | 0.923 (+0.079, +0.110)* | 0.040 (-0.006, -0.002)* | 0.932 | 0.53 | not run | 6,414 MB / n/a | 75 LID; GLiNER model tagged en | MIT + Apache-2.0 model |
| presidio | 2 | 0.532 (-0.423, -0.376)* | 0.631 (+0.083, +0.126)* | 0.266 (-0.131, -0.081)* | 0.951 (+0.107, +0.140)* | 0.075 (+0.027, +0.035)* | 0.932 | 27 | not run | 826 MB / n/a | en (default); spaCy packs when tuned | MIT |
| opf | 2 | 0.460 (-0.497, -0.448)* | 0.819 (+0.271, +0.314)* | 0.438 (+0.027, +0.103)* | 0.903 (+0.057, +0.093)* | 0.030 (-0.018, -0.011)* | 0.932 | 0.29 | not run | 26,113 MB / n/a | primarily en | Apache-2.0 |
| gliner-nvidia | 2 | 0.403 (-0.547, -0.506)* | 0.807 (+0.260, +0.300)* | 0.330 (-0.062, -0.021)* | 0.911 (+0.066, +0.099)* | 0.018 (-0.029, -0.024)* | 0.932 | 0.15 | not run | 25,732 MB / n/a | en | Apache-2.0 + NVIDIA Open Model |
| datafog | 2 | 0.161 (-0.791, -0.750)* | 0.765 (+0.218, +0.269)* | 0.155 (-0.238, -0.198)* | 0.997 (+0.152, +0.185)* | 0.008 (-0.039, -0.034)* | 0.932 | 1,787 | not run | 30 MB / n/a | regex (US shapes; DE pack) | MIT |

### Shared view, privacy-synthetic

| entry | sets | mR (entry - SCT, 95% CI) | mP (entry - SCT, 95% CI) | strict F1 (entry - SCT, 95% CI) | doc leak (entry - SCT, 95% CI) | over-mask (entry - SCT, 95% CI) | SCT mR on same sets | docs/s CPU | docs/s GPU | peak RAM / VRAM | languages | licence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sct-default | 7 | 0.949 [0.942, 0.955] | 0.613 [0.596, 0.630] | 0.358 [0.340, 0.378] | 0.171 [0.158, 0.183] | 0.038 [0.034, 0.042] | - | 1.50 | not run | 7,692 MB / n/a | 75 LID; NER en nl de es + multilingual | MIT |
| bardsai-v3 | 3 | 0.963 (+0.006, +0.030)* | 0.635 (+0.099, +0.149)* | 0.794 (+0.462, +0.515)* | 0.074 (-0.108, -0.069)* | 0.043 (-0.031, -0.017)* | 0.943 | 2.52 | not run | 1,738 MB / n/a | 24 EU | Apache-2.0 |
| sct-gliner | 7 | 0.952 (+0.000, +0.006)* | 0.479 (-0.145, -0.122)* | 0.352 (-0.008, -0.002)* | 0.163 (-0.013, -0.003)* | 0.068 (+0.028, +0.032)* | 0.949 | 0.53 | not run | 6,414 MB / n/a | 75 LID; GLiNER model tagged en | MIT + Apache-2.0 model |
| gliner2-fastino | 7 | 0.920 (-0.042, -0.016)* | 0.623 (-0.010, +0.028) | 0.834 (+0.448, +0.501)* | 0.134 (-0.057, -0.017)* | 0.037 (-0.004, +0.003) | 0.949 | 0.52 | not run | 6,832 MB / n/a | 7 EU | Apache-2.0 |
| opf | 7 | 0.915 (-0.050, -0.018)* | 0.771 (+0.138, +0.179)* | 0.751 (+0.365, +0.419)* | 0.123 (-0.068, -0.027)* | 0.047 (+0.006, +0.013)* | 0.949 | 0.29 | not run | 26,113 MB / n/a | primarily en | Apache-2.0 |
| presidio | 7 | 0.861 (-0.101, -0.074)* | 0.636 (+0.006, +0.039)* | 0.664 (+0.279, +0.336)* | 0.279 (+0.088, +0.128)* | 0.081 (+0.039, +0.048)* | 0.949 | 27 | not run | 826 MB / n/a | en (default); spaCy packs when tuned | MIT |
| gliner-nvidia | 3 | 0.790 (-0.179, -0.128)* | 0.538 (+0.004, +0.046)* | 0.698 (+0.365, +0.421)* | 0.272 (+0.077, +0.141)* | 0.075 (+0.001, +0.017)* | 0.943 | 0.15 | not run | 25,732 MB / n/a | en | Apache-2.0 + NVIDIA Open Model |
| datafog | 7 | 0.678 (-0.295, -0.246)* | 0.828 (+0.193, +0.241)* | 0.597 (+0.211, +0.265)* | 0.507 (+0.309, +0.362)* | 0.009 (-0.033, -0.026)* | 0.949 | 1,787 | not run | 30 MB / n/a | regex (US shapes; DE pack) | MIT |

### Shared view, privacy-real

| entry | sets | mR (entry - SCT, 95% CI) | mP (entry - SCT, 95% CI) | strict F1 (entry - SCT, 95% CI) | doc leak (entry - SCT, 95% CI) | over-mask (entry - SCT, 95% CI) | SCT mR on same sets | docs/s CPU | docs/s GPU | peak RAM / VRAM | languages | licence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sct-default | 2 | 0.982 [0.978, 0.987] | 0.195 [0.159, 0.238] | 0.155 [0.141, 0.177] | 0.181 [0.161, 0.205] | 0.044 [0.042, 0.047] | - | 1.50 | not run | 7,692 MB / n/a | 75 LID; NER en nl de es + multilingual | MIT |
| sct-gliner | 2 | 0.980 (-0.004, -0.000)* | 0.206 (+0.002, +0.021)* | 0.152 (-0.004, -0.002)* | 0.182 (-0.007, +0.009) | 0.040 (-0.006, -0.002)* | 0.982 | 0.53 | not run | 6,414 MB / n/a | 75 LID; GLiNER model tagged en | MIT + Apache-2.0 model |
| datafog | 2 | 0.768 (-0.332, -0.139)* | 0.750 (+0.514, +0.599)* | 0.351 (+0.174, +0.210)* | 0.812 (+0.609, +0.653)* | 0.008 (-0.039, -0.034)* | 0.982 | 1,787 | not run | 30 MB / n/a | regex (US shapes; DE pack) | MIT |
| gliner2-fastino | 2 | 0.487 (-0.520, -0.469)* | 0.181 (-0.050, +0.021) | 0.450 (+0.264, +0.319)* | 0.615 (+0.410, +0.456)* | 0.055 (+0.008, +0.013)* | 0.982 | 0.52 | not run | 6,832 MB / n/a | 7 EU | Apache-2.0 |
| opf | 2 | 0.469 (-0.525, -0.501)* | 0.426 (+0.005, +0.460)* | 0.391 (+0.209, +0.264)* | 0.622 (+0.413, +0.467)* | 0.030 (-0.018, -0.011)* | 0.982 | 0.29 | not run | 26,113 MB / n/a | primarily en | Apache-2.0 |
| presidio | 2 | 0.456 (-0.550, -0.509)* | 0.172 (-0.048, +0.001) | 0.325 (+0.149, +0.186)* | 0.721 (+0.515, +0.563)* | 0.075 (+0.027, +0.035)* | 0.982 | 27 | not run | 826 MB / n/a | en (default); spaCy packs when tuned | MIT |
| bardsai-v3 | 2 | 0.455 (-0.537, -0.518)* | 0.193 (-0.043, +0.037) | 0.415 (+0.237, +0.277)* | 0.646 (+0.440, +0.491)* | 0.027 (-0.020, -0.014)* | 0.982 | 2.52 | not run | 1,738 MB / n/a | 24 EU | Apache-2.0 |
| gliner-nvidia | 2 | 0.401 (-0.626, -0.480)* | 0.478 (+0.226, +0.340)* | 0.520 (+0.288, +0.483)* | 0.513 (+0.282, +0.383)* | 0.018 (-0.029, -0.024)* | 0.982 | 0.15 | not run | 25,732 MB / n/a | en | Apache-2.0 + NVIDIA Open Model |

## Per dataset (full view, masked F2, completed docs, all cells including in-domain)

Masked F2 = F2 of mR and mP (recall-weighted), the tuning objective. In-domain cells are in the
matrix below; read openpii, gretel_en, nemotron and argilla_pii for bardsai-v3 and gliner-nvidia
with it.

### Tuned

| dataset | bardsai-v3 | datafog | gliner-nvidia | gliner2-fastino | opf | presidio | sct-default | sct-gliner |
|---|---|---|---|---|---|---|---|---|
| openpii | 0.969 | 0.338 | 0.957 | 0.891 | 0.883 | 0.647 | 0.908 | 0.858 |
| gretel | 0.660 | 0.150 | 0.717 | 0.688 | 0.570 | 0.472 | 0.576 | 0.568 |
| gretel_en | 0.994 | 0.487 | 0.957 | 0.919 | 0.912 | 0.865 | 0.932 | 0.917 |
| nemotron | 0.979 | 0.357 | 0.916 | 0.873 | 0.681 | 0.780 | 0.870 | 0.818 |
| argilla_pii | 0.983 | 0.233 | 0.934 | 0.845 | 0.799 | 0.880 | 0.963 | 0.901 |
| thai_pii | 0.889 | 0.561 | 0.782 | 0.952 | 0.898 | 0.486 | 0.969 | 0.708 |
| bayanat | 0.893 | 0.527 | 0.731 | 0.920 | 0.898 | 0.751 | 0.926 | 0.742 |
| tab | 0.638 | 0.093 | 0.363 | 0.776 | 0.284 | 0.486 | 0.830 | 0.647 |
| redact | 0.801 | 0.274 | 0.611 | 0.808 | 0.743 | 0.549 | 0.778 | 0.678 |
| iahlt | 0.433 | 0.000 | 0.748 | 0.869 | 0.181 | 0.188 | 0.898 | 0.490 |
| thainer | 0.394 | 0.004 | 0.215 | 0.588 | 0.502 | 0.107 | 0.864 | 0.216 |
| naamapadam | 0.391 | 0.000 | 0.120 | 0.884 | 0.369 | 0.102 | 0.715 | 0.090 |
| multiconer | 0.225 | 0.000 | 0.313 | 0.642 | 0.269 | 0.076 | 0.746 | 0.229 |
| stockmark | 0.024 | 0.000 | 0.648 | 0.750 | 0.085 | 0.051 | 0.837 | 0.620 |

### Default

| dataset | bardsai-v3 | datafog | gliner-nvidia | gliner2-fastino | opf | presidio | sct-default | sct-gliner |
|---|---|---|---|---|---|---|---|---|
| openpii | 0.969 | 0.338 | 0.836 | 0.854 | 0.839 | 0.647 | 0.905 | 0.859 |
| gretel | 0.660 | 0.150 | 0.713 | 0.690 | 0.540 | 0.472 | 0.576 | 0.562 |
| gretel_en | 0.994 | 0.487 | 0.933 | 0.908 | 0.903 | 0.865 | 0.933 | 0.929 |
| nemotron | 0.979 | 0.357 | 0.867 | 0.879 | 0.649 | 0.780 | 0.870 | 0.844 |
| argilla_pii | 0.983 | 0.233 | 0.895 | 0.856 | 0.747 | 0.880 | 0.964 | 0.935 |
| thai_pii | 0.889 | 0.561 | 0.439 | 0.930 | 0.844 | 0.486 | 0.967 | 0.687 |
| bayanat | 0.893 | 0.527 | 0.722 | 0.914 | 0.887 | 0.751 | 0.927 | 0.772 |
| tab | 0.638 | 0.093 | 0.308 | 0.783 | 0.229 | 0.486 | 0.831 | 0.558 |
| redact | 0.801 | 0.274 | 0.562 | 0.802 | 0.708 | 0.549 | 0.777 | 0.677 |
| iahlt | 0.433 | 0.000 | 0.598 | 0.864 | 0.118 | 0.188 | 0.899 | 0.447 |
| thainer | 0.394 | 0.004 | 0.077 | 0.568 | 0.325 | 0.107 | 0.860 | 0.139 |
| naamapadam | 0.391 | 0.000 | 0.030 | 0.883 | 0.233 | 0.102 | 0.707 | 0.051 |
| multiconer | 0.225 | 0.000 | 0.211 | 0.628 | 0.185 | 0.076 | 0.740 | 0.194 |
| stockmark | 0.024 | 0.000 | 0.608 | 0.687 | 0.056 | 0.051 | 0.832 | 0.550 |

## Per language (tuned, full view, masked recall, completed docs)

Each cell: the mean over the group's datasets that have the language of that dataset's score
for the language. NER-group languages and the default track are in `TABLES.md`.

### Privacy-synthetic

| lang | datasets | bardsai-v3 | datafog | gliner-nvidia | gliner2-fastino | opf | presidio | sct-default | sct-gliner |
|---|---|---|---|---|---|---|---|---|---|
| ar-egyptian | bayanat | 0.846 | 0.423 | 0.750 | 0.949 | 0.904 | 0.673 | 1.000 | 0.942 |
| ar-gulf | bayanat | 0.906 | 0.592 | 0.718 | 0.919 | 0.933 | 0.758 | 0.996 | 0.942 |
| ar-mixed_en | bayanat | 0.953 | 0.470 | 0.926 | 0.993 | 0.987 | 0.933 | 1.000 | 0.980 |
| ar-msa | bayanat | 0.914 | 0.441 | 0.775 | 0.945 | 0.921 | 0.664 | 0.986 | 0.856 |
| bg | openpii | 0.984 | 0.408 | 0.987 | 0.872 | 0.887 | 0.677 | 0.938 | 0.929 |
| cs | openpii | 0.989 | 0.298 | 0.991 | 0.914 | 0.814 | 0.632 | 0.917 | 0.923 |
| da | openpii | 0.991 | 0.383 | 0.981 | 0.953 | 0.827 | 0.704 | 0.935 | 0.938 |
| de | gretel, openpii | 0.819 | 0.189 | 0.826 | 0.862 | 0.739 | 0.640 | 0.828 | 0.831 |
| el | openpii | 0.988 | 0.262 | 0.986 | 0.911 | 0.756 | 0.659 | 0.887 | 0.867 |
| en | argilla_pii, gretel, gretel_en, nemotron, openpii | 0.924 | 0.294 | 0.931 | 0.928 | 0.761 | 0.762 | 0.927 | 0.923 |
| es | gretel, openpii | 0.812 | 0.218 | 0.848 | 0.825 | 0.778 | 0.552 | 0.783 | 0.795 |
| et | openpii | 0.988 | 0.352 | 0.986 | 0.932 | 0.859 | 0.752 | 0.933 | 0.923 |
| fi | openpii | 0.991 | 0.340 | 0.990 | 0.940 | 0.779 | 0.729 | 0.940 | 0.922 |
| fr | gretel, openpii | 0.845 | 0.226 | 0.899 | 0.917 | 0.808 | 0.566 | 0.822 | 0.836 |
| hr | openpii | 0.993 | 0.368 | 0.984 | 0.908 | 0.866 | 0.703 | 0.928 | 0.929 |
| hu | openpii | 0.987 | 0.251 | 0.980 | 0.919 | 0.848 | 0.588 | 0.854 | 0.836 |
| id | openpii | 0.968 | 0.322 | 0.976 | 0.883 | 0.911 | 0.708 | 0.882 | 0.819 |
| it | gretel, openpii | 0.861 | 0.216 | 0.883 | 0.865 | 0.801 | 0.622 | 0.845 | 0.839 |
| ja | openpii | 0.595 | 0.157 | 0.766 | 0.697 | 0.868 | 0.430 | 0.874 | 0.836 |
| ko | openpii | 0.931 | 0.217 | 0.972 | 0.875 | 0.840 | 0.409 | 0.847 | 0.823 |
| lt | openpii | 0.988 | 0.338 | 0.982 | 0.910 | 0.787 | 0.753 | 0.933 | 0.905 |
| lv | openpii | 0.989 | 0.306 | 0.960 | 0.911 | 0.745 | 0.715 | 0.934 | 0.874 |
| ms | openpii | 0.979 | 0.319 | 0.981 | 0.894 | 0.910 | 0.749 | 0.883 | 0.848 |
| nl | gretel, openpii | 0.766 | 0.196 | 0.810 | 0.810 | 0.705 | 0.607 | 0.753 | 0.778 |
| pl | openpii | 0.991 | 0.212 | 0.975 | 0.903 | 0.885 | 0.651 | 0.890 | 0.899 |
| pt | openpii | 0.996 | 0.297 | 0.974 | 0.879 | 0.955 | 0.757 | 0.953 | 0.944 |
| ro | openpii | 0.993 | 0.233 | 0.989 | 0.884 | 0.838 | 0.727 | 0.895 | 0.861 |
| sk | openpii | 0.990 | 0.247 | 0.973 | 0.895 | 0.880 | 0.683 | 0.863 | 0.869 |
| sl | openpii | 0.989 | 0.213 | 0.986 | 0.918 | 0.824 | 0.712 | 0.905 | 0.921 |
| sr | openpii | 0.991 | 0.376 | 0.983 | 0.905 | 0.853 | 0.721 | 0.932 | 0.910 |
| sv | gretel, openpii | 0.810 | 0.207 | 0.839 | 0.846 | 0.681 | 0.608 | 0.792 | 0.808 |
| th | thai_pii | 0.884 | 0.512 | 0.766 | 0.953 | 0.914 | 0.440 | 0.977 | 0.674 |
| tl | openpii | 0.978 | 0.255 | 0.983 | 0.917 | 0.879 | 0.733 | 0.903 | 0.867 |
| vi | openpii | 0.963 | 0.295 | 0.972 | 0.863 | 0.922 | 0.695 | 0.888 | 0.864 |
| zh | openpii | 0.712 | 0.150 | 0.698 | 0.696 | 0.898 | 0.399 | 0.872 | 0.750 |

### Privacy-real

| lang | datasets | bardsai-v3 | datafog | gliner-nvidia | gliner2-fastino | opf | presidio | sct-default | sct-gliner |
|---|---|---|---|---|---|---|---|---|---|
| ar | redact | 0.895 | 0.351 | 0.687 | 0.967 | 0.811 | 0.530 | 0.950 | 0.752 |
| cs | redact | 0.917 | 0.211 | 0.601 | 0.902 | 0.726 | 0.723 | 0.939 | 0.809 |
| da | redact | 0.831 | 0.191 | 0.567 | 0.857 | 0.782 | 0.690 | 0.952 | 0.717 |
| de | redact | 0.864 | 0.242 | 0.705 | 0.877 | 0.785 | 0.754 | 0.926 | 0.879 |
| en | redact, tab | 0.713 | 0.163 | 0.439 | 0.914 | 0.523 | 0.612 | 0.949 | 0.799 |
| es | redact | 0.929 | 0.231 | 0.559 | 0.948 | 0.848 | 0.776 | 0.909 | 0.907 |
| fi | redact | 0.905 | 0.330 | 0.699 | 0.920 | 0.725 | 0.776 | 0.948 | 0.923 |
| fr | redact | 0.893 | 0.202 | 0.673 | 0.928 | 0.824 | 0.650 | 0.925 | 0.890 |
| fr_ca | redact | 0.918 | 0.309 | 0.676 | 0.934 | 0.839 | 0.740 | 0.936 | 0.949 |
| he | redact | 0.881 | 0.310 | 0.603 | 0.955 | 0.654 | 0.555 | 0.946 | 0.630 |
| hi | redact | 0.887 | 0.240 | 0.345 | 0.905 | 0.782 | 0.542 | 0.905 | 0.622 |
| hu | redact | 0.906 | 0.212 | 0.686 | 0.920 | 0.834 | 0.643 | 0.949 | 0.877 |
| it | redact | 0.879 | 0.283 | 0.593 | 0.871 | 0.838 | 0.687 | 0.896 | 0.856 |
| ja | redact | 0.615 | 0.151 | 0.595 | 0.744 | 0.835 | 0.348 | 0.886 | 0.858 |
| ko | redact | 0.885 | 0.159 | 0.718 | 0.906 | 0.838 | 0.312 | 0.923 | 0.697 |
| nl | redact | 0.897 | 0.239 | 0.539 | 0.901 | 0.830 | 0.816 | 0.876 | 0.832 |
| no | redact | 0.919 | 0.249 | 0.608 | 0.924 | 0.751 | 0.779 | 0.947 | 0.905 |
| pt_br | redact | 0.917 | 0.220 | 0.610 | 0.915 | 0.898 | 0.807 | 0.873 | 0.850 |
| pt_eu | redact | 0.854 | 0.230 | 0.575 | 0.894 | 0.807 | 0.767 | 0.953 | 0.876 |
| ru | redact | 0.875 | 0.170 | 0.706 | 0.920 | 0.817 | 0.585 | 0.972 | 0.907 |
| sv | redact | 0.910 | 0.313 | 0.586 | 0.935 | 0.797 | 0.827 | 0.930 | 0.903 |
| th | redact | 0.884 | 0.421 | 0.454 | 0.926 | 0.796 | 0.370 | 0.972 | 0.542 |
| tr | redact | 0.938 | 0.208 | 0.591 | 0.941 | 0.820 | 0.694 | 0.866 | 0.823 |
| zh_cn | redact | 0.546 | 0.206 | 0.465 | 0.673 | 0.774 | 0.259 | 0.969 | 0.699 |
| zh_tw | redact | 0.519 | 0.201 | 0.550 | 0.744 | 0.761 | 0.258 | 0.945 | 0.785 |

## Contamination matrix

T = trained on it (T-family: same generator), E = vendor evaluates on it, D = SCT developed
against it in earlier rounds, U = training data undisclosed, - = none known (model cards at the
pinned revisions, checked 2026-11-02). Bold = in-domain: left out of every headline pair above.
The older "out-of-domain" rows in `TABLES.md` are stricter (only `-` cells: SCT keeps 2 sets,
opf and sct-gliner none).

| entry | openpii | gretel | gretel_en | nemotron | argilla_pii | thai_pii | bayanat | tab | redact | iahlt | thainer | naamapadam | multiconer | stockmark |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sct-default | D | D | - | D | - | D | D | D | D | D | D | D | D | D |
| sct-gliner | U, D | U, D | U | U, D | U | U, D | U, D | U, D | U, D | U, D | U, D | U, D | U, D | U, D |
| presidio | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| gliner-nvidia | **E** | - | **E** | **T** | **E** | - | - | - | - | - | - | - | - | - |
| gliner2-fastino | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| opf | U | U | U | U | U | U | U | U | U | U | U | U | U | U |
| bardsai-v3 | **T** | - | **T** | **T** | **T-family** | - | - | - | - | - | - | - | - | - |
| datafog | - | - | - | - | - | - | - | - | - | - | - | - | - | - |

## Feature matrix (not scored)

| entry | languages (claimed) | types mapped to SCT's 9 | char spans | reversible | fake values | text cleaning | language ID | batch API | GPU | offline | licence (code / model) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| sct-default | 75 (LID); NER en, nl, de, es + multilingual WikiNEural | 9 | via reversible mode | yes (`TokenStore`) | yes (`synthetic` extra) | yes | yes (Lingua) | `process_batch` | optional (CUDA ONNX) | yes | MIT / per NER model card |
| sct-gliner | as SCT; GLiNER model tagged en | 9 | via reversible mode | yes | yes | yes | yes | `process_batch` | optional | yes | MIT / Apache-2.0 |
| presidio | en by default; spaCy packs per language | 8 (no ORG with spaCy) | yes | encrypt / decrypt operator | custom operator | no | no | `BatchAnalyzerEngine` | optional (transformers engine) | yes | MIT / MIT (spaCy model) |
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
sct-gliner: gretel 658, openpii 250, redact 206. On gretel, 155 of the 169 e-mail mentions the
detection track counts as missed for sct-default are masked in SCT's own output (markdown links
`[x](mailto:x)` that SCT's cleaning rewrites), so this costs SCT recall it does not lose in use.

Other problems: no GPU run (driver vs pinned torch wheel, see limits); gliner-nvidia batch 8 on
CPU peaks at ~26 GB RSS and is slower than batch 1 (GLiNER pads the batch); another job of the
same account held cores 60-63 during the quality runs, so the sct-gliner default run was finished
on cores 0-15 (quality does not depend on core placement).

## Where SCT loses, and the likely cause

Ranked by size and certainty, tuned track, in-domain cells out. No code was changed for this report.

1. **Exact spans and types (strict F1).** Every model entry beats SCT, on every view and group:
   full synthetic SCT 0.481 vs gliner2-fastino 0.683 (+0.19 to +0.22), bardsai-v3 +0.26 to +0.30
   on its 3 clean sets, opf +0.06 to +0.09; shared synthetic SCT 0.357 vs +0.28 to +0.52 for all
   models; real (full) gliner2-fastino +0.13 to +0.16. Likely cause: type confusion between
   PHONE and NUMBER (on openpii SCT puts a PHONE prediction on 1,884 credit-card, 1,384 tax,
   1,343 social-security and 1,100 passport numbers; on nemotron on account, card, customer and
   routing numbers), so the value is masked but typed wrong; plus span edges shifted by recovering
   offsets through reversible placeholders after cleaning. Privacy is not hurt (masked recall is
   type-agnostic); users who need typed spans are.
2. **Precision on numbers (shared view, and real data).** SCT masks most digit runs. Shared view
   mP: synthetic SCT 0.611 vs opf 0.720 (+0.09 to +0.13), bardsai-v3 +0.10 to +0.15, datafog
   0.828; real SCT 0.193 vs datafog 0.750, opf 0.462, gliner-nvidia 0.367. Full-view real mP:
   SCT 0.523 vs bardsai-v3 0.809, gliner-nvidia 0.831, opf 0.775. Per type, SCT's NUMBER
   precision is 0.016 on gretel, 0.05 on REDACT, 0.13 on nemotron; PHONE 0.10 on gretel.
   Likely cause: the NUMBER/PHONE regexes fire on case numbers, statute and section numbers,
   legal citations and survey age brackets (688 NUMBER predictions on nemotron overlap no gold at
   all), which the datasets do not annotate. This is SCT's recall-first design, but it costs
   utility of the masked text.
3. **Documents with any exposed mention (synthetic, shared view).** At equal masked recall
   (SCT 0.949 vs gliner2-fastino 0.944, opf 0.944, differences not significant), SCT leaves a
   mention partly visible in more documents: doc leak 0.171 vs gliner2-fastino 0.097, opf 0.086,
   bardsai-v3 0.074 (all CIs exclude 0); character survival 0.064 vs 0.044 (fastino). Likely
   cause: partial coverage of e-mail/phone/ID mentions (span edges leave digits, e.g. a
   country code or a bracketed copy) and the unplaced spans above (a scoring artefact for the
   markdown e-mails). In the full view SCT's survival is the lowest of all entries; the gap is
   specific to structured identifiers.
4. **gretel finance, the one privacy set clean for every model.** Full-view masked F2: SCT
   0.576 vs gliner-nvidia 0.717, gliner2-fastino 0.688, bardsai-v3 0.660; shared-view masked
   recall per-dataset bootstrap: bardsai-v3, gliner-nvidia, gliner2-fastino significantly above
   SCT. Likely causes: (a) markdown mailto links: SCT's e-mail mR there is 0.67 vs >= 0.96 for
   the models, but 155 of the 169 misses are masked in SCT's own output (harness cannot place
   them: scoring artefact); (b) ORGANISATION recall 0.74 vs gliner2-fastino 0.95; (c) NUMBER and
   PHONE precision 0.016 and 0.10. PERSON recall is low for every entry (0.57-0.66): gretel gold
   comes from an LLM plus GLiNER.
5. **Speed and memory.** CPU, 16 threads, batch 8: SCT 1.50 docs/s vs bardsai-v3 2.52, presidio
   27, datafog 1,787 (it is faster than gliner2-fastino 0.52, sct-gliner 0.53, opf 0.29 and
   gliner-nvidia 0.15). Single thread: median 4.2 s per doc, slower than everything but
   gliner-nvidia. Peak RSS 7.7 GB at 16 threads (5.9 GB at 1) vs bardsai-v3 1.7 GB, presidio
   0.8 GB. Likely cause: the speed runs use native routing, so each document goes through Lingua
   LID, cleaning, and an ensemble of full-precision ONNX NER models (a per-language model plus
   WikiNEural), with a model session per language loaded; bardsai-v3 is one INT8 model.

Also below SCT, smaller or narrower: naamapadam (NER, Indic): gliner2-fastino masked F2 0.884 vs
SCT 0.715 (recall 0.96 vs 0.71; SCT has no Indic NER model); REDACT: gliner2-fastino 0.808 and
bardsai-v3 0.801 vs SCT 0.778 masked F2 (SCT has the higher recall, 0.93 vs 0.89, and loses on
precision). openpii, gretel_en, nemotron and argilla_pii losses to bardsai-v3 and gliner-nvidia
are in-domain cells and not counted here; on the two sets held out for SCT (gretel_en,
argilla_pii) SCT is first among the entries without exposure there.

Where SCT leads (for balance): full-view masked recall on both privacy groups, all CIs exclude 0
against every entry (shared synthetic view: no significant difference to opf, gliner2-fastino
and sct-gliner, and bardsai-v3 +0.005 to +0.029 on its 3 clean sets); real-data recall 0.933 vs
next best 0.895 (gliner2-fastino); on synthetic data the lowest over-masking of the model entries
except bardsai-v3 (on real data bardsai-v3 and gliner-nvidia over-mask less); and every NER set
except naamapadam.

## Limits of this comparison

- **No GPU numbers.** The pinned torch 2.15.0 wheels are CUDA 13.2 builds and the box's driver is
  CUDA 12.6, so no torch entry could see the GPU; tuning, test and speed all ran on CPU (recorded
  as `device: cpu` in every result). gliner-nvidia ran fp32 (its fp16 is the CUDA path). GPU
  docs/s and VRAM columns are "not run".
- **Contamination is from cards, not data.** T/E flags come from model cards; opf and the
  knowledgator model behind sct-gliner do not disclose training data (U), so their cells stay in
  the headline. SCT's D flags (developed against most sets in earlier rounds) also stay in; only
  gretel_en and argilla_pii are fully held out for SCT. bardsai-v3 and gliner-nvidia keep 3
  synthetic sets each, so their synthetic rows rest on gretel, thai_pii and bayanat.
- **SCT is scored through its reversible view.** SCT returns rewritten text; its spans are
  recovered from placeholders. Where cleaning rewrites the text (markdown links), spans cannot be
  placed and count against SCT (gretel 455 unplaced). Native-track leaks (SCT's own output) are
  in each result file's `native` field and are lower than the detection-track numbers suggest.
- **Narrow shared view.** EMAIL, PHONE, NUMBER only (DataFog supports nothing else; DATE out).
  Read it with the full view, where unsupported types count as missed.
- **Gold quality.** gretel finance (LLM + GLiNER), gretel_en (generator strings placed at every
  exact occurrence), argilla_pii (unreviewed suggestions), nemotron (1.5% span mismatches). The
  blind annotation audit (rule 10) was not done, so precision is a lower bound on these sets.
- **Template leakage.** openpii, argilla_pii and ai4privacy-derived sets have no template id: the
  dev/test split is by record, so dev tuning may have seen near-duplicates of test templates.
- **Not run** (protocol change log): the native language track (SCT's own LID vs gold), the
  common chunker for long documents, matched operating points (mR 0.90/0.95), PR and
  leak-vs-over-masking plots, GPU repeat runs, downstream utility beyond chrF (rule 5).
- **Speed sample.** 100 FineWeb-2 docs (not 100 per language), single-thread run 1 repeat
  (3 would take ~4 h). Peak RSS is VmHWM of the adapter process.
- **One box, one seed.** Bootstrap CIs cover document sampling only, not seeds, hardware or
  dataset choice. 300 docs per language; REDACT has 27-40 per language, TAB 106 docs.
