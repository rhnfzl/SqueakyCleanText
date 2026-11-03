## default track, full view, privacy-synthetic: each entry on its own clean sets, claimed languages

| entry | clean sets with claimed-language docs (all clean sets) | mR [95% CI] | mP [95% CI] | strict F1 [95% CI] | doc leak [95% CI] | over-mask [95% CI] | mR clean, outside claimed langs | mR unknown (U) sets | mR U sets, outside claimed langs | (mR excluded sets, info only) | docs/s CPU | peak RAM | claimed languages | licence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gliner2-fastino | argilla_pii, gretel, gretel_en, nemotron, openpii (7) | 0.872 [0.861, 0.882] | 0.746 [0.727, 0.765] | 0.702 [0.688, 0.717] | 0.335 [0.315, 0.355] | 0.036 [0.033, 0.040] | 0.849 (4) | - | - | - | 0.52 | 6,832 MB | en fr es de it pt nl | Apache-2.0 |
| presidio | argilla_pii, gretel, gretel_en, nemotron, openpii (7) | 0.762 [0.742, 0.781] | 0.803 [0.773, 0.838] | 0.540 [0.521, 0.559] | 0.608 [0.584, 0.633] | 0.023 [0.019, 0.029] | 0.605 (4) | - | - | - | 27 | 826 MB | en | MIT |
| datafog | argilla_pii, bayanat, gretel, gretel_en, nemotron, openpii, thai_pii (7) | 0.337 [0.325, 0.348] | 0.854 [0.833, 0.878] | 0.342 [0.328, 0.355] | 0.919 [0.908, 0.929] | 0.009 [0.007, 0.010] | - | - | - | - | 1,787 | 30 MB | all (regex, no language claim) | MIT |
| sct-gliner | none (0) | - | - | - | - | - | - | - | - | (0.930 (5); w/o recovery 0.914 (5)) | 0.53 | 6,414 MB | en | MIT + Apache-2.0 model |
| sct-default | none (0) | - | - | - | - | - | - | - | - | (0.934 (7); w/o recovery 0.926 (7)) | 1.50 | 7,692 MB | all (multilingual claim) | MIT |
| opf | none (0) | - | - | - | - | - | - | 0.722 (5) | 0.767 (4) | - | 0.29 | 26,113 MB | en | Apache-2.0 |
| gliner-nvidia | none (2) | - | - | - | - | - | 0.563 (2) | - | - | (0.889 (5)) | 0.15 | 25,732 MB | en | Apache-2.0 + NVIDIA Open Model |
| bardsai-v3 | none (2) | - | - | - | - | - | 0.895 (2) | - | - | (0.919 (5)) | 2.52 | 1,738 MB | 24 EU | Apache-2.0 |

### same-set (every entry)

No dataset in this group is clean for every one of: sct-default, sct-gliner, presidio, gliner-nvidia, gliner2-fastino, opf, bardsai-v3, datafog (in languages all of them claim).

### same-set (non-SCT entries with clean sets)

No dataset in this group is clean for every one of: presidio, gliner-nvidia, gliner2-fastino, bardsai-v3, datafog (in languages all of them claim).

### SCT vs each entry, datasets clean for both, languages both claim

No dataset in this group is clean for both sct-default and any other entry (languages both claim).

### SCT vs each entry, datasets unknown for both, languages both claim

No dataset in this group is unknown for both sct-default and any other entry (languages both claim).

## default track, full view, privacy-real: each entry on its own clean sets, claimed languages

| entry | clean sets with claimed-language docs (all clean sets) | mR [95% CI] | mP [95% CI] | strict F1 [95% CI] | doc leak [95% CI] | over-mask [95% CI] | mR clean, outside claimed langs | mR unknown (U) sets | mR U sets, outside claimed langs | (mR excluded sets, info only) | docs/s CPU | peak RAM | claimed languages | licence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gliner2-fastino | redact, tab (2) | 0.880 [0.858, 0.900] | 0.590 [0.570, 0.611] | 0.544 [0.526, 0.562] | 0.865 [0.839, 0.890] | 0.056 [0.053, 0.059] | 0.849 (1) | - | - | - | 0.52 | 6,832 MB | en fr es de it pt nl | Apache-2.0 |
| bardsai-v3 | redact, tab (2) | 0.739 [0.714, 0.764] | 0.817 [0.804, 0.831] | 0.512 [0.495, 0.529] | 0.830 [0.806, 0.852] | 0.022 [0.020, 0.024] | 0.804 (1) | - | - | - | 2.52 | 1,738 MB | 24 EU | Apache-2.0 |
| presidio | redact, tab (2) | 0.612 [0.576, 0.648] | 0.651 [0.605, 0.697] | 0.319 [0.285, 0.349] | 0.922 [0.850, 0.984] | 0.045 [0.034, 0.057] | 0.620 (1) | - | - | - | 27 | 826 MB | en | MIT |
| gliner-nvidia | redact, tab (2) | 0.391 [0.335, 0.462] | 0.793 [0.752, 0.828] | 0.288 [0.245, 0.333] | 0.953 [0.891, 1.000] | 0.016 [0.012, 0.022] | 0.544 (1) | - | - | - | 0.15 | 25,732 MB | en | Apache-2.0 + NVIDIA Open Model |
| datafog | redact, tab (2) | 0.161 [0.144, 0.182] | 0.765 [0.751, 0.790] | 0.155 [0.147, 0.162] | 0.997 [0.994, 1.000] | 0.008 [0.007, 0.009] | - | - | - | - | 1,787 | 30 MB | all (regex, no language claim) | MIT |
| sct-gliner | none (0) | - | - | - | - | - | - | - | - | (0.770 (2); w/o recovery 0.737 (2)) | 0.53 | 6,414 MB | en | MIT + Apache-2.0 model |
| sct-default | none (0) | - | - | - | - | - | - | - | - | (0.956 (2); w/o recovery 0.932 (2)) | 1.50 | 7,692 MB | all (multilingual claim) | MIT |
| opf | none (0) | - | - | - | - | - | - | 0.458 (2) | 0.728 (1) | - | 0.29 | 26,113 MB | en | Apache-2.0 |

### same-set (every entry)

No dataset in this group is clean for every one of: sct-default, sct-gliner, presidio, gliner-nvidia, gliner2-fastino, opf, bardsai-v3, datafog (in languages all of them claim).

### same-set (non-SCT entries with clean sets)

Datasets: redact, tab (clean for all: tab, redact); only docs in languages every entry claims; point values.

| entry | mR | mP | strict F1 | doc leak | over-mask |
|---|---|---|---|---|---|
| gliner2-fastino | 0.889 | 0.546 | 0.501 | 0.828 | 0.061 |
| bardsai-v3 | 0.713 | 0.758 | 0.479 | 0.859 | 0.028 |
| presidio | 0.612 | 0.651 | 0.319 | 0.922 | 0.045 |
| gliner-nvidia | 0.391 | 0.793 | 0.288 | 0.953 | 0.016 |
| datafog | 0.163 | 0.756 | 0.143 | 1.000 | 0.007 |

### SCT vs each entry, datasets clean for both, languages both claim

No dataset in this group is clean for both sct-default and any other entry (languages both claim).

### SCT vs each entry, datasets unknown for both, languages both claim

No dataset in this group is unknown for both sct-default and any other entry (languages both claim).

## default track, full view, ner: each entry on its own clean sets, claimed languages

| entry | clean sets with claimed-language docs (all clean sets) | mR [95% CI] | mP [95% CI] | strict F1 [95% CI] | doc leak [95% CI] | over-mask [95% CI] | mR clean, outside claimed langs | mR unknown (U) sets | mR U sets, outside claimed langs | (mR excluded sets, info only) | docs/s CPU | peak RAM | claimed languages | licence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| datafog | iahlt, multiconer, naamapadam, stockmark, thainer (5) | 0.001 [0.000, 0.002] | 0.333 [0.000, 1.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | - | - | - | - | 1,787 | 30 MB | all (regex, no language claim) | MIT |
| sct-gliner | none (0) | - | - | - | - | - | - | - | - | - | 0.53 | 6,414 MB | en | MIT + Apache-2.0 model |
| sct-default | none (0) | - | - | - | - | - | - | - | - | (0.848 (5); w/o recovery 0.848 (5)) | 1.50 | 7,692 MB | all (multilingual claim) | MIT |
| presidio | none (5) | - | - | - | - | - | 0.095 (5) | - | - | - | 27 | 826 MB | en | MIT |
| opf | none (0) | - | - | - | - | - | - | - | 0.157 (5) | - | 0.29 | 26,113 MB | en | Apache-2.0 |
| gliner2-fastino | none (5) | - | - | - | - | - | 0.739 (5) | - | - | - | 0.52 | 6,832 MB | en fr es de it pt nl | Apache-2.0 |
| gliner-nvidia | none (5) | - | - | - | - | - | 0.287 (5) | - | - | - | 0.15 | 25,732 MB | en | Apache-2.0 + NVIDIA Open Model |
| bardsai-v3 | none (5) | - | - | - | - | - | 0.256 (5) | - | - | - | 2.52 | 1,738 MB | 24 EU | Apache-2.0 |

### same-set (every entry)

No dataset in this group is clean for every one of: sct-default, sct-gliner, presidio, gliner-nvidia, gliner2-fastino, opf, bardsai-v3, datafog (in languages all of them claim).

### same-set (non-SCT entries with clean sets)

No dataset in this group is clean for every one of: presidio, gliner-nvidia, gliner2-fastino, bardsai-v3, datafog (in languages all of them claim).

### SCT vs each entry, datasets clean for both, languages both claim

No dataset in this group is clean for both sct-default and any other entry (languages both claim).

### SCT vs each entry, datasets unknown for both, languages both claim

No dataset in this group is unknown for both sct-default and any other entry (languages both claim).

## default track, shared view, privacy-synthetic: each entry on its own clean sets, claimed languages

| entry | clean sets with claimed-language docs (all clean sets) | mR [95% CI] | mP [95% CI] | strict F1 [95% CI] | doc leak [95% CI] | over-mask [95% CI] | mR clean, outside claimed langs | mR unknown (U) sets | mR U sets, outside claimed langs | (mR excluded sets, info only) | docs/s CPU | peak RAM | claimed languages | licence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gliner2-fastino | argilla_pii, gretel, gretel_en, nemotron, openpii (7) | 0.917 [0.902, 0.930] | 0.528 [0.504, 0.553] | 0.827 [0.793, 0.858] | 0.147 [0.129, 0.166] | 0.036 [0.033, 0.040] | 0.904 (4) | - | - | - | 0.52 | 6,832 MB | en fr es de it pt nl | Apache-2.0 |
| presidio | argilla_pii, gretel, gretel_en, nemotron, openpii (7) | 0.854 [0.831, 0.873] | 0.682 [0.650, 0.721] | 0.698 [0.656, 0.752] | 0.318 [0.289, 0.349] | 0.023 [0.019, 0.029] | 0.821 (4) | - | - | - | 27 | 826 MB | en | MIT |
| datafog | argilla_pii, bayanat, gretel, gretel_en, nemotron, openpii, thai_pii (7) | 0.678 [0.656, 0.700] | 0.828 [0.808, 0.852] | 0.597 [0.572, 0.623] | 0.507 [0.482, 0.532] | 0.009 [0.007, 0.010] | - | - | - | - | 1,787 | 30 MB | all (regex, no language claim) | MIT |
| sct-gliner | none (0) | - | - | - | - | - | - | - | - | (0.946 (5); w/o recovery 0.928 (5)) | 0.53 | 6,414 MB | en | MIT + Apache-2.0 model |
| sct-default | none (0) | - | - | - | - | - | - | - | - | (0.961 (7); w/o recovery 0.949 (7)) | 1.50 | 7,692 MB | all (multilingual claim) | MIT |
| opf | none (0) | - | - | - | - | - | - | 0.877 (5) | 0.917 (4) | - | 0.29 | 26,113 MB | en | Apache-2.0 |
| gliner-nvidia | none (2) | - | - | - | - | - | 0.722 (2) | - | - | (0.901 (5)) | 0.15 | 25,732 MB | en | Apache-2.0 + NVIDIA Open Model |
| bardsai-v3 | none (2) | - | - | - | - | - | 0.985 (2) | - | - | (0.978 (5)) | 2.52 | 1,738 MB | 24 EU | Apache-2.0 |

### same-set (every entry)

No dataset in this group is clean for every one of: sct-default, sct-gliner, presidio, gliner-nvidia, gliner2-fastino, opf, bardsai-v3, datafog (in languages all of them claim).

### same-set (non-SCT entries with clean sets)

No dataset in this group is clean for every one of: presidio, gliner-nvidia, gliner2-fastino, bardsai-v3, datafog (in languages all of them claim).

### SCT vs each entry, datasets clean for both, languages both claim

No dataset in this group is clean for both sct-default and any other entry (languages both claim).

### SCT vs each entry, datasets unknown for both, languages both claim

No dataset in this group is unknown for both sct-default and any other entry (languages both claim).

## default track, shared view, privacy-real: each entry on its own clean sets, claimed languages

| entry | clean sets with claimed-language docs (all clean sets) | mR [95% CI] | mP [95% CI] | strict F1 [95% CI] | doc leak [95% CI] | over-mask [95% CI] | mR clean, outside claimed langs | mR unknown (U) sets | mR U sets, outside claimed langs | (mR excluded sets, info only) | docs/s CPU | peak RAM | claimed languages | licence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| datafog | redact, tab (2) | 0.768 [0.648, 0.846] | 0.750 [0.736, 0.775] | 0.351 [0.341, 0.360] | 0.812 [0.795, 0.830] | 0.008 [0.007, 0.009] | - | - | - | - | 1,787 | 30 MB | all (regex, no language claim) | MIT |
| gliner2-fastino | redact, tab (2) | 0.495 [0.467, 0.524] | 0.166 [0.149, 0.186] | 0.452 [0.429, 0.475] | 0.600 [0.574, 0.625] | 0.056 [0.053, 0.059] | 0.916 (1) | - | - | - | 0.52 | 6,832 MB | en fr es de it pt nl | Apache-2.0 |
| bardsai-v3 | redact, tab (2) | 0.472 [0.464, 0.479] | 0.198 [0.180, 0.220] | 0.433 [0.421, 0.447] | 0.621 [0.601, 0.643] | 0.022 [0.020, 0.024] | 0.864 (1) | - | - | - | 2.52 | 1,738 MB | 24 EU | Apache-2.0 |
| presidio | redact, tab (2) | 0.466 [0.429, 0.498] | 0.155 [0.112, 0.204] | 0.290 [0.234, 0.354] | 0.714 [0.625, 0.803] | 0.045 [0.034, 0.057] | 0.852 (1) | - | - | - | 27 | 826 MB | en | MIT |
| gliner-nvidia | redact, tab (2) | 0.398 [0.311, 0.520] | 0.457 [0.397, 0.516] | 0.515 [0.423, 0.642] | 0.480 [0.380, 0.587] | 0.016 [0.012, 0.022] | 0.570 (1) | - | - | - | 0.15 | 25,732 MB | en | Apache-2.0 + NVIDIA Open Model |
| sct-gliner | none (0) | - | - | - | - | - | - | - | - | (0.986 (2); w/o recovery 0.986 (2)) | 0.53 | 6,414 MB | en | MIT + Apache-2.0 model |
| sct-default | none (0) | - | - | - | - | - | - | - | - | (0.985 (2); w/o recovery 0.982 (2)) | 1.50 | 7,692 MB | all (multilingual claim) | MIT |
| opf | none (0) | - | - | - | - | - | - | 0.452 (2) | 0.926 (1) | - | 0.29 | 26,113 MB | en | Apache-2.0 |

### same-set (every entry)

No dataset in this group is clean for every one of: sct-default, sct-gliner, presidio, gliner-nvidia, gliner2-fastino, opf, bardsai-v3, datafog (in languages all of them claim).

### same-set (non-SCT entries with clean sets)

Datasets: redact, tab (clean for all: tab, redact); only docs in languages every entry claims; point values.

| entry | mR | mP | strict F1 | doc leak | over-mask |
|---|---|---|---|---|---|
| datafog | 0.758 | 0.724 | 0.342 | 0.822 | 0.007 |
| gliner2-fastino | 0.507 | 0.127 | 0.434 | 0.572 | 0.061 |
| bardsai-v3 | 0.468 | 0.145 | 0.418 | 0.625 | 0.028 |
| presidio | 0.466 | 0.155 | 0.290 | 0.714 | 0.045 |
| gliner-nvidia | 0.398 | 0.457 | 0.515 | 0.480 | 0.016 |

### SCT vs each entry, datasets clean for both, languages both claim

No dataset in this group is clean for both sct-default and any other entry (languages both claim).

### SCT vs each entry, datasets unknown for both, languages both claim

No dataset in this group is unknown for both sct-default and any other entry (languages both claim).

## default track, shared view, ner: each entry on its own clean sets, claimed languages

| entry | clean sets with claimed-language docs (all clean sets) | mR [95% CI] | mP [95% CI] | strict F1 [95% CI] | doc leak [95% CI] | over-mask [95% CI] | mR clean, outside claimed langs | mR unknown (U) sets | mR U sets, outside claimed langs | (mR excluded sets, info only) | docs/s CPU | peak RAM | claimed languages | licence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sct-gliner | none (0) | - | - | - | - | - | - | - | - | - | 0.53 | 6,414 MB | en | MIT + Apache-2.0 model |
| sct-default | none (0) | - | - | - | - | - | - | - | - | (1.000 (5); w/o recovery 1.000 (5)) | 1.50 | 7,692 MB | all (multilingual claim) | MIT |
| presidio | none (5) | - | - | - | - | - | 0.583 (5) | - | - | - | 27 | 826 MB | en | MIT |
| opf | none (0) | - | - | - | - | - | - | - | 0.583 (5) | - | 0.29 | 26,113 MB | en | Apache-2.0 |
| gliner2-fastino | none (5) | - | - | - | - | - | 0.833 (5) | - | - | - | 0.52 | 6,832 MB | en fr es de it pt nl | Apache-2.0 |
| gliner-nvidia | none (5) | - | - | - | - | - | 0.417 (5) | - | - | - | 0.15 | 25,732 MB | en | Apache-2.0 + NVIDIA Open Model |
| datafog | iahlt, multiconer, naamapadam, stockmark, thainer (5) | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | - | - | - | - | 1,787 | 30 MB | all (regex, no language claim) | MIT |
| bardsai-v3 | none (5) | - | - | - | - | - | 0.500 (5) | - | - | - | 2.52 | 1,738 MB | 24 EU | Apache-2.0 |

### same-set (every entry)

No dataset in this group is clean for every one of: sct-default, sct-gliner, presidio, gliner-nvidia, gliner2-fastino, opf, bardsai-v3, datafog (in languages all of them claim).

### same-set (non-SCT entries with clean sets)

No dataset in this group is clean for every one of: presidio, gliner-nvidia, gliner2-fastino, bardsai-v3, datafog (in languages all of them claim).

### SCT vs each entry, datasets clean for both, languages both claim

No dataset in this group is clean for both sct-default and any other entry (languages both claim).

### SCT vs each entry, datasets unknown for both, languages both claim

No dataset in this group is unknown for both sct-default and any other entry (languages both claim).

## default track, per language, full view, masked recall, privacy-synthetic, clean sets (° = outside the entry's claimed languages)

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

## default track, per language, full view, masked recall, privacy-synthetic, unknown sets (° = outside the entry's claimed languages)

| lang | datasets | opf |
|---|---|---|
| ar-egyptian | bayanat | 0.853° |
| ar-gulf | bayanat | 0.910° |
| ar-mixed_en | bayanat | 0.973° |
| ar-msa | bayanat | 0.882° |
| bg | openpii | 0.834° |
| cs | openpii | 0.741° |
| da | openpii | 0.770° |
| de | gretel, openpii | 0.696° |
| el | openpii | 0.657° |
| en | argilla_pii, gretel, gretel_en, nemotron, openpii | 0.722 |
| es | gretel, openpii | 0.739° |
| et | openpii | 0.799° |
| fi | openpii | 0.689° |
| fr | gretel, openpii | 0.776° |
| hr | openpii | 0.813° |
| hu | openpii | 0.770° |
| id | openpii | 0.863° |
| it | gretel, openpii | 0.758° |
| ja | openpii | 0.798° |
| ko | openpii | 0.780° |
| lt | openpii | 0.693° |
| lv | openpii | 0.661° |
| ms | openpii | 0.849° |
| nl | gretel, openpii | 0.652° |
| pl | openpii | 0.823° |
| pt | openpii | 0.912° |
| ro | openpii | 0.767° |
| sk | openpii | 0.810° |
| sl | openpii | 0.774° |
| sr | openpii | 0.784° |
| sv | gretel, openpii | 0.623° |
| th | thai_pii | 0.836° |
| tl | openpii | 0.812° |
| vi | openpii | 0.890° |
| zh | openpii | 0.847° |

## default track, per language, full view, masked recall, privacy-real, clean sets (° = outside the entry's claimed languages)

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

## default track, per language, full view, masked recall, privacy-real, unknown sets (° = outside the entry's claimed languages)

| lang | datasets | opf |
|---|---|---|
| ar | redact | 0.742° |
| cs | redact | 0.642° |
| da | redact | 0.695° |
| de | redact | 0.725° |
| en | redact, tab | 0.458 |
| es | redact | 0.817° |
| fi | redact | 0.619° |
| fr | redact | 0.763° |
| fr_ca | redact | 0.783° |
| he | redact | 0.564° |
| hi | redact | 0.706° |
| hu | redact | 0.735° |
| it | redact | 0.806° |
| ja | redact | 0.704° |
| ko | redact | 0.788° |
| nl | redact | 0.788° |
| no | redact | 0.675° |
| pt_br | redact | 0.831° |
| pt_eu | redact | 0.770° |
| ru | redact | 0.765° |
| sv | redact | 0.741° |
| th | redact | 0.727° |
| tr | redact | 0.751° |
| zh_cn | redact | 0.695° |
| zh_tw | redact | 0.632° |

## default track, per language, full view, masked recall, ner, clean sets (° = outside the entry's claimed languages)

| lang | datasets | bardsai-v3 | datafog | gliner-nvidia | gliner2-fastino | presidio |
|---|---|---|---|---|---|---|
| ar | iahlt | 0.380° | 0.000 | 0.569° | 0.891° | 0.189° |
| bn | multiconer, naamapadam | 0.188° | 0.000 | 0.015° | 0.895° | 0.078° |
| hi | multiconer, naamapadam | 0.388° | 0.000 | 0.015° | 0.950° | 0.088° |
| ja | stockmark | 0.019° | 0.000 | 0.581° | 0.667° | 0.042° |
| ta | naamapadam | 0.302° | 0.000 | 0.029° | 0.959° | 0.074° |
| th | thainer | 0.344° | 0.003 | 0.063° | 0.521° | 0.089° |
| zh | multiconer | 0.163° | 0.000 | 0.575° | 0.193° | 0.052° |

## default track, per language, full view, masked recall, ner, unknown sets (° = outside the entry's claimed languages)

| lang | datasets | opf |
|---|---|---|
| ar | iahlt | 0.097° |
| bn | multiconer, naamapadam | 0.282° |
| hi | multiconer, naamapadam | 0.140° |
| ja | stockmark | 0.045° |
| ta | naamapadam | 0.150° |
| th | thainer | 0.284° |
| zh | multiconer | 0.088° |

## default track, per dataset, full view, masked F2, claimed languages only (completed docs; (x) = excluded, information only; x? = unknown training data; ° = no doc in a claimed language)

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

### default track, per dataset, outside claimed languages (information only, not a loss; - = every doc is in a claimed language)

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

## default track, completion

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

## default track, SCT with and without recovered dropped copies (full view)

| entry | dataset | masked_recall with | without | doc leak with | without | copies recovered |
|---|---|---|---|---|---|---|
| sct-default | openpii | (0.903) | (0.903) | (0.473) | (0.474) | 14 |
| sct-default | gretel | (0.744) | (0.706) | (0.658) | (0.698) | 873 |
| sct-default | gretel_en | (0.980) | (0.980) | (0.270) | (0.270) | 0 |
| sct-default | nemotron | (0.973) | (0.956) | (0.201) | (0.239) | 34 |
| sct-default | argilla_pii | (0.967) | (0.967) | (0.057) | (0.057) | 0 |
| sct-default | thai_pii | (0.975) | (0.975) | (0.782) | (0.782) | 0 |
| sct-default | bayanat | (0.995) | (0.995) | (0.050) | (0.050) | 0 |
| sct-default | tab | (0.946) | (0.935) | (1.000) | (1.000) | 189 |
| sct-default | redact | (0.966) | (0.928) | (0.612) | (0.656) | 2350 |
| sct-default | iahlt | (0.937) | (0.937) | (0.295) | (0.295) | 0 |
| sct-default | thainer | (0.877) | (0.877) | (0.498) | (0.498) | 0 |
| sct-default | naamapadam | (0.701) | (0.700) | (0.432) | (0.432) | 2 |
| sct-default | multiconer | (0.819) | (0.819) | (0.310) | (0.310) | 0 |
| sct-default | stockmark | (0.908) | (0.908) | (0.282) | (0.282) | 0 |
| sct-gliner | openpii | (0.840) | (0.840) | (0.599) | (0.599) | 64 |
| sct-gliner | gretel | (0.886) | (0.818) | (0.712) | (0.763) | 1833 |
| sct-gliner | gretel_en | (0.983) | (0.983) | (0.274) | (0.274) | 0 |
| sct-gliner | nemotron | (0.967) | (0.953) | (0.285) | (0.306) | 50 |
| sct-gliner | argilla_pii | (0.975) | (0.975) | (0.131) | (0.131) | 0 |
| sct-gliner | thai_pii | ° | ° | ° | ° | 0 |
| sct-gliner | bayanat | ° | ° | ° | ° | 0 |
| sct-gliner | tab | (0.579) | (0.565) | (1.000) | (1.000) | 164 |
| sct-gliner | redact | (0.961) | (0.909) | (0.906) | (0.938) | 2855 |
| sct-gliner | iahlt | ° | ° | ° | ° | 13 |
| sct-gliner | thainer | ° | ° | ° | ° | 0 |
| sct-gliner | naamapadam | ° | ° | ° | ° | 0 |
| sct-gliner | multiconer | ° | ° | ° | ° | 0 |
| sct-gliner | stockmark | ° | ° | ° | ° | 0 |

### default track, audit of the recovered copies (seeded sample, counts only)

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

## tuned track, full view, privacy-synthetic: each entry on its own clean sets, claimed languages

| entry | clean sets with claimed-language docs (all clean sets) | mR [95% CI] | mP [95% CI] | strict F1 [95% CI] | doc leak [95% CI] | over-mask [95% CI] | mR clean, outside claimed langs | mR unknown (U) sets | mR U sets, outside claimed langs | (mR excluded sets, info only) | docs/s CPU | peak RAM | claimed languages | licence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gliner2-fastino | argilla_pii, gretel, gretel_en, nemotron, openpii (7) | 0.897 [0.886, 0.907] | 0.702 [0.682, 0.721] | 0.680 [0.665, 0.696] | 0.285 [0.265, 0.304] | 0.046 [0.042, 0.050] | 0.880 (4) | - | - | - | 0.52 | 6,832 MB | en fr es de it pt nl | Apache-2.0 |
| presidio | argilla_pii, gretel, gretel_en, nemotron, openpii (7) | 0.762 [0.742, 0.781] | 0.803 [0.773, 0.838] | 0.540 [0.521, 0.559] | 0.608 [0.584, 0.633] | 0.023 [0.019, 0.029] | 0.605 (4) | - | - | - | 27 | 826 MB | en | MIT |
| datafog | argilla_pii, bayanat, gretel, gretel_en, nemotron, openpii, thai_pii (7) | 0.337 [0.325, 0.348] | 0.854 [0.833, 0.878] | 0.342 [0.328, 0.355] | 0.919 [0.908, 0.929] | 0.009 [0.007, 0.010] | - | - | - | - | 1,787 | 30 MB | all (regex, no language claim) | MIT |
| sct-gliner | none (0) | - | - | - | - | - | - | - | - | (0.944 (5); w/o recovery 0.923 (5)) | 0.53 | 6,414 MB | en | MIT + Apache-2.0 model |
| sct-default | none (0) | - | - | - | - | - | - | - | - | (0.935 (7); w/o recovery 0.927 (7)) | 1.50 | 7,692 MB | all (multilingual claim) | MIT |
| opf | none (0) | - | - | - | - | - | - | 0.761 (5) | 0.823 (4) | - | 0.29 | 26,113 MB | en | Apache-2.0 |
| gliner-nvidia | none (2) | - | - | - | - | - | 0.779 (2) | - | - | (0.931 (5)) | 0.15 | 25,732 MB | en | Apache-2.0 + NVIDIA Open Model |
| bardsai-v3 | none (2) | - | - | - | - | - | 0.895 (2) | - | - | (0.919 (5)) | 2.52 | 1,738 MB | 24 EU | Apache-2.0 |

### same-set (every entry)

No dataset in this group is clean for every one of: sct-default, sct-gliner, presidio, gliner-nvidia, gliner2-fastino, opf, bardsai-v3, datafog (in languages all of them claim).

### same-set (non-SCT entries with clean sets)

No dataset in this group is clean for every one of: presidio, gliner-nvidia, gliner2-fastino, bardsai-v3, datafog (in languages all of them claim).

### SCT vs each entry, datasets clean for both, languages both claim

No dataset in this group is clean for both sct-default and any other entry (languages both claim).

### SCT vs each entry, datasets unknown for both, languages both claim

No dataset in this group is unknown for both sct-default and any other entry (languages both claim).

## tuned track, full view, privacy-real: each entry on its own clean sets, claimed languages

| entry | clean sets with claimed-language docs (all clean sets) | mR [95% CI] | mP [95% CI] | strict F1 [95% CI] | doc leak [95% CI] | over-mask [95% CI] | mR clean, outside claimed langs | mR unknown (U) sets | mR U sets, outside claimed langs | (mR excluded sets, info only) | docs/s CPU | peak RAM | claimed languages | licence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gliner2-fastino | redact, tab (2) | 0.904 [0.884, 0.922] | 0.542 [0.522, 0.562] | 0.516 [0.499, 0.534] | 0.833 [0.807, 0.858] | 0.071 [0.068, 0.075] | 0.884 (1) | - | - | - | 0.52 | 6,832 MB | en fr es de it pt nl | Apache-2.0 |
| bardsai-v3 | redact, tab (2) | 0.739 [0.714, 0.764] | 0.817 [0.804, 0.831] | 0.512 [0.495, 0.529] | 0.830 [0.806, 0.852] | 0.022 [0.020, 0.024] | 0.804 (1) | - | - | - | 2.52 | 1,738 MB | 24 EU | Apache-2.0 |
| presidio | redact, tab (2) | 0.612 [0.576, 0.648] | 0.651 [0.605, 0.697] | 0.319 [0.285, 0.349] | 0.922 [0.850, 0.984] | 0.045 [0.034, 0.057] | 0.620 (1) | - | - | - | 27 | 826 MB | en | MIT |
| gliner-nvidia | redact, tab (2) | 0.439 [0.382, 0.506] | 0.821 [0.783, 0.856] | 0.164 [0.134, 0.198] | 0.969 [0.922, 1.000] | 0.017 [0.012, 0.023] | 0.600 (1) | - | - | - | 0.15 | 25,732 MB | en | Apache-2.0 + NVIDIA Open Model |
| datafog | redact, tab (2) | 0.161 [0.144, 0.182] | 0.765 [0.751, 0.790] | 0.155 [0.147, 0.162] | 0.997 [0.994, 1.000] | 0.008 [0.007, 0.009] | - | - | - | - | 1,787 | 30 MB | all (regex, no language claim) | MIT |
| sct-gliner | none (0) | - | - | - | - | - | - | - | - | (0.839 (2); w/o recovery 0.799 (2)) | 0.53 | 6,414 MB | en | MIT + Apache-2.0 model |
| sct-default | none (0) | - | - | - | - | - | - | - | - | (0.958 (2); w/o recovery 0.933 (2)) | 1.50 | 7,692 MB | all (multilingual claim) | MIT |
| opf | none (0) | - | - | - | - | - | - | 0.523 (2) | 0.799 (1) | - | 0.29 | 26,113 MB | en | Apache-2.0 |

### same-set (every entry)

No dataset in this group is clean for every one of: sct-default, sct-gliner, presidio, gliner-nvidia, gliner2-fastino, opf, bardsai-v3, datafog (in languages all of them claim).

### same-set (non-SCT entries with clean sets)

Datasets: redact, tab (clean for all: tab, redact); only docs in languages every entry claims; point values.

| entry | mR | mP | strict F1 | doc leak | over-mask |
|---|---|---|---|---|---|
| gliner2-fastino | 0.914 | 0.490 | 0.461 | 0.812 | 0.080 |
| bardsai-v3 | 0.713 | 0.758 | 0.479 | 0.859 | 0.028 |
| presidio | 0.612 | 0.651 | 0.319 | 0.922 | 0.045 |
| gliner-nvidia | 0.439 | 0.821 | 0.164 | 0.969 | 0.017 |
| datafog | 0.163 | 0.756 | 0.143 | 1.000 | 0.007 |

### SCT vs each entry, datasets clean for both, languages both claim

No dataset in this group is clean for both sct-default and any other entry (languages both claim).

### SCT vs each entry, datasets unknown for both, languages both claim

No dataset in this group is unknown for both sct-default and any other entry (languages both claim).

## tuned track, full view, ner: each entry on its own clean sets, claimed languages

| entry | clean sets with claimed-language docs (all clean sets) | mR [95% CI] | mP [95% CI] | strict F1 [95% CI] | doc leak [95% CI] | over-mask [95% CI] | mR clean, outside claimed langs | mR unknown (U) sets | mR U sets, outside claimed langs | (mR excluded sets, info only) | docs/s CPU | peak RAM | claimed languages | licence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| datafog | iahlt, multiconer, naamapadam, stockmark, thainer (5) | 0.001 [0.000, 0.002] | 0.333 [0.000, 1.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | - | - | - | - | 1,787 | 30 MB | all (regex, no language claim) | MIT |
| sct-gliner | none (0) | - | - | - | - | - | - | - | - | - | 0.53 | 6,414 MB | en | MIT + Apache-2.0 model |
| sct-default | none (0) | - | - | - | - | - | - | - | - | (0.855 (5); w/o recovery 0.855 (5)) | 1.50 | 7,692 MB | all (multilingual claim) | MIT |
| presidio | none (5) | - | - | - | - | - | 0.095 (5) | - | - | - | 27 | 826 MB | en | MIT |
| opf | none (0) | - | - | - | - | - | - | - | 0.253 (5) | - | 0.29 | 26,113 MB | en | Apache-2.0 |
| gliner2-fastino | none (5) | - | - | - | - | - | 0.776 (5) | - | - | - | 0.52 | 6,832 MB | en fr es de it pt nl | Apache-2.0 |
| gliner-nvidia | none (5) | - | - | - | - | - | 0.400 (5) | - | - | - | 0.15 | 25,732 MB | en | Apache-2.0 + NVIDIA Open Model |
| bardsai-v3 | none (5) | - | - | - | - | - | 0.256 (5) | - | - | - | 2.52 | 1,738 MB | 24 EU | Apache-2.0 |

### same-set (every entry)

No dataset in this group is clean for every one of: sct-default, sct-gliner, presidio, gliner-nvidia, gliner2-fastino, opf, bardsai-v3, datafog (in languages all of them claim).

### same-set (non-SCT entries with clean sets)

No dataset in this group is clean for every one of: presidio, gliner-nvidia, gliner2-fastino, bardsai-v3, datafog (in languages all of them claim).

### SCT vs each entry, datasets clean for both, languages both claim

No dataset in this group is clean for both sct-default and any other entry (languages both claim).

### SCT vs each entry, datasets unknown for both, languages both claim

No dataset in this group is unknown for both sct-default and any other entry (languages both claim).

## tuned track, shared view, privacy-synthetic: each entry on its own clean sets, claimed languages

| entry | clean sets with claimed-language docs (all clean sets) | mR [95% CI] | mP [95% CI] | strict F1 [95% CI] | doc leak [95% CI] | over-mask [95% CI] | mR clean, outside claimed langs | mR unknown (U) sets | mR U sets, outside claimed langs | (mR excluded sets, info only) | docs/s CPU | peak RAM | claimed languages | licence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gliner2-fastino | argilla_pii, gretel, gretel_en, nemotron, openpii (7) | 0.939 [0.925, 0.951] | 0.477 [0.455, 0.500] | 0.817 [0.781, 0.852] | 0.110 [0.094, 0.129] | 0.046 [0.042, 0.050] | 0.937 (4) | - | - | - | 0.52 | 6,832 MB | en fr es de it pt nl | Apache-2.0 |
| presidio | argilla_pii, gretel, gretel_en, nemotron, openpii (7) | 0.854 [0.831, 0.873] | 0.682 [0.650, 0.721] | 0.698 [0.656, 0.752] | 0.318 [0.289, 0.349] | 0.023 [0.019, 0.029] | 0.821 (4) | - | - | - | 27 | 826 MB | en | MIT |
| datafog | argilla_pii, bayanat, gretel, gretel_en, nemotron, openpii, thai_pii (7) | 0.678 [0.656, 0.700] | 0.828 [0.808, 0.852] | 0.597 [0.572, 0.623] | 0.507 [0.482, 0.532] | 0.009 [0.007, 0.010] | - | - | - | - | 1,787 | 30 MB | all (regex, no language claim) | MIT |
| sct-gliner | none (0) | - | - | - | - | - | - | - | - | (0.947 (5); w/o recovery 0.926 (5)) | 0.53 | 6,414 MB | en | MIT + Apache-2.0 model |
| sct-default | none (0) | - | - | - | - | - | - | - | - | (0.962 (7); w/o recovery 0.949 (7)) | 1.50 | 7,692 MB | all (multilingual claim) | MIT |
| opf | none (0) | - | - | - | - | - | - | 0.900 (5) | 0.953 (4) | - | 0.29 | 26,113 MB | en | Apache-2.0 |
| gliner-nvidia | none (2) | - | - | - | - | - | 0.845 (2) | - | - | (0.973 (5)) | 0.15 | 25,732 MB | en | Apache-2.0 + NVIDIA Open Model |
| bardsai-v3 | none (2) | - | - | - | - | - | 0.985 (2) | - | - | (0.978 (5)) | 2.52 | 1,738 MB | 24 EU | Apache-2.0 |

### same-set (every entry)

No dataset in this group is clean for every one of: sct-default, sct-gliner, presidio, gliner-nvidia, gliner2-fastino, opf, bardsai-v3, datafog (in languages all of them claim).

### same-set (non-SCT entries with clean sets)

No dataset in this group is clean for every one of: presidio, gliner-nvidia, gliner2-fastino, bardsai-v3, datafog (in languages all of them claim).

### SCT vs each entry, datasets clean for both, languages both claim

No dataset in this group is clean for both sct-default and any other entry (languages both claim).

### SCT vs each entry, datasets unknown for both, languages both claim

No dataset in this group is unknown for both sct-default and any other entry (languages both claim).

## tuned track, shared view, privacy-real: each entry on its own clean sets, claimed languages

| entry | clean sets with claimed-language docs (all clean sets) | mR [95% CI] | mP [95% CI] | strict F1 [95% CI] | doc leak [95% CI] | over-mask [95% CI] | mR clean, outside claimed langs | mR unknown (U) sets | mR U sets, outside claimed langs | (mR excluded sets, info only) | docs/s CPU | peak RAM | claimed languages | licence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| datafog | redact, tab (2) | 0.768 [0.648, 0.846] | 0.750 [0.736, 0.775] | 0.351 [0.341, 0.360] | 0.812 [0.795, 0.830] | 0.008 [0.007, 0.009] | - | - | - | - | 1,787 | 30 MB | all (regex, no language claim) | MIT |
| gliner2-fastino | redact, tab (2) | 0.533 [0.482, 0.581] | 0.147 [0.129, 0.166] | 0.469 [0.421, 0.528] | 0.587 [0.564, 0.611] | 0.071 [0.068, 0.075] | 0.933 (1) | - | - | - | 0.52 | 6,832 MB | en fr es de it pt nl | Apache-2.0 |
| bardsai-v3 | redact, tab (2) | 0.472 [0.464, 0.479] | 0.198 [0.180, 0.220] | 0.433 [0.421, 0.447] | 0.621 [0.601, 0.643] | 0.022 [0.020, 0.024] | 0.864 (1) | - | - | - | 2.52 | 1,738 MB | 24 EU | Apache-2.0 |
| presidio | redact, tab (2) | 0.466 [0.429, 0.498] | 0.155 [0.112, 0.204] | 0.290 [0.234, 0.354] | 0.714 [0.625, 0.803] | 0.045 [0.034, 0.057] | 0.852 (1) | - | - | - | 27 | 826 MB | en | MIT |
| gliner-nvidia | redact, tab (2) | 0.275 [0.194, 0.365] | 0.336 [0.250, 0.441] | 0.272 [0.192, 0.350] | 0.882 [0.805, 0.948] | 0.017 [0.012, 0.023] | 0.575 (1) | - | - | - | 0.15 | 25,732 MB | en | Apache-2.0 + NVIDIA Open Model |
| sct-gliner | none (0) | - | - | - | - | - | - | - | - | (0.983 (2); w/o recovery 0.983 (2)) | 0.53 | 6,414 MB | en | MIT + Apache-2.0 model |
| sct-default | none (0) | - | - | - | - | - | - | - | - | (0.985 (2); w/o recovery 0.983 (2)) | 1.50 | 7,692 MB | all (multilingual claim) | MIT |
| opf | none (0) | - | - | - | - | - | - | 0.541 (2) | 0.963 (1) | - | 0.29 | 26,113 MB | en | Apache-2.0 |

### same-set (every entry)

No dataset in this group is clean for every one of: sct-default, sct-gliner, presidio, gliner-nvidia, gliner2-fastino, opf, bardsai-v3, datafog (in languages all of them claim).

### same-set (non-SCT entries with clean sets)

Datasets: redact, tab (clean for all: tab, redact); only docs in languages every entry claims; point values.

| entry | mR | mP | strict F1 | doc leak | over-mask |
|---|---|---|---|---|---|
| datafog | 0.758 | 0.724 | 0.342 | 0.822 | 0.007 |
| gliner2-fastino | 0.548 | 0.106 | 0.451 | 0.572 | 0.080 |
| bardsai-v3 | 0.468 | 0.145 | 0.418 | 0.625 | 0.028 |
| presidio | 0.466 | 0.155 | 0.290 | 0.714 | 0.045 |
| gliner-nvidia | 0.275 | 0.336 | 0.272 | 0.882 | 0.017 |

### SCT vs each entry, datasets clean for both, languages both claim

No dataset in this group is clean for both sct-default and any other entry (languages both claim).

### SCT vs each entry, datasets unknown for both, languages both claim

No dataset in this group is unknown for both sct-default and any other entry (languages both claim).

## tuned track, shared view, ner: each entry on its own clean sets, claimed languages

| entry | clean sets with claimed-language docs (all clean sets) | mR [95% CI] | mP [95% CI] | strict F1 [95% CI] | doc leak [95% CI] | over-mask [95% CI] | mR clean, outside claimed langs | mR unknown (U) sets | mR U sets, outside claimed langs | (mR excluded sets, info only) | docs/s CPU | peak RAM | claimed languages | licence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sct-gliner | none (0) | - | - | - | - | - | - | - | - | - | 0.53 | 6,414 MB | en | MIT + Apache-2.0 model |
| sct-default | none (0) | - | - | - | - | - | - | - | - | (1.000 (5); w/o recovery 1.000 (5)) | 1.50 | 7,692 MB | all (multilingual claim) | MIT |
| presidio | none (5) | - | - | - | - | - | 0.583 (5) | - | - | - | 27 | 826 MB | en | MIT |
| opf | none (0) | - | - | - | - | - | - | - | 0.833 (5) | - | 0.29 | 26,113 MB | en | Apache-2.0 |
| gliner2-fastino | none (5) | - | - | - | - | - | 0.917 (5) | - | - | - | 0.52 | 6,832 MB | en fr es de it pt nl | Apache-2.0 |
| gliner-nvidia | none (5) | - | - | - | - | - | 0.833 (5) | - | - | - | 0.15 | 25,732 MB | en | Apache-2.0 + NVIDIA Open Model |
| datafog | iahlt, multiconer, naamapadam, stockmark, thainer (5) | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | - | - | - | - | 1,787 | 30 MB | all (regex, no language claim) | MIT |
| bardsai-v3 | none (5) | - | - | - | - | - | 0.500 (5) | - | - | - | 2.52 | 1,738 MB | 24 EU | Apache-2.0 |

### same-set (every entry)

No dataset in this group is clean for every one of: sct-default, sct-gliner, presidio, gliner-nvidia, gliner2-fastino, opf, bardsai-v3, datafog (in languages all of them claim).

### same-set (non-SCT entries with clean sets)

No dataset in this group is clean for every one of: presidio, gliner-nvidia, gliner2-fastino, bardsai-v3, datafog (in languages all of them claim).

### SCT vs each entry, datasets clean for both, languages both claim

No dataset in this group is clean for both sct-default and any other entry (languages both claim).

### SCT vs each entry, datasets unknown for both, languages both claim

No dataset in this group is unknown for both sct-default and any other entry (languages both claim).

## tuned track, per language, full view, masked recall, privacy-synthetic, clean sets (° = outside the entry's claimed languages)

| lang | datasets | bardsai-v3 | datafog | gliner-nvidia | gliner2-fastino | presidio |
|---|---|---|---|---|---|---|
| ar-egyptian | bayanat | 0.846° | 0.423 | 0.750° | 0.949° | 0.673° |
| ar-gulf | bayanat | 0.906° | 0.592 | 0.718° | 0.919° | 0.758° |
| ar-mixed_en | bayanat | 0.953° | 0.470 | 0.926° | 0.993° | 0.933° |
| ar-msa | bayanat | 0.914° | 0.441 | 0.775° | 0.945° | 0.664° |
| bg | openpii | - | 0.408 | - | 0.872° | 0.677° |
| cs | openpii | - | 0.298 | - | 0.914° | 0.632° |
| da | openpii | - | 0.383 | - | 0.953° | 0.704° |
| de | gretel, openpii | - | 0.189 | - | 0.862 | 0.640° |
| el | openpii | - | 0.262 | - | 0.911° | 0.659° |
| en | argilla_pii, gretel, gretel_en, nemotron, openpii | - | 0.294 | - | 0.928 | 0.762 |
| es | gretel, openpii | - | 0.218 | - | 0.825 | 0.552° |
| et | openpii | - | 0.352 | - | 0.932° | 0.752° |
| fi | openpii | - | 0.340 | - | 0.940° | 0.729° |
| fr | gretel, openpii | - | 0.226 | - | 0.917 | 0.566° |
| hr | openpii | - | 0.368 | - | 0.908° | 0.703° |
| hu | openpii | - | 0.251 | - | 0.919° | 0.588° |
| id | openpii | - | 0.322 | - | 0.883° | 0.708° |
| it | gretel, openpii | - | 0.216 | - | 0.865 | 0.622° |
| ja | openpii | - | 0.157 | - | 0.697° | 0.430° |
| ko | openpii | - | 0.217 | - | 0.875° | 0.409° |
| lt | openpii | - | 0.338 | - | 0.910° | 0.753° |
| lv | openpii | - | 0.306 | - | 0.911° | 0.715° |
| ms | openpii | - | 0.319 | - | 0.894° | 0.749° |
| nl | gretel, openpii | - | 0.196 | - | 0.810 | 0.607° |
| pl | openpii | - | 0.212 | - | 0.903° | 0.651° |
| pt | openpii | - | 0.297 | - | 0.879 | 0.757° |
| ro | openpii | - | 0.233 | - | 0.884° | 0.727° |
| sk | openpii | - | 0.247 | - | 0.895° | 0.683° |
| sl | openpii | - | 0.213 | - | 0.918° | 0.712° |
| sr | openpii | - | 0.376 | - | 0.905° | 0.721° |
| sv | gretel, openpii | - | 0.207 | - | 0.846° | 0.608° |
| th | thai_pii | 0.884° | 0.512 | 0.766° | 0.953° | 0.440° |
| tl | openpii | - | 0.255 | - | 0.917° | 0.733° |
| vi | openpii | - | 0.295 | - | 0.863° | 0.695° |
| zh | openpii | - | 0.150 | - | 0.696° | 0.399° |

## tuned track, per language, full view, masked recall, privacy-synthetic, unknown sets (° = outside the entry's claimed languages)

| lang | datasets | opf |
|---|---|---|
| ar-egyptian | bayanat | 0.904° |
| ar-gulf | bayanat | 0.933° |
| ar-mixed_en | bayanat | 0.987° |
| ar-msa | bayanat | 0.921° |
| bg | openpii | 0.887° |
| cs | openpii | 0.814° |
| da | openpii | 0.827° |
| de | gretel, openpii | 0.739° |
| el | openpii | 0.756° |
| en | argilla_pii, gretel, gretel_en, nemotron, openpii | 0.761 |
| es | gretel, openpii | 0.778° |
| et | openpii | 0.859° |
| fi | openpii | 0.779° |
| fr | gretel, openpii | 0.808° |
| hr | openpii | 0.866° |
| hu | openpii | 0.848° |
| id | openpii | 0.911° |
| it | gretel, openpii | 0.801° |
| ja | openpii | 0.868° |
| ko | openpii | 0.840° |
| lt | openpii | 0.787° |
| lv | openpii | 0.745° |
| ms | openpii | 0.910° |
| nl | gretel, openpii | 0.705° |
| pl | openpii | 0.885° |
| pt | openpii | 0.955° |
| ro | openpii | 0.838° |
| sk | openpii | 0.880° |
| sl | openpii | 0.824° |
| sr | openpii | 0.853° |
| sv | gretel, openpii | 0.681° |
| th | thai_pii | 0.914° |
| tl | openpii | 0.879° |
| vi | openpii | 0.922° |
| zh | openpii | 0.898° |

## tuned track, per language, full view, masked recall, privacy-real, clean sets (° = outside the entry's claimed languages)

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

## tuned track, per language, full view, masked recall, privacy-real, unknown sets (° = outside the entry's claimed languages)

| lang | datasets | opf |
|---|---|---|
| ar | redact | 0.811° |
| cs | redact | 0.726° |
| da | redact | 0.782° |
| de | redact | 0.785° |
| en | redact, tab | 0.523 |
| es | redact | 0.848° |
| fi | redact | 0.725° |
| fr | redact | 0.824° |
| fr_ca | redact | 0.839° |
| he | redact | 0.654° |
| hi | redact | 0.782° |
| hu | redact | 0.834° |
| it | redact | 0.838° |
| ja | redact | 0.835° |
| ko | redact | 0.838° |
| nl | redact | 0.830° |
| no | redact | 0.751° |
| pt_br | redact | 0.898° |
| pt_eu | redact | 0.807° |
| ru | redact | 0.817° |
| sv | redact | 0.797° |
| th | redact | 0.796° |
| tr | redact | 0.820° |
| zh_cn | redact | 0.774° |
| zh_tw | redact | 0.761° |

## tuned track, per language, full view, masked recall, ner, clean sets (° = outside the entry's claimed languages)

| lang | datasets | bardsai-v3 | datafog | gliner-nvidia | gliner2-fastino | presidio |
|---|---|---|---|---|---|---|
| ar | iahlt | 0.380° | 0.000 | 0.794° | 0.922° | 0.189° |
| bn | multiconer, naamapadam | 0.188° | 0.000 | 0.103° | 0.939° | 0.078° |
| hi | multiconer, naamapadam | 0.388° | 0.000 | 0.127° | 0.964° | 0.088° |
| ja | stockmark | 0.019° | 0.000 | 0.622° | 0.740° | 0.042° |
| ta | naamapadam | 0.302° | 0.000 | 0.066° | 0.977° | 0.074° |
| th | thainer | 0.344° | 0.003 | 0.186° | 0.547° | 0.089° |
| zh | multiconer | 0.163° | 0.000 | 0.657° | 0.235° | 0.052° |

## tuned track, per language, full view, masked recall, ner, unknown sets (° = outside the entry's claimed languages)

| lang | datasets | opf |
|---|---|---|
| ar | iahlt | 0.152° |
| bn | multiconer, naamapadam | 0.434° |
| hi | multiconer, naamapadam | 0.230° |
| ja | stockmark | 0.069° |
| ta | naamapadam | 0.275° |
| th | thainer | 0.468° |
| zh | multiconer | 0.124° |

## tuned track, per dataset, full view, masked F2, claimed languages only (completed docs; (x) = excluded, information only; x? = unknown training data; ° = no doc in a claimed language)

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

### tuned track, per dataset, outside claimed languages (information only, not a loss; - = every doc is in a claimed language)

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

## tuned track, completion

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

## tuned track, SCT with and without recovered dropped copies (full view)

| entry | dataset | masked_recall with | without | doc leak with | without | copies recovered |
|---|---|---|---|---|---|---|
| sct-default | openpii | (0.907) | (0.907) | (0.462) | (0.463) | 18 |
| sct-default | gretel | (0.745) | (0.707) | (0.647) | (0.690) | 899 |
| sct-default | gretel_en | (0.978) | (0.978) | (0.266) | (0.266) | 0 |
| sct-default | nemotron | (0.973) | (0.956) | (0.197) | (0.236) | 33 |
| sct-default | argilla_pii | (0.967) | (0.967) | (0.057) | (0.057) | 0 |
| sct-default | thai_pii | (0.977) | (0.977) | (0.782) | (0.782) | 0 |
| sct-default | bayanat | (0.995) | (0.995) | (0.050) | (0.050) | 0 |
| sct-default | tab | (0.947) | (0.935) | (1.000) | (1.000) | 198 |
| sct-default | redact | (0.968) | (0.930) | (0.597) | (0.647) | 2379 |
| sct-default | iahlt | (0.937) | (0.937) | (0.291) | (0.291) | 0 |
| sct-default | thainer | (0.882) | (0.882) | (0.490) | (0.490) | 0 |
| sct-default | naamapadam | (0.712) | (0.711) | (0.411) | (0.411) | 2 |
| sct-default | multiconer | (0.829) | (0.829) | (0.296) | (0.296) | 0 |
| sct-default | stockmark | (0.917) | (0.917) | (0.267) | (0.267) | 0 |
| sct-gliner | openpii | (0.857) | (0.857) | (0.566) | (0.566) | 153 |
| sct-gliner | gretel | (0.912) | (0.834) | (0.588) | (0.685) | 3055 |
| sct-gliner | gretel_en | (0.983) | (0.983) | (0.239) | (0.239) | 1 |
| sct-gliner | nemotron | (0.981) | (0.958) | (0.222) | (0.271) | 105 |
| sct-gliner | argilla_pii | (0.985) | (0.985) | (0.102) | (0.102) | 0 |
| sct-gliner | thai_pii | ° | ° | ° | ° | 0 |
| sct-gliner | bayanat | ° | ° | ° | ° | 0 |
| sct-gliner | tab | (0.706) | (0.684) | (1.000) | (1.000) | 786 |
| sct-gliner | redact | (0.973) | (0.915) | (0.875) | (0.906) | 3725 |
| sct-gliner | iahlt | ° | ° | ° | ° | 33 |
| sct-gliner | thainer | ° | ° | ° | ° | 0 |
| sct-gliner | naamapadam | ° | ° | ° | ° | 0 |
| sct-gliner | multiconer | ° | ° | ° | ° | 0 |
| sct-gliner | stockmark | ° | ° | ° | ° | 0 |

### tuned track, audit of the recovered copies (seeded sample, counts only)

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

## contamination (bold = excluded, italic = unknown, plain = clean)

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

## speed (speed-16t)

device cpu, cpus 0-15, threads 16, 100 docs, 3 repeats, median; peak RSS from getrusage (on Linux it never reads below the runner's own ~2 GB: values at that floor are an upper bound)

| entry | cold start s | latency median ms | p95 ms | batch-8 docs/s | chars/s | peak RSS MB | venv MB | failed |
|---|---|---|---|---|---|---|---|---|
| datafog | 0.101 | 0.32 | 2.2 | 1787 | 3.518e+06 | 2014 | 9.3 | 0 |
| presidio | 1.504 | 20.61 | 118.8 | 27.39 | 5.392e+04 | 2014 | 11398.6 | 0 |
| bardsai-v3 | 3.614 | 252.6 | 1106 | 2.523 | 4966 | 2014 | 5416.3 | 0 |
| sct-default | 8.372 | 610 | 2192 | 1.495 | 2943 | 7692 | 473.7 | 0 |
| sct-gliner | 7.378 | 836.4 | 5656 | 0.529 | 1042 | 6414 | 5769.2 | 0 |
| gliner2-fastino | 6.666 | 196 | 7431 | 0.524 | 1032 | 6832 | 5402.3 | 0 |
| opf | 4.371 | 639.2 | 4913 | 0.293 | 577.1 | 2.611e+04 | 5382.7 | 0 |
| gliner-nvidia | 7.071 | 866.7 | 5298 | 0.148 | 291.4 | 2.573e+04 | 5388.7 | 0 |

## speed (speed-16t-rss)

device cpu, cpus 0-15, threads 16, 100 docs, 3 repeats, median; peak RSS from /proc VmHWM

| entry | cold start s | latency median ms | p95 ms | batch-8 docs/s | chars/s | peak RSS MB | venv MB | failed |
|---|---|---|---|---|---|---|---|---|
| datafog | 0.101 | 0.31 | 2.17 | 1793 | 3.529e+06 | 30.3 | 9.3 | 0 |
| presidio | 1.509 | 20.76 | 119.4 | 27.28 | 5.37e+04 | 826.1 | 11398.6 | 0 |
| bardsai-v3 | 3.624 | 252.6 | 1220 | 2.514 | 4948 | 1738 | 5416.3 | 0 |

## speed (speed-1t)

device cpu, cpus 0, threads 1, 30 docs, 1 repeats, median; peak RSS from /proc VmHWM

| entry | cold start s | latency median ms | p95 ms | batch-8 docs/s | chars/s | peak RSS MB | venv MB | failed |
|---|---|---|---|---|---|---|---|---|
| datafog | 0.104 | 0.29 | 1.21 | 2530 | 3.374e+06 | 30.3 | 9.3 | 0 |
| presidio | 1.513 | 18.21 | 91.13 | 37.77 | 5.037e+04 | 777.6 | 11398.6 | 0 |
| bardsai-v3 | 3.48 | 314.9 | 1951 | 1.65 | 2200 | 1272 | 5416.3 | 0 |
| sct-default | 8.77 | 4244 | 2.014e+04 | 0.194 | 258.6 | 5919 | 474.2 | 0 |
| opf | 4.134 | 701.4 | 8131 | 0.164 | 218.5 | 8406 | 5382.7 | 0 |
| gliner2-fastino | 7.162 | 1429 | 2.656e+04 | 0.156 | 208.5 | 4476 | 5402.3 | 0 |
| sct-gliner | 7.553 | 2953 | 1.825e+04 | 0.145 | 193.1 | 5327 | 5769.2 | 0 |
| gliner-nvidia | 8.377 | 5958 | 3.809e+04 | 0.025 | 33.4 | 1.757e+04 | 5388.7 | 0 |
