## default track, full view, r2-privacy-synthetic: each entry on its own clean sets, claimed languages

† = under 99% complete on a dataset in this row or cell (PROTOCOL section 15; see the completion table).

| entry | clean sets with claimed-language docs (all clean sets) | mR [95% CI] | mP [95% CI] | strict F1 [95% CI] | doc leak [95% CI] | over-mask [95% CI] | mR clean, outside claimed langs | mR unknown (U) sets | mR U sets, outside claimed langs | (mR excluded sets, info only) | docs/s CPU | peak RAM | GPU: device, docs/s, peak VRAM | claimed languages | licence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sct-default | dialogpii, europii, greek_deid, meddeid, neonredact, newmean, pl_pii, tonic (8) | 0.888 [0.881, 0.894] | 0.750 [0.735, 0.767] | 0.548 [0.537, 0.560] | 0.580 [0.561, 0.598] | 0.051 [0.048, 0.055] | - | - | - | - | 1.55 | 10,023 MB | n/a | all (multilingual claim) | MIT |
| gliner2-fastino | dialogpii, europii, meddeid, newmean, tonic (8) | 0.864 [0.854, 0.873] | 0.640 [0.627, 0.653] | 0.642 [0.630, 0.654] | 0.437 [0.421, 0.454] | 0.062 [0.059, 0.065] | 0.831 (6) | - | - | - | 0.63 | 34,356 MB | cuda, fail docs/s, - MB | en fr es de it pt nl | Apache-2.0 |
| sct-gliner | dialogpii, europii, meddeid, newmean, tonic (8) | 0.856 [0.842, 0.869] | 0.626 [0.607, 0.642] | 0.288 [0.271, 0.305] | 0.558 [0.534, 0.581] | 0.057 [0.053, 0.060] | 0.775 (7) | - | - | - | 0.61 | 6,715 MB | cuda, fail docs/s, - MB | en | MIT + Apache-2.0 model |
| bardsai-v3 | dialogpii, europii, greek_deid, meddeid, newmean, pl_pii, tonic (8) | 0.852 [0.841, 0.861] | 0.822 [0.804, 0.840] | 0.674 [0.659, 0.689] | 0.567 [0.549, 0.582] | 0.033 [0.031, 0.036] | 0.699 (3) | - | - | - | 2.42 | 6,975 MB | n/a | 24 EU | Apache-2.0 |
| sct-gliner-nvidia | dialogpii, europii, meddeid, newmean, tonic (8) | 0.841 [0.825, 0.857] | 0.714 [0.694, 0.735] | 0.304 [0.284, 0.327] | 0.546 [0.523, 0.572] | 0.047 [0.044, 0.050] | 0.812 (7) | - | - | - | n/a | n/a | cuda, fail docs/s, - MB | en | MIT + NVIDIA Open Model |
| sct-training | dialogpii, europii, greek_deid, meddeid, neonredact, newmean, pl_pii, tonic (8) | 0.838 [0.829, 0.846] | 0.816 [0.799, 0.832] | 0.553 [0.542, 0.565] | 0.639 [0.619, 0.658] | 0.030 [0.027, 0.033] | - | - | - | - | 1.56 | 10,205 MB | n/a | all (multilingual claim) | MIT |
| pii-tracer | dialogpii, europii, greek_deid, meddeid, neonredact, newmean, pl_pii, tonic (8) | 0.837 [0.818, 0.854] | 0.781 [0.768, 0.793] | 0.629 [0.614, 0.643] | 0.502 [0.481, 0.521] | 0.055 [0.052, 0.059] | - | - | - | - | 1.33 | 3,098 MB | cuda, 19.8 docs/s, 1,826 MB | all (model card: en + multilingual) | MIT |
| gliner-knowledgator | dialogpii, europii, meddeid, newmean, tonic (8) | 0.803 [0.789, 0.819] | 0.593 [0.578, 0.608] | 0.593 [0.577, 0.607] | 0.505 [0.477, 0.529] | 0.082 [0.078, 0.086] | 0.787 (7) | - | - | - | n/a | n/a | cuda, fail docs/s, - MB | en | Apache-2.0 |
| gliner-nvidia | dialogpii, europii, meddeid, newmean, tonic (8) | 0.760 [0.742, 0.782] | 0.681 [0.661, 0.702] | 0.540 [0.523, 0.558] | 0.448 [0.424, 0.472] | 0.062 [0.059, 0.065] | 0.675 (7) | - | - | - | n/a | n/a | cuda, fail docs/s, 5,354 MB | en | Apache-2.0 + NVIDIA Open Model |
| openmed | dialogpii, europii, meddeid, neonredact, newmean, tonic (8) | 0.756 [0.745, 0.767] | 0.753 [0.742, 0.765] | 0.360 [0.349, 0.370] | 0.648 [0.631, 0.663] | 0.049 [0.047, 0.052] | 0.727 (5) | - | - | - | 0.42 | 25,637 MB | cuda, 9.9 docs/s, 5,100 MB | ar bn de en es fr hi it ja ko nl pt te tr vi zh | Apache-2.0 |
| presidio | dialogpii, europii, meddeid, newmean, tonic (8) | 0.746 [0.729, 0.765] | 0.569 [0.552, 0.589] | 0.496 [0.479, 0.514] | 0.606 [0.583, 0.629] | 0.066 [0.063, 0.069] | 0.669 (7) | - | - | - | 26 | 1,080 MB | n/a | en | MIT |
| opf | dialogpii, europii, meddeid, newmean, tonic (8) | 0.727 [0.707, 0.746] | 0.776 [0.757, 0.796] | 0.638 [0.617, 0.657] | 0.540 [0.516, 0.563] | 0.045 [0.043, 0.047] | 0.519 (7) | - | - | - | n/a | n/a | cuda, 17.0 docs/s, 4,626 MB | en | Apache-2.0 |
| presidio-full | dialogpii, europii, greek_deid, meddeid, neonredact, newmean, pl_pii, tonic (8) | 0.711 [0.701, 0.719] | 0.548 [0.528, 0.569] | 0.362 [0.354, 0.370] | 0.726 [0.715, 0.738] | 0.088 [0.084, 0.091] | - | - | - | - | 14 | 19,753 MB | n/a | all (spaCy engine per gold language + recognizers) | MIT |
| scrubadub | dialogpii, europii, meddeid, newmean, tonic (8) | 0.660 [0.643, 0.679] | 0.771 [0.749, 0.795] | 0.591 [0.575, 0.606] | 0.740 [0.720, 0.760] | 0.016 [0.015, 0.018] | 0.438 (7) | - | - | - | 1.20 | 5,180 MB | n/a | en | Apache-2.0 / MIT + MIT spaCy model |
| llm-guard | dialogpii, europii, meddeid, newmean, tonic (8) | 0.553 [0.530, 0.578] | 0.895 [0.876, 0.912] | 0.296 [0.279, 0.315] | 0.739 [0.715, 0.762] | 0.005 [0.004, 0.007] | 0.519 (7) | - | - | - | 1.40 | 2,271 MB | cuda, 11.4 docs/s, 1,322 MB | en | MIT + CC-BY-NC-4.0 model |
| piiranha | dialogpii, europii, meddeid, newmean, tonic (8) | 0.488 [0.472, 0.504] | 0.829 [0.809, 0.848] | 0.252 [0.238, 0.265] | 0.892 [0.879, 0.905] | 0.015 [0.014, 0.016] | 0.440 (6) | - | - | - | 1.80 | 2,418 MB | cuda, 41.8 docs/s, 1,884 MB | en it fr de nl es | CC-BY-NC-ND-4.0 model (benchmark only) |
| ai4privacy | dialogpii, europii, meddeid, newmean, tonic (8) | 0.357 [0.346, 0.369] | 0.656 [0.635, 0.680] | 0.108 [0.100, 0.116] | 0.963 [0.956, 0.969] | 0.032 [0.031, 0.033] | 0.351 (6) | - | - | - | 1.37 | 1,534 MB | cuda, 20.8 docs/s, 1,226 MB | fr en de te hi it es nl | MIT |
| datafog | dialogpii, europii, greek_deid, meddeid, neonredact, newmean, pl_pii, tonic (8) | 0.212 [0.202, 0.222] | 0.775 [0.743, 0.812] | 0.220 [0.209, 0.232] | 0.951 [0.944, 0.958] | 0.011 [0.010, 0.013] | - | - | - | - | 1,555 | 34 MB | n/a | all (regex, no language claim) | MIT |

### same-set (every entry)

Datasets: dialogpii, europii, meddeid, newmean, tonic (clean for all: newmean, dialogpii, tonic, meddeid, greek_deid, neonredact, pl_pii, europii); only docs in languages every entry claims; point values.

| entry | mR | mP | strict F1 | doc leak | over-mask |
|---|---|---|---|---|---|
| sct-default | 0.879 | 0.741 | 0.568 | 0.514 | 0.045 |
| presidio-full | 0.877 | 0.534 | 0.463 | 0.516 | 0.081 |
| gliner2-fastino | 0.872 | 0.607 | 0.620 | 0.431 | 0.073 |
| sct-gliner | 0.856 | 0.626 | 0.288 | 0.558 | 0.057 |
| bardsai-v3 | 0.853 | 0.782 | 0.691 | 0.477 | 0.040 |
| pii-tracer | 0.846 | 0.798 | 0.656 | 0.406 | 0.061 |
| openmed | 0.845 | 0.771 | 0.392 | 0.542 | 0.048 |
| sct-training | 0.843 | 0.800 | 0.568 | 0.581 | 0.023 |
| sct-gliner-nvidia | 0.841 | 0.714 | 0.304 | 0.546 | 0.047 |
| gliner-knowledgator | 0.803 | 0.593 | 0.593 | 0.505 | 0.082 |
| gliner-nvidia | 0.760 | 0.681 | 0.540 | 0.448 | 0.062 |
| presidio | 0.746 | 0.569 | 0.496 | 0.606 | 0.066 |
| opf | 0.727 | 0.776 | 0.638 | 0.540 | 0.045 |
| scrubadub | 0.660 | 0.771 | 0.591 | 0.740 | 0.016 |
| piiranha | 0.560 | 0.844 | 0.273 | 0.824 | 0.013 |
| llm-guard | 0.553 | 0.895 | 0.296 | 0.739 | 0.005 |
| ai4privacy | 0.474 | 0.700 | 0.172 | 0.928 | 0.032 |
| datafog | 0.173 | 0.760 | 0.182 | 0.954 | 0.012 |

### same-set (non-SCT entries with clean sets)

Datasets: dialogpii, europii, meddeid, newmean, tonic (clean for all: newmean, dialogpii, tonic, meddeid, greek_deid, neonredact, pl_pii, europii); only docs in languages every entry claims; point values.

| entry | mR | mP | strict F1 | doc leak | over-mask |
|---|---|---|---|---|---|
| presidio-full | 0.877 | 0.534 | 0.463 | 0.516 | 0.081 |
| gliner2-fastino | 0.872 | 0.607 | 0.620 | 0.431 | 0.073 |
| bardsai-v3 | 0.853 | 0.782 | 0.691 | 0.477 | 0.040 |
| pii-tracer | 0.846 | 0.798 | 0.656 | 0.406 | 0.061 |
| openmed | 0.845 | 0.771 | 0.392 | 0.542 | 0.048 |
| gliner-knowledgator | 0.803 | 0.593 | 0.593 | 0.505 | 0.082 |
| gliner-nvidia | 0.760 | 0.681 | 0.540 | 0.448 | 0.062 |
| presidio | 0.746 | 0.569 | 0.496 | 0.606 | 0.066 |
| opf | 0.727 | 0.776 | 0.638 | 0.540 | 0.045 |
| scrubadub | 0.660 | 0.771 | 0.591 | 0.740 | 0.016 |
| piiranha | 0.560 | 0.844 | 0.273 | 0.824 | 0.013 |
| llm-guard | 0.553 | 0.895 | 0.296 | 0.739 | 0.005 |
| ai4privacy | 0.474 | 0.700 | 0.172 | 0.928 | 0.032 |
| datafog | 0.173 | 0.760 | 0.182 | 0.954 | 0.012 |

### SCT vs each entry, datasets clean for both, languages both claim

| entry | sets | mR (entry - SCT, 95% CI) | mP (entry - SCT, 95% CI) | strict F1 (entry - SCT, 95% CI) | doc leak (entry - SCT, 95% CI) | over-mask (entry - SCT, 95% CI) | SCT mR, same docs | SCT mP, same docs |
|---|---|---|---|---|---|---|---|---|
| gliner2-fastino | newmean, dialogpii, tonic, meddeid, europii | 0.864 (-0.012, +0.009) | 0.640 (-0.113, -0.088)* | 0.642 (+0.043, +0.066)* | 0.437 (-0.150, -0.110)* | 0.062 (+0.013, +0.017)* | 0.866 | 0.739 |
| sct-gliner | newmean, dialogpii, tonic, meddeid, europii | 0.856 (-0.035, -0.012)* | 0.626 (-0.132, -0.101)* | 0.288 (-0.295, -0.263)* | 0.558 (+0.020, +0.067)* | 0.057 (+0.010, +0.014)* | 0.879 | 0.741 |
| bardsai-v3 | newmean, dialogpii, tonic, meddeid, greek_deid, pl_pii, europii | 0.852 (-0.045, -0.024)* | 0.822 (+0.060, +0.080)* | 0.674 (+0.093, +0.124)* | 0.567 (-0.044, -0.003)* | 0.033 (-0.012, -0.006)* | 0.886 | 0.752 |
| sct-gliner-nvidia | newmean, dialogpii, tonic, meddeid, europii | 0.841 (-0.051, -0.024)* | 0.714 (-0.042, -0.013)* | 0.304 (-0.283, -0.244)* | 0.546 (+0.009, +0.056)* | 0.047 (+0.001, +0.004)* | 0.879 | 0.741 |
| sct-training | newmean, dialogpii, tonic, meddeid, greek_deid, neonredact, pl_pii, europii | 0.838 (-0.054, -0.045)* | 0.816 (+0.056, +0.075)* | 0.553 (-0.000, +0.009) | 0.639 (+0.050, +0.068)* | 0.030 (-0.024, -0.020)* | 0.888 | 0.750 |
| pii-tracer | newmean, dialogpii, tonic, meddeid, greek_deid, neonredact, pl_pii, europii | 0.837 (-0.070, -0.033)* | 0.781 (+0.015, +0.045)* | 0.629 (+0.063, +0.097)* | 0.502 (-0.101, -0.057)* | 0.055 (+0.000, +0.007)* | 0.888 | 0.750 |
| gliner-knowledgator | newmean, dialogpii, tonic, meddeid, europii | 0.803 (-0.092, -0.057)* | 0.593 (-0.169, -0.130)* | 0.593 (+0.004, +0.041)* | 0.505 (-0.043, +0.022) | 0.082 (+0.034, +0.041)* | 0.879 | 0.741 |
| gliner-nvidia | newmean, dialogpii, tonic, meddeid, europii | 0.760 (-0.140, -0.090)* | 0.681 (-0.078, -0.043)* | 0.540 (-0.048, -0.009)* | 0.448 (-0.096, -0.034)* | 0.062 (+0.014, +0.021)* | 0.879 | 0.741 |
| openmed | newmean, dialogpii, tonic, meddeid, neonredact, europii | 0.756 (-0.128, -0.104)* | 0.753 (+0.005, +0.020)* | 0.360 (-0.216, -0.189)* | 0.648 (+0.071, +0.112)* | 0.049 (-0.011, -0.006)* | 0.872 | 0.740 |
| presidio | newmean, dialogpii, tonic, meddeid, europii | 0.746 (-0.151, -0.113)* | 0.569 (-0.189, -0.155)* | 0.496 (-0.091, -0.054)* | 0.606 (+0.063, +0.121)* | 0.066 (+0.019, +0.025)* | 0.879 | 0.741 |
| opf | newmean, dialogpii, tonic, meddeid, europii | 0.727 (-0.174, -0.130)* | 0.776 (+0.013, +0.058)* | 0.638 (+0.042, +0.095)* | 0.540 (-0.003, +0.055) | 0.045 (-0.002, +0.004) | 0.879 | 0.741 |
| presidio-full | newmean, dialogpii, tonic, meddeid, greek_deid, neonredact, pl_pii, europii | 0.711 (-0.188, -0.167)* | 0.548 (-0.217, -0.186)* | 0.362 (-0.200, -0.174)* | 0.726 (+0.126, +0.165)* | 0.088 (+0.033, +0.040)* | 0.888 | 0.750 |
| scrubadub | newmean, dialogpii, tonic, meddeid, europii | 0.660 (-0.236, -0.201)* | 0.771 (+0.011, +0.049)* | 0.591 (+0.004, +0.042)* | 0.740 (+0.199, +0.253)* | 0.016 (-0.031, -0.025)* | 0.879 | 0.741 |
| llm-guard | newmean, dialogpii, tonic, meddeid, europii | 0.553 (-0.351, -0.300)* | 0.895 (+0.129, +0.175)* | 0.296 (-0.294, -0.248)* | 0.739 (+0.196, +0.255)* | 0.005 (-0.043, -0.036)* | 0.879 | 0.741 |
| piiranha | newmean, dialogpii, tonic, meddeid, europii | 0.488 (-0.390, -0.358)* | 0.829 (+0.074, +0.103)* | 0.252 (-0.353, -0.317)* | 0.892 (+0.308, +0.346)* | 0.015 (-0.033, -0.029)* | 0.863 | 0.739 |
| ai4privacy | newmean, dialogpii, tonic, meddeid, europii | 0.357 (-0.517, -0.487)* | 0.656 (-0.104, -0.060)* | 0.108 (-0.489, -0.456)* | 0.963 (+0.376, +0.414)* | 0.032 (-0.017, -0.012)* | 0.859 | 0.738 |
| datafog | newmean, dialogpii, tonic, meddeid, greek_deid, neonredact, pl_pii, europii | 0.212 (-0.688, -0.664)* | 0.775 (-0.004, +0.058) | 0.220 (-0.345, -0.311)* | 0.951 (+0.351, +0.390)* | 0.011 (-0.043, -0.037)* | 0.888 | 0.750 |

### SCT vs each entry, datasets unknown for both, languages both claim

No dataset in this group is unknown for both sct-default and any other entry (languages both claim).

## default track, full view, r2-privacy-real: each entry on its own clean sets, claimed languages

† = under 99% complete on a dataset in this row or cell (PROTOCOL section 15; see the completion table).

| entry | clean sets with claimed-language docs (all clean sets) | mR [95% CI] | mP [95% CI] | strict F1 [95% CI] | doc leak [95% CI] | over-mask [95% CI] | mR clean, outside claimed langs | mR unknown (U) sets | mR U sets, outside claimed langs | (mR excluded sets, info only) | docs/s CPU | peak RAM | GPU: device, docs/s, peak VRAM | claimed languages | licence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sct-default | kdpii, nus_lt, redmadrobot (3) | 0.807 [0.766, 0.842] | 0.843 [0.799, 0.884] | 0.526 [0.479, 0.573] | 0.587 [0.535, 0.639] | 0.013 [0.010, 0.017] | - | - | - | - | 1.55 | 10,023 MB | n/a | all (multilingual claim) | MIT |
| sct-training | kdpii, nus_lt, redmadrobot (3) | 0.724 [0.679, 0.765] | 0.883 [0.851, 0.914] | 0.521 [0.469, 0.572] | 0.636 [0.586, 0.689] | 0.009 [0.007, 0.012] | - | - | - | - | 1.56 | 10,205 MB | n/a | all (multilingual claim) | MIT |
| presidio-full | kdpii, nus_lt, redmadrobot (3) | 0.625 [0.589, 0.662] | 0.877 [0.848, 0.902] | 0.426 [0.385, 0.463] | 0.732 [0.690, 0.776] | 0.016 [0.012, 0.019] | - | - | - | - | 14 | 19,753 MB | n/a | all (spaCy engine per gold language + recognizers) | MIT |
| bardsai-v3 | nus_lt (3) | 0.505 [0.463, 0.556] | 0.916 [0.825, 0.984] | 0.462 [0.412, 0.520] | 0.979 [0.949, 1.000] | 0.005 [0.002, 0.007] | 0.590 (2) | - | - | - | 2.42 | 6,975 MB | n/a | 24 EU | Apache-2.0 |
| pii-tracer | kdpii, nus_lt, redmadrobot (3) | 0.461 [0.417, 0.506] | 0.787 [0.739, 0.837] | 0.301 [0.259, 0.347] | 0.612 [0.567, 0.659] | 0.014 [0.010, 0.019] | - | - | - | - | 1.33 | 3,098 MB | cuda, 19.8 docs/s, 1,826 MB | all (model card: en + multilingual) | MIT |
| datafog | kdpii, nus_lt, redmadrobot (3) | 0.073 [0.051, 0.101] | 0.944 [0.908, 0.977] | 0.082 [0.051, 0.116] | 0.949 [0.923, 0.970] | 0.001 [0.000, 0.001] | - | - | - | - | 1,555 | 34 MB | n/a | all (regex, no language claim) | MIT |
| sct-gliner-nvidia | none (3) | - | - | - | - | - | 0.710 (3) | - | - | - | n/a | n/a | cuda, fail docs/s, - MB | en | MIT + NVIDIA Open Model |
| sct-gliner | none (3) | - | - | - | - | - | 0.646 (3) | - | - | - | 0.61 | 6,715 MB | cuda, fail docs/s, - MB | en | MIT + Apache-2.0 model |
| scrubadub | none (3) | - | - | - | - | - | 0.293 (3) | - | - | - | 1.20 | 5,180 MB | n/a | en | Apache-2.0 / MIT + MIT spaCy model |
| presidio | none (3) | - | - | - | - | - | 0.500 (3) | - | - | - | 26 | 1,080 MB | n/a | en | MIT |
| piiranha | none (3) | - | - | - | - | - | 0.431 (3) | - | - | - | 1.80 | 2,418 MB | cuda, 41.8 docs/s, 1,884 MB | en it fr de nl es | CC-BY-NC-ND-4.0 model (benchmark only) |
| opf | none (2) | - | - | - | - | - | 0.384 (2) | - | 0.338 (1) | - | n/a | n/a | cuda, 17.0 docs/s, 4,626 MB | en | Apache-2.0 |
| openmed | none (2) | - | - | - | - | - | 0.503 (2) | 0.613 (1) | - | - | 0.42 | 25,637 MB | cuda, 9.9 docs/s, 5,100 MB | ar bn de en es fr hi it ja ko nl pt te tr vi zh | Apache-2.0 |
| llm-guard | none (3) | - | - | - | - | - | 0.306 (3) | - | - | - | 1.40 | 2,271 MB | cuda, 11.4 docs/s, 1,322 MB | en | MIT + CC-BY-NC-4.0 model |
| gliner2-fastino | none (3) | - | - | - | - | - | 0.776 (3) | - | - | - | 0.63 | 34,356 MB | cuda, fail docs/s, - MB | en fr es de it pt nl | Apache-2.0 |
| gliner-nvidia | none (3) | - | - | - | - | - | 0.508 (3) | - | - | - | n/a | n/a | cuda, fail docs/s, 5,354 MB | en | Apache-2.0 + NVIDIA Open Model |
| gliner-knowledgator† | none (3) | - | - | - | - | - | 0.724 (3) | - | - | - | n/a | n/a | cuda, fail docs/s, - MB | en | Apache-2.0 |
| ai4privacy | none (3) | - | - | - | - | - | 0.246 (3) | - | - | - | 1.37 | 1,534 MB | cuda, 20.8 docs/s, 1,226 MB | fr en de te hi it es nl | MIT |

### same-set (every entry)

No dataset in this group is clean for every one of: sct-default, sct-gliner, presidio, gliner-nvidia, gliner2-fastino, opf, bardsai-v3, datafog, gliner-knowledgator, piiranha, llm-guard, scrubadub, ai4privacy, openmed, pii-tracer, presidio-full, sct-gliner-nvidia, sct-training (in languages all of them claim).

### same-set (non-SCT entries with clean sets)

No dataset in this group is clean for every one of: presidio, gliner-nvidia, gliner2-fastino, opf, bardsai-v3, datafog, gliner-knowledgator, piiranha, llm-guard, scrubadub, ai4privacy, openmed, pii-tracer, presidio-full (in languages all of them claim).

### SCT vs each entry, datasets clean for both, languages both claim

| entry | sets | mR (entry - SCT, 95% CI) | mP (entry - SCT, 95% CI) | strict F1 (entry - SCT, 95% CI) | doc leak (entry - SCT, 95% CI) | over-mask (entry - SCT, 95% CI) | SCT mR, same docs | SCT mP, same docs |
|---|---|---|---|---|---|---|---|---|
| sct-training | kdpii, nus_lt, redmadrobot | 0.724 (-0.106, -0.064)* | 0.883 (+0.013, +0.072)* | 0.521 (-0.031, +0.022) | 0.636 (+0.028, +0.071)* | 0.009 (-0.006, -0.002)* | 0.807 | 0.843 |
| presidio-full | kdpii, nus_lt, redmadrobot | 0.625 (-0.220, -0.146)* | 0.877 (-0.004, +0.074) | 0.426 (-0.151, -0.049)* | 0.732 (+0.095, +0.198)* | 0.016 (-0.001, +0.004) | 0.807 | 0.843 |
| bardsai-v3 | nus_lt | 0.505 (-0.472, -0.378)* | 0.916 (-0.019, +0.060) | 0.462 (-0.187, -0.098)* | 0.979 (+0.052, +0.200)* | 0.005 (-0.010, -0.004)* | 0.934 | 0.898 |
| pii-tracer | kdpii, nus_lt, redmadrobot | 0.461 (-0.388, -0.298)* | 0.787 (-0.109, +0.002) | 0.301 (-0.287, -0.160)* | 0.612 (-0.032, +0.084) | 0.014 (-0.003, +0.004) | 0.807 | 0.843 |
| datafog | kdpii, nus_lt, redmadrobot | 0.073 (-0.774, -0.689)* | 0.944 (+0.051, +0.156)* | 0.082 (-0.499, -0.387)* | 0.949 (+0.309, +0.414)* | 0.001 (-0.016, -0.010)* | 0.807 | 0.843 |

No clean dataset in common with sct-default: ai4privacy, gliner-knowledgator, gliner-nvidia, gliner2-fastino, llm-guard, openmed, opf, piiranha, presidio, scrubadub, sct-gliner, sct-gliner-nvidia.

### SCT vs each entry, datasets unknown for both, languages both claim

| entry | sets | mR (entry - SCT, 95% CI) | mP (entry - SCT, 95% CI) | strict F1 (entry - SCT, 95% CI) | doc leak (entry - SCT, 95% CI) | over-mask (entry - SCT, 95% CI) | SCT mR, same docs | SCT mP, same docs |
|---|---|---|---|---|---|---|---|---|
| openmed | kdpii | 0.613 (-0.185, +0.056) | 0.447 (-0.548, -0.335)* | 0.274 (-0.319, -0.043)* | 0.486 (-0.116, +0.122) | 0.025 (+0.013, +0.026)* | 0.675 | 0.889 |

No unknown dataset in common with sct-default: ai4privacy, bardsai-v3, datafog, gliner-knowledgator, gliner-nvidia, gliner2-fastino, llm-guard, opf, pii-tracer, piiranha, presidio, presidio-full, scrubadub, sct-gliner, sct-gliner-nvidia, sct-training.

## default track, full view, r2-untyped: each entry on its own clean sets, claimed languages

† = under 99% complete on a dataset in this row or cell (PROTOCOL section 15; see the completion table).

| entry | clean sets with claimed-language docs (all clean sets) | mR [95% CI] | mP [95% CI] | strict F1 [95% CI] | doc leak [95% CI] | over-mask [95% CI] | mR clean, outside claimed langs | mR unknown (U) sets | mR U sets, outside claimed langs | (mR excluded sets, info only) | docs/s CPU | peak RAM | GPU: device, docs/s, peak VRAM | claimed languages | licence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bardsai-v3 | redactionbench (1) | 0.745 [0.644, 0.815] | 0.420 [0.355, 0.516] | - | 0.744 [0.681, 0.806] | 0.051 [0.028, 0.079] | - | - | - | - | 2.42 | 6,975 MB | n/a | 24 EU | Apache-2.0 |
| presidio-full | redactionbench (1) | 0.720 [0.614, 0.818] | 0.235 [0.164, 0.302] | - | 0.884 [0.832, 0.931] | 0.123 [0.102, 0.146] | - | - | - | - | 14 | 19,753 MB | n/a | all (spaCy engine per gold language + recognizers) | MIT |
| openmed | redactionbench (1) | 0.692 [0.603, 0.768] | 0.451 [0.386, 0.533] | - | 0.793 [0.732, 0.849] | 0.049 [0.029, 0.076] | - | - | - | - | 0.42 | 25,637 MB | cuda, 9.9 docs/s, 5,100 MB | ar bn de en es fr hi it ja ko nl pt te tr vi zh | Apache-2.0 |
| presidio | redactionbench (1) | 0.692 [0.574, 0.799] | 0.262 [0.187, 0.338] | - | 0.902 [0.854, 0.946] | 0.090 [0.069, 0.113] | - | - | - | - | 26 | 1,080 MB | n/a | en | MIT |
| sct-default | redactionbench (1) | 0.687 [0.552, 0.809] | 0.297 [0.240, 0.359] | - | 0.848 [0.792, 0.903] | 0.063 [0.045, 0.084] | - | - | - | - | 1.55 | 10,023 MB | n/a | all (multilingual claim) | MIT |
| sct-gliner | redactionbench (1) | 0.681 [0.549, 0.797] | 0.213 [0.154, 0.271] | - | 0.896 [0.846, 0.939] | 0.089 [0.069, 0.112] | - | - | - | - | 0.61 | 6,715 MB | cuda, fail docs/s, - MB | en | MIT + Apache-2.0 model |
| sct-gliner-nvidia† | redactionbench (1) | 0.668 [0.535, 0.791] | 0.320 [0.263, 0.389] | - | 0.896 [0.850, 0.939] | 0.061 [0.043, 0.085] | - | - | - | - | n/a | n/a | cuda, fail docs/s, - MB | en | MIT + NVIDIA Open Model |
| sct-training | redactionbench (1) | 0.662 [0.537, 0.782] | 0.343 [0.228, 0.468] | - | 0.915 [0.873, 0.957] | 0.044 [0.033, 0.058] | - | - | - | - | 1.56 | 10,205 MB | n/a | all (multilingual claim) | MIT |
| gliner2-fastino† | redactionbench (1) | 0.662 [0.530, 0.781] | 0.243 [0.167, 0.324] | - | 0.902 [0.854, 0.941] | 0.063 [0.052, 0.074] | - | - | - | - | 0.63 | 34,356 MB | cuda, fail docs/s, - MB | en fr es de it pt nl | Apache-2.0 |
| llm-guard | redactionbench (1) | 0.657 [0.471, 0.778] | 0.408 [0.277, 0.566] | - | 0.963 [0.932, 0.988] | 0.062 [0.028, 0.100] | - | - | - | - | 1.40 | 2,271 MB | cuda, 11.4 docs/s, 1,322 MB | en | MIT + CC-BY-NC-4.0 model |
| ai4privacy | redactionbench (1) | 0.573 [0.450, 0.697] | 0.323 [0.248, 0.406] | - | 0.994 [0.981, 1.000] | 0.021 [0.013, 0.029] | - | - | - | - | 1.37 | 1,534 MB | cuda, 20.8 docs/s, 1,226 MB | fr en de te hi it es nl | MIT |
| datafog | redactionbench (1) | 0.531 [0.376, 0.671] | 0.498 [0.384, 0.636] | - | 0.976 [0.951, 0.994] | 0.011 [0.007, 0.017] | - | - | - | - | 1,555 | 34 MB | n/a | all (regex, no language claim) | MIT |
| piiranha | redactionbench (1) | 0.375 [0.254, 0.498] | 0.781 [0.675, 0.870] | - | 0.982 [0.958, 1.000] | 0.004 [0.002, 0.006] | - | - | - | - | 1.80 | 2,418 MB | cuda, 41.8 docs/s, 1,884 MB | en it fr de nl es | CC-BY-NC-ND-4.0 model (benchmark only) |
| opf | redactionbench (1) | 0.351 [0.270, 0.481] | 0.519 [0.361, 0.730] | - | 0.860 [0.807, 0.908] | 0.021 [0.010, 0.035] | - | - | - | - | n/a | n/a | cuda, 17.0 docs/s, 4,626 MB | en | Apache-2.0 |
| pii-tracer | redactionbench (1) | 0.332 [0.231, 0.483] | 0.720 [0.653, 0.783] | - | 0.665 [0.589, 0.737] | 0.008 [0.006, 0.012] | - | - | - | - | 1.33 | 3,098 MB | cuda, 19.8 docs/s, 1,826 MB | all (model card: en + multilingual) | MIT |
| scrubadub | redactionbench (1) | 0.211 [0.146, 0.319] | 0.208 [0.130, 0.330] | - | 0.939 [0.901, 0.975] | 0.041 [0.029, 0.054] | - | - | - | - | 1.20 | 5,180 MB | n/a | en | Apache-2.0 / MIT + MIT spaCy model |
| gliner-knowledgator† | redactionbench (1) | 0.197 [0.127, 0.327] | 0.550 [0.478, 0.622] | - | 0.884 [0.835, 0.932] | 0.007 [0.005, 0.009] | - | - | - | - | n/a | n/a | cuda, fail docs/s, - MB | en | Apache-2.0 |
| gliner-nvidia | redactionbench (1) | 0.086 [0.057, 0.138] | 0.649 [0.579, 0.718] | - | 0.988 [0.969, 1.000] | 0.003 [0.003, 0.004] | - | - | - | - | n/a | n/a | cuda, fail docs/s, 5,354 MB | en | Apache-2.0 + NVIDIA Open Model |

### same-set (every entry)

Datasets: redactionbench (clean for all: redactionbench); only docs in languages every entry claims; point values.

| entry | mR | mP | strict F1 | doc leak | over-mask |
|---|---|---|---|---|---|
| bardsai-v3 | 0.745 | 0.420 | n/a | 0.744 | 0.051 |
| presidio-full | 0.720 | 0.235 | n/a | 0.884 | 0.123 |
| openmed | 0.692 | 0.451 | n/a | 0.793 | 0.049 |
| presidio | 0.692 | 0.262 | n/a | 0.902 | 0.090 |
| sct-default | 0.687 | 0.297 | n/a | 0.848 | 0.063 |
| sct-gliner | 0.681 | 0.213 | n/a | 0.896 | 0.089 |
| sct-gliner-nvidia† | 0.668 | 0.320 | n/a | 0.896 | 0.061 |
| sct-training | 0.662 | 0.343 | n/a | 0.915 | 0.044 |
| gliner2-fastino† | 0.662 | 0.243 | n/a | 0.902 | 0.063 |
| llm-guard | 0.657 | 0.408 | n/a | 0.963 | 0.062 |
| ai4privacy | 0.573 | 0.323 | n/a | 0.994 | 0.021 |
| datafog | 0.531 | 0.498 | n/a | 0.976 | 0.011 |
| piiranha | 0.375 | 0.781 | n/a | 0.982 | 0.004 |
| opf | 0.351 | 0.519 | n/a | 0.860 | 0.021 |
| pii-tracer | 0.332 | 0.720 | n/a | 0.665 | 0.008 |
| scrubadub | 0.211 | 0.208 | n/a | 0.939 | 0.041 |
| gliner-knowledgator† | 0.197 | 0.550 | n/a | 0.884 | 0.007 |
| gliner-nvidia | 0.086 | 0.649 | n/a | 0.988 | 0.003 |

### same-set (non-SCT entries with clean sets)

Datasets: redactionbench (clean for all: redactionbench); only docs in languages every entry claims; point values.

| entry | mR | mP | strict F1 | doc leak | over-mask |
|---|---|---|---|---|---|
| bardsai-v3 | 0.745 | 0.420 | n/a | 0.744 | 0.051 |
| presidio-full | 0.720 | 0.235 | n/a | 0.884 | 0.123 |
| openmed | 0.692 | 0.451 | n/a | 0.793 | 0.049 |
| presidio | 0.692 | 0.262 | n/a | 0.902 | 0.090 |
| gliner2-fastino† | 0.662 | 0.243 | n/a | 0.902 | 0.063 |
| llm-guard | 0.657 | 0.408 | n/a | 0.963 | 0.062 |
| ai4privacy | 0.573 | 0.323 | n/a | 0.994 | 0.021 |
| datafog | 0.531 | 0.498 | n/a | 0.976 | 0.011 |
| piiranha | 0.375 | 0.781 | n/a | 0.982 | 0.004 |
| opf | 0.351 | 0.519 | n/a | 0.860 | 0.021 |
| pii-tracer | 0.332 | 0.720 | n/a | 0.665 | 0.008 |
| scrubadub | 0.211 | 0.208 | n/a | 0.939 | 0.041 |
| gliner-knowledgator† | 0.197 | 0.550 | n/a | 0.884 | 0.007 |
| gliner-nvidia | 0.086 | 0.649 | n/a | 0.988 | 0.003 |

### SCT vs each entry, datasets clean for both, languages both claim

| entry | sets | mR (entry - SCT, 95% CI) | mP (entry - SCT, 95% CI) | strict F1 (entry - SCT, 95% CI) | doc leak (entry - SCT, 95% CI) | over-mask (entry - SCT, 95% CI) | SCT mR, same docs | SCT mP, same docs |
|---|---|---|---|---|---|---|---|---|
| bardsai-v3 | redactionbench | 0.745 (-0.055, +0.175) | 0.420 (+0.058, +0.204)* | n/a | 0.744 (-0.159, -0.050)* | 0.051 (-0.024, +0.002) | 0.687 | 0.297 |
| presidio-full | redactionbench | 0.720 (-0.022, +0.096) | 0.235 (-0.119, -0.007)* | n/a | 0.884 (-0.012, +0.083) | 0.123 (+0.046, +0.077)* | 0.687 | 0.297 |
| openmed | redactionbench | 0.692 (-0.077, +0.098) | 0.451 (+0.111, +0.211)* | n/a | 0.793 (-0.110, -0.000) | 0.049 (-0.022, -0.004)* | 0.687 | 0.297 |
| presidio | redactionbench | 0.692 (-0.053, +0.068) | 0.262 (-0.098, +0.021) | n/a | 0.902 (+0.006, +0.104)* | 0.090 (+0.014, +0.044)* | 0.687 | 0.297 |
| sct-gliner | redactionbench | 0.681 (-0.033, +0.018) | 0.213 (-0.126, -0.044)* | n/a | 0.896 (+0.012, +0.086)* | 0.089 (+0.016, +0.038)* | 0.687 | 0.297 |
| sct-gliner-nvidia† | redactionbench | 0.668 (-0.048, +0.001) | 0.320 (-0.016, +0.064) | n/a | 0.896 (+0.018, +0.085)* | 0.061 (-0.011, +0.007) | 0.687 | 0.297 |
| sct-training | redactionbench | 0.662 (-0.076, +0.036) | 0.343 (-0.033, +0.134) | n/a | 0.915 (+0.031, +0.107)* | 0.044 (-0.032, -0.008)* | 0.687 | 0.297 |
| gliner2-fastino† | redactionbench | 0.662 (-0.078, +0.038) | 0.243 (-0.109, +0.012) | n/a | 0.902 (+0.012, +0.103)* | 0.063 (-0.021, +0.019) | 0.687 | 0.297 |
| llm-guard | redactionbench | 0.657 (-0.221, +0.155) | 0.408 (-0.008, +0.260) | n/a | 0.963 (+0.069, +0.169)* | 0.062 (-0.041, +0.043) | 0.687 | 0.297 |
| ai4privacy | redactionbench | 0.573 (-0.190, -0.065)* | 0.323 (-0.033, +0.088) | n/a | 0.994 (+0.092, +0.202)* | 0.021 (-0.056, -0.030)* | 0.687 | 0.297 |
| datafog | redactionbench | 0.531 (-0.283, -0.075)* | 0.498 (+0.078, +0.351)* | n/a | 0.976 (+0.073, +0.184)* | 0.011 (-0.072, -0.035)* | 0.687 | 0.297 |
| piiranha | redactionbench | 0.375 (-0.390, -0.227)* | 0.781 (+0.386, +0.573)* | n/a | 0.982 (+0.079, +0.190)* | 0.004 (-0.080, -0.043)* | 0.687 | 0.297 |
| opf | redactionbench | 0.351 (-0.461, -0.175)* | 0.519 (+0.067, +0.421)* | n/a | 0.860 (-0.053, +0.072) | 0.021 (-0.059, -0.026)* | 0.687 | 0.297 |
| pii-tracer | redactionbench | 0.332 (-0.557, -0.152)* | 0.720 (+0.352, +0.497)* | n/a | 0.665 (-0.247, -0.117)* | 0.008 (-0.075, -0.038)* | 0.687 | 0.297 |
| scrubadub | redactionbench | 0.211 (-0.630, -0.317)* | 0.208 (-0.151, +0.002) | n/a | 0.939 (+0.051, +0.139)* | 0.041 (-0.038, -0.007)* | 0.687 | 0.297 |
| gliner-knowledgator† | redactionbench | 0.197 (-0.641, -0.330)* | 0.550 (+0.190, +0.312)* | n/a | 0.884 (-0.024, +0.094) | 0.007 (-0.077, -0.039)* | 0.687 | 0.297 |
| gliner-nvidia | redactionbench | 0.086 (-0.729, -0.460)* | 0.649 (+0.277, +0.425)* | n/a | 0.988 (+0.085, +0.196)* | 0.003 (-0.081, -0.042)* | 0.687 | 0.297 |

### SCT vs each entry, datasets unknown for both, languages both claim

No dataset in this group is unknown for both sct-default and any other entry (languages both claim).

## default track, shared view, r2-privacy-synthetic: each entry on its own clean sets, claimed languages

† = under 99% complete on a dataset in this row or cell (PROTOCOL section 15; see the completion table).

| entry | clean sets with claimed-language docs (all clean sets) | mR [95% CI] | mP [95% CI] | strict F1 [95% CI] | doc leak [95% CI] | over-mask [95% CI] | mR clean, outside claimed langs | mR unknown (U) sets | mR U sets, outside claimed langs | (mR excluded sets, info only) | docs/s CPU | peak RAM | GPU: device, docs/s, peak VRAM | claimed languages | licence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pii-tracer | dialogpii, europii, greek_deid, meddeid, neonredact, newmean, pl_pii, tonic (8) | 0.940 [0.918, 0.959] | 0.622 [0.598, 0.644] | 0.717 [0.700, 0.735] | 0.110 [0.094, 0.129] | 0.055 [0.052, 0.059] | - | - | - | - | 1.33 | 3,098 MB | cuda, 19.8 docs/s, 1,826 MB | all (model card: en + multilingual) | MIT |
| bardsai-v3 | dialogpii, europii, greek_deid, meddeid, newmean, pl_pii, tonic (8) | 0.880 [0.870, 0.891] | 0.663 [0.636, 0.693] | 0.786 [0.771, 0.799] | 0.224 [0.206, 0.241] | 0.033 [0.031, 0.036] | 0.714 (3) | - | - | - | 2.42 | 6,975 MB | n/a | 24 EU | Apache-2.0 |
| opf | dialogpii, europii, meddeid, newmean, tonic (8) | 0.852 [0.829, 0.874] | 0.630 [0.594, 0.663] | 0.754 [0.733, 0.775] | 0.245 [0.214, 0.274] | 0.045 [0.043, 0.047] | 0.659 (7) | - | - | - | n/a | n/a | cuda, 17.0 docs/s, 4,626 MB | en | Apache-2.0 |
| openmed | dialogpii, europii, meddeid, neonredact, newmean, tonic (8) | 0.839 [0.830, 0.850] | 0.489 [0.470, 0.510] | 0.718 [0.705, 0.729] | 0.262 [0.246, 0.278] | 0.049 [0.047, 0.052] | 0.819 (5) | - | - | - | 0.42 | 25,637 MB | cuda, 9.9 docs/s, 5,100 MB | ar bn de en es fr hi it ja ko nl pt te tr vi zh | Apache-2.0 |
| presidio-full | dialogpii, europii, greek_deid, meddeid, neonredact, newmean, pl_pii, tonic (8) | 0.812 [0.800, 0.823] | 0.398 [0.375, 0.422] | 0.509 [0.490, 0.529] | 0.439 [0.415, 0.461] | 0.088 [0.084, 0.091] | - | - | - | - | 14 | 19,753 MB | n/a | all (spaCy engine per gold language + recognizers) | MIT |
| sct-default | dialogpii, europii, greek_deid, meddeid, neonredact, newmean, pl_pii, tonic (8) | 0.805 [0.789, 0.820] | 0.509 [0.488, 0.532] | 0.504 [0.487, 0.524] | 0.321 [0.302, 0.340] | 0.051 [0.048, 0.055] | - | - | - | - | 1.55 | 10,023 MB | n/a | all (multilingual claim) | MIT |
| presidio | dialogpii, europii, meddeid, newmean, tonic (8) | 0.795 [0.767, 0.825] | 0.378 [0.355, 0.403] | 0.515 [0.484, 0.548] | 0.386 [0.348, 0.421] | 0.066 [0.063, 0.069] | 0.737 (7) | - | - | - | 26 | 1,080 MB | n/a | en | MIT |
| sct-gliner | dialogpii, europii, meddeid, newmean, tonic (8) | 0.785 [0.756, 0.817] | 0.327 [0.308, 0.345] | 0.437 [0.411, 0.470] | 0.339 [0.304, 0.371] | 0.057 [0.053, 0.060] | 0.792 (7) | - | - | - | 0.61 | 6,715 MB | cuda, fail docs/s, - MB | en | MIT + Apache-2.0 model |
| sct-gliner-nvidia | dialogpii, europii, meddeid, newmean, tonic (8) | 0.776 [0.747, 0.809] | 0.407 [0.384, 0.433] | 0.471 [0.441, 0.508] | 0.326 [0.292, 0.360] | 0.047 [0.044, 0.050] | 0.792 (7) | - | - | - | n/a | n/a | cuda, fail docs/s, - MB | en | MIT + NVIDIA Open Model |
| gliner2-fastino | dialogpii, europii, meddeid, newmean, tonic (8) | 0.732 [0.709, 0.756] | 0.349 [0.332, 0.370] | 0.705 [0.686, 0.722] | 0.370 [0.345, 0.396] | 0.062 [0.059, 0.065] | 0.727 (6) | - | - | - | 0.63 | 34,356 MB | cuda, fail docs/s, - MB | en fr es de it pt nl | Apache-2.0 |
| ai4privacy | dialogpii, europii, meddeid, newmean, tonic (8) | 0.697 [0.679, 0.719] | 0.583 [0.558, 0.612] | 0.172 [0.153, 0.193] | 0.800 [0.778, 0.821] | 0.032 [0.031, 0.033] | 0.737 (6) | - | - | - | 1.37 | 1,534 MB | cuda, 20.8 docs/s, 1,226 MB | fr en de te hi it es nl | MIT |
| gliner-nvidia | dialogpii, europii, meddeid, newmean, tonic (8) | 0.687 [0.651, 0.738] | 0.428 [0.400, 0.460] | 0.616 [0.586, 0.653] | 0.375 [0.341, 0.408] | 0.062 [0.059, 0.065] | 0.535 (7) | - | - | - | n/a | n/a | cuda, fail docs/s, 5,354 MB | en | Apache-2.0 + NVIDIA Open Model |
| sct-training | dialogpii, europii, greek_deid, meddeid, neonredact, newmean, pl_pii, tonic (8) | 0.661 [0.641, 0.681] | 0.555 [0.527, 0.587] | 0.462 [0.443, 0.484] | 0.567 [0.547, 0.587] | 0.030 [0.027, 0.033] | - | - | - | - | 1.56 | 10,205 MB | n/a | all (multilingual claim) | MIT |
| gliner-knowledgator | dialogpii, europii, meddeid, newmean, tonic (8) | 0.633 [0.605, 0.663] | 0.335 [0.307, 0.358] | 0.566 [0.537, 0.591] | 0.514 [0.481, 0.545] | 0.082 [0.078, 0.086] | 0.729 (7) | - | - | - | n/a | n/a | cuda, fail docs/s, - MB | en | Apache-2.0 |
| piiranha | dialogpii, europii, meddeid, newmean, tonic (8) | 0.561 [0.538, 0.586] | 0.648 [0.620, 0.679] | 0.469 [0.444, 0.494] | 0.614 [0.585, 0.640] | 0.015 [0.014, 0.016] | 0.423 (6) | - | - | - | 1.80 | 2,418 MB | cuda, 41.8 docs/s, 1,884 MB | en it fr de nl es | CC-BY-NC-ND-4.0 model (benchmark only) |
| datafog | dialogpii, europii, greek_deid, meddeid, neonredact, newmean, pl_pii, tonic (8) | 0.522 [0.501, 0.539] | 0.766 [0.735, 0.803] | 0.459 [0.439, 0.480] | 0.692 [0.672, 0.711] | 0.011 [0.010, 0.013] | - | - | - | - | 1,555 | 34 MB | n/a | all (regex, no language claim) | MIT |
| llm-guard | dialogpii, europii, meddeid, newmean, tonic (8) | 0.478 [0.444, 0.515] | 0.637 [0.594, 0.676] | 0.294 [0.259, 0.344] | 0.613 [0.580, 0.646] | 0.005 [0.004, 0.007] | 0.402 (7) | - | - | - | 1.40 | 2,271 MB | cuda, 11.4 docs/s, 1,322 MB | en | MIT + CC-BY-NC-4.0 model |
| scrubadub | dialogpii, europii, meddeid, newmean, tonic (8) | 0.363 [0.329, 0.399] | 0.458 [0.416, 0.504] | 0.379 [0.348, 0.408] | 0.738 [0.712, 0.765] | 0.016 [0.015, 0.018] | 0.275 (7) | - | - | - | 1.20 | 5,180 MB | n/a | en | Apache-2.0 / MIT + MIT spaCy model |

### same-set (every entry)

Datasets: dialogpii, europii, meddeid, newmean, tonic (clean for all: newmean, dialogpii, tonic, meddeid, greek_deid, neonredact, pl_pii, europii); only docs in languages every entry claims; point values.

| entry | mR | mP | strict F1 | doc leak | over-mask |
|---|---|---|---|---|---|
| pii-tracer | 0.945 | 0.659 | 0.759 | 0.143 | 0.061 |
| openmed | 0.876 | 0.502 | 0.720 | 0.245 | 0.048 |
| opf | 0.852 | 0.630 | 0.754 | 0.245 | 0.045 |
| presidio-full | 0.843 | 0.330 | 0.473 | 0.347 | 0.081 |
| bardsai-v3 | 0.842 | 0.605 | 0.765 | 0.230 | 0.040 |
| presidio | 0.795 | 0.378 | 0.515 | 0.386 | 0.066 |
| sct-gliner | 0.785 | 0.327 | 0.437 | 0.339 | 0.057 |
| sct-gliner-nvidia | 0.776 | 0.407 | 0.471 | 0.326 | 0.047 |
| sct-default | 0.772 | 0.510 | 0.459 | 0.349 | 0.045 |
| ai4privacy | 0.763 | 0.599 | 0.191 | 0.684 | 0.032 |
| gliner2-fastino | 0.719 | 0.303 | 0.641 | 0.384 | 0.073 |
| gliner-nvidia | 0.687 | 0.428 | 0.616 | 0.375 | 0.062 |
| sct-training | 0.661 | 0.550 | 0.407 | 0.467 | 0.023 |
| gliner-knowledgator | 0.633 | 0.335 | 0.566 | 0.514 | 0.082 |
| piiranha | 0.591 | 0.654 | 0.523 | 0.553 | 0.013 |
| datafog | 0.489 | 0.749 | 0.424 | 0.668 | 0.012 |
| llm-guard | 0.478 | 0.637 | 0.294 | 0.613 | 0.005 |
| scrubadub | 0.363 | 0.458 | 0.379 | 0.738 | 0.016 |

### same-set (non-SCT entries with clean sets)

Datasets: dialogpii, europii, meddeid, newmean, tonic (clean for all: newmean, dialogpii, tonic, meddeid, greek_deid, neonredact, pl_pii, europii); only docs in languages every entry claims; point values.

| entry | mR | mP | strict F1 | doc leak | over-mask |
|---|---|---|---|---|---|
| pii-tracer | 0.945 | 0.659 | 0.759 | 0.143 | 0.061 |
| openmed | 0.876 | 0.502 | 0.720 | 0.245 | 0.048 |
| opf | 0.852 | 0.630 | 0.754 | 0.245 | 0.045 |
| presidio-full | 0.843 | 0.330 | 0.473 | 0.347 | 0.081 |
| bardsai-v3 | 0.842 | 0.605 | 0.765 | 0.230 | 0.040 |
| presidio | 0.795 | 0.378 | 0.515 | 0.386 | 0.066 |
| ai4privacy | 0.763 | 0.599 | 0.191 | 0.684 | 0.032 |
| gliner2-fastino | 0.719 | 0.303 | 0.641 | 0.384 | 0.073 |
| gliner-nvidia | 0.687 | 0.428 | 0.616 | 0.375 | 0.062 |
| gliner-knowledgator | 0.633 | 0.335 | 0.566 | 0.514 | 0.082 |
| piiranha | 0.591 | 0.654 | 0.523 | 0.553 | 0.013 |
| datafog | 0.489 | 0.749 | 0.424 | 0.668 | 0.012 |
| llm-guard | 0.478 | 0.637 | 0.294 | 0.613 | 0.005 |
| scrubadub | 0.363 | 0.458 | 0.379 | 0.738 | 0.016 |

### SCT vs each entry, datasets clean for both, languages both claim

| entry | sets | mR (entry - SCT, 95% CI) | mP (entry - SCT, 95% CI) | strict F1 (entry - SCT, 95% CI) | doc leak (entry - SCT, 95% CI) | over-mask (entry - SCT, 95% CI) | SCT mR, same docs | SCT mP, same docs |
|---|---|---|---|---|---|---|---|---|
| pii-tracer | newmean, dialogpii, tonic, meddeid, greek_deid, neonredact, pl_pii, europii | 0.940 (+0.107, +0.161)* | 0.622 (+0.093, +0.129)* | 0.717 (+0.188, +0.234)* | 0.110 (-0.230, -0.189)* | 0.055 (+0.000, +0.007)* | 0.805 | 0.509 |
| bardsai-v3 | newmean, dialogpii, tonic, meddeid, greek_deid, pl_pii, europii | 0.880 (+0.077, +0.105)* | 0.663 (+0.136, +0.168)* | 0.786 (+0.255, +0.295)* | 0.224 (-0.147, -0.111)* | 0.033 (-0.012, -0.006)* | 0.789 | 0.512 |
| opf | newmean, dialogpii, tonic, meddeid, europii | 0.852 (+0.052, +0.106)* | 0.630 (+0.088, +0.148)* | 0.754 (+0.251, +0.330)* | 0.245 (-0.143, -0.072)* | 0.045 (-0.002, +0.004) | 0.772 | 0.510 |
| openmed | newmean, dialogpii, tonic, meddeid, neonredact, europii | 0.839 (+0.037, +0.075)* | 0.489 (-0.020, +0.003) | 0.718 (+0.187, +0.229)* | 0.262 (-0.106, -0.068)* | 0.049 (-0.011, -0.006)* | 0.783 | 0.497 |
| presidio-full | newmean, dialogpii, tonic, meddeid, greek_deid, neonredact, pl_pii, europii | 0.812 (-0.004, +0.019) | 0.398 (-0.129, -0.095)* | 0.509 (-0.024, +0.032) | 0.439 (+0.102, +0.134)* | 0.088 (+0.033, +0.040)* | 0.805 | 0.509 |
| presidio | newmean, dialogpii, tonic, meddeid, europii | 0.795 (-0.002, +0.051) | 0.378 (-0.154, -0.112)* | 0.515 (+0.013, +0.095)* | 0.386 (+0.007, +0.065)* | 0.066 (+0.019, +0.025)* | 0.772 | 0.510 |
| sct-gliner | newmean, dialogpii, tonic, meddeid, europii | 0.785 (+0.003, +0.023)* | 0.327 (-0.207, -0.163)* | 0.437 (-0.031, -0.016)* | 0.339 (-0.024, +0.002) | 0.057 (+0.010, +0.014)* | 0.772 | 0.510 |
| sct-gliner-nvidia | newmean, dialogpii, tonic, meddeid, europii | 0.776 (-0.011, +0.018) | 0.407 (-0.124, -0.083)* | 0.471 (+0.001, +0.024)* | 0.326 (-0.038, -0.008)* | 0.047 (+0.001, +0.004)* | 0.772 | 0.510 |
| gliner2-fastino | newmean, dialogpii, tonic, meddeid, europii | 0.732 (-0.034, -0.009)* | 0.349 (-0.161, -0.132)* | 0.705 (+0.170, +0.220)* | 0.370 (-0.059, -0.022)* | 0.062 (+0.013, +0.017)* | 0.754 | 0.495 |
| ai4privacy | newmean, dialogpii, tonic, meddeid, europii | 0.697 (-0.076, -0.021)* | 0.583 (+0.070, +0.120)* | 0.172 (-0.378, -0.305)* | 0.800 (+0.369, +0.426)* | 0.032 (-0.017, -0.012)* | 0.745 | 0.489 |
| gliner-nvidia | newmean, dialogpii, tonic, meddeid, europii | 0.687 (-0.119, -0.042)* | 0.428 (-0.115, -0.049)* | 0.616 (+0.125, +0.187)* | 0.375 (-0.008, +0.060) | 0.062 (+0.014, +0.021)* | 0.772 | 0.510 |
| sct-training | newmean, dialogpii, tonic, meddeid, greek_deid, neonredact, pl_pii, europii | 0.661 (-0.158, -0.131)* | 0.555 (+0.030, +0.065)* | 0.462 (-0.053, -0.032)* | 0.567 (+0.234, +0.263)* | 0.030 (-0.024, -0.020)* | 0.805 | 0.509 |
| gliner-knowledgator | newmean, dialogpii, tonic, meddeid, europii | 0.633 (-0.169, -0.111)* | 0.335 (-0.205, -0.150)* | 0.566 (+0.062, +0.141)* | 0.514 (+0.127, +0.200)* | 0.082 (+0.034, +0.041)* | 0.772 | 0.510 |
| piiranha | newmean, dialogpii, tonic, meddeid, europii | 0.561 (-0.201, -0.172)* | 0.648 (+0.135, +0.178)* | 0.469 (-0.085, -0.008)* | 0.614 (+0.189, +0.230)* | 0.015 (-0.033, -0.029)* | 0.748 | 0.493 |
| datafog | newmean, dialogpii, tonic, meddeid, greek_deid, neonredact, pl_pii, europii | 0.522 (-0.305, -0.265)* | 0.766 (+0.231, +0.289)* | 0.459 (-0.068, -0.024)* | 0.692 (+0.355, +0.389)* | 0.011 (-0.043, -0.037)* | 0.805 | 0.509 |
| llm-guard | newmean, dialogpii, tonic, meddeid, europii | 0.478 (-0.324, -0.265)* | 0.637 (+0.085, +0.164)* | 0.294 (-0.192, -0.129)* | 0.613 (+0.236, +0.297)* | 0.005 (-0.043, -0.036)* | 0.772 | 0.510 |
| scrubadub | newmean, dialogpii, tonic, meddeid, europii | 0.363 (-0.439, -0.381)* | 0.458 (-0.083, -0.018)* | 0.379 (-0.119, -0.049)* | 0.738 (+0.359, +0.420)* | 0.016 (-0.031, -0.025)* | 0.772 | 0.510 |

### SCT vs each entry, datasets unknown for both, languages both claim

No dataset in this group is unknown for both sct-default and any other entry (languages both claim).

## default track, shared view, r2-privacy-real: each entry on its own clean sets, claimed languages

† = under 99% complete on a dataset in this row or cell (PROTOCOL section 15; see the completion table).

| entry | clean sets with claimed-language docs (all clean sets) | mR [95% CI] | mP [95% CI] | strict F1 [95% CI] | doc leak [95% CI] | over-mask [95% CI] | mR clean, outside claimed langs | mR unknown (U) sets | mR U sets, outside claimed langs | (mR excluded sets, info only) | docs/s CPU | peak RAM | GPU: device, docs/s, peak VRAM | claimed languages | licence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pii-tracer | kdpii, nus_lt, redmadrobot (3) | 0.783 [0.728, 0.859] | 0.718 [0.661, 0.779] | 0.554 [0.459, 0.662] | 0.296 [0.230, 0.366] | 0.014 [0.010, 0.019] | - | - | - | - | 1.33 | 3,098 MB | cuda, 19.8 docs/s, 1,826 MB | all (model card: en + multilingual) | MIT |
| sct-default | kdpii, nus_lt, redmadrobot (3) | 0.774 [0.718, 0.837] | 0.517 [0.436, 0.602] | 0.326 [0.245, 0.411] | 0.520 [0.424, 0.603] | 0.013 [0.010, 0.017] | - | - | - | - | 1.55 | 10,023 MB | n/a | all (multilingual claim) | MIT |
| presidio-full | kdpii, nus_lt, redmadrobot (3) | 0.683 [0.621, 0.752] | 0.610 [0.554, 0.677] | 0.489 [0.385, 0.598] | 0.628 [0.536, 0.722] | 0.016 [0.012, 0.019] | - | - | - | - | 14 | 19,753 MB | n/a | all (spaCy engine per gold language + recognizers) | MIT |
| bardsai-v3 | nus_lt (3) | 0.616 [0.430, 0.803] | 0.353 [0.135, 0.786] | 0.508 [0.317, 0.741] | 0.679 [0.500, 0.846] | 0.005 [0.002, 0.007] | 0.769 (2) | - | - | - | 2.42 | 6,975 MB | n/a | 24 EU | Apache-2.0 |
| sct-training | kdpii, nus_lt, redmadrobot (3) | 0.489 [0.417, 0.555] | 0.497 [0.423, 0.571] | 0.232 [0.157, 0.307] | 0.693 [0.617, 0.767] | 0.009 [0.007, 0.012] | - | - | - | - | 1.56 | 10,205 MB | n/a | all (multilingual claim) | MIT |
| datafog | kdpii, nus_lt, redmadrobot (3) | 0.289 [0.215, 0.367] | 0.931 [0.889, 0.970] | 0.239 [0.147, 0.342] | 0.868 [0.800, 0.925] | 0.001 [0.000, 0.001] | - | - | - | - | 1,555 | 34 MB | n/a | all (regex, no language claim) | MIT |
| sct-gliner-nvidia | none (3) | - | - | - | - | - | 0.794 (3) | - | - | - | n/a | n/a | cuda, fail docs/s, - MB | en | MIT + NVIDIA Open Model |
| sct-gliner | none (3) | - | - | - | - | - | 0.803 (3) | - | - | - | 0.61 | 6,715 MB | cuda, fail docs/s, - MB | en | MIT + Apache-2.0 model |
| scrubadub | none (3) | - | - | - | - | - | 0.206 (3) | - | - | - | 1.20 | 5,180 MB | n/a | en | Apache-2.0 / MIT + MIT spaCy model |
| presidio | none (3) | - | - | - | - | - | 0.673 (3) | - | - | - | 26 | 1,080 MB | n/a | en | MIT |
| piiranha | none (3) | - | - | - | - | - | 0.528 (3) | - | - | - | 1.80 | 2,418 MB | cuda, 41.8 docs/s, 1,884 MB | en it fr de nl es | CC-BY-NC-ND-4.0 model (benchmark only) |
| opf | none (2) | - | - | - | - | - | 0.539 (2) | - | 0.826 (1) | - | n/a | n/a | cuda, 17.0 docs/s, 4,626 MB | en | Apache-2.0 |
| openmed | none (2) | - | - | - | - | - | 0.673 (2) | 0.957 (1) | - | - | 0.42 | 25,637 MB | cuda, 9.9 docs/s, 5,100 MB | ar bn de en es fr hi it ja ko nl pt te tr vi zh | Apache-2.0 |
| llm-guard | none (3) | - | - | - | - | - | 0.406 (3) | - | - | - | 1.40 | 2,271 MB | cuda, 11.4 docs/s, 1,322 MB | en | MIT + CC-BY-NC-4.0 model |
| gliner2-fastino | none (3) | - | - | - | - | - | 0.667 (3) | - | - | - | 0.63 | 34,356 MB | cuda, fail docs/s, - MB | en fr es de it pt nl | Apache-2.0 |
| gliner-nvidia | none (3) | - | - | - | - | - | 0.600 (3) | - | - | - | n/a | n/a | cuda, fail docs/s, 5,354 MB | en | Apache-2.0 + NVIDIA Open Model |
| gliner-knowledgator† | none (3) | - | - | - | - | - | 0.731 (3) | - | - | - | n/a | n/a | cuda, fail docs/s, - MB | en | Apache-2.0 |
| ai4privacy | none (3) | - | - | - | - | - | 0.752 (3) | - | - | - | 1.37 | 1,534 MB | cuda, 20.8 docs/s, 1,226 MB | fr en de te hi it es nl | MIT |

### same-set (every entry)

No dataset in this group is clean for every one of: sct-default, sct-gliner, presidio, gliner-nvidia, gliner2-fastino, opf, bardsai-v3, datafog, gliner-knowledgator, piiranha, llm-guard, scrubadub, ai4privacy, openmed, pii-tracer, presidio-full, sct-gliner-nvidia, sct-training (in languages all of them claim).

### same-set (non-SCT entries with clean sets)

No dataset in this group is clean for every one of: presidio, gliner-nvidia, gliner2-fastino, opf, bardsai-v3, datafog, gliner-knowledgator, piiranha, llm-guard, scrubadub, ai4privacy, openmed, pii-tracer, presidio-full (in languages all of them claim).

### SCT vs each entry, datasets clean for both, languages both claim

| entry | sets | mR (entry - SCT, 95% CI) | mP (entry - SCT, 95% CI) | strict F1 (entry - SCT, 95% CI) | doc leak (entry - SCT, 95% CI) | over-mask (entry - SCT, 95% CI) | SCT mR, same docs | SCT mP, same docs |
|---|---|---|---|---|---|---|---|---|
| pii-tracer | kdpii, nus_lt, redmadrobot | 0.783 (-0.058, +0.087) | 0.718 (+0.122, +0.280)* | 0.554 (+0.107, +0.354)* | 0.296 (-0.308, -0.144)* | 0.014 (-0.003, +0.004) | 0.774 | 0.517 |
| presidio-full | kdpii, nus_lt, redmadrobot | 0.683 (-0.162, -0.023)* | 0.610 (+0.025, +0.161)* | 0.489 (+0.047, +0.274)* | 0.628 (+0.020, +0.195)* | 0.016 (-0.001, +0.004) | 0.774 | 0.517 |
| bardsai-v3 | nus_lt | 0.616 (-0.201, +0.118) | 0.353 (+0.033, +0.451)* | 0.508 (+0.039, +0.569)* | 0.679 (-0.200, +0.118) | 0.005 (-0.010, -0.004)* | 0.672 | 0.203 |
| sct-training | kdpii, nus_lt, redmadrobot | 0.489 (-0.365, -0.221)* | 0.497 (-0.068, +0.025) | 0.232 (-0.144, -0.046)* | 0.693 (+0.111, +0.242)* | 0.009 (-0.006, -0.002)* | 0.774 | 0.517 |
| datafog | kdpii, nus_lt, redmadrobot | 0.289 (-0.587, -0.392)* | 0.931 (+0.324, +0.499)* | 0.239 (-0.206, +0.029) | 0.868 (+0.254, +0.436)* | 0.001 (-0.016, -0.010)* | 0.774 | 0.517 |

No clean dataset in common with sct-default: ai4privacy, gliner-knowledgator, gliner-nvidia, gliner2-fastino, llm-guard, openmed, opf, piiranha, presidio, scrubadub, sct-gliner, sct-gliner-nvidia.

### SCT vs each entry, datasets unknown for both, languages both claim

| entry | sets | mR (entry - SCT, 95% CI) | mP (entry - SCT, 95% CI) | strict F1 (entry - SCT, 95% CI) | doc leak (entry - SCT, 95% CI) | over-mask (entry - SCT, 95% CI) | SCT mR, same docs | SCT mP, same docs |
|---|---|---|---|---|---|---|---|---|
| openmed | kdpii | 0.957 (-0.111, +0.200) | 0.244 (-0.664, -0.375)* | 0.727 (+0.059, +0.589)* | 0.217 (-0.304, -0.000) | 0.025 (+0.013, +0.026)* | 0.913 | 0.767 |

No unknown dataset in common with sct-default: ai4privacy, bardsai-v3, datafog, gliner-knowledgator, gliner-nvidia, gliner2-fastino, llm-guard, opf, pii-tracer, piiranha, presidio, presidio-full, scrubadub, sct-gliner, sct-gliner-nvidia, sct-training.

## default track, shared view, r2-untyped: each entry on its own clean sets, claimed languages

† = under 99% complete on a dataset in this row or cell (PROTOCOL section 15; see the completion table).

| entry | clean sets with claimed-language docs (all clean sets) | mR [95% CI] | mP [95% CI] | strict F1 [95% CI] | doc leak [95% CI] | over-mask [95% CI] | mR clean, outside claimed langs | mR unknown (U) sets | mR U sets, outside claimed langs | (mR excluded sets, info only) | docs/s CPU | peak RAM | GPU: device, docs/s, peak VRAM | claimed languages | licence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bardsai-v3 | redactionbench (1) | 0.745 [0.644, 0.815] | 0.420 [0.355, 0.516] | - | 0.744 [0.681, 0.806] | 0.051 [0.028, 0.079] | - | - | - | - | 2.42 | 6,975 MB | n/a | 24 EU | Apache-2.0 |
| presidio-full | redactionbench (1) | 0.720 [0.614, 0.818] | 0.235 [0.164, 0.302] | - | 0.884 [0.832, 0.931] | 0.123 [0.102, 0.146] | - | - | - | - | 14 | 19,753 MB | n/a | all (spaCy engine per gold language + recognizers) | MIT |
| openmed | redactionbench (1) | 0.692 [0.603, 0.768] | 0.451 [0.386, 0.533] | - | 0.793 [0.732, 0.849] | 0.049 [0.029, 0.076] | - | - | - | - | 0.42 | 25,637 MB | cuda, 9.9 docs/s, 5,100 MB | ar bn de en es fr hi it ja ko nl pt te tr vi zh | Apache-2.0 |
| presidio | redactionbench (1) | 0.692 [0.574, 0.799] | 0.262 [0.187, 0.338] | - | 0.902 [0.854, 0.946] | 0.090 [0.069, 0.113] | - | - | - | - | 26 | 1,080 MB | n/a | en | MIT |
| sct-default | redactionbench (1) | 0.687 [0.552, 0.809] | 0.297 [0.240, 0.359] | - | 0.848 [0.792, 0.903] | 0.063 [0.045, 0.084] | - | - | - | - | 1.55 | 10,023 MB | n/a | all (multilingual claim) | MIT |
| sct-gliner | redactionbench (1) | 0.681 [0.549, 0.797] | 0.213 [0.154, 0.271] | - | 0.896 [0.846, 0.939] | 0.089 [0.069, 0.112] | - | - | - | - | 0.61 | 6,715 MB | cuda, fail docs/s, - MB | en | MIT + Apache-2.0 model |
| sct-gliner-nvidia† | redactionbench (1) | 0.668 [0.535, 0.791] | 0.320 [0.263, 0.389] | - | 0.896 [0.850, 0.939] | 0.061 [0.043, 0.085] | - | - | - | - | n/a | n/a | cuda, fail docs/s, - MB | en | MIT + NVIDIA Open Model |
| sct-training | redactionbench (1) | 0.662 [0.537, 0.782] | 0.343 [0.228, 0.468] | - | 0.915 [0.873, 0.957] | 0.044 [0.033, 0.058] | - | - | - | - | 1.56 | 10,205 MB | n/a | all (multilingual claim) | MIT |
| gliner2-fastino† | redactionbench (1) | 0.662 [0.530, 0.781] | 0.243 [0.167, 0.324] | - | 0.902 [0.854, 0.941] | 0.063 [0.052, 0.074] | - | - | - | - | 0.63 | 34,356 MB | cuda, fail docs/s, - MB | en fr es de it pt nl | Apache-2.0 |
| llm-guard | redactionbench (1) | 0.657 [0.471, 0.778] | 0.408 [0.277, 0.566] | - | 0.963 [0.932, 0.988] | 0.062 [0.028, 0.100] | - | - | - | - | 1.40 | 2,271 MB | cuda, 11.4 docs/s, 1,322 MB | en | MIT + CC-BY-NC-4.0 model |
| ai4privacy | redactionbench (1) | 0.573 [0.450, 0.697] | 0.323 [0.248, 0.406] | - | 0.994 [0.981, 1.000] | 0.021 [0.013, 0.029] | - | - | - | - | 1.37 | 1,534 MB | cuda, 20.8 docs/s, 1,226 MB | fr en de te hi it es nl | MIT |
| datafog | redactionbench (1) | 0.531 [0.376, 0.671] | 0.498 [0.384, 0.636] | - | 0.976 [0.951, 0.994] | 0.011 [0.007, 0.017] | - | - | - | - | 1,555 | 34 MB | n/a | all (regex, no language claim) | MIT |
| piiranha | redactionbench (1) | 0.375 [0.254, 0.498] | 0.781 [0.675, 0.870] | - | 0.982 [0.958, 1.000] | 0.004 [0.002, 0.006] | - | - | - | - | 1.80 | 2,418 MB | cuda, 41.8 docs/s, 1,884 MB | en it fr de nl es | CC-BY-NC-ND-4.0 model (benchmark only) |
| opf | redactionbench (1) | 0.351 [0.270, 0.481] | 0.519 [0.361, 0.730] | - | 0.860 [0.807, 0.908] | 0.021 [0.010, 0.035] | - | - | - | - | n/a | n/a | cuda, 17.0 docs/s, 4,626 MB | en | Apache-2.0 |
| pii-tracer | redactionbench (1) | 0.332 [0.231, 0.483] | 0.720 [0.653, 0.783] | - | 0.665 [0.589, 0.737] | 0.008 [0.006, 0.012] | - | - | - | - | 1.33 | 3,098 MB | cuda, 19.8 docs/s, 1,826 MB | all (model card: en + multilingual) | MIT |
| scrubadub | redactionbench (1) | 0.211 [0.146, 0.319] | 0.208 [0.130, 0.330] | - | 0.939 [0.901, 0.975] | 0.041 [0.029, 0.054] | - | - | - | - | 1.20 | 5,180 MB | n/a | en | Apache-2.0 / MIT + MIT spaCy model |
| gliner-knowledgator† | redactionbench (1) | 0.197 [0.127, 0.327] | 0.550 [0.478, 0.622] | - | 0.884 [0.835, 0.932] | 0.007 [0.005, 0.009] | - | - | - | - | n/a | n/a | cuda, fail docs/s, - MB | en | Apache-2.0 |
| gliner-nvidia | redactionbench (1) | 0.086 [0.057, 0.138] | 0.649 [0.579, 0.718] | - | 0.988 [0.969, 1.000] | 0.003 [0.003, 0.004] | - | - | - | - | n/a | n/a | cuda, fail docs/s, 5,354 MB | en | Apache-2.0 + NVIDIA Open Model |

### same-set (every entry)

Datasets: redactionbench (clean for all: redactionbench); only docs in languages every entry claims; point values.

| entry | mR | mP | strict F1 | doc leak | over-mask |
|---|---|---|---|---|---|
| bardsai-v3 | 0.745 | 0.420 | n/a | 0.744 | 0.051 |
| presidio-full | 0.720 | 0.235 | n/a | 0.884 | 0.123 |
| openmed | 0.692 | 0.451 | n/a | 0.793 | 0.049 |
| presidio | 0.692 | 0.262 | n/a | 0.902 | 0.090 |
| sct-default | 0.687 | 0.297 | n/a | 0.848 | 0.063 |
| sct-gliner | 0.681 | 0.213 | n/a | 0.896 | 0.089 |
| sct-gliner-nvidia† | 0.668 | 0.320 | n/a | 0.896 | 0.061 |
| sct-training | 0.662 | 0.343 | n/a | 0.915 | 0.044 |
| gliner2-fastino† | 0.662 | 0.243 | n/a | 0.902 | 0.063 |
| llm-guard | 0.657 | 0.408 | n/a | 0.963 | 0.062 |
| ai4privacy | 0.573 | 0.323 | n/a | 0.994 | 0.021 |
| datafog | 0.531 | 0.498 | n/a | 0.976 | 0.011 |
| piiranha | 0.375 | 0.781 | n/a | 0.982 | 0.004 |
| opf | 0.351 | 0.519 | n/a | 0.860 | 0.021 |
| pii-tracer | 0.332 | 0.720 | n/a | 0.665 | 0.008 |
| scrubadub | 0.211 | 0.208 | n/a | 0.939 | 0.041 |
| gliner-knowledgator† | 0.197 | 0.550 | n/a | 0.884 | 0.007 |
| gliner-nvidia | 0.086 | 0.649 | n/a | 0.988 | 0.003 |

### same-set (non-SCT entries with clean sets)

Datasets: redactionbench (clean for all: redactionbench); only docs in languages every entry claims; point values.

| entry | mR | mP | strict F1 | doc leak | over-mask |
|---|---|---|---|---|---|
| bardsai-v3 | 0.745 | 0.420 | n/a | 0.744 | 0.051 |
| presidio-full | 0.720 | 0.235 | n/a | 0.884 | 0.123 |
| openmed | 0.692 | 0.451 | n/a | 0.793 | 0.049 |
| presidio | 0.692 | 0.262 | n/a | 0.902 | 0.090 |
| gliner2-fastino† | 0.662 | 0.243 | n/a | 0.902 | 0.063 |
| llm-guard | 0.657 | 0.408 | n/a | 0.963 | 0.062 |
| ai4privacy | 0.573 | 0.323 | n/a | 0.994 | 0.021 |
| datafog | 0.531 | 0.498 | n/a | 0.976 | 0.011 |
| piiranha | 0.375 | 0.781 | n/a | 0.982 | 0.004 |
| opf | 0.351 | 0.519 | n/a | 0.860 | 0.021 |
| pii-tracer | 0.332 | 0.720 | n/a | 0.665 | 0.008 |
| scrubadub | 0.211 | 0.208 | n/a | 0.939 | 0.041 |
| gliner-knowledgator† | 0.197 | 0.550 | n/a | 0.884 | 0.007 |
| gliner-nvidia | 0.086 | 0.649 | n/a | 0.988 | 0.003 |

### SCT vs each entry, datasets clean for both, languages both claim

| entry | sets | mR (entry - SCT, 95% CI) | mP (entry - SCT, 95% CI) | strict F1 (entry - SCT, 95% CI) | doc leak (entry - SCT, 95% CI) | over-mask (entry - SCT, 95% CI) | SCT mR, same docs | SCT mP, same docs |
|---|---|---|---|---|---|---|---|---|
| bardsai-v3 | redactionbench | 0.745 (-0.055, +0.175) | 0.420 (+0.058, +0.204)* | n/a | 0.744 (-0.159, -0.050)* | 0.051 (-0.024, +0.002) | 0.687 | 0.297 |
| presidio-full | redactionbench | 0.720 (-0.022, +0.096) | 0.235 (-0.119, -0.007)* | n/a | 0.884 (-0.012, +0.083) | 0.123 (+0.046, +0.077)* | 0.687 | 0.297 |
| openmed | redactionbench | 0.692 (-0.077, +0.098) | 0.451 (+0.111, +0.211)* | n/a | 0.793 (-0.110, -0.000) | 0.049 (-0.022, -0.004)* | 0.687 | 0.297 |
| presidio | redactionbench | 0.692 (-0.053, +0.068) | 0.262 (-0.098, +0.021) | n/a | 0.902 (+0.006, +0.104)* | 0.090 (+0.014, +0.044)* | 0.687 | 0.297 |
| sct-gliner | redactionbench | 0.681 (-0.033, +0.018) | 0.213 (-0.126, -0.044)* | n/a | 0.896 (+0.012, +0.086)* | 0.089 (+0.016, +0.038)* | 0.687 | 0.297 |
| sct-gliner-nvidia† | redactionbench | 0.668 (-0.048, +0.001) | 0.320 (-0.016, +0.064) | n/a | 0.896 (+0.018, +0.085)* | 0.061 (-0.011, +0.007) | 0.687 | 0.297 |
| sct-training | redactionbench | 0.662 (-0.076, +0.036) | 0.343 (-0.033, +0.134) | n/a | 0.915 (+0.031, +0.107)* | 0.044 (-0.032, -0.008)* | 0.687 | 0.297 |
| gliner2-fastino† | redactionbench | 0.662 (-0.078, +0.038) | 0.243 (-0.109, +0.012) | n/a | 0.902 (+0.012, +0.103)* | 0.063 (-0.021, +0.019) | 0.687 | 0.297 |
| llm-guard | redactionbench | 0.657 (-0.221, +0.155) | 0.408 (-0.008, +0.260) | n/a | 0.963 (+0.069, +0.169)* | 0.062 (-0.041, +0.043) | 0.687 | 0.297 |
| ai4privacy | redactionbench | 0.573 (-0.190, -0.065)* | 0.323 (-0.033, +0.088) | n/a | 0.994 (+0.092, +0.202)* | 0.021 (-0.056, -0.030)* | 0.687 | 0.297 |
| datafog | redactionbench | 0.531 (-0.283, -0.075)* | 0.498 (+0.078, +0.351)* | n/a | 0.976 (+0.073, +0.184)* | 0.011 (-0.072, -0.035)* | 0.687 | 0.297 |
| piiranha | redactionbench | 0.375 (-0.390, -0.227)* | 0.781 (+0.386, +0.573)* | n/a | 0.982 (+0.079, +0.190)* | 0.004 (-0.080, -0.043)* | 0.687 | 0.297 |
| opf | redactionbench | 0.351 (-0.461, -0.175)* | 0.519 (+0.067, +0.421)* | n/a | 0.860 (-0.053, +0.072) | 0.021 (-0.059, -0.026)* | 0.687 | 0.297 |
| pii-tracer | redactionbench | 0.332 (-0.557, -0.152)* | 0.720 (+0.352, +0.497)* | n/a | 0.665 (-0.247, -0.117)* | 0.008 (-0.075, -0.038)* | 0.687 | 0.297 |
| scrubadub | redactionbench | 0.211 (-0.630, -0.317)* | 0.208 (-0.151, +0.002) | n/a | 0.939 (+0.051, +0.139)* | 0.041 (-0.038, -0.007)* | 0.687 | 0.297 |
| gliner-knowledgator† | redactionbench | 0.197 (-0.641, -0.330)* | 0.550 (+0.190, +0.312)* | n/a | 0.884 (-0.024, +0.094) | 0.007 (-0.077, -0.039)* | 0.687 | 0.297 |
| gliner-nvidia | redactionbench | 0.086 (-0.729, -0.460)* | 0.649 (+0.277, +0.425)* | n/a | 0.988 (+0.085, +0.196)* | 0.003 (-0.081, -0.042)* | 0.687 | 0.297 |

### SCT vs each entry, datasets unknown for both, languages both claim

No dataset in this group is unknown for both sct-default and any other entry (languages both claim).

## default track, per language, full view, masked recall, r2-privacy-synthetic, clean sets (° = outside the entry's claimed languages)

† = under 99% complete on a dataset in this row or cell (PROTOCOL section 15; see the completion table).

| lang | datasets | ai4privacy | bardsai-v3 | datafog | gliner-knowledgator | gliner-nvidia | gliner2-fastino | llm-guard | openmed | opf | pii-tracer | piiranha | presidio | presidio-full | scrubadub | sct-default | sct-gliner | sct-gliner-nvidia | sct-training |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ar | dialogpii, newmean | 0.178° | 0.539° | 0.147 | 0.477° | 0.551° | 0.761° | 0.291° | 0.499 | 0.439° | 0.742 | 0.381° | 0.431° | 0.304 | 0.081° | 0.772 | 0.419° | 0.674° | 0.762 |
| ar-asr | dialogpii | 0.042° | 0.254° | 0.013 | 0.141° | 0.198° | 0.575° | 0.120° | 0.202 | 0.200° | 0.612 | 0.274° | 0.217° | 0.088 | 0.005° | 0.691 | 0.207° | 0.560° | 0.667 |
| az | newmean | 0.311° | 0.784° | 0.129 | 0.855° | 0.743° | 0.735° | 0.494° | 0.772° | 0.685° | 0.867 | 0.573° | 0.668° | 0.772 | 0.357° | 0.734 | 0.651° | 0.701° | 0.718 |
| bg | newmean | 0.322° | 0.820 | 0.210 | 0.880° | 0.811° | 0.807° | 0.524° | 0.730° | 0.708° | 0.923 | 0.481° | 0.669° | 0.760 | 0.189° | 0.854 | 0.828° | 0.841° | 0.824 |
| bn | newmean | 0.282° | 0.617° | 0.247 | 0.374° | 0.414° | 0.757° | 0.269° | 0.626 | 0.656° | 0.841 | 0.498° | 0.498° | 0.555 | 0.167° | 0.745 | 0.498° | 0.507° | 0.665 |
| ca | newmean | 0.400° | 0.844° | 0.141 | 0.844° | 0.844° | 0.810° | 0.419° | 0.756° | 0.668° | 0.888 | 0.610° | 0.663° | 0.834 | 0.322° | 0.707 | 0.702° | 0.683° | 0.600 |
| cs | europii, newmean | 0.389° | 0.915 | 0.270 | 0.866° | 0.907° | 0.921° | 0.670° | 0.860° | 0.736° | 0.934 | 0.461° | 0.757° | 0.847 | 0.484° | 0.913 | 0.899° | 0.901° | 0.834 |
| da | europii, newmean | 0.423° | 0.853 | 0.306 | 0.843° | 0.871° | 0.872° | 0.607° | 0.872° | 0.793° | 0.935 | 0.481° | 0.761° | 0.877 | 0.485° | 0.902 | 0.896° | 0.875° | 0.773 |
| de | dialogpii, europii, newmean | 0.349 | 0.841 | 0.289 | 0.869° | 0.730° | 0.839 | 0.606° | 0.828 | 0.707° | 0.862 | 0.476 | 0.761° | 0.766 | 0.517° | 0.842 | 0.833° | 0.821° | 0.804 |
| de-asr | dialogpii | 0.092 | 0.733 | 0.011 | 0.792° | 0.360° | 0.777 | 0.458° | 0.633 | 0.427° | 0.699 | 0.429 | 0.750° | 0.671 | 0.639° | 0.841 | 0.791° | 0.741° | 0.816 |
| el | greek_deid, newmean | 0.430° | 0.804 | 0.299 | 0.750° | 0.620° | 0.841° | 0.353° | 0.687° | 0.254° | 0.862 | 0.470° | 0.624° | 0.759 | 0.137° | 0.890 | 0.785° | 0.825° | 0.805 |
| en | dialogpii, europii, newmean, tonic | 0.492 | 0.827 | 0.168 | 0.806 | 0.730 | 0.880 | 0.555 | 0.846 | 0.726 | 0.827 | 0.545 | 0.761 | 0.887 | 0.662 | 0.868 | 0.852 | 0.825 | 0.834 |
| en-asr | dialogpii | 0.441 | 0.774 | 0.015 | 0.624 | 0.393 | 0.791 | 0.426 | 0.772 | 0.584 | 0.738 | 0.485 | 0.692 | 0.844 | 0.698 | 0.832 | 0.775 | 0.749 | 0.809 |
| en-gb | meddeid | 0.439 | 0.976 | 0.175 | 0.793 | 0.859 | 0.856 | 0.527 | 0.853 | 0.701 | 0.898 | 0.570 | 0.698 | 0.855 | 0.632 | 0.935 | 0.869 | 0.892 | 0.925 |
| en-us | meddeid | 0.446 | 0.965 | 0.247 | 0.837 | 0.909 | 0.846 | 0.567 | 0.845 | 0.711 | 0.920 | 0.691 | 0.679 | 0.841 | 0.661 | 0.934 | 0.872 | 0.917 | 0.869 |
| es | dialogpii, europii, newmean | 0.383 | 0.855 | 0.194 | 0.884° | 0.759° | 0.862 | 0.560° | 0.738 | 0.726° | 0.865 | 0.435 | 0.689° | 0.816 | 0.548° | 0.843 | 0.820° | 0.801° | 0.745 |
| es-asr | dialogpii | 0.182 | 0.782 | 0.018 | 0.853° | 0.423° | 0.787 | 0.442° | 0.642 | 0.491° | 0.698 | 0.347 | 0.623° | 0.733 | 0.638° | 0.834 | 0.766° | 0.755° | 0.806 |
| et | newmean | 0.306° | 0.808 | 0.147 | 0.816° | 0.784° | 0.796° | 0.498° | 0.727° | 0.559° | 0.865 | 0.441° | 0.620° | 0.702 | 0.437° | 0.796 | 0.784° | 0.771° | 0.763 |
| fa | newmean | 0.237° | 0.691° | 0.212 | 0.682° | 0.699° | 0.821° | 0.347° | 0.657° | 0.763° | 0.886 | 0.572° | 0.534° | 0.631 | 0.157° | 0.822 | 0.610° | 0.814° | 0.797 |
| fi | dialogpii, europii, newmean | 0.319° | 0.846 | 0.233 | 0.853° | 0.721° | 0.834° | 0.601° | 0.803° | 0.527° | 0.869 | 0.510° | 0.728° | 0.839 | 0.447° | 0.865 | 0.800° | 0.820° | 0.815 |
| fi-asr | dialogpii | 0.100° | 0.684 | 0.025 | 0.721° | 0.429° | 0.692° | 0.379° | 0.535° | 0.161° | 0.671 | 0.441° | 0.720° | 0.737 | 0.530° | 0.736 | 0.629° | 0.709° | 0.713 |
| fr | dialogpii, europii, newmean | 0.368 | 0.849 | 0.191 | 0.890° | 0.716° | 0.867 | 0.640° | 0.841 | 0.678° | 0.865 | 0.466 | 0.662° | 0.804 | 0.479° | 0.880 | 0.850° | 0.826° | 0.838 |
| fr-asr | dialogpii | 0.176 | 0.734 | 0.008 | 0.816° | 0.368° | 0.807 | 0.463° | 0.699 | 0.414° | 0.694 | 0.332 | 0.535° | 0.724 | 0.591° | 0.796 | 0.782° | 0.732° | 0.782 |
| ga | newmean | 0.341° | 0.780 | 0.168 | 0.832° | 0.836° | 0.776° | 0.467° | 0.804° | 0.406° | 0.841 | 0.556° | 0.622° | 0.776 | 0.276° | 0.785 | 0.724° | 0.701° | 0.771 |
| he | newmean | 0.392° | 0.650° | 0.286 | 0.733° | 0.701° | 0.793° | 0.406° | 0.544° | 0.511° | 0.871 | 0.465° | 0.548° | 0.521 | 0.101° | 0.843 | 0.664° | 0.811° | 0.802 |
| hi | dialogpii, newmean | 0.210 | 0.604° | 0.110 | 0.231° | 0.228° | 0.771° | 0.318° | 0.344 | 0.338° | 0.725 | 0.374° | 0.299° | 0.285 | 0.092° | 0.764 | 0.281° | 0.307° | 0.735 |
| hi-asr | dialogpii | 0.003 | 0.480° | 0.001 | 0.000° | 0.008° | 0.736° | 0.322° | 0.021 | 0.064° | 0.524 | 0.260° | 0.108° | 0.121 | 0.000° | 0.721 | 0.032° | 0.076° | 0.719 |
| hr | newmean | 0.315° | 0.831 | 0.249 | 0.901° | 0.911° | 0.784° | 0.554° | 0.737° | 0.671° | 0.920 | 0.535° | 0.742° | 0.822 | 0.493° | 0.803 | 0.826° | 0.798° | 0.812 |
| hu | newmean | 0.297° | 0.817 | 0.154 | 0.874° | 0.858° | 0.820° | 0.532° | 0.699° | 0.581° | 0.906 | 0.528° | 0.667° | 0.728 | 0.463° | 0.833 | 0.789° | 0.829° | 0.805 |
| id | newmean | 0.376° | 0.770° | 0.150 | 0.836° | 0.770° | 0.787° | 0.469° | 0.819° | 0.642° | 0.814 | 0.606° | 0.765° | 0.761 | 0.411° | 0.810 | 0.770° | 0.774° | 0.779 |
| is | newmean | 0.328° | 0.742° | 0.151 | 0.849° | 0.773° | 0.811° | 0.429° | 0.707° | 0.384° | 0.843 | 0.455° | 0.550° | 0.727 | 0.429° | 0.788 | 0.763° | 0.722° | 0.778 |
| it | dialogpii, europii, newmean | 0.325 | 0.836 | 0.223 | 0.870° | 0.734° | 0.862 | 0.590° | 0.819 | 0.747° | 0.859 | 0.430 | 0.693° | 0.817 | 0.528° | 0.829 | 0.796° | 0.780° | 0.791 |
| it-asr | dialogpii | 0.098 | 0.741 | 0.021 | 0.832° | 0.408° | 0.788 | 0.452° | 0.687 | 0.420° | 0.688 | 0.424 | 0.661° | 0.754 | 0.632° | 0.811 | 0.749° | 0.732° | 0.782 |
| ja | newmean | 0.338° | 0.603° | 0.114 | 0.639° | 0.598° | 0.594° | 0.484° | 0.772 | 0.690° | 0.840 | 0.525° | 0.461° | 0.767 | 0.164° | 0.794 | 0.744° | 0.744° | 0.758 |
| ka | newmean | 0.382° | 0.631° | 0.124 | 0.636° | 0.627° | 0.817° | 0.196° | 0.502° | 0.631° | 0.773 | 0.462° | 0.422° | 0.573 | 0.093° | 0.702 | 0.520° | 0.662° | 0.622 |
| kk | newmean | 0.368° | 0.752° | 0.107 | 0.806° | 0.711° | 0.740° | 0.347° | 0.665° | 0.628° | 0.893 | 0.636° | 0.595° | 0.719 | 0.165° | 0.785 | 0.727° | 0.748° | 0.694 |
| km | newmean | 0.378° | 0.575° | 0.197 | 0.468° | 0.545° | 0.793° | 0.348° | 0.631° | 0.841° | 0.760 | 0.455° | 0.416° | 0.541 | 0.120° | 0.798 | 0.511° | 0.571° | 0.772 |
| ko | newmean | 0.342° | 0.694° | 0.149 | 0.806° | 0.752° | 0.734° | 0.356° | 0.640 | 0.595° | 0.833 | 0.482° | 0.347° | 0.329 | 0.158° | 0.743 | 0.698° | 0.685° | 0.703 |
| lt | newmean | 0.335° | 0.814 | 0.186 | 0.847° | 0.752° | 0.810° | 0.516° | 0.740° | 0.583° | 0.884 | 0.545° | 0.645° | 0.715 | 0.475° | 0.756 | 0.773° | 0.756° | 0.711 |
| lv | newmean | 0.355° | 0.763 | 0.197 | 0.842° | 0.776° | 0.798° | 0.465° | 0.711° | 0.504° | 0.842 | 0.478° | 0.610° | 0.662 | 0.377° | 0.715 | 0.715° | 0.689° | 0.680 |
| mr | newmean | 0.339° | 0.665° | 0.234 | 0.454° | 0.458° | 0.736° | 0.335° | 0.573° | 0.612° | 0.837 | 0.423° | 0.454° | 0.529 | 0.216° | 0.736 | 0.471° | 0.485° | 0.718 |
| ms | newmean | 0.367° | 0.808° | 0.214 | 0.769° | 0.799° | 0.778° | 0.449° | 0.756° | 0.632° | 0.885 | 0.598° | 0.752° | 0.709 | 0.423° | 0.765 | 0.765° | 0.761° | 0.714 |
| mt | newmean | 0.306° | 0.780 | 0.099 | 0.828° | 0.681° | 0.736° | 0.388° | 0.754° | 0.569° | 0.828 | 0.522° | 0.651° | 0.806 | 0.422° | 0.698 | 0.776° | 0.724° | 0.685 |
| nl | europii, meddeid, newmean | 0.412 | 0.885 | 0.245 | 0.882° | 0.853° | 0.852 | 0.610° | 0.698 | 0.731° | 0.918 | 0.514 | 0.731° | 0.880 | 0.460° | 0.840 | 0.793° | 0.800° | 0.825 |
| no | newmean | 0.311° | 0.880° | 0.149 | 0.913° | 0.871° | 0.847° | 0.539° | 0.822° | 0.635° | 0.946 | 0.556° | 0.627° | 0.763 | 0.477° | 0.801 | 0.772° | 0.759° | 0.718 |
| pl | dialogpii, europii, newmean, pl_pii | 0.303° | 0.862 | 0.171 | 0.857° | 0.724° | 0.798° | 0.583° | 0.751° | 0.576° | 0.837 | 0.380° | 0.713° | 0.321 | 0.521° | 0.876 | 0.840° | 0.825° | 0.831 |
| pl-asr | dialogpii | 0.069° | 0.658 | 0.009 | 0.828° | 0.377° | 0.731° | 0.454° | 0.625° | 0.238° | 0.678 | 0.436° | 0.621° | 0.068 | 0.598° | 0.771 | 0.697° | 0.701° | 0.762 |
| pt | dialogpii, europii, newmean | 0.325° | 0.851 | 0.236 | 0.890° | 0.739° | 0.866 | 0.587° | 0.819 | 0.726° | 0.862 | 0.431° | 0.713° | 0.892 | 0.531° | 0.895 | 0.874° | 0.860° | 0.865 |
| pt-asr | dialogpii | 0.131° | 0.773 | 0.018 | 0.838° | 0.394° | 0.794 | 0.425° | 0.667 | 0.429° | 0.683 | 0.329° | 0.624° | 0.806 | 0.612° | 0.817 | 0.781° | 0.724° | 0.799 |
| ro | newmean | 0.367° | 0.808 | 0.138 | 0.933° | 0.871° | 0.777° | 0.533° | 0.754° | 0.608° | 0.892 | 0.504° | 0.658° | 0.817 | 0.350° | 0.800 | 0.804° | 0.779° | 0.804 |
| ru | newmean | 0.418° | 0.837° | 0.153 | 0.934° | 0.816° | 0.827° | 0.551° | 0.801° | 0.765° | 0.929 | 0.571° | 0.587° | 0.765 | 0.214° | 0.816 | 0.837° | 0.781° | 0.750 |
| sk | newmean | 0.257° | 0.705 | 0.143 | 0.844° | 0.848° | 0.785° | 0.544° | 0.743° | 0.595° | 0.873 | 0.485° | 0.658° | 0.722 | 0.481° | 0.717 | 0.751° | 0.764° | 0.696 |
| sl | newmean | 0.291° | 0.844 | 0.135 | 0.890° | 0.865° | 0.797° | 0.565° | 0.764° | 0.646° | 0.873 | 0.481° | 0.764° | 0.806 | 0.413° | 0.852 | 0.806° | 0.819° | 0.831 |
| sq | newmean | 0.344° | 0.736° | 0.172 | 0.864° | 0.744° | 0.724° | 0.464° | 0.748° | 0.648° | 0.916 | 0.492° | 0.620° | 0.720 | 0.372° | 0.800 | 0.772° | 0.792° | 0.700 |
| sr | newmean | 0.305° | 0.779° | 0.150 | 0.854° | 0.810° | 0.783° | 0.553° | 0.730° | 0.655° | 0.911 | 0.496° | 0.743° | 0.646 | 0.420° | 0.752 | 0.792° | 0.770° | 0.739 |
| sv | europii, newmean | 0.451° | 0.906 | 0.247 | 0.888° | 0.889° | 0.910° | 0.659° | 0.864° | 0.793° | 0.957 | 0.507° | 0.784° | 0.536 | 0.497° | 0.869 | 0.902° | 0.860° | 0.816 |
| sw | newmean | 0.353° | 0.767° | 0.195 | 0.837° | 0.753° | 0.795° | 0.372° | 0.786° | 0.726° | 0.888 | 0.516° | 0.740° | 0.791 | 0.400° | 0.777 | 0.744° | 0.763° | 0.758 |
| te | newmean | 0.290 | 0.690° | 0.250 | 0.490° | 0.505° | 0.745° | 0.300° | 0.590 | 0.740° | 0.890 | 0.510° | 0.480° | 0.560 | 0.290° | 0.780 | 0.495° | 0.540° | 0.775 |
| th | newmean | 0.357° | 0.705° | 0.281 | 0.357° | 0.438° | 0.695° | 0.290° | 0.624° | 0.676° | 0.886 | 0.552° | 0.424° | 0.371 | 0.114° | 0.767 | 0.514° | 0.548° | 0.686 |
| tl | newmean | 0.396° | 0.846° | 0.108 | 0.858° | 0.887° | 0.829° | 0.446° | 0.758° | 0.713° | 0.867 | 0.629° | 0.671° | 0.721 | 0.438° | 0.804 | 0.787° | 0.771° | 0.775 |
| tr | dialogpii, neonredact, newmean | 0.277° | 0.809° | 0.117 | 0.826° | 0.700° | 0.830° | 0.507° | 0.741 | 0.577° | 0.847 | 0.508° | 0.667° | 0.746 | 0.543° | 0.853 | 0.741° | 0.806° | 0.810 |
| tr-asr | dialogpii | 0.072° | 0.667° | 0.020 | 0.714° | 0.421° | 0.750° | 0.402° | 0.589 | 0.361° | 0.669 | 0.327° | 0.616° | 0.682 | 0.657° | 0.774 | 0.534° | 0.729° | 0.745 |
| uk | newmean | 0.332° | 0.753° | 0.206 | 0.879° | 0.789° | 0.749° | 0.489° | 0.744° | 0.700° | 0.897 | 0.457° | 0.641° | 0.758 | 0.166° | 0.798 | 0.821° | 0.812° | 0.749 |
| ur | newmean | 0.390° | 0.717° | 0.195 | 0.673° | 0.751° | 0.790° | 0.390° | 0.756° | 0.678° | 0.859 | 0.424° | 0.624° | 0.688 | 0.171° | 0.732 | 0.629° | 0.707° | 0.702 |
| uz | newmean | 0.344° | 0.802° | 0.165 | 0.811° | 0.703° | 0.741° | 0.509° | 0.764° | 0.665° | 0.868 | 0.684° | 0.712° | 0.835 | 0.467° | 0.807 | 0.769° | 0.774° | 0.759 |
| vi | newmean | 0.339° | 0.778° | 0.138 | 0.849° | 0.803° | 0.807° | 0.469° | 0.837 | 0.674° | 0.891 | 0.506° | 0.766° | 0.695 | 0.389° | 0.833 | 0.782° | 0.812° | 0.837 |
| zh | newmean | 0.439° | 0.577° | 0.217 | 0.518° | 0.565° | 0.557° | 0.360° | 0.708 | 0.644° | 0.933 | 0.478° | 0.463° | 0.617 | 0.202° | 0.802 | 0.640° | 0.640° | 0.783 |

## default track, per language, full view, masked recall, r2-privacy-synthetic, unknown sets (° = outside the entry's claimed languages)

† = under 99% complete on a dataset in this row or cell (PROTOCOL section 15; see the completion table).

No unknown dataset in this group.

## default track, per language, full view, masked recall, r2-privacy-real, clean sets (° = outside the entry's claimed languages)

† = under 99% complete on a dataset in this row or cell (PROTOCOL section 15; see the completion table).

| lang | datasets | ai4privacy | bardsai-v3 | datafog | gliner-knowledgator | gliner-nvidia | gliner2-fastino | llm-guard | openmed | opf | pii-tracer | piiranha | presidio | presidio-full | scrubadub | sct-default | sct-gliner | sct-gliner-nvidia | sct-training |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ko | kdpii | 0.300° | 0.487° | 0.100 | 0.787° | 0.575° | 0.750° | 0.237° | - | - | 0.487 | 0.450° | 0.237° | 0.263 | 0.075° | 0.675 | 0.688° | 0.575° | 0.637 |
| lt | nus_lt | 0.029° | 0.505 | 0.021 | 0.505°† | 0.222° | 0.871° | 0.364° | 0.385° | 0.076° | 0.059 | 0.375° | 0.661° | 0.882 | 0.659° | 0.934 | 0.443° | 0.776° | 0.918 |
| ru | redmadrobot | 0.408° | 0.692° | 0.100 | 0.895° | 0.728° | 0.708° | 0.316° | 0.621° | 0.692° | 0.837 | 0.468° | 0.601° | 0.731 | 0.143° | 0.811 | 0.808° | 0.780° | 0.617 |

## default track, per language, full view, masked recall, r2-privacy-real, unknown sets (° = outside the entry's claimed languages)

† = under 99% complete on a dataset in this row or cell (PROTOCOL section 15; see the completion table).

| lang | datasets | openmed | opf |
|---|---|---|---|
| ko | kdpii | 0.613 | 0.338° |

## default track, per language, full view, masked recall, r2-untyped, clean sets (° = outside the entry's claimed languages)

† = under 99% complete on a dataset in this row or cell (PROTOCOL section 15; see the completion table).

| lang | datasets | ai4privacy | bardsai-v3 | datafog | gliner-knowledgator | gliner-nvidia | gliner2-fastino | llm-guard | openmed | opf | pii-tracer | piiranha | presidio | presidio-full | scrubadub | sct-default | sct-gliner | sct-gliner-nvidia | sct-training |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| en | redactionbench | 0.573 | 0.745 | 0.531 | 0.199† | 0.086 | 0.663† | 0.657 | 0.692 | 0.351 | 0.332 | 0.375 | 0.692 | 0.720 | 0.211 | 0.687 | 0.681 | 0.682† | 0.662 |

## default track, per language, full view, masked recall, r2-untyped, unknown sets (° = outside the entry's claimed languages)

† = under 99% complete on a dataset in this row or cell (PROTOCOL section 15; see the completion table).

No unknown dataset in this group.

## default track, per dataset, full view, masked F2, claimed languages only (completed docs; (x) = excluded, information only; x? = unknown training data; ° = no doc in a claimed language)

† = under 99% complete on a dataset in this row or cell (PROTOCOL section 15; see the completion table).

| dataset | ai4privacy | bardsai-v3 | datafog | gliner-knowledgator | gliner-nvidia | gliner2-fastino | llm-guard | openmed | opf | pii-tracer | piiranha | presidio | presidio-full | scrubadub | sct-default | sct-gliner | sct-gliner-nvidia | sct-training |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| newmean | 0.389 | 0.752 | 0.199 | 0.497 | 0.627 | 0.596 | 0.543 | 0.668 | 0.748 | 0.750 | 0.524 | 0.587 | 0.516 | 0.509 | 0.695 | 0.630 | 0.686 | 0.704 |
| dialogpii | 0.209 | 0.783 | 0.043 | 0.691 | 0.441 | 0.795 | 0.480 | 0.591 | 0.558 | 0.714 | 0.465 | 0.690 | 0.566 | 0.673 | 0.839 | 0.726 | 0.743 | 0.820 |
| tonic | 0.294 | 0.711 | 0.214 | 0.794 | 0.670 | 0.815 | 0.598 | 0.728 | 0.646 | 0.725 | 0.435 | 0.655 | 0.778 | 0.896 | 0.820 | 0.809 | 0.794 | 0.843 |
| meddeid | 0.432 | 0.925 | 0.237 | 0.807 | 0.899 | 0.821 | 0.592 | 0.812 | 0.748 | 0.897 | 0.681 | 0.668 | 0.727 | 0.671 | 0.898 | 0.867 | 0.913 | 0.862 |
| greek_deid | ° | 0.888 | 0.475 | ° | ° | ° | ° | ° | ° | 0.872 | ° | ° | 0.753 | ° | 0.939 | ° | ° | 0.866 |
| neonredact | ° | ° | 0.185 | ° | ° | ° | ° | 0.786 | ° | 0.880 | ° | ° | 0.726 | ° | 0.886 | ° | ° | 0.872 |
| pl_pii | ° | 0.905 | 0.144 | ° | ° | ° | ° | ° | ° | 0.771 | ° | ° | 0.274 | ° | 0.854 | ° | ° | 0.834 |
| europii | 0.570 | 0.921 | 0.448 | 0.722 | 0.906 | 0.917 | 0.773 | 0.898 | 0.906 | 0.901 | 0.541 | 0.876 | 0.829 | 0.602 | 0.878 | 0.901 | 0.902 | 0.848 |
| kdpii | ° | ° | 0.122 | ° | ° | ° | ° | 0.570? | ° | 0.501 | ° | ° | 0.308 | ° | 0.709 | ° | ° | 0.675 |
| nus_lt | ° | 0.555 | 0.026 | ° | ° | ° | ° | ° | ° | 0.073 | ° | ° | 0.884 | ° | 0.927 | ° | ° | 0.920 |
| redmadrobot | ° | ° | 0.121 | ° | ° | ° | ° | ° | ° | 0.829 | ° | ° | 0.733 | ° | 0.797 | ° | ° | 0.651 |
| redactionbench | 0.496 | 0.645 | 0.524 | 0.228† | 0.104 | 0.493† | 0.585 | 0.626 | 0.375 | 0.372 | 0.418 | 0.521 | 0.510 | 0.210 | 0.544 | 0.473 | 0.556† | 0.558 |

### default track, per dataset, outside claimed languages (information only, not a loss; - = every doc is in a claimed language)

| dataset | ai4privacy | bardsai-v3 | datafog | gliner-knowledgator | gliner-nvidia | gliner2-fastino | llm-guard | openmed | opf | pii-tracer | piiranha | presidio | presidio-full | scrubadub | sct-default | sct-gliner | sct-gliner-nvidia | sct-training |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| newmean | 0.352 | 0.701 | - | 0.472 | 0.577 | 0.602 | 0.405 | 0.654 | 0.618 | - | 0.542 | 0.432 | - | 0.297 | - | 0.555 | 0.595 | - |
| dialogpii | 0.103 | 0.543 | - | 0.670 | 0.390 | 0.741 | 0.424 | 0.637 | 0.355 | - | 0.408 | 0.367 | - | 0.459 | - | 0.584 | 0.638 | - |
| tonic | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| meddeid | - | - | - | 0.851 | 0.758 | - | 0.502 | - | 0.612 | - | - | 0.512 | - | 0.511 | - | 0.756 | 0.803 | - |
| greek_deid | 0.547 | - | - | 0.594 | 0.490 | 0.862 | 0.345 | 0.722 | 0.061 | - | 0.566 | 0.508 | - | 0.159 | - | 0.753 | 0.833 | - |
| neonredact | 0.384 | 0.854 | - | 0.802 | 0.790 | 0.862 | 0.609 | - | 0.675 | - | 0.640 | 0.695 | - | 0.513 | - | 0.783 | 0.845 | - |
| pl_pii | 0.325 | - | - | 0.750 | 0.701 | 0.746 | 0.600 | 0.618 | 0.453 | - | 0.247 | 0.639 | - | 0.617 | - | 0.731 | 0.779 | - |
| europii | 0.529 | - | - | 0.837 | 0.891 | 0.905 | 0.821 | 0.910 | 0.878 | - | 0.483 | 0.773 | - | 0.515 | - | 0.872 | 0.897 | - |
| kdpii | 0.348 | 0.543 | - | 0.455 | 0.429 | 0.611 | 0.256 | - | 0.386? | - | 0.501 | 0.268 | - | 0.092 | - | 0.508 | 0.446 | - |
| nus_lt | 0.035 | - | - | 0.554† | 0.261 | 0.859 | 0.362 | 0.432 | 0.093 | - | 0.427 | 0.542 | - | 0.564 | - | 0.487 | 0.769 | - |
| redmadrobot | 0.450 | 0.723 | - | 0.775 | 0.713 | 0.695 | 0.349 | 0.647 | 0.714 | - | 0.516 | 0.566 | - | 0.171 | - | 0.744 | 0.771 | - |
| redactionbench | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |

## default track, completion

| entry | docs | failed | flagged datasets (< 99% complete) |
|---|---|---|---|
| ai4privacy | 20817 | 0 | - |
| bardsai-v3 | 20817 | 0 | - |
| datafog | 20817 | 0 | - |
| gliner-knowledgator | 20817 | 5 | nus_lt 0.980, redactionbench 0.983 |
| gliner-nvidia | 20817 | 0 | - |
| gliner2-fastino | 20817 | 22 | redactionbench 0.989 |
| llm-guard | 20817 | 0 | - |
| openmed | 20817 | 0 | - |
| opf | 20817 | 0 | - |
| pii-tracer | 20817 | 0 | - |
| piiranha | 20817 | 0 | - |
| presidio | 20817 | 0 | - |
| presidio-full | 20817 | 0 | - |
| scrubadub | 20817 | 2 | - |
| sct-default | 20817 | 0 | - |
| sct-gliner | 20817 | 0 | - |
| sct-gliner-nvidia | 20817 | 7 | redactionbench 0.960 |
| sct-training | 20817 | 0 | - |

## contamination (bold = excluded, italic = unknown, plain = clean)

| entry | openpii | gretel | gretel_en | nemotron | argilla_pii | thai_pii | bayanat | tab | redact | iahlt | thainer | naamapadam | multiconer | stockmark | newmean | dialogpii | tonic | meddeid | greek_deid | neonredact | pl_pii | europii | kdpii | nus_lt | redmadrobot | carmen | redactionbench |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sct-default | **D** | **D** | **D-family** | **D** | **D-family** | **D** | **D** | **D** | **D** | **D** | **D** | **D** | **D** | **D** | - | - | - | - | - | - | - | - | - | - | - | - | - |
| sct-gliner | **U, D** | **U, D** | **U, D-family** | **U, D** | **U, D-family** | **U, D** | **U, D** | **U, D** | **U, D** | **U, D** | **U, D** | **U, D** | **U, D** | **U, D** | - | - | - | - | - | - | - | - | - | - | - | - | - |
| presidio | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| gliner-nvidia | **E** | **T-family, E-family** | **E, T-family** | **T, E-family** | **E** | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| gliner2-fastino | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| opf | _U_ | _U_ | _U_ | _U_ | _U_ | _U_ | _U_ | _U_ | _U_ | _U_ | _U_ | _U_ | _U_ | _U_ | - | - | - | - | - | - | - | - | _U_ | - | - | - | - |
| bardsai-v3 | **T** | **T-family** | **T** | **T** | **T-family** | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| datafog | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| gliner-knowledgator | _U_ | _U_ | _U_ | _U_ | _U_ | _U_ | _U_ | _U_ | _U_ | _U_ | _U_ | _U_ | _U_ | _U_ | - | - | - | - | - | - | - | - | - | - | - | - | - |
| piiranha | **T-family** | - | - | - | **T-family** | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| llm-guard | **T-family** | - | - | - | **T** | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| scrubadub | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| ai4privacy | **T-family, E-family** | - | - | - | **T-family, E-family** | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| openmed | **T** | **U, T-family** | **T** | **T** | **T** | _U_ | _U_ | _U_ | _U_ | _U_ | _U_ | _U_ | _U_ | _U_ | - | - | - | - | - | - | - | - | _U_ | - | - | - | - |
| pii-tracer | **T-family, E-family** | **E-family** | **E** | **E** | **T-family, E-family** | - | - | **E** | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| presidio-full | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| sct-training | **D** | **D** | **D-family** | **D** | **D-family** | **D** | **D** | **D** | **D** | **D** | **D** | **D** | **D** | **D** | - | - | - | - | - | - | - | - | - | - | - | - | - |
| sct-gliner-nvidia | **E, D** | **D, T-family, E-family** | **E, T-family, D-family** | **T, D, E-family** | **E, D-family** | **D** | **D** | **D** | **D** | **D** | **D** | **D** | **D** | **D** | - | - | - | - | - | - | - | - | - | - | - | - | - |

## speed (speed-16t)

run device cpu (gpu None), cpus 0-15, threads 16, 1000 docs, 3 repeats, median; peak RSS from /proc VmHWM

| entry | device | cold start s | latency median ms | p95 ms | batch-8 docs/s | chars/s | peak RSS MB | peak VRAM MB | venv MB | failed |
|---|---|---|---|---|---|---|---|---|---|---|
| datafog | cpu | 0.101 | 0.37 | 1.99 | 1555 | 3.547e+06 | 34.3 | - | 9.3 | 0 |
| presidio | cpu | 1.458 | 22.87 | 124.1 | 26.07 | 5.946e+04 | 1080 | - | 11397.8 | 0 |
| presidio-full | cpu | 31.02 | 32.45 | 204.5 | 13.71 | 3.126e+04 | 1.975e+04 | - | 11398.5 | 0 |
| bardsai-v3 | cpu | 3.771 | 342.6 | 1060 | 2.416 | 5510 | 6975 | - | 6483.5 | 0 |
| piiranha | cpu | 4.417 | 286.6 | 1448 | 1.8 | 4104 | 2418 | - | 6449.8 | 0 |
| sct-training | cpu | 8.046 | 588.3 | 2107 | 1.56 | 3557 | 1.02e+04 | - | 3860.4 | 0 |
| sct-default | cpu | 8.184 | 581.1 | 2013 | 1.549 | 3534 | 1.002e+04 | - | 473.8 | 0 |
| llm-guard | cpu | 4.727 | 434.3 | 2516 | 1.4 | 3192 | 2271 | - | 6800.6 | 0 |
| ai4privacy | cpu | 3.466 | 485 | 2467 | 1.366 | 3115 | 1534 | - | 6449.8 | 0 |
| pii-tracer | cpu | 3.868 | 368.6 | 3712 | 1.331 | 3036 | 3098 | - | 6449.4 | 0 |
| scrubadub | cpu | 5.926 | 352.9 | 3232 | 1.204 | 2747 | 5180 | - | 7348.2 | 0 |
| gliner2-fastino | cpu | 7.177 | 320.9 | 4282 | 0.63 | 1437 | 3.436e+04 | - | 6468.6 | 0 |
| sct-gliner | cpu | 8.034 | 794 | 4317 | 0.607 | 1384 | 6715 | - | 6835.6 | 0 |
| openmed | cpu | 4.095 | 403.5 | 3463 | 0.421 | 960.6 | 2.564e+04 | - | 6465.2 | 1 |
| gliner-knowledgator | cpu | fail | fail | fail | fail | fail | fail | - | 6454.9 | 3 |
| gliner-nvidia | cpu | fail | fail | fail | fail | fail | fail | - | 6454.9 | 3 |
| opf | cpu | fail | fail | fail | fail | fail | fail | - | 6467.0 | 3 |
| sct-gliner-nvidia | cpu | fail | fail | fail | fail | fail | fail | - | 6835.6 | 3 |

## speed (speed-gpu)

run device cuda (gpu 3), cpus 0-15, threads 16, 1000 docs, 3 repeats, median; peak RSS from /proc VmHWM

| entry | device | cold start s | latency median ms | p95 ms | batch-8 docs/s | chars/s | peak RSS MB | peak VRAM MB | venv MB | failed |
|---|---|---|---|---|---|---|---|---|---|---|
| piiranha | cuda | 4.27 | 29.18 | 149 | 41.8 | 9.532e+04 | 2117 | 1,884 | 6449.8 | 0 |
| ai4privacy | cuda | 3.265 | 37.12 | 192.4 | 20.82 | 4.747e+04 | 1369 | 1,226 | 6449.8 | 0 |
| pii-tracer | cuda | 3.721 | 34.42 | 154.4 | 19.83 | 4.523e+04 | 1918 | 1,826 | 6449.4 | 0 |
| opf | cuda | 7.637 | 217 | 272.6 | 17.03 | 3.884e+04 | 3739 | 4,626 | 6467.0 | 0 |
| llm-guard | cuda | 4.272 | 55.04 | 276.1 | 11.43 | 2.607e+04 | 2083 | 1,322 | 6800.6 | 0 |
| openmed | cuda | 7.743 | 136.9 | 331.4 | 9.945 | 2.268e+04 | 3742 | 5,100 | 6465.2 | 0 |
| gliner-knowledgator | cuda | fail | fail | fail | fail | fail | fail | - | 6454.9 | 3 |
| gliner-nvidia | cuda | 6.128 | 24.29 | 93.17 | fail | fail | 2759 | 5,354 | 6454.9 | 0 |
| gliner2-fastino | cuda | fail | fail | fail | fail | fail | fail | - | 6468.6 | 3 |
| sct-gliner | cuda | fail | fail | fail | fail | fail | fail | - | 6835.6 | 3 |
| sct-gliner-nvidia | cuda | fail | fail | fail | fail | fail | fail | - | 6835.6 | 3 |
