# SCT 0.7.1 benchmark

Commit `None` (dirty: None). 64 CPUs.
Device `cpu`, ONNX threads 32, workers 2, seed 13. Command: `python -m benchmarks.run --suite speed --threads 32 --workers 2 --seed 13 --out benchmarks/results/0.7.1`

Percentages. Strict = exact span + type. Partial = overlap + type. Leak = gold PII value (or a 4+ char word of it) still in the output; lower is better.

## PII

| dataset | config | docs | strict F1 | partial P | partial R | leak | docs/s |
|---|---|---|---|---|---|---|---|
| bayanat | default | 845 | 42.0 | 44.5 | 64.6 | 7.7 | 19.388 |
| bayanat | ner-off | 845 | 28.8 | 35.5 | 43.3 | 34.8 | 5196.905 |
| bayanat | training | 845 | 51.6 | 52.4 | 57.9 | 8.9 | 18.175 |
| gretel | default | 2100 | 12.9 | 19.1 | 52.4 | 35.3 | 2.444 |
| gretel | ner-off | 2100 | 4.2 | 8.8 | 19.8 | 84.8 | 405.285 |
| gretel | training | 2100 | 27.3 | 42.8 | 42.9 | 40.4 | 2.634 |
| iahlt | default | 300 | 63.8 | 77.0 | 71.9 | 19.5 | 11.817 |
| iahlt | ner-off | 300 | 0.0 | 0.0 | 0.0 | 99.9 | 1779.658 |
| iahlt | training | 300 | 66.8 | 79.4 | 76.6 | 18.4 | 10.385 |
| multiconer | default | 900 | 40.4 | 53.1 | 49.0 | 16.7 | 20.906 |
| multiconer | ner-off | 900 | 0.0 | 0.0 | 0.0 | 89.9 | 4873.122 |
| multiconer | training | 900 | 49.2 | 61.5 | 59.8 | 16.6 | 20.628 |
| naamapadam | default | 900 | 48.2 | 71.7 | 56.8 | 15.1 | 17.092 |
| naamapadam | ner-off | 900 | 0.0 | 0.0 | 0.0 | 96.7 | 3666.061 |
| naamapadam | training | 900 | 59.1 | 79.7 | 70.9 | 13.4 | 17.278 |
| nemotron | default | 300 | 41.8 | 48.2 | 84.1 | 15.0 | 5.893 |
| nemotron | ner-off | 300 | 18.9 | 34.3 | 44.1 | 64.7 | 686.15 |
| nemotron | training | 300 | 67.3 | 82.3 | 66.5 | 19.1 | 5.67 |
| openpii | default | 9000 | 34.6 | 51.3 | 62.5 | 10.3 | 5.703 |
| openpii | ner-off | 9000 | 20.4 | 44.0 | 40.1 | 49.9 | 1305.003 |
| openpii | training | 9000 | 43.2 | 63.3 | 40.9 | 18.3 | 5.904 |
| redact | default | 1000 | 23.5 | 29.9 | 63.2 | 16.0 | 1.483 |
| redact | ner-off | 1000 | 10.3 | 18.1 | 27.3 | 67.2 | 263.058 |
| redact | training | 1000 | 48.1 | 59.3 | 63.6 | 20.8 | 1.391 |
| stockmark | default | 300 | 59.9 | 72.0 | 62.7 | 12.5 | 14.418 |
| stockmark | ner-off | 300 | 0.0 | 0.0 | 0.0 | 98.9 | 4423.176 |
| stockmark | training | 300 | 61.0 | 70.8 | 63.8 | 12.2 | 15.474 |
| tab | default | 127 | 22.1 | 54.2 | 77.5 | 42.3 | 0.831 |
| tab | ner-off | 127 | 4.3 | 42.9 | 38.5 | 91.3 | 120.668 |
| tab | training | 127 | 33.8 | 76.2 | 42.0 | 49.0 | 0.974 |
| thai_pii | default | 300 | 24.3 | 56.8 | 80.1 | 15.4 | 10.559 |
| thai_pii | ner-off | 300 | 18.2 | 46.7 | 36.0 | 59.3 | 2737.629 |
| thai_pii | training | 300 | 37.9 | 74.5 | 71.1 | 32.2 | 9.887 |
| thainer | default | 300 | 24.9 | 53.3 | 55.0 | 23.4 | 10.52 |
| thainer | ner-off | 300 | 0.4 | 3.5 | 1.1 | 87.5 | 2322.048 |
| thainer | training | 300 | 35.4 | 82.2 | 61.7 | 23.3 | 10.794 |

### bayanat (default): by type and language

| | gold | partial R | partial P | strict F1 | leak |
|---|---|---|---|---|---|
| EMAIL | 73 | 100.0 | 100.0 | 100.0 | 0.0 |
| NUMBER | 402 | 23.1 | 24.5 | 0.0 | 18.2 |
| PERSON | 309 | 79.3 | 80.5 | 73.9 | 3.9 |
| PHONE | 288 | 96.2 | 34.4 | 45.9 | 0.0 |
| URL | 31 | 80.7 | 100.0 | 78.6 | 0.0 |
| ar-egyptian | 201 | 69.7 | 52.0 | 45.8 | 6.0 |
| ar-gulf | 289 | 61.9 | 39.8 | 41.8 | 5.9 |
| ar-mixed_en | 200 | 70.5 | 49.3 | 45.5 | 11.5 |
| ar-msa | 413 | 61.3 | 42.4 | 38.8 | 8.0 |

### gretel (default): by type and language

| | gold | partial R | partial P | strict F1 | leak |
|---|---|---|---|---|---|
| DATE | 2647 | 61.8 | 65.8 | 8.7 | 42.4 |
| EMAIL | 487 | 63.2 | 98.4 | 75.8 | 24.9 |
| IP | 113 | 77.9 | 93.8 | 83.8 | 1.8 |
| LOCATION | 1544 | 64.9 | 46.3 | 0.3 | 31.1 |
| NUMBER | 343 | 34.1 | 1.0 | 0.0 | 6.4 |
| ORGANISATION | 2107 | 57.2 | 42.1 | 33.7 | 31.3 |
| PERSON | 4769 | 38.7 | 73.6 | 43.2 | 44.8 |
| PHONE | 334 | 80.5 | 8.2 | 13.6 | 1.5 |
| de | 1739 | 53.6 | 19.1 | 13.7 | 39.1 |
| en | 1888 | 56.4 | 19.4 | 14.3 | 22.0 |
| es | 1717 | 48.5 | 18.3 | 11.6 | 30.9 |
| fr | 1579 | 56.4 | 20.3 | 14.2 | 37.2 |
| it | 1642 | 54.2 | 18.3 | 10.4 | 34.8 |
| nl | 1905 | 48.0 | 19.1 | 11.8 | 41.2 |
| sv | 1874 | 50.5 | 19.1 | 14.5 | 39.7 |

### iahlt (default): by type and language

| | gold | partial R | partial P | strict F1 | leak |
|---|---|---|---|---|---|
| LOCATION | 414 | 75.4 | 81.4 | 66.6 | 17.6 |
| ORGANISATION | 191 | 56.5 | 62.0 | 42.5 | 39.3 |
| PERSON | 229 | 78.6 | 80.8 | 75.9 | 6.6 |
| ar | 834 | 71.9 | 77.0 | 63.8 | 19.5 |

### multiconer (default): by type and language

| | gold | partial R | partial P | strict F1 | leak |
|---|---|---|---|---|---|
| LOCATION | 296 | 52.4 | 62.2 | 45.7 | 19.3 |
| ORGANISATION | 204 | 45.1 | 65.5 | 40.8 | 24.0 |
| PERSON | 325 | 48.3 | 42.3 | 36.1 | 9.8 |
| bn | 287 | 54.7 | 58.3 | 40.1 | 15.7 |
| hi | 258 | 63.2 | 66.7 | 57.8 | 15.9 |
| zh | 280 | 30.0 | 34.3 | 24.1 | 18.6 |

### naamapadam (default): by type and language

| | gold | partial R | partial P | strict F1 | leak |
|---|---|---|---|---|---|
| LOCATION | 510 | 56.7 | 70.9 | 45.4 | 18.6 |
| ORGANISATION | 405 | 44.9 | 70.0 | 41.2 | 22.0 |
| PERSON | 738 | 63.3 | 73.0 | 53.5 | 8.9 |
| bn | 516 | 44.8 | 67.0 | 34.5 | 23.8 |
| hi | 671 | 62.7 | 76.6 | 63.2 | 10.4 |
| ta | 466 | 61.4 | 69.1 | 40.7 | 12.2 |

### nemotron (default): by type and language

| | gold | partial R | partial P | strict F1 | leak |
|---|---|---|---|---|---|
| DATE | 285 | 90.2 | 97.0 | 4.4 | 9.5 |
| EMAIL | 161 | 98.8 | 100.0 | 99.1 | 6.8 |
| IP | 25 | 88.0 | 75.9 | 81.5 | 0.0 |
| LOCATION | 264 | 86.0 | 92.5 | 69.6 | 5.3 |
| NUMBER | 135 | 48.1 | 8.8 | 2.2 | 6.7 |
| ORGANISATION | 135 | 77.8 | 72.5 | 72.9 | 31.9 |
| PERSON | 350 | 83.4 | 98.0 | 88.3 | 18.0 |
| PHONE | 104 | 91.3 | 39.6 | 53.9 | 0.0 |
| URL | 94 | 89.4 | 98.8 | 52.5 | 53.2 |
| en | 1553 | 84.1 | 48.2 | 41.8 | 15.0 |

### openpii (default): by type and language

| | gold | partial R | partial P | strict F1 | leak |
|---|---|---|---|---|---|
| DATE | 7583 | 82.7 | 97.6 | 0.0 | 60.5 |
| EMAIL | 4837 | 97.3 | 99.8 | 92.6 | 0.6 |
| LOCATION | 7478 | 63.1 | 69.8 | 51.8 | 14.2 |
| NUMBER | 13280 | 27.2 | 22.5 | 10.9 | 16.2 |
| PERSON | 8489 | 67.0 | 74.3 | 64.4 | 7.5 |
| PHONE | 3632 | 92.0 | 32.5 | 37.6 | 0.0 |
| bg | 1549 | 64.7 | 51.8 | 37.2 | 14.2 |
| cs | 1501 | 63.8 | 53.1 | 28.7 | 7.5 |
| da | 1543 | 64.9 | 50.9 | 41.6 | 5.5 |
| de | 1378 | 68.3 | 51.6 | 38.1 | 6.1 |
| el | 1451 | 59.7 | 55.9 | 33.9 | 19.3 |
| en | 1561 | 65.4 | 56.2 | 40.7 | 13.7 |
| es | 1480 | 59.5 | 48.1 | 35.8 | 3.3 |
| et | 1499 | 61.4 | 49.8 | 35.3 | 12.0 |
| fi | 1597 | 65.0 | 49.7 | 36.0 | 11.0 |
| fr | 1559 | 64.8 | 56.2 | 37.1 | 12.6 |
| hr | 1453 | 65.6 | 51.9 | 39.5 | 8.9 |
| hu | 1564 | 62.5 | 54.3 | 40.9 | 10.2 |
| id | 1528 | 61.9 | 49.4 | 37.8 | 10.3 |
| it | 1627 | 66.0 | 50.7 | 36.7 | 5.0 |
| ja | 1548 | 54.1 | 43.6 | 20.2 | 17.7 |
| ko | 1388 | 55.4 | 43.8 | 22.8 | 22.3 |
| lt | 1595 | 58.9 | 46.1 | 32.9 | 6.2 |
| lv | 1485 | 61.6 | 49.6 | 29.3 | 9.7 |
| ms | 1536 | 64.5 | 53.2 | 34.8 | 10.2 |
| nl | 1545 | 59.9 | 53.8 | 38.1 | 5.4 |
| pl | 1515 | 62.2 | 58.5 | 30.3 | 12.1 |
| pt | 1503 | 71.1 | 54.0 | 31.5 | 1.9 |
| ro | 1484 | 58.9 | 50.2 | 37.8 | 6.4 |
| sk | 1483 | 65.0 | 56.6 | 32.6 | 11.6 |
| sl | 1479 | 61.9 | 51.9 | 40.6 | 8.5 |
| sr | 1422 | 61.7 | 48.9 | 37.3 | 8.6 |
| sv | 1413 | 64.8 | 52.2 | 34.2 | 6.4 |
| tl | 1622 | 66.8 | 53.2 | 39.6 | 16.0 |
| vi | 1615 | 56.4 | 45.6 | 31.4 | 8.9 |
| zh | 1376 | 59.6 | 47.1 | 25.5 | 18.6 |

### redact (default): by type and language

| | gold | partial R | partial P | strict F1 | leak |
|---|---|---|---|---|---|
| DATE | 1992 | 68.6 | 40.6 | 0.3 | 20.3 |
| EMAIL | 1481 | 73.0 | 97.2 | 80.0 | 10.9 |
| IP | 103 | 84.5 | 70.8 | 67.0 | 1.0 |
| LOCATION | 1574 | 67.3 | 55.1 | 19.1 | 24.3 |
| NUMBER | 937 | 40.9 | 4.1 | 0.7 | 14.9 |
| ORGANISATION | 957 | 56.1 | 24.7 | 24.5 | 19.1 |
| PERSON | 5529 | 59.3 | 66.9 | 50.6 | 17.3 |
| PHONE | 923 | 80.1 | 47.2 | 51.2 | 1.1 |
| ar | 551 | 53.5 | 30.8 | 24.3 | 17.8 |
| cs | 540 | 63.7 | 29.4 | 22.2 | 16.7 |
| da | 555 | 67.9 | 33.7 | 26.4 | 20.9 |
| de | 578 | 64.4 | 33.0 | 26.5 | 6.6 |
| en | 607 | 60.5 | 23.3 | 19.9 | 20.9 |
| es | 607 | 72.2 | 34.6 | 28.1 | 10.0 |
| fi | 505 | 61.8 | 23.4 | 16.2 | 13.3 |
| fr | 477 | 67.3 | 28.6 | 25.4 | 13.9 |
| fr_ca | 523 | 61.8 | 39.4 | 25.3 | 14.1 |
| he | 511 | 57.3 | 28.5 | 21.0 | 14.1 |
| hi | 468 | 72.0 | 29.8 | 24.8 | 23.1 |
| hu | 514 | 71.2 | 33.6 | 24.1 | 11.0 |
| it | 578 | 59.3 | 24.1 | 17.5 | 24.4 |
| ja | 565 | 62.8 | 30.7 | 24.6 | 17.7 |
| ko | 575 | 55.3 | 28.6 | 23.0 | 26.5 |
| nl | 579 | 66.7 | 34.3 | 28.0 | 22.3 |
| no | 478 | 59.8 | 25.1 | 23.6 | 14.1 |
| pt_br | 691 | 67.0 | 33.1 | 21.4 | 18.0 |
| pt_eu | 521 | 68.9 | 27.4 | 21.8 | 11.6 |
| ru | 435 | 51.7 | 30.5 | 18.3 | 22.5 |
| sv | 551 | 66.8 | 28.3 | 26.1 | 12.5 |
| th | 375 | 61.1 | 30.0 | 18.3 | 13.5 |
| tr | 577 | 59.5 | 28.8 | 21.8 | 10.4 |
| zh_cn | 632 | 65.3 | 35.4 | 31.5 | 8.1 |
| zh_tw | 503 | 59.4 | 30.4 | 30.0 | 15.2 |

### stockmark (default): by type and language

| | gold | partial R | partial P | strict F1 | leak |
|---|---|---|---|---|---|
| LOCATION | 192 | 50.0 | 60.5 | 47.6 | 16.2 |
| ORGANISATION | 283 | 65.0 | 77.2 | 60.5 | 14.5 |
| PERSON | 157 | 73.9 | 75.8 | 72.9 | 4.5 |
| ja | 632 | 62.7 | 72.0 | 59.9 | 12.5 |

### tab (default): by type and language

| | gold | partial R | partial P | strict F1 | leak |
|---|---|---|---|---|---|
| DATE | 2647 | 82.0 | 95.9 | 10.5 | 82.3 |
| LOCATION | 523 | 94.7 | 61.2 | 61.5 | 18.4 |
| NUMBER | 334 | 99.7 | 14.6 | 0.9 | 1.8 |
| ORGANISATION | 1947 | 63.2 | 75.4 | 57.1 | 66.3 |
| PERSON | 1078 | 77.2 | 89.9 | 10.6 | 23.0 |
| en | 6529 | 77.5 | 54.2 | 22.1 | 42.3 |

TAB masking recall (any type, macro over 12 annotators): all 31.1, direct 41.5, quasi 30.1.
Partial recall by start offset: 0-1000: 76.0 (1457), 1000-2000: 83.2 (1465), 2000-3000: 78.3 (1184), 3000-4000: 75.2 (792), 4000-5000: 71.2 (454), 5000-6000: 69.4 (333), 6000-7000: 76.4 (344), 7000-8000: 74.1 (263), 8000-9000: 82.7 (139), 9000+: 86.7 (98)

### thai_pii (default): by type and language

| | gold | partial R | partial P | strict F1 | leak |
|---|---|---|---|---|---|
| EMAIL | 61 | 100.0 | 100.0 | 100.0 | 0.0 |
| LOCATION | 207 | 73.4 | 77.2 | 0.3 | 50.2 |
| NUMBER | 139 | 74.8 | 40.5 | 0.0 | 0.0 |
| ORGANISATION | 66 | 68.2 | 51.1 | 54.5 | 18.2 |
| PERSON | 281 | 77.6 | 79.3 | 49.6 | 9.2 |
| PHONE | 167 | 94.6 | 59.0 | 67.1 | 0.0 |
| th | 921 | 80.1 | 56.8 | 24.3 | 15.4 |

### thainer (default): by type and language

| | gold | partial R | partial P | strict F1 | leak |
|---|---|---|---|---|---|
| DATE | 90 | 3.3 | 60.0 | 0.0 | 42.2 |
| LOCATION | 222 | 72.1 | 82.3 | 35.2 | 18.5 |
| NUMBER | 0 | 0.0 | 0.0 | 0.0 | - |
| ORGANISATION | 274 | 58.4 | 79.5 | 41.8 | 30.3 |
| PERSON | 156 | 55.8 | 60.4 | 11.3 | 19.2 |
| PHONE | 7 | 28.6 | 100.0 | 0.0 | 0.0 |
| URL | 3 | 66.7 | 100.0 | 40.0 | 33.3 |
| th | 752 | 55.0 | 53.3 | 24.9 | 23.4 |

## Cleaning: injected noise removed (fineweb2-clean)

| config | kind | injected | removed |
|---|---|---|---|
| default | emoji | 1000 | 100.0 |
| default | html | 1000 | 100.0 |
| default | mojibake | 1000 | 100.0 |
| default | url | 1000 | 100.0 |
| ner-off | emoji | 1000 | 100.0 |
| ner-off | html | 1000 | 100.0 |
| ner-off | mojibake | 1000 | 100.0 |
| ner-off | url | 1000 | 100.0 |
| training | emoji | 1000 | 5.0 |
| training | html | 1000 | 0.9 |
| training | mojibake | 1000 | 99.5 |
| training | url | 1000 | 100.0 |

## Speed (fineweb2)

| config | script | docs | docs/s | chars/s |
|---|---|---|---|---|
| default | all | 1000 | 1.185 | 2703.0 |
| default | arb_Arab (ARABIC) | 100 | 1.072 | 2913.3 |
| default | cmn_Hani (CJK) | 100 | 0.895 | 1080.7 |
| default | ell_Grek (GREEK) | 100 | 1.546 | 4676.0 |
| default | heb_Hebr (HEBREW) | 99 | 0.942 | 2532.0 |
| default | heb_Hebr (LATIN) | 1 | 0.437 | 3722.7 |
| default | hin_Deva (DEVANAGARI) | 98 | 1.559 | 3270.4 |
| default | hin_Deva (LATIN) | 2 | 1.722 | 4463.4 |
| default | jpn_Jpan (CJK) | 14 | 1.972 | 1728.0 |
| default | jpn_Jpan (HIRAGANA) | 78 | 1.224 | 1787.6 |
| default | jpn_Jpan (KATAKANA) | 7 | 1.379 | 1958.2 |
| default | jpn_Jpan (LATIN) | 1 | 0.924 | 1855.8 |
| default | kor_Hang (HANGUL) | 99 | 1.345 | 1818.8 |
| default | kor_Hang (LATIN) | 1 | 6.384 | 3492.0 |
| default | nld_Latn (LATIN) | 100 | 1.083 | 3038.3 |
| default | rus_Cyrl (CYRILLIC) | 99 | 1.424 | 4282.7 |
| default | rus_Cyrl (LATIN) | 1 | 2.085 | 5873.3 |
| default | tha_Thai (THAI) | 100 | 1.118 | 2753.7 |
| ner-off | all | 1000 | 211.805 | 483020.0 |
| ner-off | arb_Arab (ARABIC) | 100 | 215.046 | 584341.1 |
| ner-off | cmn_Hani (CJK) | 100 | 161.482 | 195039.8 |
| ner-off | ell_Grek (GREEK) | 100 | 199.86 | 604585.6 |
| ner-off | heb_Hebr (HEBREW) | 99 | 207.613 | 558042.4 |
| ner-off | heb_Hebr (LATIN) | 1 | 53.26 | 453451.5 |
| ner-off | hin_Deva (DEVANAGARI) | 98 | 242.322 | 508285.2 |
| ner-off | hin_Deva (LATIN) | 2 | 187.06 | 484954.2 |
| ner-off | jpn_Jpan (CJK) | 14 | 278.033 | 243676.0 |
| ner-off | jpn_Jpan (HIRAGANA) | 78 | 180.505 | 263648.6 |
| ner-off | jpn_Jpan (KATAKANA) | 7 | 176.584 | 250824.3 |
| ner-off | jpn_Jpan (LATIN) | 1 | 160.31 | 321901.9 |
| ner-off | kor_Hang (HANGUL) | 99 | 248.708 | 336260.6 |
| ner-off | kor_Hang (LATIN) | 1 | 579.428 | 316946.9 |
| ner-off | nld_Latn (LATIN) | 100 | 239.959 | 673474.8 |
| ner-off | rus_Cyrl (CYRILLIC) | 99 | 202.723 | 609843.2 |
| ner-off | rus_Cyrl (LATIN) | 1 | 214.787 | 605055.4 |
| ner-off | tha_Thai (THAI) | 100 | 257.675 | 634600.1 |
| training | all | 1000 | 1.675 | 3820.1 |
| training | arb_Arab (ARABIC) | 100 | 1.786 | 4854.1 |
| training | cmn_Hani (CJK) | 100 | 1.716 | 2072.6 |
| training | ell_Grek (GREEK) | 100 | 1.638 | 4955.4 |
| training | heb_Hebr (HEBREW) | 99 | 1.626 | 4371.1 |
| training | heb_Hebr (LATIN) | 1 | 0.597 | 5079.8 |
| training | hin_Deva (DEVANAGARI) | 98 | 2.318 | 4861.4 |
| training | hin_Deva (LATIN) | 2 | 1.929 | 5000.3 |
| training | jpn_Jpan (CJK) | 14 | 2.617 | 2293.4 |
| training | jpn_Jpan (HIRAGANA) | 78 | 1.607 | 2347.2 |
| training | jpn_Jpan (KATAKANA) | 7 | 1.735 | 2463.9 |
| training | jpn_Jpan (LATIN) | 1 | 1.301 | 2611.9 |
| training | kor_Hang (HANGUL) | 99 | 1.861 | 2516.0 |
| training | kor_Hang (LATIN) | 1 | 8.142 | 4453.7 |
| training | nld_Latn (LATIN) | 100 | 1.06 | 2973.8 |
| training | rus_Cyrl (CYRILLIC) | 99 | 1.688 | 5079.1 |
| training | rus_Cyrl (LATIN) | 1 | 2.577 | 7259.1 |
| training | tha_Thai (THAI) | 100 | 1.985 | 4888.8 |

## Language ID (wili)

Accuracy 5.5, macro 5.5 over 7200 in-set docs. Out-of-set docs given a language: 66.5 of 16300.

| script | docs | accuracy |
|---|---|---|
| ARABIC | 301 | 0.0 |
| ARMENIAN | 99 | 0.0 |
| BENGALI | 81 | 0.0 |
| CJK | 146 | 0.0 |
| CYRILLIC | 782 | 0.0 |
| DEVANAGARI | 193 | 0.0 |
| GEORGIAN | 99 | 0.0 |
| GREEK | 98 | 0.0 |
| GUJARATI | 94 | 0.0 |
| GURMUKHI | 98 | 0.0 |
| HANGUL | 99 | 0.0 |
| HEBREW | 100 | 0.0 |
| HIRAGANA | 34 | 0.0 |
| KATAKANA | 10 | 0.0 |
| LATIN | 4673 | 8.4 |
| TAMIL | 98 | 0.0 |
| TELUGU | 96 | 0.0 |
| THAI | 99 | 0.0 |
