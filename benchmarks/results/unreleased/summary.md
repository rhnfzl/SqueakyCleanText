# SCT 0.8.0 benchmark

Commit `None` (dirty: None). 64 CPUs.
Device `cpu`, ONNX threads 32, workers 2, seed 13. Command: `python -m benchmarks.run --suite speed --threads 32 --workers 2 --seed 13 --configs default training --out benchmarks/results/unreleased`

Percentages. Strict = exact span + type. Partial = overlap + type. Masked R = overlap, any type. Leak = gold PII value (or a 4+ char word of it) still in the output; lower is better.

## PII

| dataset | config | docs | strict F1 | partial P | partial R | masked R | leak | docs/s |
|---|---|---|---|---|---|---|---|---|
| bayanat | default | 845 | 56.7 | 54.9 | 73.7 | 99.0 | 2.3 | 56.745 |
| bayanat | training | 845 | 63.4 | 64.1 | 70.6 | 98.6 | 2.7 | 56.994 |
| gretel | default | 2100 | 20.7 | 24.1 | 61.9 | 72.1 | 28.3 | 3.58 |
| gretel | training | 2100 | 32.3 | 49.6 | 52.5 | 60.0 | 33.7 | 3.553 |
| iahlt | default | 300 | 69.6 | 88.8 | 90.2 | 94.0 | 12.7 | 18.21 |
| iahlt | training | 300 | 70.4 | 89.2 | 91.4 | 95.0 | 12.2 | 18.695 |
| multiconer | default | 900 | 57.5 | 76.2 | 77.1 | 82.2 | 20.1 | 48.349 |
| multiconer | training | 900 | 58.8 | 75.3 | 78.7 | 83.5 | 18.9 | 50.44 |
| naamapadam | default | 900 | 64.1 | 81.3 | 67.8 | 71.9 | 17.5 | 38.859 |
| naamapadam | training | 900 | 75.3 | 89.4 | 84.3 | 89.0 | 14.6 | 37.825 |
| nemotron | default | 300 | 63.7 | 59.6 | 89.5 | 96.1 | 7.7 | 6.675 |
| nemotron | training | 300 | 73.5 | 84.4 | 71.7 | 79.1 | 11.7 | 6.416 |
| openpii | default | 9000 | 60.0 | 73.6 | 74.4 | 90.3 | 10.6 | 9.779 |
| openpii | training | 9000 | 50.1 | 75.1 | 51.1 | 67.5 | 18.6 | 10.757 |
| redact | default | 1000 | 39.2 | 41.1 | 85.6 | 92.2 | 8.2 | 2.196 |
| redact | training | 1000 | 54.0 | 66.5 | 75.3 | 81.7 | 12.8 | 2.197 |
| stockmark | default | 300 | 64.9 | 86.9 | 83.9 | 89.2 | 6.5 | 31.382 |
| stockmark | training | 300 | 66.6 | 85.8 | 84.5 | 89.6 | 6.7 | 33.355 |
| tab | default | 127 | 48.1 | 62.8 | 85.9 | 92.7 | 11.7 | 1.325 |
| tab | training | 127 | 33.7 | 66.9 | 46.6 | 51.6 | 19.0 | 1.448 |
| thai_pii | default | 300 | 30.9 | 63.4 | 89.4 | 95.8 | 13.7 | 21.697 |
| thai_pii | training | 300 | 47.0 | 82.8 | 79.3 | 85.3 | 31.1 | 24.843 |
| thainer | default | 300 | 32.8 | 69.0 | 73.3 | 84.6 | 15.1 | 20.508 |
| thainer | training | 300 | 37.1 | 91.6 | 71.7 | 74.7 | 15.4 | 23.784 |

### bayanat (default): by type and language

| | gold | partial R | masked R | partial P | strict F1 | leak |
|---|---|---|---|---|---|---|
| EMAIL | 73 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0 |
| NUMBER | 402 | 31.8 | 100.0 | 36.1 | 23.7 | 4.5 |
| PERSON | 309 | 98.4 | 98.4 | 94.4 | 90.3 | 2.3 |
| PHONE | 288 | 96.2 | 97.9 | 39.9 | 51.9 | 0.0 |
| URL | 31 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0 |
| ar-egyptian | 201 | 83.6 | 100.0 | 63.8 | 58.0 | 2.5 |
| ar-gulf | 289 | 70.9 | 99.0 | 49.3 | 54.9 | 1.7 |
| ar-mixed_en | 200 | 76.0 | 99.5 | 59.5 | 63.0 | 0.5 |
| ar-msa | 413 | 69.7 | 98.3 | 52.5 | 54.3 | 3.4 |

### gretel (default): by type and language

| | gold | partial R | masked R | partial P | strict F1 | leak |
|---|---|---|---|---|---|---|
| DATE | 2647 | 64.6 | 75.9 | 66.9 | 45.5 | 35.7 |
| EMAIL | 487 | 65.1 | 65.5 | 98.1 | 77.3 | 25.5 |
| IP | 113 | 92.0 | 95.6 | 94.7 | 92.0 | 1.8 |
| LOCATION | 1544 | 77.0 | 93.3 | 52.1 | 0.4 | 22.5 |
| NUMBER | 343 | 47.8 | 95.9 | 1.4 | 0.6 | 7.0 |
| ORGANISATION | 2107 | 72.6 | 75.0 | 46.8 | 39.3 | 22.0 |
| PERSON | 4769 | 48.6 | 58.5 | 85.9 | 53.1 | 37.3 |
| PHONE | 334 | 91.6 | 95.2 | 9.1 | 15.2 | 1.8 |
| de | 1739 | 62.2 | 71.0 | 23.7 | 20.8 | 34.4 |
| en | 1888 | 65.9 | 76.8 | 23.4 | 21.7 | 11.6 |
| es | 1717 | 58.9 | 71.4 | 23.5 | 19.7 | 25.6 |
| fr | 1579 | 68.8 | 77.9 | 26.9 | 23.5 | 26.5 |
| it | 1642 | 65.0 | 76.7 | 23.4 | 18.0 | 26.9 |
| nl | 1905 | 57.6 | 65.3 | 24.8 | 20.5 | 36.0 |
| sv | 1874 | 56.0 | 66.7 | 23.2 | 20.7 | 33.8 |

### iahlt (default): by type and language

| | gold | partial R | masked R | partial P | strict F1 | leak |
|---|---|---|---|---|---|---|
| LOCATION | 414 | 91.3 | 95.2 | 91.4 | 70.0 | 12.1 |
| ORGANISATION | 191 | 79.1 | 86.9 | 78.1 | 51.7 | 28.3 |
| PERSON | 229 | 97.4 | 97.8 | 93.3 | 83.8 | 0.9 |
| ar | 834 | 90.2 | 94.0 | 88.8 | 69.6 | 12.7 |

### multiconer (default): by type and language

| | gold | partial R | masked R | partial P | strict F1 | leak |
|---|---|---|---|---|---|---|
| LOCATION | 296 | 75.0 | 79.4 | 79.8 | 55.5 | 20.6 |
| ORGANISATION | 204 | 60.8 | 73.5 | 79.6 | 45.6 | 33.8 |
| PERSON | 325 | 89.2 | 90.1 | 72.6 | 64.8 | 11.1 |
| bn | 287 | 76.3 | 80.8 | 74.3 | 50.4 | 24.7 |
| hi | 258 | 76.4 | 83.3 | 77.2 | 66.0 | 20.2 |
| zh | 280 | 78.6 | 82.5 | 77.1 | 56.6 | 15.4 |

### naamapadam (default): by type and language

| | gold | partial R | masked R | partial P | strict F1 | leak |
|---|---|---|---|---|---|---|
| LOCATION | 510 | 69.8 | 73.7 | 81.1 | 65.6 | 17.2 |
| ORGANISATION | 405 | 54.6 | 64.2 | 78.6 | 52.6 | 33.6 |
| PERSON | 738 | 73.7 | 74.8 | 82.6 | 68.7 | 8.9 |
| bn | 516 | 56.6 | 59.9 | 79.8 | 54.4 | 25.2 |
| hi | 671 | 70.8 | 74.5 | 82.5 | 69.0 | 13.7 |
| ta | 466 | 76.0 | 81.3 | 80.9 | 66.6 | 14.6 |

### nemotron (default): by type and language

| | gold | partial R | masked R | partial P | strict F1 | leak |
|---|---|---|---|---|---|---|
| DATE | 285 | 90.5 | 96.1 | 97.0 | 91.3 | 2.1 |
| EMAIL | 161 | 98.8 | 98.8 | 100.0 | 99.1 | 7.4 |
| IP | 25 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0 |
| LOCATION | 264 | 90.9 | 94.7 | 95.0 | 71.5 | 3.4 |
| NUMBER | 135 | 49.6 | 100.0 | 13.2 | 3.9 | 6.7 |
| ORGANISATION | 135 | 89.6 | 89.6 | 75.6 | 77.5 | 9.6 |
| PERSON | 350 | 93.1 | 94.9 | 97.7 | 94.1 | 2.3 |
| PHONE | 104 | 97.1 | 99.0 | 41.7 | 57.0 | 0.0 |
| URL | 94 | 98.9 | 98.9 | 98.9 | 98.9 | 50.0 |
| en | 1553 | 89.5 | 96.1 | 59.6 | 63.7 | 7.7 |

### openpii (default): by type and language

| | gold | partial R | masked R | partial P | strict F1 | leak |
|---|---|---|---|---|---|---|
| DATE | 7583 | 99.5 | 100.0 | 97.7 | 92.5 | 5.1 |
| EMAIL | 4837 | 99.9 | 99.9 | 99.6 | 94.3 | 0.6 |
| LOCATION | 7478 | 84.8 | 88.5 | 94.6 | 70.5 | 12.2 |
| NUMBER | 13280 | 28.2 | 79.0 | 52.7 | 18.6 | 20.9 |
| PERSON | 8489 | 91.0 | 91.3 | 96.0 | 72.9 | 3.2 |
| PHONE | 3632 | 96.1 | 100.0 | 32.6 | 42.7 | 0.0 |
| bg | 1549 | 74.5 | 94.0 | 71.0 | 62.8 | 8.6 |
| cs | 1501 | 75.9 | 91.9 | 76.1 | 55.0 | 9.1 |
| da | 1543 | 74.5 | 93.5 | 72.6 | 68.8 | 5.6 |
| de | 1378 | 79.0 | 94.9 | 71.2 | 63.4 | 5.4 |
| el | 1451 | 70.6 | 86.4 | 77.0 | 60.9 | 13.6 |
| en | 1561 | 73.9 | 90.3 | 73.7 | 62.6 | 11.5 |
| es | 1480 | 74.9 | 90.7 | 69.9 | 55.8 | 3.2 |
| et | 1499 | 74.5 | 93.1 | 73.4 | 61.0 | 11.4 |
| fi | 1597 | 78.8 | 93.2 | 75.8 | 63.1 | 11.6 |
| fr | 1559 | 73.9 | 90.6 | 76.3 | 62.0 | 10.8 |
| hr | 1453 | 75.1 | 92.6 | 73.2 | 64.8 | 10.2 |
| hu | 1564 | 73.7 | 85.8 | 78.8 | 67.4 | 16.6 |
| id | 1528 | 75.6 | 90.2 | 73.3 | 64.3 | 11.3 |
| it | 1627 | 77.2 | 94.3 | 68.6 | 57.9 | 4.5 |
| ja | 1548 | 66.7 | 86.9 | 67.2 | 48.6 | 15.7 |
| ko | 1388 | 62.0 | 83.4 | 57.7 | 36.2 | 19.0 |
| lt | 1595 | 75.4 | 93.7 | 70.9 | 59.1 | 8.6 |
| lv | 1485 | 73.5 | 91.6 | 73.9 | 55.2 | 11.0 |
| ms | 1536 | 79.3 | 89.6 | 79.1 | 63.0 | 11.4 |
| nl | 1545 | 74.1 | 89.3 | 74.7 | 57.0 | 4.4 |
| pl | 1515 | 74.1 | 85.9 | 81.2 | 55.0 | 14.0 |
| pt | 1503 | 85.6 | 95.7 | 78.2 | 60.2 | 3.6 |
| ro | 1484 | 73.7 | 88.4 | 76.2 | 67.4 | 12.6 |
| sk | 1483 | 75.2 | 86.7 | 80.9 | 57.9 | 14.3 |
| sl | 1479 | 71.4 | 88.6 | 74.1 | 65.8 | 10.3 |
| sr | 1422 | 70.2 | 90.2 | 68.6 | 61.1 | 9.6 |
| sv | 1413 | 75.0 | 93.2 | 74.7 | 61.4 | 6.4 |
| tl | 1622 | 76.1 | 88.2 | 74.1 | 65.0 | 13.9 |
| vi | 1615 | 72.6 | 89.4 | 71.5 | 60.6 | 13.0 |
| zh | 1376 | 72.4 | 86.3 | 75.0 | 55.3 | 16.2 |

### redact (default): by type and language

| | gold | partial R | masked R | partial P | strict F1 | leak |
|---|---|---|---|---|---|---|
| DATE | 1992 | 83.1 | 88.8 | 38.7 | 38.5 | 10.7 |
| EMAIL | 1481 | 99.0 | 99.2 | 96.5 | 95.2 | 10.1 |
| IP | 103 | 93.2 | 100.0 | 69.0 | 70.2 | 1.0 |
| LOCATION | 1574 | 88.9 | 95.4 | 62.3 | 23.0 | 14.9 |
| NUMBER | 937 | 40.1 | 90.6 | 5.1 | 0.9 | 17.1 |
| ORGANISATION | 957 | 85.7 | 93.2 | 30.3 | 32.6 | 13.0 |
| PERSON | 5529 | 88.0 | 89.3 | 85.0 | 69.2 | 4.7 |
| PHONE | 923 | 94.4 | 99.5 | 45.2 | 56.5 | 1.3 |
| ar | 551 | 84.6 | 93.1 | 51.1 | 49.8 | 13.0 |
| cs | 540 | 89.6 | 93.0 | 38.4 | 34.4 | 10.3 |
| da | 555 | 90.6 | 95.3 | 43.8 | 42.4 | 4.8 |
| de | 578 | 86.7 | 92.4 | 43.5 | 42.2 | 5.8 |
| en | 607 | 89.1 | 94.1 | 32.3 | 33.1 | 11.8 |
| es | 607 | 85.8 | 90.6 | 46.2 | 42.6 | 3.7 |
| fi | 505 | 88.3 | 94.3 | 36.1 | 33.5 | 7.7 |
| fr | 477 | 88.9 | 92.7 | 39.9 | 39.5 | 5.1 |
| fr_ca | 523 | 87.2 | 93.5 | 50.0 | 43.4 | 10.0 |
| he | 511 | 87.9 | 95.5 | 39.6 | 41.9 | 9.9 |
| hi | 468 | 88.9 | 92.3 | 40.1 | 39.2 | 9.3 |
| hu | 514 | 86.8 | 95.1 | 40.7 | 34.9 | 9.5 |
| it | 578 | 83.7 | 90.1 | 37.4 | 36.8 | 13.3 |
| ja | 565 | 81.8 | 86.9 | 40.1 | 40.5 | 9.4 |
| ko | 575 | 83.3 | 87.5 | 41.5 | 40.0 | 13.0 |
| nl | 579 | 82.6 | 87.1 | 39.9 | 37.3 | 5.9 |
| no | 478 | 81.4 | 94.3 | 35.0 | 37.8 | 7.9 |
| pt_br | 691 | 85.1 | 89.3 | 45.8 | 37.5 | 11.0 |
| pt_eu | 521 | 81.2 | 93.1 | 35.8 | 35.2 | 6.0 |
| ru | 435 | 84.6 | 94.0 | 48.2 | 40.4 | 5.6 |
| sv | 551 | 81.3 | 91.6 | 39.1 | 40.3 | 4.9 |
| th | 375 | 92.0 | 97.1 | 40.9 | 33.1 | 9.7 |
| tr | 577 | 80.4 | 87.5 | 41.6 | 34.9 | 10.3 |
| zh_cn | 632 | 87.2 | 95.1 | 46.8 | 48.7 | 2.7 |
| zh_tw | 503 | 83.7 | 92.6 | 43.9 | 46.9 | 4.1 |

### stockmark (default): by type and language

| | gold | partial R | masked R | partial P | strict F1 | leak |
|---|---|---|---|---|---|---|
| LOCATION | 192 | 78.1 | 87.0 | 84.8 | 63.5 | 9.4 |
| ORGANISATION | 283 | 82.0 | 86.9 | 89.4 | 64.1 | 6.7 |
| PERSON | 157 | 94.3 | 96.2 | 85.3 | 68.0 | 2.5 |
| ja | 632 | 83.9 | 89.2 | 86.9 | 64.9 | 6.5 |

### tab (default): by type and language

| | gold | partial R | masked R | partial P | strict F1 | leak |
|---|---|---|---|---|---|---|
| DATE | 2647 | 87.0 | 91.6 | 96.1 | 81.5 | 30.3 |
| LOCATION | 523 | 97.7 | 98.3 | 59.6 | 60.6 | 5.0 |
| NUMBER | 334 | 99.7 | 99.7 | 24.1 | 1.5 | 2.1 |
| ORGANISATION | 1947 | 75.9 | 91.5 | 58.5 | 52.9 | 16.4 |
| PERSON | 1078 | 91.0 | 92.3 | 97.2 | 13.7 | 9.5 |
| en | 6529 | 85.9 | 92.7 | 62.8 | 48.1 | 11.7 |

TAB masking recall (any type, macro over 12 annotators): all 66.1, direct 43.6, quasi 68.0.
Partial recall by start offset: 0-1000: 82.4 (1457), 1000-2000: 90.6 (1465), 2000-3000: 87.3 (1184), 3000-4000: 85.9 (792), 4000-5000: 80.0 (454), 5000-6000: 78.4 (333), 6000-7000: 85.8 (344), 7000-8000: 87.1 (263), 8000-9000: 89.9 (139), 9000+: 93.9 (98)

### thai_pii (default): by type and language

| | gold | partial R | masked R | partial P | strict F1 | leak |
|---|---|---|---|---|---|---|
| EMAIL | 61 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0 |
| LOCATION | 207 | 88.9 | 96.6 | 97.8 | 1.1 | 45.4 |
| NUMBER | 139 | 76.3 | 100.0 | 39.6 | 0.8 | 0.0 |
| ORGANISATION | 66 | 83.3 | 84.9 | 57.9 | 62.1 | 15.2 |
| PERSON | 281 | 89.3 | 92.2 | 94.0 | 82.1 | 7.8 |
| PHONE | 167 | 99.4 | 100.0 | 60.4 | 68.3 | 0.0 |
| th | 921 | 89.4 | 95.8 | 63.4 | 30.9 | 13.7 |

### thainer (default): by type and language

| | gold | partial R | masked R | partial P | strict F1 | leak |
|---|---|---|---|---|---|---|
| DATE | 90 | 17.8 | 75.6 | 84.2 | 25.7 | 32.2 |
| LOCATION | 222 | 85.6 | 90.1 | 90.0 | 37.4 | 8.6 |
| NUMBER | 0 | 0.0 | 0.0 | 0.0 | 0.0 | - |
| ORGANISATION | 274 | 72.6 | 77.0 | 90.1 | 46.4 | 23.7 |
| PERSON | 156 | 90.4 | 94.2 | 96.5 | 27.9 | 9.6 |
| PHONE | 7 | 28.6 | 100.0 | 100.0 | 0.0 | 0.0 |
| URL | 3 | 100.0 | 100.0 | 100.0 | 66.7 | 33.3 |
| th | 752 | 73.3 | 84.6 | 69.0 | 32.8 | 15.1 |

## Cleaning: injected noise removed (fineweb2-clean)

| config | kind | injected | removed |
|---|---|---|---|
| default | emoji | 1000 | 4.9 |
| default | html | 1000 | 100.0 |
| default | mojibake | 1000 | 99.6 |
| default | url | 1000 | 100.0 |
| training | emoji | 1000 | 5.1 |
| training | html | 1000 | 0.9 |
| training | mojibake | 1000 | 99.5 |
| training | url | 1000 | 100.0 |

## Speed (fineweb2)

| config | script | docs | docs/s | chars/s |
|---|---|---|---|---|
| default | all | 1000 | 2.618 | 5969.6 |
| default | arb_Arab (ARABIC) | 100 | 2.728 | 7412.7 |
| default | cmn_Hani (CJK) | 100 | 2.677 | 3233.3 |
| default | ell_Grek (GREEK) | 100 | 2.538 | 7678.2 |
| default | heb_Hebr (HEBREW) | 99 | 2.552 | 6859.0 |
| default | heb_Hebr (LATIN) | 1 | 0.702 | 5980.2 |
| default | hin_Deva (DEVANAGARI) | 98 | 3.56 | 7468.3 |
| default | hin_Deva (LATIN) | 2 | 2.215 | 5741.4 |
| default | jpn_Jpan (CJK) | 14 | 3.611 | 3164.8 |
| default | jpn_Jpan (HIRAGANA) | 78 | 2.632 | 3844.1 |
| default | jpn_Jpan (KATAKANA) | 7 | 2.469 | 3507.0 |
| default | jpn_Jpan (LATIN) | 1 | 1.4 | 2811.9 |
| default | kor_Hang (HANGUL) | 99 | 3.097 | 4186.6 |
| default | kor_Hang (LATIN) | 1 | 6.92 | 3785.5 |
| default | nld_Latn (LATIN) | 100 | 1.824 | 5118.5 |
| default | rus_Cyrl (CYRILLIC) | 99 | 2.82 | 8483.9 |
| default | rus_Cyrl (LATIN) | 1 | 2.464 | 6939.9 |
| default | tha_Thai (THAI) | 100 | 2.473 | 6090.0 |
| training | all | 1000 | 2.778 | 6336.3 |
| training | arb_Arab (ARABIC) | 100 | 2.863 | 7779.8 |
| training | cmn_Hani (CJK) | 100 | 2.826 | 3413.4 |
| training | ell_Grek (GREEK) | 100 | 2.632 | 7961.8 |
| training | heb_Hebr (HEBREW) | 99 | 2.673 | 7185.6 |
| training | heb_Hebr (LATIN) | 1 | 0.76 | 6467.0 |
| training | hin_Deva (DEVANAGARI) | 98 | 3.784 | 7937.4 |
| training | hin_Deva (LATIN) | 2 | 2.422 | 6279.9 |
| training | jpn_Jpan (CJK) | 14 | 3.831 | 3357.7 |
| training | jpn_Jpan (HIRAGANA) | 78 | 2.776 | 4054.2 |
| training | jpn_Jpan (KATAKANA) | 7 | 2.429 | 3450.2 |
| training | jpn_Jpan (LATIN) | 1 | 1.77 | 3553.6 |
| training | kor_Hang (HANGUL) | 99 | 3.258 | 4404.6 |
| training | kor_Hang (LATIN) | 1 | 7.488 | 4095.9 |
| training | nld_Latn (LATIN) | 100 | 1.96 | 5499.6 |
| training | rus_Cyrl (CYRILLIC) | 99 | 2.974 | 8947.2 |
| training | rus_Cyrl (LATIN) | 1 | 2.429 | 6843.8 |
| training | tha_Thai (THAI) | 100 | 2.772 | 6826.3 |
