# SCT 0.8.0 benchmark

Commit `None` (dirty: None). 64 CPUs.
Device `cpu`, ONNX threads 32, workers 2, seed 13. Command: `python -m benchmarks.run --suite speed --threads 32 --workers 2 --seed 13 --out benchmarks/results/0.8.0`

Percentages. Strict = exact span + type. Partial = overlap + type. Leak = gold PII value (or a 4+ char word of it) still in the output; lower is better.

## PII

| dataset | config | docs | strict F1 | partial P | partial R | leak | docs/s |
|---|---|---|---|---|---|---|---|
| bayanat | default | 845 | 47.1 | 48.4 | 68.2 | 7.5 | 56.215 |
| bayanat | ner-off | 845 | 30.0 | 37.2 | 41.8 | 34.1 | 1776.317 |
| bayanat | training | 845 | 54.0 | 54.1 | 61.5 | 7.6 | 58.793 |
| gretel | default | 2100 | 19.7 | 23.1 | 57.0 | 37.8 | 3.724 |
| gretel | ner-off | 2100 | 9.8 | 10.2 | 20.5 | 84.7 | 127.723 |
| gretel | training | 2100 | 31.1 | 49.7 | 48.4 | 42.6 | 3.642 |
| iahlt | default | 300 | 67.0 | 88.8 | 84.5 | 20.4 | 19.006 |
| iahlt | ner-off | 300 | 0.0 | 0.0 | 0.0 | 99.9 | 809.827 |
| iahlt | training | 300 | 68.1 | 89.2 | 85.6 | 19.1 | 19.357 |
| multiconer | default | 900 | 55.4 | 76.9 | 73.1 | 19.0 | 44.742 |
| multiconer | ner-off | 900 | 0.0 | 0.0 | 0.0 | 99.6 | 2304.966 |
| multiconer | training | 900 | 56.9 | 75.9 | 74.3 | 17.9 | 47.479 |
| naamapadam | default | 900 | 55.9 | 81.6 | 63.9 | 16.2 | 36.375 |
| naamapadam | ner-off | 900 | 0.0 | 0.0 | 0.0 | 96.7 | 1749.212 |
| naamapadam | training | 900 | 67.7 | 90.1 | 79.6 | 15.0 | 35.645 |
| nemotron | default | 300 | 59.7 | 58.6 | 85.5 | 15.1 | 6.885 |
| nemotron | ner-off | 300 | 37.2 | 44.6 | 44.8 | 64.4 | 170.979 |
| nemotron | training | 300 | 67.9 | 83.4 | 67.7 | 18.9 | 6.558 |
| openpii | default | 9000 | 55.6 | 70.2 | 69.2 | 13.1 | 12.374 |
| openpii | ner-off | 9000 | 39.3 | 59.1 | 40.0 | 50.0 | 317.558 |
| openpii | training | 9000 | 48.2 | 73.9 | 48.9 | 21.0 | 12.724 |
| redact | default | 1000 | 36.2 | 41.1 | 72.1 | 17.9 | 2.54 |
| redact | ner-off | 1000 | 19.5 | 23.8 | 27.8 | 67.1 | 85.628 |
| redact | training | 1000 | 54.0 | 69.0 | 72.5 | 21.6 | 2.202 |
| stockmark | default | 300 | 62.3 | 86.7 | 78.6 | 13.1 | 29.086 |
| stockmark | ner-off | 300 | 0.0 | 0.0 | 0.0 | 98.9 | 3087.153 |
| stockmark | training | 300 | 63.7 | 85.6 | 79.6 | 12.8 | 31.433 |
| tab | default | 127 | 49.5 | 67.3 | 81.4 | 42.4 | 1.19 |
| tab | ner-off | 127 | 35.9 | 58.0 | 40.4 | 91.5 | 38.117 |
| tab | training | 127 | 34.6 | 79.4 | 43.1 | 49.4 | 1.298 |
| thai_pii | default | 300 | 26.5 | 62.6 | 87.3 | 17.5 | 20.258 |
| thai_pii | ner-off | 300 | 17.8 | 46.9 | 36.0 | 59.5 | 1447.267 |
| thai_pii | training | 300 | 41.7 | 82.2 | 77.2 | 32.8 | 23.067 |
| thainer | default | 300 | 34.1 | 65.6 | 68.6 | 24.6 | 20.976 |
| thainer | ner-off | 300 | 1.0 | 3.5 | 1.1 | 96.8 | 1382.707 |
| thainer | training | 300 | 40.3 | 91.8 | 69.3 | 24.2 | 22.103 |

### bayanat (default): by type and language

| | gold | partial R | partial P | strict F1 | leak |
|---|---|---|---|---|---|
| EMAIL | 73 | 100.0 | 100.0 | 100.0 | 0.0 |
| NUMBER | 402 | 19.9 | 27.4 | 0.0 | 15.7 |
| PERSON | 309 | 94.2 | 94.5 | 89.1 | 6.5 |
| PHONE | 288 | 96.2 | 33.2 | 45.5 | 0.0 |
| URL | 31 | 100.0 | 100.0 | 90.3 | 0.0 |
| ar-egyptian | 201 | 79.6 | 58.4 | 53.5 | 5.0 |
| ar-gulf | 289 | 65.4 | 44.3 | 48.3 | 4.5 |
| ar-mixed_en | 200 | 70.0 | 49.6 | 47.1 | 11.5 |
| ar-msa | 413 | 63.7 | 46.1 | 43.2 | 9.0 |

### gretel (default): by type and language

| | gold | partial R | partial P | strict F1 | leak |
|---|---|---|---|---|---|
| DATE | 2647 | 64.2 | 67.3 | 45.5 | 35.6 |
| EMAIL | 487 | 64.1 | 98.4 | 76.6 | 25.1 |
| IP | 113 | 85.8 | 94.3 | 88.6 | 1.8 |
| LOCATION | 1544 | 67.5 | 51.0 | 0.4 | 35.4 |
| NUMBER | 343 | 31.8 | 1.1 | 0.0 | 7.3 |
| ORGANISATION | 2107 | 65.3 | 51.8 | 39.9 | 34.9 |
| PERSON | 4769 | 43.9 | 88.2 | 49.4 | 46.6 |
| PHONE | 334 | 91.6 | 9.1 | 15.2 | 1.5 |
| de | 1739 | 59.5 | 23.2 | 20.2 | 38.1 |
| en | 1888 | 61.0 | 22.8 | 20.9 | 24.8 |
| es | 1717 | 54.7 | 22.6 | 18.7 | 34.0 |
| fr | 1579 | 60.4 | 25.0 | 21.2 | 38.6 |
| it | 1642 | 57.4 | 21.7 | 16.3 | 41.0 |
| nl | 1905 | 54.3 | 24.0 | 20.1 | 41.3 |
| sv | 1874 | 52.2 | 22.5 | 20.2 | 44.3 |

### iahlt (default): by type and language

| | gold | partial R | partial P | strict F1 | leak |
|---|---|---|---|---|---|
| LOCATION | 414 | 87.2 | 91.0 | 67.5 | 18.1 |
| ORGANISATION | 191 | 69.6 | 78.0 | 48.4 | 39.3 |
| PERSON | 229 | 92.1 | 93.4 | 80.9 | 8.7 |
| ar | 834 | 84.5 | 88.8 | 67.0 | 20.4 |

### multiconer (default): by type and language

| | gold | partial R | partial P | strict F1 | leak |
|---|---|---|---|---|---|
| LOCATION | 296 | 71.0 | 79.4 | 53.2 | 22.0 |
| ORGANISATION | 204 | 55.4 | 81.5 | 42.5 | 27.9 |
| PERSON | 325 | 86.2 | 73.7 | 63.4 | 10.8 |
| bn | 287 | 71.4 | 74.6 | 47.1 | 20.6 |
| hi | 258 | 71.7 | 78.9 | 63.7 | 18.2 |
| zh | 280 | 76.1 | 77.4 | 56.2 | 18.2 |

### naamapadam (default): by type and language

| | gold | partial R | partial P | strict F1 | leak |
|---|---|---|---|---|---|
| LOCATION | 510 | 65.9 | 82.0 | 56.2 | 17.8 |
| ORGANISATION | 405 | 49.6 | 78.5 | 43.6 | 23.9 |
| PERSON | 738 | 70.5 | 82.6 | 61.8 | 10.8 |
| bn | 516 | 54.1 | 80.2 | 45.4 | 23.8 |
| hi | 671 | 65.7 | 82.4 | 64.5 | 13.0 |
| ta | 466 | 72.3 | 81.7 | 54.4 | 12.4 |

### nemotron (default): by type and language

| | gold | partial R | partial P | strict F1 | leak |
|---|---|---|---|---|---|
| DATE | 285 | 90.5 | 97.0 | 91.3 | 2.1 |
| EMAIL | 161 | 98.8 | 100.0 | 99.1 | 6.8 |
| IP | 25 | 88.0 | 100.0 | 93.6 | 0.0 |
| LOCATION | 264 | 87.9 | 94.8 | 70.2 | 7.6 |
| NUMBER | 135 | 47.4 | 13.1 | 3.3 | 6.7 |
| ORGANISATION | 135 | 76.3 | 73.0 | 71.4 | 34.1 |
| PERSON | 350 | 84.9 | 98.0 | 89.1 | 16.9 |
| PHONE | 104 | 97.1 | 40.7 | 56.0 | 0.0 |
| URL | 94 | 96.8 | 98.9 | 54.8 | 50.0 |
| en | 1553 | 85.5 | 58.6 | 59.7 | 15.1 |

### openpii (default): by type and language

| | gold | partial R | partial P | strict F1 | leak |
|---|---|---|---|---|---|
| DATE | 7583 | 83.5 | 97.3 | 83.2 | 20.1 |
| EMAIL | 4837 | 98.2 | 99.8 | 93.5 | 0.6 |
| LOCATION | 7478 | 81.0 | 94.7 | 68.4 | 16.8 |
| NUMBER | 13280 | 26.9 | 45.9 | 16.2 | 20.9 |
| PERSON | 8489 | 84.3 | 96.2 | 68.5 | 10.6 |
| PHONE | 3632 | 96.1 | 32.1 | 42.1 | 0.0 |
| bg | 1549 | 66.4 | 66.9 | 57.3 | 15.1 |
| cs | 1501 | 70.0 | 73.0 | 50.1 | 13.4 |
| da | 1543 | 68.2 | 68.2 | 63.3 | 8.9 |
| de | 1378 | 74.6 | 68.0 | 59.3 | 6.4 |
| el | 1451 | 62.0 | 72.9 | 53.9 | 20.7 |
| en | 1561 | 68.2 | 69.5 | 57.7 | 13.0 |
| es | 1480 | 70.5 | 66.4 | 51.7 | 3.9 |
| et | 1499 | 69.2 | 69.6 | 56.3 | 13.5 |
| fi | 1597 | 74.7 | 73.0 | 59.3 | 13.3 |
| fr | 1559 | 67.5 | 72.2 | 56.8 | 14.5 |
| hr | 1453 | 71.0 | 70.2 | 61.9 | 12.0 |
| hu | 1564 | 68.5 | 74.9 | 62.7 | 18.8 |
| id | 1528 | 70.4 | 69.4 | 59.8 | 12.7 |
| it | 1627 | 71.7 | 65.5 | 53.4 | 8.0 |
| ja | 1548 | 63.0 | 64.0 | 45.0 | 16.9 |
| ko | 1388 | 61.2 | 57.2 | 35.5 | 19.6 |
| lt | 1595 | 70.4 | 67.9 | 55.3 | 10.9 |
| lv | 1485 | 68.4 | 70.9 | 50.9 | 13.9 |
| ms | 1536 | 74.7 | 75.9 | 59.1 | 12.8 |
| nl | 1545 | 69.5 | 71.0 | 52.8 | 5.9 |
| pl | 1515 | 69.2 | 78.4 | 50.9 | 17.0 |
| pt | 1503 | 81.2 | 75.1 | 56.4 | 5.0 |
| ro | 1484 | 68.8 | 72.5 | 63.1 | 14.5 |
| sk | 1483 | 69.9 | 78.1 | 53.8 | 17.5 |
| sl | 1479 | 65.8 | 69.9 | 60.9 | 13.2 |
| sr | 1422 | 64.5 | 64.8 | 56.2 | 12.9 |
| sv | 1413 | 70.0 | 71.3 | 57.1 | 9.2 |
| tl | 1622 | 70.5 | 70.0 | 60.1 | 16.7 |
| vi | 1615 | 67.1 | 67.7 | 56.1 | 15.5 |
| zh | 1376 | 67.8 | 71.6 | 50.9 | 18.2 |

### redact (default): by type and language

| | gold | partial R | partial P | strict F1 | leak |
|---|---|---|---|---|---|
| DATE | 1992 | 72.4 | 41.2 | 36.7 | 10.8 |
| EMAIL | 1481 | 75.7 | 97.2 | 83.4 | 9.7 |
| IP | 103 | 85.4 | 70.8 | 67.8 | 1.0 |
| LOCATION | 1574 | 76.4 | 65.0 | 21.8 | 28.9 |
| NUMBER | 937 | 35.2 | 5.1 | 0.5 | 15.2 |
| ORGANISATION | 957 | 66.8 | 32.2 | 31.3 | 23.9 |
| PERSON | 5529 | 74.9 | 86.6 | 63.2 | 19.5 |
| PHONE | 923 | 83.4 | 44.9 | 54.0 | 1.1 |
| ar | 551 | 70.6 | 48.8 | 44.3 | 23.1 |
| cs | 540 | 76.3 | 40.4 | 34.2 | 23.5 |
| da | 555 | 71.5 | 43.8 | 37.4 | 29.8 |
| de | 578 | 70.2 | 42.3 | 38.6 | 8.1 |
| en | 607 | 66.4 | 30.8 | 29.3 | 24.2 |
| es | 607 | 82.0 | 46.5 | 42.0 | 12.0 |
| fi | 505 | 75.6 | 36.9 | 31.5 | 17.0 |
| fr | 477 | 75.3 | 41.1 | 38.1 | 13.9 |
| fr_ca | 523 | 64.4 | 49.7 | 37.3 | 14.6 |
| he | 511 | 67.5 | 37.3 | 34.9 | 13.9 |
| hi | 468 | 79.9 | 39.1 | 36.4 | 20.8 |
| hu | 514 | 76.3 | 40.8 | 32.8 | 18.0 |
| it | 578 | 68.2 | 34.8 | 31.3 | 23.8 |
| ja | 565 | 71.0 | 40.8 | 39.1 | 20.1 |
| ko | 575 | 64.9 | 40.2 | 35.7 | 21.2 |
| nl | 579 | 70.0 | 44.8 | 37.4 | 19.9 |
| no | 478 | 66.7 | 34.0 | 34.1 | 15.1 |
| pt_br | 691 | 76.8 | 45.2 | 35.4 | 25.0 |
| pt_eu | 521 | 75.2 | 37.3 | 34.5 | 17.6 |
| ru | 435 | 75.2 | 49.0 | 37.6 | 16.1 |
| sv | 551 | 71.1 | 40.2 | 40.5 | 16.7 |
| th | 375 | 71.7 | 37.8 | 23.4 | 11.0 |
| tr | 577 | 70.5 | 43.9 | 33.0 | 18.9 |
| zh_cn | 632 | 72.9 | 47.5 | 45.8 | 6.8 |
| zh_tw | 503 | 73.8 | 43.7 | 44.6 | 10.9 |

### stockmark (default): by type and language

| | gold | partial R | partial P | strict F1 | leak |
|---|---|---|---|---|---|
| LOCATION | 192 | 72.4 | 85.9 | 61.0 | 16.7 |
| ORGANISATION | 283 | 75.6 | 88.5 | 60.8 | 14.8 |
| PERSON | 157 | 91.7 | 84.9 | 66.2 | 5.7 |
| ja | 632 | 78.6 | 86.7 | 62.3 | 13.1 |

### tab (default): by type and language

| | gold | partial R | partial P | strict F1 | leak |
|---|---|---|---|---|---|
| DATE | 2647 | 87.0 | 96.0 | 81.4 | 30.3 |
| LOCATION | 523 | 94.5 | 61.5 | 61.7 | 22.8 |
| NUMBER | 334 | 99.7 | 24.1 | 1.5 | 2.1 |
| ORGANISATION | 1947 | 64.9 | 79.4 | 59.6 | 66.3 |
| PERSON | 1078 | 85.2 | 97.5 | 13.5 | 21.2 |
| en | 6529 | 81.4 | 67.3 | 49.5 | 42.4 |

TAB masking recall (any type, macro over 12 annotators): all 64.1, direct 43.3, quasi 65.9.
Partial recall by start offset: 0-1000: 79.1 (1457), 1000-2000: 87.2 (1465), 2000-3000: 82.6 (1184), 3000-4000: 79.5 (792), 4000-5000: 74.2 (454), 5000-6000: 71.8 (333), 6000-7000: 80.8 (344), 7000-8000: 79.5 (263), 8000-9000: 88.5 (139), 9000+: 90.8 (98)

### thai_pii (default): by type and language

| | gold | partial R | partial P | strict F1 | leak |
|---|---|---|---|---|---|
| EMAIL | 61 | 100.0 | 100.0 | 100.0 | 0.0 |
| LOCATION | 207 | 87.0 | 97.7 | 0.9 | 55.1 |
| NUMBER | 139 | 74.8 | 40.8 | 0.0 | 0.0 |
| ORGANISATION | 66 | 81.8 | 58.1 | 60.4 | 16.7 |
| PERSON | 281 | 84.7 | 93.7 | 58.3 | 12.8 |
| PHONE | 167 | 100.0 | 60.1 | 67.9 | 0.0 |
| th | 921 | 87.3 | 62.6 | 26.5 | 17.5 |

### thainer (default): by type and language

| | gold | partial R | partial P | strict F1 | leak |
|---|---|---|---|---|---|
| DATE | 90 | 3.3 | 60.0 | 6.3 | 43.3 |
| LOCATION | 222 | 82.9 | 90.8 | 42.9 | 16.7 |
| NUMBER | 0 | 0.0 | 0.0 | 0.0 | - |
| ORGANISATION | 274 | 67.5 | 90.5 | 55.8 | 36.5 |
| PERSON | 156 | 89.1 | 96.5 | 21.4 | 16.0 |
| PHONE | 7 | 28.6 | 100.0 | 0.0 | 0.0 |
| URL | 3 | 100.0 | 100.0 | 66.7 | 33.3 |
| th | 752 | 68.6 | 65.6 | 34.1 | 24.6 |

## Cleaning: injected noise removed (fineweb2-clean)

| config | kind | injected | removed |
|---|---|---|---|
| default | emoji | 1000 | 5.1 |
| default | html | 1000 | 100.0 |
| default | mojibake | 1000 | 99.6 |
| default | url | 1000 | 100.0 |
| ner-off | emoji | 1000 | 0.4 |
| ner-off | html | 1000 | 100.0 |
| ner-off | mojibake | 1000 | 99.6 |
| ner-off | url | 1000 | 100.0 |
| training | emoji | 1000 | 4.9 |
| training | html | 1000 | 0.9 |
| training | mojibake | 1000 | 99.5 |
| training | url | 1000 | 100.0 |

## Speed (fineweb2)

| config | script | docs | docs/s | chars/s |
|---|---|---|---|---|
| default | all | 1000 | 2.646 | 6034.5 |
| default | arb_Arab (ARABIC) | 100 | 2.84 | 7718.3 |
| default | cmn_Hani (CJK) | 100 | 2.702 | 3263.9 |
| default | ell_Grek (GREEK) | 100 | 2.548 | 7708.2 |
| default | heb_Hebr (HEBREW) | 99 | 2.589 | 6958.0 |
| default | heb_Hebr (LATIN) | 1 | 0.697 | 5935.8 |
| default | hin_Deva (DEVANAGARI) | 98 | 3.575 | 7498.5 |
| default | hin_Deva (LATIN) | 2 | 2.341 | 6068.9 |
| default | jpn_Jpan (CJK) | 14 | 3.936 | 3449.3 |
| default | jpn_Jpan (HIRAGANA) | 78 | 2.652 | 3873.2 |
| default | jpn_Jpan (KATAKANA) | 7 | 2.479 | 3521.2 |
| default | jpn_Jpan (LATIN) | 1 | 1.472 | 2956.8 |
| default | kor_Hang (HANGUL) | 99 | 3.128 | 4228.5 |
| default | kor_Hang (LATIN) | 1 | 5.333 | 2917.3 |
| default | nld_Latn (LATIN) | 100 | 1.822 | 5114.9 |
| default | rus_Cyrl (CYRILLIC) | 99 | 2.822 | 8489.4 |
| default | rus_Cyrl (LATIN) | 1 | 2.447 | 6894.5 |
| default | tha_Thai (THAI) | 100 | 2.511 | 6183.6 |
| ner-off | all | 1000 | 100.088 | 228250.4 |
| ner-off | arb_Arab (ARABIC) | 100 | 78.677 | 213788.4 |
| ner-off | cmn_Hani (CJK) | 100 | 195.295 | 235879.1 |
| ner-off | ell_Grek (GREEK) | 100 | 91.266 | 276083.0 |
| ner-off | heb_Hebr (HEBREW) | 99 | 90.297 | 242708.9 |
| ner-off | heb_Hebr (LATIN) | 1 | 16.848 | 143441.7 |
| ner-off | hin_Deva (DEVANAGARI) | 98 | 96.326 | 202049.9 |
| ner-off | hin_Deva (LATIN) | 2 | 84.394 | 218791.7 |
| ner-off | jpn_Jpan (CJK) | 14 | 267.041 | 234042.6 |
| ner-off | jpn_Jpan (HIRAGANA) | 78 | 172.77 | 252349.9 |
| ner-off | jpn_Jpan (KATAKANA) | 7 | 171.826 | 244066.3 |
| ner-off | jpn_Jpan (LATIN) | 1 | 127.805 | 256631.6 |
| ner-off | kor_Hang (HANGUL) | 99 | 167.39 | 226316.0 |
| ner-off | kor_Hang (LATIN) | 1 | 416.449 | 227797.6 |
| ner-off | nld_Latn (LATIN) | 100 | 67.028 | 188123.7 |
| ner-off | rus_Cyrl (CYRILLIC) | 99 | 72.264 | 217389.5 |
| ner-off | rus_Cyrl (LATIN) | 1 | 45.74 | 128849.9 |
| ner-off | tha_Thai (THAI) | 100 | 112.917 | 278090.3 |
| training | all | 1000 | 2.772 | 6322.3 |
| training | arb_Arab (ARABIC) | 100 | 2.852 | 7750.7 |
| training | cmn_Hani (CJK) | 100 | 2.817 | 3402.8 |
| training | ell_Grek (GREEK) | 100 | 2.62 | 7927.0 |
| training | heb_Hebr (HEBREW) | 99 | 2.664 | 7161.8 |
| training | heb_Hebr (LATIN) | 1 | 0.763 | 6492.8 |
| training | hin_Deva (DEVANAGARI) | 98 | 3.798 | 7967.3 |
| training | hin_Deva (LATIN) | 2 | 2.437 | 6318.7 |
| training | jpn_Jpan (CJK) | 14 | 3.902 | 3419.4 |
| training | jpn_Jpan (HIRAGANA) | 78 | 2.746 | 4011.4 |
| training | jpn_Jpan (KATAKANA) | 7 | 2.374 | 3371.6 |
| training | jpn_Jpan (LATIN) | 1 | 1.408 | 2827.2 |
| training | kor_Hang (HANGUL) | 99 | 3.27 | 4421.3 |
| training | kor_Hang (LATIN) | 1 | 6.413 | 3507.7 |
| training | nld_Latn (LATIN) | 100 | 1.953 | 5482.2 |
| training | rus_Cyrl (CYRILLIC) | 99 | 2.98 | 8964.8 |
| training | rus_Cyrl (LATIN) | 1 | 1.873 | 5277.3 |
| training | tha_Thai (THAI) | 100 | 2.787 | 6864.4 |

## Language ID (wili)

Accuracy 95.1, macro 95.1 over 7200 in-set docs. Out-of-set docs given a language: 96.6 of 16300.

| script | docs | accuracy |
|---|---|---|
| ARABIC | 301 | 95.3 |
| ARMENIAN | 99 | 100.0 |
| BENGALI | 81 | 100.0 |
| CJK | 146 | 98.6 |
| CYRILLIC | 782 | 99.4 |
| DEVANAGARI | 193 | 96.9 |
| GEORGIAN | 99 | 100.0 |
| GREEK | 98 | 100.0 |
| GUJARATI | 94 | 100.0 |
| GURMUKHI | 98 | 100.0 |
| HANGUL | 99 | 100.0 |
| HEBREW | 100 | 100.0 |
| HIRAGANA | 34 | 100.0 |
| KATAKANA | 10 | 100.0 |
| LATIN | 4673 | 93.1 |
| TAMIL | 98 | 100.0 |
| TELUGU | 96 | 100.0 |
| THAI | 99 | 99.0 |
