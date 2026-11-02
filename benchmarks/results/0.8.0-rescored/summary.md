# SCT 0.8.0 benchmark

Commit `None` (dirty: None). AMD Ryzen Threadripper PRO 7985WX 64-Cores, 64 CPUs, GPUs: NVIDIA GeForce RTX 4090, 24564 MiB, 17869 MiB, 560.35.05, NVIDIA GeForce RTX 4090, 24564 MiB, 19167 MiB, 560.35.05, NVIDIA GeForce RTX 4090, 24564 MiB, 18981 MiB, 560.35.05, NVIDIA GeForce RTX 4090, 24564 MiB, 16502 MiB, 560.35.05.
Device `cpu`, ONNX threads 32, workers 2, seed 13. Command: `python -m benchmarks.run --suite speed --threads 32 --workers 2 --seed 13 --configs default training --out benchmarks/results/0.8.0-rescored`

Percentages. Strict = exact span + type. Partial = overlap + type. Masked R = overlap, any type. Leak = gold PII value (or a 4+ char word of it) still in the output; lower is better.

## PII

| dataset | config | docs | strict F1 | partial P | partial R | masked R | leak | docs/s |
|---|---|---|---|---|---|---|---|---|
| bayanat | default | 845 | 47.1 | 48.4 | 68.2 | 97.8 | 7.5 | 55.52 |
| bayanat | training | 845 | 54.0 | 54.1 | 61.5 | 97.9 | 7.6 | 58.817 |
| gretel | default | 2100 | 19.7 | 23.1 | 57.0 | 67.8 | 37.8 | 3.681 |
| gretel | training | 2100 | 31.1 | 49.7 | 48.4 | 55.8 | 42.6 | 3.566 |
| iahlt | default | 300 | 67.0 | 88.8 | 84.5 | 88.1 | 20.5 | 18.719 |
| iahlt | training | 300 | 68.1 | 89.2 | 85.6 | 89.0 | 19.2 | 19.276 |
| multiconer | default | 900 | 55.4 | 76.9 | 73.1 | 77.3 | 25.8 | 44.455 |
| multiconer | training | 900 | 56.9 | 75.9 | 74.3 | 78.8 | 24.4 | 46.898 |
| naamapadam | default | 900 | 55.9 | 81.6 | 63.9 | 67.5 | 22.2 | 36.059 |
| naamapadam | training | 900 | 67.7 | 90.1 | 79.6 | 83.5 | 20.4 | 35.135 |
| nemotron | default | 300 | 59.7 | 58.6 | 85.5 | 92.3 | 15.1 | 6.721 |
| nemotron | training | 300 | 67.9 | 83.4 | 67.7 | 75.3 | 18.9 | 6.415 |
| openpii | default | 9000 | 55.6 | 70.2 | 69.2 | 88.2 | 13.1 | 12.058 |
| openpii | training | 9000 | 48.2 | 73.9 | 48.9 | 65.7 | 21.0 | 12.43 |
| redact | default | 1000 | 36.2 | 41.1 | 72.1 | 78.1 | 18.1 | 2.497 |
| redact | training | 1000 | 54.0 | 69.0 | 72.5 | 78.7 | 21.6 | 2.187 |
| stockmark | default | 300 | 62.3 | 86.7 | 78.6 | 84.0 | 13.1 | 28.6 |
| stockmark | training | 300 | 63.7 | 85.6 | 79.6 | 84.3 | 12.8 | 31.431 |
| tab | default | 127 | 49.5 | 67.3 | 81.4 | 88.1 | 42.4 | 1.177 |
| tab | training | 127 | 34.6 | 79.4 | 43.1 | 47.9 | 49.4 | 1.315 |
| thai_pii | default | 300 | 26.5 | 62.6 | 87.3 | 94.0 | 18.9 | 20.028 |
| thai_pii | training | 300 | 41.7 | 82.2 | 77.2 | 83.3 | 33.3 | 23.281 |
| thainer | default | 300 | 34.1 | 65.6 | 68.6 | 81.0 | 20.5 | 20.678 |
| thainer | training | 300 | 40.3 | 91.8 | 69.3 | 71.7 | 19.9 | 21.905 |

### bayanat (default): by type and language

| | gold | partial R | masked R | partial P | strict F1 | leak |
|---|---|---|---|---|---|---|
| EMAIL | 73 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0 |
| NUMBER | 402 | 19.9 | 100.0 | 27.4 | 0.0 | 15.7 |
| PERSON | 309 | 94.2 | 94.2 | 94.5 | 89.1 | 6.5 |
| PHONE | 288 | 96.2 | 97.9 | 33.2 | 45.5 | 0.0 |
| URL | 31 | 100.0 | 100.0 | 100.0 | 90.3 | 0.0 |
| ar-egyptian | 201 | 79.6 | 98.5 | 58.4 | 53.5 | 5.0 |
| ar-gulf | 289 | 65.4 | 97.9 | 44.3 | 48.3 | 4.5 |
| ar-mixed_en | 200 | 70.0 | 98.0 | 49.6 | 47.1 | 11.5 |
| ar-msa | 413 | 63.7 | 97.3 | 46.1 | 43.2 | 9.0 |

### gretel (default): by type and language

| | gold | partial R | masked R | partial P | strict F1 | leak |
|---|---|---|---|---|---|---|
| DATE | 2647 | 64.2 | 75.5 | 67.3 | 45.5 | 35.6 |
| EMAIL | 487 | 64.1 | 64.3 | 98.4 | 76.6 | 25.1 |
| IP | 113 | 85.8 | 89.4 | 94.3 | 88.6 | 1.8 |
| LOCATION | 1544 | 67.5 | 90.9 | 51.0 | 0.4 | 35.4 |
| NUMBER | 343 | 31.8 | 92.7 | 1.1 | 0.0 | 7.3 |
| ORGANISATION | 2107 | 65.3 | 67.6 | 51.8 | 39.9 | 34.9 |
| PERSON | 4769 | 43.9 | 52.1 | 88.2 | 49.4 | 46.6 |
| PHONE | 334 | 91.6 | 95.2 | 9.1 | 15.2 | 1.5 |
| de | 1739 | 59.5 | 69.0 | 23.2 | 20.2 | 38.1 |
| en | 1888 | 61.0 | 73.2 | 22.8 | 20.9 | 24.8 |
| es | 1717 | 54.7 | 66.7 | 22.6 | 18.7 | 34.0 |
| fr | 1579 | 60.4 | 70.4 | 25.0 | 21.2 | 38.6 |
| it | 1642 | 57.4 | 70.1 | 21.7 | 16.3 | 41.0 |
| nl | 1905 | 54.3 | 62.4 | 24.0 | 20.1 | 41.3 |
| sv | 1874 | 52.2 | 63.2 | 22.5 | 20.2 | 44.3 |

### iahlt (default): by type and language

| | gold | partial R | masked R | partial P | strict F1 | leak |
|---|---|---|---|---|---|---|
| LOCATION | 414 | 87.2 | 91.1 | 91.0 | 67.5 | 18.1 |
| ORGANISATION | 191 | 69.6 | 76.4 | 78.0 | 48.4 | 39.8 |
| PERSON | 229 | 92.1 | 92.6 | 93.4 | 80.9 | 8.7 |
| ar | 834 | 84.5 | 88.1 | 88.8 | 67.0 | 20.5 |

### multiconer (default): by type and language

| | gold | partial R | masked R | partial P | strict F1 | leak |
|---|---|---|---|---|---|---|
| LOCATION | 296 | 71.0 | 73.7 | 79.4 | 53.2 | 27.4 |
| ORGANISATION | 204 | 55.4 | 67.2 | 81.5 | 42.5 | 41.2 |
| PERSON | 325 | 86.2 | 87.1 | 73.7 | 63.4 | 14.8 |
| bn | 287 | 71.4 | 75.3 | 74.6 | 47.1 | 32.8 |
| hi | 258 | 71.7 | 77.5 | 78.9 | 63.7 | 26.4 |
| zh | 280 | 76.1 | 79.3 | 77.4 | 56.2 | 18.2 |

### naamapadam (default): by type and language

| | gold | partial R | masked R | partial P | strict F1 | leak |
|---|---|---|---|---|---|---|
| LOCATION | 510 | 65.9 | 69.0 | 82.0 | 56.2 | 22.4 |
| ORGANISATION | 405 | 49.6 | 58.8 | 78.5 | 43.6 | 38.5 |
| PERSON | 738 | 70.5 | 71.3 | 82.6 | 61.8 | 13.1 |
| bn | 516 | 54.1 | 57.2 | 80.2 | 45.4 | 28.7 |
| hi | 671 | 65.7 | 68.8 | 82.4 | 64.5 | 19.7 |
| ta | 466 | 72.3 | 77.0 | 81.7 | 54.4 | 18.7 |

### nemotron (default): by type and language

| | gold | partial R | masked R | partial P | strict F1 | leak |
|---|---|---|---|---|---|---|
| DATE | 285 | 90.5 | 96.1 | 97.0 | 91.3 | 2.1 |
| EMAIL | 161 | 98.8 | 98.8 | 100.0 | 99.1 | 6.8 |
| IP | 25 | 88.0 | 88.0 | 100.0 | 93.6 | 0.0 |
| LOCATION | 264 | 87.9 | 92.0 | 94.8 | 70.2 | 7.6 |
| NUMBER | 135 | 47.4 | 100.0 | 13.1 | 3.3 | 6.7 |
| ORGANISATION | 135 | 76.3 | 76.3 | 73.0 | 71.4 | 34.1 |
| PERSON | 350 | 84.9 | 86.6 | 98.0 | 89.1 | 16.9 |
| PHONE | 104 | 97.1 | 99.0 | 40.7 | 56.0 | 0.0 |
| URL | 94 | 96.8 | 96.8 | 98.9 | 54.8 | 50.0 |
| en | 1553 | 85.5 | 92.3 | 58.6 | 59.7 | 15.1 |

### openpii (default): by type and language

| | gold | partial R | masked R | partial P | strict F1 | leak |
|---|---|---|---|---|---|---|
| DATE | 7583 | 83.5 | 100.0 | 97.3 | 83.2 | 20.1 |
| EMAIL | 4837 | 98.2 | 98.2 | 99.8 | 93.5 | 0.6 |
| LOCATION | 7478 | 81.0 | 84.4 | 94.7 | 68.4 | 16.8 |
| NUMBER | 13280 | 26.9 | 79.0 | 45.9 | 16.2 | 20.9 |
| PERSON | 8489 | 84.3 | 84.5 | 96.2 | 68.5 | 10.6 |
| PHONE | 3632 | 96.1 | 99.9 | 32.1 | 42.1 | 0.0 |
| bg | 1549 | 66.4 | 88.7 | 66.9 | 57.3 | 15.1 |
| cs | 1501 | 70.0 | 88.8 | 73.0 | 50.1 | 13.4 |
| da | 1543 | 68.2 | 90.9 | 68.2 | 63.3 | 8.9 |
| de | 1378 | 74.6 | 93.8 | 68.0 | 59.3 | 6.4 |
| el | 1451 | 62.0 | 80.6 | 72.9 | 53.9 | 20.7 |
| en | 1561 | 68.2 | 88.7 | 69.5 | 57.7 | 13.0 |
| es | 1480 | 70.5 | 89.8 | 66.4 | 51.7 | 3.9 |
| et | 1499 | 69.2 | 91.1 | 69.6 | 56.3 | 13.5 |
| fi | 1597 | 74.7 | 91.6 | 73.0 | 59.3 | 13.3 |
| fr | 1559 | 67.5 | 87.7 | 72.2 | 56.8 | 14.5 |
| hr | 1453 | 71.0 | 90.6 | 70.2 | 61.9 | 12.0 |
| hu | 1564 | 68.5 | 84.0 | 74.9 | 62.7 | 18.8 |
| id | 1528 | 70.4 | 88.7 | 69.4 | 59.8 | 12.7 |
| it | 1627 | 71.7 | 91.6 | 65.5 | 53.4 | 8.0 |
| ja | 1548 | 63.0 | 85.9 | 64.0 | 45.0 | 16.9 |
| ko | 1388 | 61.2 | 83.1 | 57.2 | 35.5 | 19.6 |
| lt | 1595 | 70.4 | 91.6 | 67.9 | 55.3 | 10.9 |
| lv | 1485 | 68.4 | 89.0 | 70.9 | 50.9 | 13.9 |
| ms | 1536 | 74.7 | 88.0 | 75.9 | 59.1 | 12.8 |
| nl | 1545 | 69.5 | 88.0 | 71.0 | 52.8 | 5.9 |
| pl | 1515 | 69.2 | 83.6 | 78.4 | 50.9 | 17.0 |
| pt | 1503 | 81.2 | 94.6 | 75.1 | 56.4 | 5.0 |
| ro | 1484 | 68.8 | 86.7 | 72.5 | 63.1 | 14.5 |
| sk | 1483 | 69.9 | 83.7 | 78.1 | 53.8 | 17.5 |
| sl | 1479 | 65.8 | 86.2 | 69.9 | 60.9 | 13.2 |
| sr | 1422 | 64.5 | 87.8 | 64.8 | 56.2 | 12.9 |
| sv | 1413 | 70.0 | 91.2 | 71.3 | 57.1 | 9.2 |
| tl | 1622 | 70.5 | 86.3 | 70.0 | 60.1 | 16.7 |
| vi | 1615 | 67.1 | 87.2 | 67.7 | 56.1 | 15.5 |
| zh | 1376 | 67.8 | 84.7 | 71.6 | 50.9 | 18.2 |

### redact (default): by type and language

| | gold | partial R | masked R | partial P | strict F1 | leak |
|---|---|---|---|---|---|---|
| DATE | 1992 | 72.4 | 78.8 | 41.2 | 36.7 | 11.0 |
| EMAIL | 1481 | 75.7 | 76.4 | 97.2 | 83.4 | 9.7 |
| IP | 103 | 85.4 | 91.3 | 70.8 | 67.8 | 1.0 |
| LOCATION | 1574 | 76.4 | 81.4 | 65.0 | 21.8 | 29.5 |
| NUMBER | 937 | 35.2 | 80.7 | 5.1 | 0.5 | 15.2 |
| ORGANISATION | 957 | 66.8 | 72.6 | 32.2 | 31.3 | 24.8 |
| PERSON | 5529 | 74.9 | 75.9 | 86.6 | 63.2 | 19.5 |
| PHONE | 923 | 83.4 | 88.2 | 44.9 | 54.0 | 1.1 |
| ar | 551 | 70.6 | 79.1 | 48.8 | 44.3 | 23.1 |
| cs | 540 | 76.3 | 79.1 | 40.4 | 34.2 | 23.5 |
| da | 555 | 71.5 | 76.0 | 43.8 | 37.4 | 29.8 |
| de | 578 | 70.2 | 74.1 | 42.3 | 38.6 | 8.1 |
| en | 607 | 66.4 | 71.8 | 30.8 | 29.3 | 24.2 |
| es | 607 | 82.0 | 86.8 | 46.5 | 42.0 | 12.0 |
| fi | 505 | 75.6 | 81.6 | 36.9 | 31.5 | 17.0 |
| fr | 477 | 75.3 | 78.2 | 41.1 | 38.1 | 13.9 |
| fr_ca | 523 | 64.4 | 69.6 | 49.7 | 37.3 | 14.6 |
| he | 511 | 67.5 | 73.8 | 37.3 | 34.9 | 13.9 |
| hi | 468 | 79.9 | 82.7 | 39.1 | 36.4 | 23.9 |
| hu | 514 | 76.3 | 82.7 | 40.8 | 32.8 | 18.0 |
| it | 578 | 68.2 | 73.4 | 34.8 | 31.3 | 23.8 |
| ja | 565 | 71.0 | 75.4 | 40.8 | 39.1 | 20.1 |
| ko | 575 | 64.9 | 68.2 | 40.2 | 35.7 | 21.2 |
| nl | 579 | 70.0 | 73.8 | 44.8 | 37.4 | 19.9 |
| no | 478 | 66.7 | 77.8 | 34.0 | 34.1 | 15.1 |
| pt_br | 691 | 76.8 | 81.0 | 45.2 | 35.4 | 25.0 |
| pt_eu | 521 | 75.2 | 86.4 | 37.3 | 34.5 | 17.6 |
| ru | 435 | 75.2 | 84.1 | 49.0 | 37.6 | 16.1 |
| sv | 551 | 71.1 | 79.7 | 40.2 | 40.5 | 16.7 |
| th | 375 | 71.7 | 82.1 | 37.8 | 23.4 | 13.2 |
| tr | 577 | 70.5 | 77.0 | 43.9 | 33.0 | 18.9 |
| zh_cn | 632 | 72.9 | 80.4 | 47.5 | 45.8 | 6.8 |
| zh_tw | 503 | 73.8 | 81.5 | 43.7 | 44.6 | 10.9 |

### stockmark (default): by type and language

| | gold | partial R | masked R | partial P | strict F1 | leak |
|---|---|---|---|---|---|---|
| LOCATION | 192 | 72.4 | 81.8 | 85.9 | 61.0 | 16.7 |
| ORGANISATION | 283 | 75.6 | 80.2 | 88.5 | 60.8 | 14.8 |
| PERSON | 157 | 91.7 | 93.6 | 84.9 | 66.2 | 5.7 |
| ja | 632 | 78.6 | 84.0 | 86.7 | 62.3 | 13.1 |

### tab (default): by type and language

| | gold | partial R | masked R | partial P | strict F1 | leak |
|---|---|---|---|---|---|---|
| DATE | 2647 | 87.0 | 91.6 | 96.0 | 81.4 | 30.3 |
| LOCATION | 523 | 94.5 | 95.0 | 61.5 | 61.7 | 22.8 |
| NUMBER | 334 | 99.7 | 99.7 | 24.1 | 1.5 | 2.1 |
| ORGANISATION | 1947 | 64.9 | 80.7 | 79.4 | 59.6 | 66.3 |
| PERSON | 1078 | 85.2 | 85.8 | 97.5 | 13.5 | 21.2 |
| en | 6529 | 81.4 | 88.1 | 67.3 | 49.5 | 42.4 |

TAB masking recall (any type, macro over 12 annotators): all 64.1, direct 43.3, quasi 65.9.
Partial recall by start offset: 0-1000: 79.1 (1457), 1000-2000: 87.2 (1465), 2000-3000: 82.6 (1184), 3000-4000: 79.5 (792), 4000-5000: 74.2 (454), 5000-6000: 71.8 (333), 6000-7000: 80.8 (344), 7000-8000: 79.5 (263), 8000-9000: 88.5 (139), 9000+: 90.8 (98)

### thai_pii (default): by type and language

| | gold | partial R | masked R | partial P | strict F1 | leak |
|---|---|---|---|---|---|---|
| EMAIL | 61 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0 |
| LOCATION | 207 | 87.0 | 96.1 | 97.7 | 0.9 | 61.4 |
| NUMBER | 139 | 74.8 | 100.0 | 40.8 | 0.0 | 0.0 |
| ORGANISATION | 66 | 81.8 | 83.3 | 58.1 | 60.4 | 16.7 |
| PERSON | 281 | 84.7 | 87.2 | 93.7 | 58.3 | 12.8 |
| PHONE | 167 | 100.0 | 100.0 | 60.1 | 67.9 | 0.0 |
| th | 921 | 87.3 | 94.0 | 62.6 | 26.5 | 18.9 |

### thainer (default): by type and language

| | gold | partial R | masked R | partial P | strict F1 | leak |
|---|---|---|---|---|---|---|
| DATE | 90 | 3.3 | 75.6 | 60.0 | 6.3 | 45.6 |
| LOCATION | 222 | 82.9 | 87.4 | 90.8 | 42.9 | 12.2 |
| NUMBER | 0 | 0.0 | 0.0 | 0.0 | 0.0 | - |
| ORGANISATION | 274 | 67.5 | 71.9 | 90.5 | 55.8 | 31.4 |
| PERSON | 156 | 89.1 | 89.7 | 96.5 | 21.4 | 14.1 |
| PHONE | 7 | 28.6 | 100.0 | 100.0 | 0.0 | 0.0 |
| URL | 3 | 100.0 | 100.0 | 100.0 | 66.7 | 33.3 |
| th | 752 | 68.6 | 81.0 | 65.6 | 34.1 | 20.5 |

## Speed (fineweb2)

| config | script | docs | docs/s | chars/s |
|---|---|---|---|---|
| default | all | 1000 | 2.62 | 5975.7 |
| default | arb_Arab (ARABIC) | 100 | 2.795 | 7595.7 |
| default | cmn_Hani (CJK) | 100 | 2.658 | 3209.9 |
| default | ell_Grek (GREEK) | 100 | 2.506 | 7581.5 |
| default | heb_Hebr (HEBREW) | 99 | 2.55 | 6853.5 |
| default | heb_Hebr (LATIN) | 1 | 0.7 | 5955.6 |
| default | hin_Deva (DEVANAGARI) | 98 | 3.538 | 7422.2 |
| default | hin_Deva (LATIN) | 2 | 2.282 | 5916.6 |
| default | jpn_Jpan (CJK) | 14 | 3.915 | 3431.2 |
| default | jpn_Jpan (HIRAGANA) | 78 | 2.632 | 3844.9 |
| default | jpn_Jpan (KATAKANA) | 7 | 2.489 | 3535.0 |
| default | jpn_Jpan (LATIN) | 1 | 1.42 | 2851.2 |
| default | kor_Hang (HANGUL) | 99 | 3.099 | 4190.2 |
| default | kor_Hang (LATIN) | 1 | 6.693 | 3661.1 |
| default | nld_Latn (LATIN) | 100 | 1.822 | 5112.6 |
| default | rus_Cyrl (CYRILLIC) | 99 | 2.805 | 8438.2 |
| default | rus_Cyrl (LATIN) | 1 | 2.356 | 6637.7 |
| default | tha_Thai (THAI) | 100 | 2.496 | 6146.9 |
| training | all | 1000 | 2.779 | 6337.3 |
| training | arb_Arab (ARABIC) | 100 | 2.873 | 7805.6 |
| training | cmn_Hani (CJK) | 100 | 2.818 | 3403.2 |
| training | ell_Grek (GREEK) | 100 | 2.604 | 7878.3 |
| training | heb_Hebr (HEBREW) | 99 | 2.686 | 7219.4 |
| training | heb_Hebr (LATIN) | 1 | 0.79 | 6723.7 |
| training | hin_Deva (DEVANAGARI) | 98 | 3.794 | 7957.5 |
| training | hin_Deva (LATIN) | 2 | 2.457 | 6370.4 |
| training | jpn_Jpan (CJK) | 14 | 3.939 | 3451.9 |
| training | jpn_Jpan (HIRAGANA) | 78 | 2.758 | 4029.0 |
| training | jpn_Jpan (KATAKANA) | 7 | 2.39 | 3395.0 |
| training | jpn_Jpan (LATIN) | 1 | 1.726 | 3466.1 |
| training | kor_Hang (HANGUL) | 99 | 3.289 | 4446.3 |
| training | kor_Hang (LATIN) | 1 | 7.249 | 3965.4 |
| training | nld_Latn (LATIN) | 100 | 1.952 | 5479.6 |
| training | rus_Cyrl (CYRILLIC) | 99 | 2.972 | 8939.1 |
| training | rus_Cyrl (LATIN) | 1 | 2.382 | 6709.0 |
| training | tha_Thai (THAI) | 100 | 2.79 | 6870.8 |
