# Release benchmark: SCT 0.8.0

Commit `678a59b93e5bb43e9b30b60261050532e0d7c361`, macOS-27.2-arm64-arm-64bit-Mach-O, Python 3.13.12.
Dataset `release_benchmark.jsonl`: 30 documents, 84 entities. Exact label + span match.

Speed (default config, single process): **7.195 docs/s**, mean 138.985 ms, p95 623.978 ms.
Leaks: **0 / 67** PII values still in output (partial: 0).

## By label

| | TP | FP | FN | P | R | F1 |
|---|---|---|---|---|---|---|
| **overall** | 83 | 1 | 1 | 0.99 | 0.99 | 0.99 |
| EMAIL | 7 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| IP | 9 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| LOCATION | 5 | 1 | 1 | 0.83 | 0.83 | 0.83 |
| NUMBER | 8 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| ORGANISATION | 2 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| PERSON | 25 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| PHONE | 16 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| URL | 2 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| YEAR | 9 | 0 | 0 | 1.00 | 1.00 | 1.00 |

## By language

| | TP | FP | FN | P | R | F1 |
|---|---|---|---|---|---|---|
| bn | 2 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| de | 9 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| en | 33 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| es | 8 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| hi | 10 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| ja | 4 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| nl | 8 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| ta | 2 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| zh | 7 | 1 | 1 | 0.88 | 0.88 | 0.88 |

## Leaks

