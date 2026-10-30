# Release benchmark: SCT 0.7.1

Commit `ef4a651936117ce11e36f5ca7e3e8653c3c2104c`, macOS-27.2-arm64-arm-64bit-Mach-O, Python 3.13.12.
Dataset `release_benchmark.jsonl`: 30 documents, 84 entities. Exact label + span match.

Speed (default config, single process): **6.923 docs/s**, mean 144.445 ms, p95 758.89 ms.
Leaks: **0 / 67** PII values still in output (partial: 0).

## By label

| | TP | FP | FN | P | R | F1 |
|---|---|---|---|---|---|---|
| **overall** | 73 | 11 | 11 | 0.87 | 0.87 | 0.87 |
| EMAIL | 6 | 1 | 1 | 0.86 | 0.86 | 0.86 |
| IP | 0 | 0 | 9 | 0.00 | 0.00 | 0.00 |
| IP_ADDRESS | 0 | 9 | 0 | 0.00 | 0.00 | 0.00 |
| LOCATION | 5 | 1 | 1 | 0.83 | 0.83 | 0.83 |
| NUMBER | 8 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| ORGANISATION | 2 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| PERSON | 25 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| PHONE_NUMBER | 16 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| URL | 2 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| YEAR | 9 | 0 | 0 | 1.00 | 1.00 | 1.00 |

## By language

| | TP | FP | FN | P | R | F1 |
|---|---|---|---|---|---|---|
| bn | 2 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| de | 9 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| en | 24 | 9 | 9 | 0.73 | 0.73 | 0.73 |
| es | 8 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| hi | 10 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| ja | 4 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| nl | 8 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| ta | 2 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| zh | 6 | 2 | 2 | 0.75 | 0.75 | 0.75 |

## Leaks

