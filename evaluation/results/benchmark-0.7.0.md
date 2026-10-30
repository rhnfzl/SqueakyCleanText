# Release benchmark: SCT 0.7.0

Commit `c3a44461cc365b4aa7d1bcf9f83aabb502d6b1a3`, macOS-27.2-arm64-arm-64bit-Mach-O, Python 3.13.12.
Dataset `release_benchmark.jsonl`: 30 documents, 84 entities. Exact label + span match.

Speed (default config, single process): **7.708 docs/s**, mean 129.73 ms, p95 844.744 ms.
Leaks: **6 / 67** PII values still in output (partial: 4).

## By label

| | TP | FP | FN | P | R | F1 |
|---|---|---|---|---|---|---|
| **overall** | 52 | 40 | 32 | 0.57 | 0.62 | 0.59 |
| EMAIL | 6 | 1 | 1 | 0.86 | 0.86 | 0.86 |
| IP | 0 | 0 | 9 | 0.00 | 0.00 | 0.00 |
| LOCATION | 5 | 1 | 1 | 0.83 | 0.83 | 0.83 |
| MISC | 0 | 1 | 0 | 0.00 | 0.00 | 0.00 |
| NUMBER | 5 | 33 | 3 | 0.13 | 0.62 | 0.22 |
| ORGANISATION | 2 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| PERSON | 25 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| PHONE_NUMBER | 1 | 2 | 15 | 0.33 | 0.06 | 0.11 |
| URL | 2 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| YEAR | 6 | 2 | 3 | 0.75 | 0.67 | 0.71 |

## By language

| | TP | FP | FN | P | R | F1 |
|---|---|---|---|---|---|---|
| bn | 1 | 0 | 1 | 1.00 | 0.50 | 0.67 |
| de | 8 | 3 | 1 | 0.73 | 0.89 | 0.80 |
| en | 18 | 24 | 15 | 0.43 | 0.55 | 0.48 |
| es | 6 | 2 | 2 | 0.75 | 0.75 | 0.75 |
| hi | 6 | 2 | 4 | 0.75 | 0.60 | 0.67 |
| ja | 3 | 0 | 1 | 1.00 | 0.75 | 0.86 |
| nl | 6 | 7 | 2 | 0.46 | 0.75 | 0.57 |
| ta | 1 | 0 | 1 | 1.00 | 0.50 | 0.67 |
| zh | 3 | 2 | 5 | 0.60 | 0.38 | 0.46 |

## Leaks

- `en-glued-001` PHONE_NUMBER: 0612345678
- `zh-glued-001` PHONE_NUMBER: 13812345678
- `ja-glued-001` PHONE_NUMBER: 09012345678
- `hi-glued-001` PHONE_NUMBER: 9876543210
- `bn-glued-001` PHONE_NUMBER: 9876543210
- `ta-glued-001` PHONE_NUMBER: 9876543210
- `en-ip-001` IP (partial): 192.168.1.10
- `en-ip-002` IP (partial): fd12:3456:789a::1
- `en-ip-002` IP (partial): fe80::1
- `en-ip-003` IP (partial): 172.16.5.20
