# Release benchmark: 0.7.0 vs 0.7.1

Same machine (macOS 27.2 arm64, Python 3.13.12), same dataset (`release_benchmark.jsonl`,
30 documents, 84 entities). Full numbers: `benchmark-0.7.0.md`, `benchmark-0.7.1.md`.

| | 0.7.0 | 0.7.1 |
|---|---|---|
| Overall recall (exact label + span) | 0.62 (52/84) | 0.87 (73/84) |
| Overall precision | 0.57 | 0.87 |
| PHONE_NUMBER recall | 0.06 (1/16) | 1.00 (16/16) |
| YEAR recall | 0.67 | 1.00 |
| NUMBER precision | 0.13 (33 FP) | 1.00 (0 FP) |
| Full leaks | 6 / 67 | 0 / 67 |
| Partial leaks | 2 | 0 |
| Speed, docs/s | 7.708 | 6.923 (-10.2%) |
| Mean / p95 ms | 129.7 / 844.7 | 144.4 / 758.9 |

Recall did not drop for any label or language. EMAIL (6/7) and LOCATION (5/6) are
unchanged; every other label and every language went up or stayed at 1.00.

IP shows 0/9 in both runs, for different reasons. 0.7.0 left the 6 IPv4 values out of the
leak list only because the number and year rules broke them up, and left IPv6 partly in the
output. 0.7.1 replaces all 9 IPs at the exact expected spans, but the reversible placeholder
type is `IP_ADDRESS` (the same name the synthetic replacer used in 0.7.0) while the dataset
says `IP`, so they score as 9 FN + 9 FP. With that label counted as IP, overall recall is
82/84 (0.98). The dataset is left unchanged so both runs score the same file.

Speed: the single 0.7.1 run is 10.2% below the 0.7.0 run, just over the 10% line, so it was
checked. It is machine noise, not a code change:

- Back-to-back runs of 0.7.0 and 0.7.1 code on this machine drift from about 6.9 to 5.1
  docs/s for both versions as the machine warms, so one run per release cannot resolve 10%.
- Median of 7 runs per document, 0.7.0 code then 0.7.1 code: total 3361 ms vs 3273 ms
  (0.7.1 about 3% faster). Long documents got faster (`zh-long-001` 953 to 840 ms); a few
  short contact documents got 10 to 30 ms slower.
- Profiling `es-contact-001` (20 runs): the same 60 ONNX NER calls in both versions, ONNX
  time 3.74 s vs 3.92 s, and all Python-side regex and aggregation work under 10 ms. The
  difference is inside ONNX Runtime, not in the changed code.
