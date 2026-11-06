# SCT v0.7.1 versus v0.8.3

Released v0.8.3 improves strict span F1 on all 12 tested PII/long-document datasets.
Two checks worsen: unsupported-language false answers and TAB direct-identifier full coverage.

## Results

Default configuration plus scoring overrides. Values are percentages. Higher F1/recall and lower leak heuristic are better.

| Dataset | Docs | Strict F1 0.7.1 / 0.8.3 | Any-overlap masking recall | Leak heuristic |
|---|---:|---:|---:|---:|
| bayanat | 845 | 38.67 / 63.59 | 87.31 / 98.55 | 7.71 / 2.72 |
| gretel | 2,100 | 13.32 / 34.83 | 65.18 / 71.66 | 35.34 / 29.61 |
| iahlt | 300 | 64.31 / 69.63 | 78.90 / 94.96 | 19.66 / 12.71 |
| multiconer | 900 | 49.20 / 57.48 | 62.42 / 82.67 | 23.52 / 19.64 |
| naamapadam | 900 | 49.66 / 66.25 | 67.09 / 86.39 | 21.29 / 17.97 |
| nemotron | 300 | 42.29 / 82.35 | 93.75 / 98.58 | 14.98 / 7.02 |
| openpii | 9,000 | 34.63 / 69.35 | 80.95 / 90.57 | 10.30 / 10.14 |
| redact | 1,000 | 25.00 / 53.18 | 73.43 / 94.93 | 16.10 / 9.23 |
| stockmark | 300 | 59.91 / 65.82 | 66.77 / 89.40 | 12.50 / 6.49 |
| tab | 127 | 22.53 / 60.40 | 87.12 / 90.92 | 42.27 / 12.80 |
| thai_pii | 300 | 24.40 / 50.92 | 90.66 / 95.33 | 15.85 / 12.70 |
| thainer | 300 | 26.68 / 35.72 | 73.40 / 80.19 | 19.18 / 15.11 |

## Speed

| Configuration | v0.7.1 docs/s | v0.8.3 docs/s | Ratio |
|---|---:|---:|---:|
| default | 1.174 | 2.669 | 2.27x |
| training | 1.739 | 2.764 | 1.59x |

These are measured software stacks with different runtime versions. Read the compatibility limit below.

## Charts

- quality: [PNG](charts/quality.png), [SVG](charts/quality.svg), [interactive HTML](charts/quality.html)
- speed: [PNG](charts/speed.png), [SVG](charts/speed.svg), [interactive HTML](charts/speed.html)
- leaks: [PNG](charts/leaks.png), [SVG](charts/leaks.svg), [interactive HTML](charts/leaks.html)
- limits: [PNG](charts/limits.png), [SVG](charts/limits.svg), [interactive HTML](charts/limits.html)

## Method and limits

The run uses the released v0.8.3 source at 8afef61c74ea738a2c904374776dea34d9c1c39e. Both versions use its frozen evaluator. Sample SHA-256 values match across versions. There are 16,372 PII/long documents across 12 datasets, plus WiLI language detection.

Quality uses 16 runtime threads, two workers, seed 13, and the same CPU allocation. Both configurations override check_to_ascii_unicode=False and replacement_mode='reversible' for scoring. These are not untouched shipped defaults. The old version has no native training preset; the wrapper applies supported fields from the released preset and adapts API differences.

Speed uses 1,000 FineWeb2 documents (2,280,498 characters), 32 runtime threads, two workers, warmup, and three alternating version rounds. The CPU allocation is the same. No GPU is used. Medians and observed ranges are not confidence intervals.

Runtime versions differ: ONNX Runtime 1.23.2 versus 1.31.1 and PyArrow 26.0.0 versus 26.0.1. The compatibility-assisted legacy runtime is below SCT 0.7.1's declared ONNX Runtime minimum (>=1.24.2). Results describe these measured stacks; they do not isolate SCT code changes. A dependency-matched control would be needed for that claim.

Any-overlap masking recall counts a gold span when any part is masked. Partial masking can leave identifiers visible. The leak heuristic excludes DATE and checks literal values or words with at least four characters. Repeated words can inflate it; scripts without spaces can hide partial leaks. Neither metric proves privacy.

WiLI known-language accuracy rises from 5.47% to 95.13% on 7,200 documents. The old detector offers only four languages, so this also measures expanded language scope. Unsupported-language false answers worsen from 66.48% to 96.61% on 16,300 other documents.

TAB uses a separate full-alphanumeric-coverage metric, averaged across 12 annotators. Direct-identifier coverage falls from 41.51% to 40.22% (1,612 mentions). Quasi-identifier coverage rises from 30.35% to 66.44% (19,197 mentions). This metric differs from generic overlap recall.

The comparison has no statistical significance test or peak-memory measurement. It does not rerun the earlier comparison against other tools. No raw text or predictions are included; JSON files contain aggregates and sample hashes. Environment paths and hardware identifiers are removed. Freeze files preserve captured provenance, not install-ready requirements.

## Reproduce the charts

Chrome must be installed for static exports. Plotly and Kaleido are script-only dependencies.

```bash
uv run --script scripts/plot_version_comparison.py benchmarks/results/compare-0.7.1-0.8.3
```

The launcher and compatibility wrapper are in `scripts/watch_release_comparison.py` and `scripts/release_comparison_version.py`.
Dataset sources, revisions, licenses, model revisions and settings are recorded in the per-run aggregate JSON.
