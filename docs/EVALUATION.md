# Evaluation

Three tools measure SCT. None of them is part of the cleaning path.

- The PII evaluation harness and model candidates (below).
- The release benchmark: `python -m scripts.benchmark_release` scores exact spans and
  leaks on `evaluation/release_benchmark.jsonl` (results in `evaluation/results/`).
- Score columns, data cards and `sct-report`: see [SCORE.md](SCORE.md).

## PII evaluation harness

Model changes need ground-truth privacy evaluation, not only backend agreement.
The evaluation harness reports exact entity metrics by label, language, domain,
and named slice.

```sh
python -m scripts.evaluate_pii \
  --dataset evaluation/pii_starter.jsonl \
  --predictions path/to/model-predictions.jsonl \
  --model model-name-and-revision \
  --output evaluation/results/model-name.json
```

The included starter corpus is synthetic and only checks the evaluation path.
See [`evaluation/README.md`](https://github.com/rhnfzl/SqueakyCleanText/blob/main/evaluation/README.md) for schemas and release rules.

## Model candidates

Pinned candidates live in `sct.model_candidates`. Run one candidate with:

```sh
python -m scripts.benchmark_models \
  --dataset evaluation/pii_starter.jsonl \
  --candidate gliner2-pii \
  --output evaluation/results/gliner2-pii.json
```

## Results

The multilingual set `evaluation/multilingual_pii.jsonl` (220 synthetic records, en, nl, de,
es, zh, ja, th, ar, hi, bn, ta) scores the default ensemble at 0.93 recall / 0.95 precision
in 0.8.0, ahead of the GLiNER multilingual models; see
[docs/MULTILINGUAL_EVAL.md](MULTILINGUAL_EVAL.md). The release benchmark
(`scripts/benchmark_release.py`, 30 documents in 9 languages) gives 0.99 exact-span F1 and
0 of 67 leaks. Throughput by hardware: [docs/PERFORMANCE.md](PERFORMANCE.md).

Published third-party results are reference data only. `assess_candidate`
requires local recall, exact-span F1, latency, and ONNX parity before adoption.
