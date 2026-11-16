# SCT v0.10.0 comparison

This bundle compares released SCT v0.10.0 with fourteen pinned historical tool configurations. It contains aggregate scores, not benchmark text.

SCT uses its default detector selection, supplied gold languages and reversible scoring output. This does not measure every optional detector or adaptive routing configuration.

The test sample uses up to 300 records per language, seed 13 and inference batch size 1. The twelve fixed datasets cover 20,817 documents. Samples were seen in earlier evaluation; this run does not tune on them. They are not newly held-out data.

The social charts use eight synthetic datasets. Three real-data datasets and the separate untyped legal group remain in the full bundle.

Shared labels are EMAIL, PHONE and NUMBER. Full-label scoring treats unsupported types as misses. DATE is excluded from headline scoring.

Metrics average across languages within datasets, then across datasets. Each pair matches documents and claimed languages on datasets marked clean for both configurations. Pair subsets differ.

Intervals use 1,000 paired group-bootstrap draws without multiple-comparison adjustment. Recall credits any overlap. Document exposure uses reconstructed source-span masks, among documents containing scored mentions.

Reuse accepts 154 other-tool cells only after cache, source, sample and reconstructed-score validation. NUS-LT predictions run fresh and are deleted after scoring. Failed documents count as unmasked.

The frozen harness commit is 102e557ac8372062a8c8f6f92b26e6ca414e9436. SCT's adapter receives a compatible positional worker argument; package pins change to 0.10.0. Provenance records modified and unchanged source hashes.

Native timing compares the direct public API in SCT v0.8.2 and v0.10.0. Three alternating fresh-process pairs use the same 1,000 FineWeb-2 documents across 10 languages. Dependencies and model revisions match.

Timing uses automatic language detection, CPU execution, 16 ONNX Runtime threads and 2 workers. Throughput uses batch 8. Single-document latency follows 20 warm-up documents. Memory is process peak resident memory in MiB.

Ranges describe three repeats on a shared host. They are not confidence intervals or proof of a code-only speed change.

RedactionBench retains 134 unplaced SCT spans, versus 39 in the archived v0.8.2 run. The three largest increases reproduce cached output and show no tested glued-quote normalization mismatch. Other cases are not exhaustively diagnosed. This separate legal group is outside the social charts.

`validation.json` records acceptance checks and file hashes. `summary.json` contains full aggregate comparisons. `native-speed.json` contains all six timing trials. `provenance.json` records runtimes, models and source hashes. Per-dataset files retain source credits and licence notes.

Regenerate charts from the repository root with Python 3.12+, Matplotlib 3.11.2 and Pillow 13.0.0:

```sh
python -m scripts.plot_pii_social_release benchmarks/results/compare-2026-11-12-v010/default /tmp/sct-v010-charts
```

The renderer requires a matching validation manifest. It exports PNG, SVG, phone previews, grayscale previews and exact chart values.
