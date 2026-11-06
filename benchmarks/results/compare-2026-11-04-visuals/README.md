# SCT v0.8.2 versus other PII tools: charts

For X, use the [six redesigned social images and draft thread](social/README.md). The full charts below remain supporting evidence.

SCT prepares text datasets: it detects language, repairs text, handles PII, and scores and documents the result.

PII is one stage of that pipeline. It combines SCT’s own regex detection with language-based model routing. Configurable backends include ONNX, PyTorch, GLiNER, ensembles, and the beta Presidio/GLiNER integration. Users can bring supported models. This is backend flexibility within documented interfaces, not automatic integration of every tool benchmarked here.

## Findings

### Shared types: four tools have higher paired recall

PII-Tracer, BardsAI-v3, OPF and OpenMed have higher recall than SCT default, with recorded difference intervals below zero. Presidio and Presidio-full intervals cross zero.

[PNG](shared-recall.png) · [SVG](shared-recall.svg) · [Interactive HTML](shared-recall.html)

### Full taxonomy: SCT has stronger recall in most paired comparisons

SCT default has higher recall than 13 of the 14 non-SCT entries, with recorded difference intervals above zero. GLiNER2-Fastino’s interval crosses zero. This view also measures entity coverage.

[PNG](full-recall.png) · [SVG](full-recall.svg) · [Interactive HTML](full-recall.html)

### Precision and document exposure change the choice

Higher recall does not imply higher precision or fewer exposed documents. Several tools outperform SCT on these measures in the full-taxonomy synthetic comparisons.

[PNG](tradeoffs.png) · [SVG](tradeoffs.svg) · [Interactive HTML](tradeoffs.html)

## Method and limits

This report visualises the completed round-2 benchmark. The tested SCT version is v0.8.2. v0.8.3 is now available, but was not measured in this other-tools run. The separate version comparison cannot establish that these rankings remain unchanged.

The complete run has 18 entries, including four SCT configurations and 14 non-SCT entries, across 12 datasets. Each entry saw 20,817 documents; all 216 entry/dataset cells finished without failure. The charts here cover only the eight synthetic datasets. Real-language and long-document results remain in the source tables and are not pooled here.

Each plotted row compares SCT default with one other entry on the same documents: datasets clean for both and languages both claim. The set list can differ between rows. Do not read this as one universal leaderboard. Gold language is supplied to SCT, Presidio and Datafog; this does not test SCT language detection.

Shared types are EMAIL, PHONE and NUMBER, as specified by the protocol for the headline view. The full view includes the broader scored taxonomy and counts unsupported types as misses. DATE is excluded from both totals. Full-view recall rewards broader entity coverage, so the two views answer different questions.

Masking recall counts any overlap with a gold mention, regardless of predicted type. It does not require full coverage. Masking precision measures how many predicted mask spans overlap a gold mention. Document exposure is the share of eligible documents where at least one letter or digit of a gold mention remains visible. These measures are not privacy guarantees.

Points and intervals come from the archived summary JSON, without new inference. The dashed line is the SCT reference: zero means equal scores. The axis labels show which side favours SCT. Interactive charts show both tools’ scores on hover. Differences are SCT minus the other entry, in percentage points. A positive difference favours SCT for recall and precision; a negative difference favours SCT for document exposure. Whiskers are recorded paired bootstrap 95% intervals from 1,000 resamples (seed 13). They are not adjusted for multiple comparisons. An interval crossing zero does not establish a difference.

The default track uses vendor quick-start settings. SCT default runs on CPU. Other entries can run on CPU or a shared GPU with a memory cap; this report makes no speed ranking. The source report documents placement-code changes, completion rules, dataset provenance, licenses, and deviations from the original protocol. SCT’s RedactionBench placement fix is included in the archived run.

These charts compare detector configurations, not the value of every pipeline feature. A stronger PII recall result does not establish better text repair, data scoring, language detection, or complete redaction. Choose and validate the backend for the documents you process.

## Sources

- [Original report](https://github.com/rhnfzl/SqueakyCleanText/blob/main/benchmarks/results/compare-2026-11-04/README.md)
- [All tables](https://github.com/rhnfzl/SqueakyCleanText/blob/main/benchmarks/results/compare-2026-11-04/TABLES.md)
- Aggregate source: `../compare-2026-11-04/default/summary.json`.
- Source SHA-256: `8698487c90a6af1039b22c9237585ca8d17962675620e5889a5329a668c345b4`.
- `plot-data.json` preserves the exact selected values and paired set lists.

## Recreate

Chrome is required for PNG and SVG export. Script-only dependencies are pinned.

```bash
uv run --script scripts/plot_pii_comparison.py \
  benchmarks/results/compare-2026-11-04/default/summary.json \
  benchmarks/results/compare-2026-11-04-visuals
```
