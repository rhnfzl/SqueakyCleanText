# Social images for the other-tools benchmark

SCT prepares text datasets. PII handling is one configurable stage, combining its own rules with supported model backends.

These six images replace the dense report plots in the X draft. Each PNG is 1200 × 1350 pixels, with labels at least 36 pixels before scaling. Review the images at phone width and in the X composer before publication.

The result cards measure SCT v0.8.2 default on paired synthetic subsets. They do not measure v0.8.3 or the whole pipeline. Shared EMAIL/PHONE/NUMBER recall remains separate from the full taxonomy. DATE is excluded. Any overlap counts toward recall. Bootstrap intervals are unadjusted for multiple comparisons.

Precision and exposure show one named comparison, SCT default versus PII-Tracer. Scores are dataset averages. Each pair uses matching documents and languages both tools claim; different pairs can use different subsets.

## Images

1. **The processing pipeline:** [PNG](01-pipeline.drawio.png), [editable source](01-pipeline.drawio), [SVG](01-pipeline.drawio.svg).
2. **Configurable PII routing:** [PNG](02-pii-routing.drawio.png), [editable source](02-pii-routing.drawio), [SVG](02-pii-routing.drawio.svg).
3. **Shared-type recall:** [PNG](03-shared-recall.drawio.png), [editable source](03-shared-recall.drawio), [SVG](03-shared-recall.drawio.svg).
4. **Full-taxonomy recall:** [PNG](04-full-recall.drawio.png), [editable source](04-full-recall.drawio), [SVG](04-full-recall.drawio.svg).
5. **Precision: one paired example:** [PNG](05-precision.drawio.png), [editable source](05-precision.drawio), [SVG](05-precision.drawio.svg).
6. **Exposed text: the same pair:** [PNG](06-exposed-text.drawio.png), [editable source](06-exposed-text.drawio), [SVG](06-exposed-text.drawio.svg).

Use one PNG per post. The first two explain the product; the next four show results and limits. Keep all fourteen tools in the linked [full comparison](../README.md).

`thread.json` is the seven-post draft; nothing has been posted.

`cards.json` contains image descriptions, the source aggregate hash and derived full-taxonomy counts. The generator reads the archived summary through the existing validated extractor. No inference or benchmark scores are changed.

## Recreate

The generator uses only the Python standard library. PNG/SVG export requires the draw.io desktop CLI. Run from the repository root:

```bash
python -m scripts.build_pii_social_cards \
  benchmarks/results/compare-2026-11-04/default/summary.json \
  benchmarks/results/compare-2026-11-04-visuals/social

for assetfile in benchmarks/results/compare-2026-11-04-visuals/social/*.drawio; do
  drawio -x -f png -e --width 1200 -o "${assetfile}.png" "$assetfile"
  drawio -x -f svg -e -o "${assetfile}.svg" "$assetfile"
done
```

On macOS, the CLI can be `/Applications/draw.io.app/Contents/MacOS/draw.io`. Sources are retained, and exports embed the editable diagram. Image descriptions supplement the visible labels. This set has not been published or tested in the native X composer.
