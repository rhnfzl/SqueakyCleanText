# README diagrams

`draw_diagrams.py` (stdlib only) writes three diagrams to `resources/`, each as `.excalidraw` and `.svg`:

- `sct_overview`: what SCT takes in, the five stages, what comes out, and where to run it.
- `sct_flow`: the pipeline in the order `sct/sct.py` runs it; dashed steps are off by default.
- `sct_pii_modes`: one sentence in the three PII modes (real SCT 0.8.1 output).

Rebuild from the repo root:

```sh
python scripts/diagrams/draw_diagrams.py   # .excalidraw + .svg; fails if text overflows a box or is too small
scripts/diagrams/render.sh                 # .png at 2x with headless Chrome (CHROME=... to override the path)
```

The layout lives in code, so edit `draw_diagrams.py`, not the `.excalidraw` files. When the pipeline,
a default or an output changes, update the diagram and the text alternative under "How it works" in
`README.md`. The `sct_pii_modes` sentences come from
`TextCleaner(cfg=preset_config("training", replacement_mode=mode)).process(...)`; rerun it after NER
or synthetic changes and paste the new output.
