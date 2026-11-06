# Final round 2c bundle (code 102e557ac8372062a8c8f6f92b26e6ca414e9436)

The long-document rerun finished on 2026-11-06 at 02:19 UTC. All four SCT RedactionBench cells and the
combined rescore completed without a failed cell. Individual failed documents remain scored as unmasked.

- RedactionBench inference ran again for sct-default, sct-gliner, sct-gliner-nvidia and sct-training.
- The other 212 cells were carried forward. SCT's 11 typed datasets retain b665956 placements;
  other entries retain 9cea2d0 placements. NUS-LT inference runs again during rescoring for every entry,
  because its prediction cache is deleted after scoring. All 11 typed datasets' views, completion counts,
  unplaced counts and sample hashes match the previous bundle exactly.
- `git diff --name-only b665956..102e557` contains `scripts/benchmark_release.py` (placement),
  `benchmarks/compare/adapters/sct_adapter.py` (debug counts only), and three test files.
  The runner, scorer and other adapters did not change.
- The carried SCT caches were re-keyed only after all 40 existing keys matched the old adapter and config.
  No NUS-LT prediction cache was carried. The rescore log lists only NUS-LT inference for the typed sets.
- Speed results are unchanged from round 2b. Placement does not run in the speed benchmark.
- `code-sha.txt` and `code-manifest.sha256` identify the frozen inference tree. `TABLES.md` is rendered
  locally using this branch's reporting changes, including low-completion flags.
- Only aggregate scores and provenance were copied. Server paths and hardware fields were scrubbed again.
  A read-only search on 2026-11-06 found no JSONL prediction caches in `nus_lt-test/` directories under round-2 runs.

## Previous round 2b bundle (code b6659566cd95a2f146368491854d2defbe3c9292)

Prepared on 2026-11-04 from the round-2 results of code 9cea2d0.

- Kept from 9cea2d0: the cells and offset checks of the 14 non-SCT entries, and their raw outputs. The rescore
  reuses a raw output only when its cache key (adapter code, `_protocol.py`, venv freeze, config, docs) still matches.
- Removed and re-run on b665956: sct-default, sct-gliner, sct-gliner-nvidia and sct-training (all 12 datasets).
  sct-training had 0 unplaced spans, but it runs the same `sct_adapter.py`, so its cache key changed and the rescore
  would run its inference again anyway. Re-running its cells keeps cells and rescore on the same code.
- Regenerated: `default/` (rescore), `TABLES.md`, `code-sha.txt`, `code-manifest.sha256`.
- `git diff --stat 9cea2d0..b665956` touches only: `benchmarks/compare/adapters/sct_adapter.py`,
  `scripts/benchmark_release.py` (SCT span placement, R2-1), `benchmarks/compare/report.py` (language-claim labels,
  skipping groups not run), `benchmarks/run.py` and `scripts/run_multilingual_eval.py` (pass the cleaner config; not
  used by the comparison), and two test files. Nothing that another entry runs or is scored with changed.
- New in this bundle: `speed-16t/` (16 threads, cpus 0-15, all 18 entries; peak RSS = /proc VmHWM in the same run,
  so there is no separate `speed-16t-rss/`) and `speed-gpu/` (`--device cuda`, 5 GB cap, GPU-capable entries).
  sct-training was skipped on GPU: its ONNX Runtime CUDA sessions do not respect the cap. A batch-8 OOM is recorded
  in `common_batch_errors`.
- Scrubbed before commit: server paths (rewritten to repo-relative paths or `<hf-hub-cache>` / `<out>`), the
  hardware fields of `run.json` and `speed.json`, and the card and other-process memory figures inside OOM messages.
