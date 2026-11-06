# Release benchmark: 0.7.1 vs 0.8.0

0.8.0 is the speed release (plus features, see CHANGELOG). 0.7.1 is tag `v0.7.1`; 0.8.0 is branch `perf/v0.8.0-speed`.
Correctness and output parity were re-run at `ef3f245` (`sct.__version__` 0.8.0); the Linux
speed runs were made earlier at `c88e3ca`. Default config unless a row says otherwise.
A final re-run at `371a873` (all 0.8.0 features and fixes) gave the same TP/FP/FN, leaks
and 30 of 30 byte-identical outputs as `ef3f245`.
The final check at `5f56bec` gives the same TP/FP/FN and leaks; 11 of 30 outputs differ from
`ef3f245` only by a sentence-final `.` or a comma after a word (`，`, written `,`) that is now kept (isolated-symbol
fix, CHANGELOG).
The last re-run at `3a01635` (final 0.8.0) gives the same TP/FP/FN and leaks again. Outputs differ
from `5f56bec` only by the new names and defaults: reversible placeholders `<PHONE_n>` / `<IP_n>`
instead of `<PHONE_NUMBER_n>` / `<IP_ADDRESS_n>` (18 documents) and `，` kept in Chinese (2 documents,
`fix_character_width=False`). The dataset label `PHONE_NUMBER` is now `PHONE`, so its sha256 changed.
`benchmark-0.8.0.json` / `.md` are from that run; its speed line (7.5 docs/s) was measured on a
loaded machine (load average 9 to 16; paired runs of the code before and after the last commit
were within noise of each other) and is not comparable to the 10.3 docs/s of `5f56bec`.
The final Mac check at `4b7e9f9` (styled-character folding, same-script widening, per-language
synthetic fakes, NER reading placeholder-mode tags) gives the same TP/FP/FN, leaks and 30 of 30
byte-identical outputs as `3a01635`; only the speed line changed (9.6 docs/s, mean 104 ms, same
machine at lower load, not a code speed-up claim).
The last Mac check at `678a59b` (fuzz rounds, Codex rounds 19-25) gives 83/1/1 (was 82/2/2) and
0 / 67 leaks; 29 of 30 outputs are byte-identical to `4b7e9f9`. The one change is `zh-long-001`:
`邮箱是<EMAIL_0>` instead of `<EMAIL_0>`, because an email local part now ends where glued CJK
prose meets a Latin letter (fuzz round 3, `d6e047c`), so the predicted EMAIL span matches the gold
span. The speed line (7.2 docs/s) is a single loaded-machine run, not a code slow-down claim.
`benchmark-0.8.0.json` / `.md` are now from that run.

## Headline

- Correctness on the release benchmark: overall 83/1/1 (P = R = 0.99; 0.7.1 82/2/2), one
  more zh EMAIL span exact (`zh-long-001`, fuzz round 3); every other label, language and
  domain is equal. 0 / 67 leaks, 0 partial leaks.
- `process_batch` on 16 Linux cores: **4.2x** on short documents (3.69 -> 15.55 docs/s, 16
  workers) and **3.2x** on long documents (0.65 -> 2.08 docs/s).
- Single-document `process`: unchanged within noise (1.10x on 16 cores, 0.95x on 4 cores,
  1.09x on the Mac run).

## Correctness (Mac)

macOS 27.2 arm64, Python 3.13.12, onnxruntime 1.31.1,
lingua-language-detector 2.2.0. `python -m scripts.benchmark_release` with the committed
correctness settings (`replacement_mode="reversible"`, `check_to_ascii_unicode=False`),
dataset `release_benchmark.jsonl` (same records as the 0.7.1 run, 30 documents, 84 entities; the
`PHONE_NUMBER` label is `PHONE` since `28ed3a2`).
Full numbers: `benchmark-0.8.0.md` / `.json`.

| | 0.7.1 | 0.8.0 |
|---|---|---|
| Overall TP / FP / FN | 82 / 2 / 2 | 83 / 1 / 1 |
| Labels with a TP/FP/FN change | | EMAIL 6/1/1 -> 7/0/0 |
| Languages with a TP/FP/FN change | | zh 6/2/2 -> 7/1/1 |
| Full / partial leaks | 0 / 0 | 0 / 0 |
| Reversible `lm_text`, 30 documents | | 29 identical to the committed 0.7.1 outputs; `zh-glued-001` fixed (below) |
| Speed, docs/s (single run, default config) | 6.923 | 7.462 |
| Mean / p95 ms | 144.4 / 758.9 | 134.0 / 663.6 |

The one reversible difference is `zh-glued-001`: 0.7.1 gives the garbled
`他住<LOCATION_1>OCATION_0>。`, 0.8.0 gives `他住<LOCATION_0>。`. Two overlapping LOCATION
detections are now merged before replacement (CHANGELOG). Its spans score the same, so the
TP/FP/FN counts do not move.

The 0.7.1 run was made with lingua 2.1.x; 0.8.0 needs lingua 2.2. The other outputs are
equal anyway, so the lingua change does not move any document of this set.

## Output parity, default config

Default `TextCleanerConfig()`, `lm_text` of every dataset document, 0.7.1 code (temporary
`git worktree` of `v0.7.1`) and 0.8.0 code in the same venv (Mac), through both `process`
and `process_batch`, over the 30 documents plus the 4 long documents (every 8 documents joined
into one). 32 of 34 are byte-identical; `process` and `process_batch` agree with each other in
both versions. The two differences are `zh-long-001` and the long document that contains it
(one merged pair there), and both are equal once adjacent repeated placeholders are merged.

The one difference, `zh-long-001` (longer than one NER chunk):

```
0.7.1: <PERSON> ,<PERSON> ,<PERSON> ,<PERSON> <PERSON> ,<PERSON> <PERSON> ,<PERSON> ,<PERSON> <EMAIL>
0.8.0: <PERSON> ,<PERSON> ,<PERSON> ,<PERSON> ,<PERSON> ,<PERSON> ,<PERSON> <EMAIL>
```

Two `<PERSON> <PERSON>` pairs that sat across a chunk boundary become one `<PERSON>`. This is
the documented 0.8.0 change (CHANGELOG, "replace once per document": two same-type entities
separated only by spaces at a chunk boundary merge, as they already did inside a chunk). The
detected entities are the same: the reversible outputs used for correctness are identical.

The Linux check over the 30 documents plus the 4 long joined documents (onnxruntime 1.31.1,
4 threads) finds the same: 2 of 34 differ (`zh-long-001` and one long document), and both are
equal once adjacent repeated placeholders are merged.

## Speed (Linux)

Machine: shared Linux workstation,
Ubuntu 24.04.4, kernel 6.11, CPU only (GPUs not used). Python 3.13, `uv sync --frozen
--no-dev` of each version's own `uv.lock` (both resolve lingua 2.2.0, huggingface-hub 1.32.0),
then onnxruntime 1.31.1 in both venvs (see "Linux model load" below). Same default models and
revisions, `HF_HUB_OFFLINE=1`, `CUDA_VISIBLE_DEVICES=""`, `nice -n 10`.

Thread caps: `taskset -c 0-15` (16 cores) or `taskset -c 0-3` (4 cores), and every ONNX
session gets `intra_op_num_threads` = the core count. 0.7.1 has no thread option, so the
harness sets it for both versions by wrapping `onnxruntime.SessionOptions` (0.8.0's
`onnx_intra_op_threads`, when passed, is assigned later and wins). This is needed: with the
ONNX Runtime default (unset) the session pins one thread to every physical core of the
machine and ignores `taskset` (observed: threads on CPUs 0-127, about 5800% CPU under
`taskset -c 0-3`).

Commands (dataset `evaluation/release_benchmark.jsonl`; harness `speed.py` below):

```
SCT_CAP=16 nice -n 10 taskset -c 0-15 <venv>/bin/python speed.py single <dataset>
SCT_CAP=16 nice -n 10 taskset -c 0-15 <venv>/bin/python speed.py batch <dataset> --docs 120 --workers B
SCT_CAP=16 nice -n 10 taskset -c 0-15 <venv>/bin/python speed.py batch <dataset> --long --docs 16 --workers B
```

`single`: one warmup pass, then 3 timed passes of `process` over the 30 documents.
`batch`: `process_batch(docs, batch_size=B)` (B = worker threads) over the dataset cycled to
120 documents, or over the long set (every 8 documents joined into one, 4 long documents,
cycled to 16), one warmup call, 3 timed calls. Cell = median docs/s (the three runs). Output
hashes are equal across all runs of one version and mode.

| CPUs (taskset, ORT threads) | Run | 0.7.1 docs/s | 0.8.0 docs/s | 0.8.0 / 0.7.1 |
|---|---|---|---|---|
| 0-15 (16) | `single` | 4.52 (4.42, 4.61, 4.52) | 4.96 (4.77, 4.96, 5.03) | 1.10x |
| 0-3 (4) | `single` | 3.84 (3.84, 3.93, 3.82) | 3.65 (3.64, 3.65, 3.65) | 0.95x |
| 0-15 (16) | `batch --docs 120 --workers 1` | 4.45 (4.45, 4.40, 4.50) | 13.57 (13.57, 13.59, 13.41) | 3.05x |
| 0-15 (16) | `batch --docs 120 --workers 4` | 3.54 (3.54, 3.46, 3.62) | 14.35 (14.42, 14.32, 14.35) | 4.06x |
| 0-15 (16) | `batch --docs 120 --workers 16` | 3.69 (3.61, 3.69, 3.78) | 15.55 (15.17, 15.61, 15.55) | 4.21x |
| 0-15 (16) | `batch --long --docs 16 --workers 1` | 1.03 (1.08, 1.03, 1.03) | 1.98 (1.99, 1.95, 1.98) | 1.93x |
| 0-15 (16) | `batch --long --docs 16 --workers 4` | 0.64 (0.64, 0.64, 0.63) | 2.07 (2.04, 2.07, 2.07) | 3.22x |
| 0-15 (16) | `batch --long --docs 16 --workers 16` | 0.65 (0.65, 0.65, 0.64) | 2.08 (2.08, 2.11, 2.08) | 3.23x |
| 0-3 (4) | `batch --docs 120 --workers 4` | 3.10 (3.10, 3.10, 3.11) | 4.81 (4.81, 4.86, 4.81) | 1.55x |
| 0-15 (16) | `batch --docs 120 --workers 16 --ner-batch 1` | n/a | 10.31 (10.31, 10.01, 10.36) |  |
| 0-15 (16) | `batch --long --docs 16 --workers 16 --ner-batch 1` | n/a | 1.82 (1.84, 1.82, 1.75) |  |
| 0-15 (16) | `batch --docs 120 --workers 16 --intra 1` | n/a | 4.48 (4.48, 4.48, 4.53) |  |
| 0-15 (16) | `batch --long --docs 16 --workers 16 --intra 1` | n/a | 0.57 (0.57, 0.57, 0.57) |  |
| 0-15 (16) | `batch --docs 120 --workers 4 --intra 4` | n/a | 7.46 (7.41, 7.50, 7.46) |  |
| 0-15 (16) | `batch --long --docs 16 --workers 4 --intra 4` | n/a | 1.04 (1.04, 1.04, 1.03) |  |

The 0.8.0-only rows use the default `ner_batch_size=8` unless `--ner-batch 1` is shown;
`--intra T` passes `onnx_intra_op_threads=T`.

Notes:

- The gain is in `process_batch`: in 0.7.1 the NER calls of all worker threads queue on one
  lock per model, so more workers make it slower (4.45 -> 3.69 docs/s from 1 to 16 workers);
  0.8.0 runs one batched NER pass over all documents, which already gives 3.0x with one
  worker. `process` of one document is not batched across documents and stays the same.
- Batching helps here: `ner_batch_size=8` vs `1` with 16 workers is 15.55 vs 10.31 docs/s
  (short) and 2.08 vs 1.82 (long). The CHANGELOG measured the opposite on long documents with
  many workers on another 16-core CPU (2.38 vs 2.66) and recommends `ner_batch_size=1` there;
  that advice does not hold on this machine.
- Fewer ONNX threads per session is slower here, not faster: 16 workers x 1 thread is 4.48
  docs/s and 4 workers x 4 threads 7.46, against 15.55 with 16 threads. The CHANGELOG tip to
  set `onnx_intra_op_threads` so workers x threads is about the core count does not help with
  the 0.8.0 batched pass, which runs model calls as a few large batches rather than one call
  per worker. Its real use is capping CPU use (see thread caps above).
- Uncapped runs (ONNX Runtime default threads, about 64 cores in use) gave the same picture:
  short 16 workers 11.1 -> 25.8 docs/s (2.3x), long 2.14 -> 3.42 (1.6x), single 10.9 -> 11.4.
  They are not in the table because they broke the core cap on a shared box.

### Linux model load

With the locked huggingface-hub 1.32.0, the Linux cache stores `model.onnx` and
`model.onnx_data` as blobs in different directories, and onnxruntime (1.30.0 locked, and
1.31.1) refuses the external data file ("External data path escapes model directory"). This
fails for the XLM-RoBERTa-large default models in both 0.7.1 and 0.8.0, so it is not a 0.8.0
regression. For the benchmark the two files were hard-linked into each snapshot directory.
The Mac venv (huggingface-hub 2.4.0) is not affected. Cause: hub 1.32.0 stores each blob in
its own directory (`hub/blobs/<xx>/<sha256>`), and onnxruntime only allows external data in
the real directory of `model.onnx`. Fixed later in 0.8.0: SCT hard-links both files into one
directory in that layout (see CHANGELOG); the timings above were measured before the fix.

### Harness

`speed.py` (run with each version's venv; `SCT_CAP` sets the ONNX thread default):

```python
import argparse, hashlib, json, os, statistics, time
import onnxruntime as ort

CAP = int(os.environ.get("SCT_CAP", "0"))
if CAP:
    _SessionOptions = ort.SessionOptions
    class _Capped(_SessionOptions):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.intra_op_num_threads = CAP
    ort.SessionOptions = _Capped

from sct.config import TextCleanerConfig
from sct.evaluation import load_evaluation_jsonl
from sct.sct import TextCleaner
# parse mode, dataset, --workers, --docs, --long, --ner-batch, --intra, --runs;
# long = " ".join of every 8 texts; build TextCleaner(cfg=TextCleanerConfig(**overrides));
# single: warmup pass, then time `process` over all texts per run;
# batch: warmup `process_batch(docs, batch_size=workers)`, then time it per run;
# print median docs/s, the runs and a hash of the outputs.
```
