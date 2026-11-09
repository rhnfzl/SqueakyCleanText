# Performance and hardware

Measured with SqueakyCleanText 0.7.1 and 0.8.0, default config (full multilingual NER
ensemble), on the release benchmark set (`evaluation/release_benchmark.jsonl`). "Short" docs
are the benchmark documents (up to about 900 characters); "long" docs join 8 of them.
Numbers are docs per second, median of 3 runs. Output text is identical on CPU and GPU.

Machine: Linux x86_64 workstation with a high-end GPU,
onnxruntime 1.30/1.31 (CPU) and onnxruntime-gpu 1.29 (CUDA build not recorded). CPU runs were pinned to 16 or
64 physical cores. Correctness and the 16-core CPU commands:
`evaluation/results/comparison-0.7.1-0.8.0.md`.

## Which hardware to use

- **Small jobs: any CPU.** One document at a time (`process`) runs at about 4 to 5 docs/s
  (about 200 ms each), on 4 cores or 64. More cores do not help a single document.
- **Bulk on CPU: use `process_batch` with `max_workers` about 16.** Keep
  `ner_batch_size=8` (default) and let ONNX use the cores you give it.
- **Bulk and speed: use a GPU.** With a high-end GPU, single documents are about 17x faster
  and batches about 4.5x faster than 64 CPU cores.
- **Containers or `taskset`:** set `onnx_intra_op_threads` to your core count. Without it,
  ONNX Runtime starts one thread per core of the whole machine.

## CPU

| Workload | 0.7.1, 16 cores | 0.7.1, 64 cores | 0.8.0, 16 cores | 0.8.0, 64 cores |
|---|---|---|---|---|
| `process`, one doc at a time | 4.5 | 4.3 | 5.0 | 4.7 |
| `process_batch`, short docs, 16 workers | 3.7 | 3.3 | 15.5 | 29.2 |
| `process_batch`, long docs, 16 workers | 0.65 | 0.58 | 2.1 | 3.6 |

0.8.0 tuning on 64 cores (64 workers): `ner_batch_size=8` (default) was best for short docs
(24.0 docs/s; 1 gave 17.1, 32 gave 14.9). Large `ner_batch_size` values hurt on CPU. More
than about 16 workers lowered throughput.

## GPU

Install:

```sh
pip install "squeakycleantext[gpu]"
# onnxruntime and onnxruntime-gpu install the same package, so keep only the GPU one
pip uninstall -y onnxruntime onnxruntime-gpu
pip install "onnxruntime-gpu[cuda,cudnn]"
python -c "import onnxruntime as ort; print(ort.get_available_providers())"  # lists CUDAExecutionProvider
```

Upgrading or reinstalling SqueakyCleanText can bring `onnxruntime` back; repeat the uninstall
and install steps if it does. The onnxruntime-gpu build must match your NVIDIA driver: recent PyPI wheels need CUDA 13
(driver 580 or newer); for CUDA 12 use the onnxruntime CUDA 12 package index. Call
`onnxruntime.preload_dlls()` before creating the cleaner when CUDA comes from pip packages.

If CUDA cannot be loaded, ONNX Runtime runs on CPU. SCT logs a warning
("runs on CPU: CUDA was requested") and adds `cuda_fallback` to `metadata['warnings']` of
every result. `metadata['provenance']['models']` names the provider each model ran on.

With `ner_device=None`, GLiNER in torch follows torch: it uses the GPU when torch can, even with the
CPU `onnxruntime` package. Torch and GLiNER models also run on CPU when torch cannot use the GPU, for
example with `CUDA_VISIBLE_DEVICES` empty. With `ner_backend` `gliner`, `ensemble_onnx` or
`presidio_gliner`, if ONNX Runtime has CUDA and GLiNER runs in torch, SCT logs a warning and adds
`cuda_fallback`. GLiNER in ONNX mode (`gliner_onnx=True`, as in `ner_mode="pii"`) does not depend on
torch here, so nothing changes for it. `ner_backend` `torch` and `ensemble_torch` log nothing and only
show provider `cpu`. With an explicit `ner_device="cuda"`, torch and GLiNER torch models fail to load
instead.

Pick the GPU with `cuda_device_id` (default 0) and force a device with `ner_device`
(`None` auto, `"cpu"`, `"cuda"`, or `"mps"` for the Apple GPU, see below). `onnx_cuda_provider_options` passes extra
`CUDAExecutionProvider` options, for example `{"gpu_mem_limit": 8 << 30}`.

| Setup | Workload | 0.7.1 | 0.8.0 | GPU memory (0.8.0 peak) |
|---|---|---|---|---|
| Full default ensemble | `process`, one doc | 81 | 79 | 11 GB |
| Full default ensemble | `process_batch`, short, 16 workers | 106 | 130 | 13 GB |
| Full default ensemble | `process_batch`, long, 16 workers | 16 | 16 | 14 GB |
| English only (2 models) | `process`, one doc | 191 | 190 | 2.5 GB |
| English only (2 models) | `process_batch`, short, 16 workers | 230 | 646 | 4.4 GB |
| English only (2 models) | `process_batch`, long, 16 workers | 168 | 251 | 4.7 GB |

With one language, `ner_batch_size=32` raised short-doc throughput to about 1,080 docs/s.
With the full multilingual ensemble, batch sizes above 8 gave no gain and used more memory.

GPU memory: the full default ensemble needs about 12 to 16 GB of free GPU memory. An
English-only setup needs about 3 to 5 GB.

### Older drivers

The newest wheels target CUDA 13. On an older driver, such as 560 (CUDA 12.6), torch and ONNX
Runtime silently use CPU. SCT warns and sets `cuda_fallback` as described above, except with
`ner_backend` `torch` and `ensemble_torch`, which only show provider `cpu`. To check:

```sh
nvidia-smi  # "CUDA Version" is the newest CUDA the driver supports
python -c "from onnxruntime.capi import build_and_package_info as i; print(i.cuda_version)"
python -c "import torch; print(torch.version.cuda, torch.cuda.is_available())"
```

For example, this worked on driver 560; other combinations can work too. That run also set
`LD_LIBRARY_PATH` to every `site-packages/nvidia/*/lib` folder.

```sh
pip install "torch==2.14.1+cu126" --index-url https://download.pytorch.org/whl/cu126
pip install onnxruntime-gpu==1.26.0
```

### Shared GPUs

ONNX Runtime's memory pool grows, so it can take nearly all free GPU memory: about 7.5 GB of 7.6 GB
free with default options, measured with two loaded models. With
`onnx_cuda_provider_options={"gpu_mem_limit": 2500 << 20, "arena_extend_strategy": "kSameAsRequested"}`
the same two models took 5.6 GB.

The limit applies per session, and each language can load its own models, so it does not cap the
total across models. A limit below what a model needs fails with an out-of-memory error.

## Apple silicon (M-series)

Opt in with `ner_device="mps"`. It moves the torch NER models (`ner_backend="torch"`) and
GLiNER in torch mode to the Apple GPU through PyTorch MPS. Install torch (`squeakycleantext[torch]`
or the `gliner` extra). ONNX models and GLiClass stay on CPU. Auto (`ner_device=None`) never picks
the Apple GPU: on a Mac it stays on CPU.

If torch has no MPS backend, SCT logs a warning ("NER runs on CPU: Apple GPU (mps) was
requested") and adds `mps_fallback` to `metadata['warnings']`; `metadata['provenance']['models']`
then shows `cpu` as the provider. Set `PYTORCH_ENABLE_MPS_FALLBACK=1` if a model uses an
operator MPS does not support.

Measured on an Apple Silicon Mac, GLiNER, 500 synthetic docs, `process_batch` (docs/s):

| Run pair (same machine load) | CPU | MPS | Speed-up |
|---|---|---|---|
| First run | 12.9 | 21.1 | 1.64x |
| Quiet repeat | 13.3 | 21.4 | 1.61x |
| Busy repeat (indexing running) | 3.9-4.1 | 10.3-10.9 | about 2.6x |

Output was identical on all 500 docs (largest score gap 2.1e-6). Peak memory was 1.8 GB on MPS
against 6.4 GB on CPU. Single-doc `process` gains only 1.16x, and the first call adds about 1.2 s
of GPU warm-up, so MPS pays off on batches.

ONNX on Apple GPU (CoreML) is not supported. It gave the same output but was not reliably faster
(0.6x to 1.6x CPU, median about 1.0x), its first use compiled for 35 minutes and wrote 82 GB of
cache, a cached load still took 30-46 s, and XLM-R-large crashed the process.

## Known limits

- All models of one cleaner run on one GPU (`cuda_device_id`). There is no option to spread
  models over several GPUs; run one cleaner per GPU instead. The full-ensemble GPU numbers above spread
  the models over 3 GPUs with a test harness, because 7.6 GB of free memory on one GPU was
  not enough.
- GPU numbers were measured on a shared machine; your numbers on a free GPU can be higher.

## Experimental detector routing

The calibrated cascade can skip a second detector after a fast stage covers every requested entity type.
It is opt-in. `always_run` remains the default.

The version 1 experiment compares 12 fixed confidence, margin and size combinations against the same two-stage baseline.
It uses pinned English BERT and multilingual WikiNEURAL ONNX checkpoints on the same CPU runtime.
Here `always_run` means the configured fast-plus-full model union, not SCT's default model profile.
Any measured speed difference applies only to this configured comparison.
The fast stage also runs the existing email, phone and IP format detectors.
The full protocol is `evaluation/improvements/routing-protocol.md` in the repository.

Development uses the existing 220 multilingual synthetic records, whose language outcomes were already inspected.
A separate 110-record synthetic cohort is frozen before candidate inference and evaluated after development selection.
Six mixed-entity fixtures provide an additional deterministic regression check.
Shared neural types and the full annotated taxonomy have separate reports.
Unsupported neural languages are reported as extrapolation, with all documents retained in accuracy denominators.

The benchmark measures actual ONNX session calls, median latency and nearest-rank p95 over three timed repeats.
Two document warmups precede each policy. Model loading is outside timing.
Memory observations are cumulative Linux process peak RSS, in KiB, with both models resident.
They do not show independent per-policy memory use or memory savings.

Promotion requires a nonpositive upper paired 95% exposure interval, no mixed-entity regression,
fewer full-model inference calls and lower median latency.
Every declared held-out language, entity and domain slice must also show no added exposure.
Negative and inconclusive results remain in the versioned evaluation artifacts.
Small synthetic samples cannot establish production privacy.

### Version 1 observed results

The selected cascade skips the full stage for ten English test documents without changing any saved accuracy summary.
This result covers 110 synthetic documents in eleven template groups, using the configured two-model baseline above.
Development selected `c95-m20-s256`: confidence 0.95, margin 0.20 and maximum length 256 characters.
All twelve development candidates had zero skipped stages and zero full-model call savings.
Their timing-only selection does not establish a useful development speed improvement.

| Held-out measurement | `always_run` | Selected cascade |
| --- | ---: | ---: |
| Actual fast-model calls, three repeats | 750 | 750 |
| Actual full-model calls, three repeats | 750 | 690 |
| Median document latency, ms | 314.978201 | 313.561970 |
| p95 document latency, ms | 387.812853 | 385.098284 |
| Full-taxonomy exact-span precision | 538/574 (93.7282%) | 538/574 (93.7282%) |
| Full-taxonomy exact-span recall | 538/660 (81.5152%) | 538/660 (81.5152%) |
| Shared neural-type precision | 208/244 (85.2459%) | 208/244 (85.2459%) |
| Shared neural-type recall | 208/330 (63.0303%) | 208/330 (63.0303%) |
| Exposed documents, either taxonomy | 74/110 (67.2727%) | 74/110 (67.2727%) |

Full-model calls fall 8%; total model calls fall 4%.
The median falls 1.416231 ms (0.449628%) over 330 timed document calls per policy.
Ten unique early exits produce 30 skipped stages across the three repeats.
The saved paired 95% exposure interval is `[0.0, 0.0]`, with all five experiment gates passing.
All 22 declared language, entity and domain added-exposure counts are zero.
Both policies retain identical shared/full slice summaries, including support counts and undefined values.
Language detection covers all 110 documents; 70 documents use languages outside the neural models' declared support.
Accuracy includes these documents and every routing fallback.

The six mixed-entity fixtures expose no annotated entities under either policy.
They save no calls and have a slower cascade median: 118.674737 ms versus 111.708482 ms.
All four measurements report the same cumulative process peak RSS of 1,649,020 KiB.
That value does not measure policy-specific memory savings.

Source-free results and the validation record are under `evaluation/improvements/`:
`routing-development-v1.json`, `routing-heldout-v1.json`, `routing-test-consumed-v1.json` and `routing-validation-v1.json`.
They preserve every slice; raw latency samples, predictions and per-document exposure counters were not persisted.
Validation checks saved counts, ratios, identities and gates; it cannot independently reconstruct raw scores, timing quantiles or the paired interval.
The surviving remote logs contain summaries only. Held-out inference was not replayed.
Sequential execution and host drift can affect this small timing difference; no latency uncertainty interval was measured.
This synthetic result establishes neither zero population exposure risk nor a general production speedup.
`always_run` remains the default.

### Version 2 observed results on the frozen v0.9.0 feature source

The selected cascade reduces full-model calls, but both policies leave annotated content exposed in 78 of 110 documents (70.9091%).
Passing the relative experiment gates does not establish adequate absolute privacy protection.
`always_run` remains the default.

The separate preregistration and retained evidence are under `evaluation/improvements/routing-v090-v2/`.
This experiment uses a new 110-record synthetic cohort, eleven language-template clusters and six separate regression fixtures.
The historical v1 cohort and results remain unchanged and were not replayed.

Development selected `c95-m20-s512`: confidence 0.95, margin 0.20 and maximum length 512 code points.
All twelve development candidates were eligible, but none skipped a stage or saved model calls.
The selected development median was 161.103002 ms against the baseline's 166.432187 ms, with 660 observations per policy.
Selection therefore relied on measured timing differences without development call savings; fixed order and host drift remain possible explanations.

| Held-out measurement | Configured `always_run` | Selected cascade |
| --- | ---: | ---: |
| Actual fast-model calls, three repeats | 750 | 750 |
| Actual full-model calls, three repeats | 750 | 690 |
| Total actual model calls | 1,500 | 1,440 |
| Median document latency, ms | 395.409150 | 381.347755 |
| Nearest-rank p95 latency, ms | 481.847735 | 468.015126 |
| Full-taxonomy exact-span precision | 560/614 (91.2052%) | 560/614 (91.2052%) |
| Full-taxonomy exact-span recall | 560/660 (84.8485%) | 560/660 (84.8485%) |
| Shared neural-type precision | 230/284 (80.9859%) | 230/284 (80.9859%) |
| Shared neural-type recall | 230/330 (69.6970%) | 230/330 (69.6970%) |
| Exposed documents, either taxonomy | 78/110 (70.9091%) | 78/110 (70.9091%) |

Each policy has 330 retained held-out durations: 110 documents over three fixed repeats, excluding initialization and two warmup documents.
Full-model calls fall 8%, total calls fall 4%, and median latency falls 14.061395 ms (3.556163%).
Ten English documents skip the full stage, producing 30 skipped stages across the repeats.
Every language, entity and domain added-exposure count is zero in both taxonomy views.
Both paired 95% exposure intervals are `[0.0, 0.0]`, using 1,000 template-group draws and seed 20261109.
All preregistered gates pass for this configured comparison; the intervals do not establish zero population risk.
All 110 documents remain in accuracy denominators, including 70 documents outside the models' declared neural language support.

The six mixed fixtures add no exposed gold mentions relative to baseline, but both policies expose three of six documents.
Their full-taxonomy recall is 33/36 and precision is 33/89; no stages or model calls are saved.
Across 18 durations per policy, the mixed median falls from 212.856895 to 212.251151 ms.
Its p95 increases from 873.196982 to 874.181043 ms.
All held-out and mixed measurements report cumulative Linux peak RSS of 1,660,124 KiB with both models resident.
This is not an independent memory comparison or evidence of memory savings.

Every repeat retains source-free positional predictions, durations and actual model-call deltas.
Durable completion receipts bind successful postflight verification to the manifest and every observation file.
The saved numeric audit reconstructs selection, scores, quantiles, intervals and gates without model-cache access or inference replay.
The portable frozen source is `frozen-audit-source-v2.tar.gz`; reproduction commands are in `benchmarks/compare/PROTOCOL.md`.

The result binds the experiment's frozen source, not every later repository revision.
Later fixes changed the HF integration, MCP server and warmup paths; those paths were not exercised by this benchmark.
The preserved archive retains the original bytes, and the manifest was not rebound to the later source.
This small synthetic cohort lacks independent native-language expert review.
Sequential timing and a single configured CPU runtime do not establish a general production speed improvement.
