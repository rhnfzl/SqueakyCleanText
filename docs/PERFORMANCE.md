# Performance and hardware

Measured with SqueakyCleanText 0.7.1 and 0.8.0, default config (full multilingual NER
ensemble), on the release benchmark set (`evaluation/release_benchmark.jsonl`). "Short" docs
are the benchmark documents (up to about 900 characters); "long" docs join 8 of them.
Numbers are docs per second, median of 3 runs. Output text is identical on CPU and GPU.

Machine: Linux x86_64, AMD Ryzen Threadripper PRO 7985WX (64 cores), RTX 4090 (24 GB),
onnxruntime 1.30/1.31 (CPU) and onnxruntime-gpu 1.29 (CUDA build not recorded). CPU runs were pinned to 16 or
64 physical cores. Correctness and the 16-core CPU commands:
`evaluation/results/comparison-0.7.1-0.8.0.md`.

## Which hardware to use

- **Small jobs: any CPU.** One document at a time (`process`) runs at about 4 to 5 docs/s
  (about 200 ms each), on 4 cores or 64. More cores do not help a single document.
- **Bulk on CPU: use `process_batch` with `max_workers` about 16.** Keep
  `ner_batch_size=8` (default) and let ONNX use the cores you give it.
- **Bulk and speed: use a GPU.** With an RTX 4090, single documents are about 17x faster
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

Measured on an M4 Max, GLiNER, 500 synthetic docs, `process_batch` (docs/s):

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
