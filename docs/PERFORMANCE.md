# Performance and hardware

Measured with SqueakyCleanText 0.7.1 and 0.8.0, default config (full multilingual NER
ensemble), on the release benchmark set (`evaluation/release_benchmark.jsonl`). "Short" docs
are the benchmark documents (up to about 900 characters); "long" docs join 8 of them.
Numbers are docs per second, median of 3 runs. Output text is identical on CPU and GPU.

Machine: Linux x86_64, AMD Ryzen Threadripper PRO 7985WX (64 cores), RTX 4090 (24 GB),
onnxruntime 1.30/1.31 (CPU) and onnxruntime-gpu 1.29 (CUDA 12). CPU runs were pinned to 16 or
64 physical cores. Correctness and the 16-core CPU commands:
`evaluation/results/comparison-0.7.1-0.8.0.md`.

## Which hardware to use

- **Small jobs: any CPU.** One document at a time (`process`) runs at about 4 to 5 docs/s
  (about 200 ms each), on 4 cores or 64. More cores do not help a single document.
- **Bulk on CPU: use `process_batch` with `batch_size` about 16.** Keep
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

Install: `pip install "squeakycleantext[gpu]"` and `pip install "onnxruntime-gpu[cuda,cudnn]"`.
The onnxruntime-gpu build must match your NVIDIA driver: recent PyPI wheels need CUDA 13
(driver 580 or newer); for CUDA 12 use the onnxruntime CUDA 12 package index. Call
`onnxruntime.preload_dlls()` before creating the cleaner when CUDA comes from pip packages.

If CUDA cannot be loaded, ONNX Runtime runs on CPU. SCT logs a warning
("runs on CPU: CUDA was requested") when this happens.

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

## Known limits

- SCT puts every ONNX model on the first GPU (device 0). There is no option yet to choose
  the GPU or to spread models over several GPUs. The full-ensemble GPU numbers above spread
  the models over 3 GPUs with a test harness, because 7.6 GB of free memory on one GPU was
  not enough.
- GPU numbers were measured on a shared machine; your numbers on a free GPU can be higher.
