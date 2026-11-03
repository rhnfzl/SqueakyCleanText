"""Adapter contract. Standard library only: it runs inside each tool's own venv.

Every adapter is a script ``<tool>_adapter.py`` that defines ``load(config) -> predict``
and calls ``serve(load)``. The runner starts it with the tool venv's Python:

    <venv>/bin/python benchmarks/compare/adapters/<tool>_adapter.py \\
        --input in.jsonl --output out.jsonl [--config '{"...": ...}'] [--batch-size 1] [--speed]

Input: one JSON object per line, ``{"id": str, "text": str, "lang": str}``. ``lang`` is the
gold language code (the gold-language track); an adapter may ignore it.

``predict(docs)`` takes a list of input objects and returns one dict per doc, in order:
``{"spans": [{"start", "end", "label", "score"}], "masked_text"?: str, "unplaced"?: int}``.
Offsets are Python code-point offsets into the unmodified ``text``, half-open. ``label`` is
the tool's own label (``entries.py`` maps it). ``score`` is a float, or None when the tool
gives none. ``masked_text`` is the tool's own anonymised output, where it has one.
``unplaced`` counts detections the adapter could not place in the original text (they
count as wrong predictions in scoring, never silently dropped).

Output: exactly one line per input doc, in input order, flushed after each batch:
``{"id", "spans", "masked_text"?, "unplaced"?, "error"?, "seconds"}``. A Python exception
becomes ``error`` (``"<Type>: <message>"``) with empty spans; a batch that raises is retried
doc by doc so one bad doc does not fail its neighbours. Spans that break the offset rules
make that doc an error (no silent clamping). After ``load`` returns, the adapter writes
``<output>.ready`` (load seconds), so the runner can tell a slow model load from a hung doc.
Crashes, hangs and OOM kills are the runner's job: it watches the output file and restarts
the adapter after the last finished doc.

``--speed`` writes timing lines instead (see ``speed``).
"""

import argparse
import json
import resource
import statistics
import sys
import time

WARMUP_DOCS = 20
COMMON_BATCH = 8


def check(doc: dict, result: dict) -> dict:
    """Validate one prediction against the contract; raise ValueError when it breaks it."""
    spans = []
    for span in result.get("spans", ()):
        start, end, label, score = span["start"], span["end"], span["label"], span.get("score")
        if not (isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(doc["text"])):
            raise ValueError(f"span offsets out of range: {start}-{end} for {len(doc['text'])} chars")
        if not isinstance(label, str) or not label:
            raise ValueError("span label must be a non-empty string")
        spans.append({"start": start, "end": end, "label": label, "score": None if score is None else float(score)})
    out = {"spans": spans}
    if result.get("masked_text") is not None:
        out["masked_text"] = str(result["masked_text"])
    if result.get("unplaced"):
        out["unplaced"] = int(result["unplaced"])
    return out


def cap_gpu(config: dict) -> None:
    """PROTOCOL section 16: cap a torch entry's share of the card at ``gpu_mem_gb`` (runner key) on cuda."""
    if config.get("device") != "cuda" or not config.get("gpu_mem_gb"):
        return
    import torch

    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, float(config["gpu_mem_gb"]) * 2**30 / total))


def _sync() -> None:
    """Wait for queued GPU work so timings are real (only when the tool already imported torch)."""
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()


def _timed(predict, docs: list[dict]) -> tuple[list[dict], float]:
    started = time.perf_counter()
    results = predict(docs)
    _sync()
    if len(results) != len(docs):
        raise ValueError(f"predict returned {len(results)} results for {len(docs)} docs")
    return results, time.perf_counter() - started


def _error(error: Exception) -> str:
    return f"{type(error).__name__}: {error}"[:500]


def run_quality(predict, docs: list[dict], out, batch_size: int) -> None:
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        try:
            results, seconds = _timed(predict, batch)
            lines = []
            for doc, result in zip(batch, results, strict=True):
                lines.append({"id": doc["id"], **check(doc, result), "seconds": round(seconds / len(batch), 6)})
        except Exception as error:  # noqa: BLE001  # isolate the failing doc, keep going
            if len(batch) == 1:
                lines = [{"id": batch[0]["id"], "spans": [], "error": _error(error), "seconds": 0.0}]
            else:
                lines = []
                for doc in batch:
                    try:
                        results, seconds = _timed(predict, [doc])
                        lines.append({"id": doc["id"], **check(doc, results[0]), "seconds": round(seconds, 6)})
                    except Exception as doc_error:  # noqa: BLE001
                        lines.append({"id": doc["id"], "spans": [], "error": _error(doc_error), "seconds": 0.0})
        for line in lines:
            out.write(json.dumps(line, ensure_ascii=False) + "\n")
        out.flush()


def _peak_rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(rss / 2**20 if sys.platform == "darwin" else rss / 2**10, 1)  # bytes on macOS, KiB on Linux


def run_speed(predict, docs: list[dict], out, batch_size: int, load_seconds: float) -> None:
    """Timing lines: ``first`` (after one prediction; the runner times the cold start from spawn),
    ``latency`` (batch 1 after warmup), ``throughput`` (common batch 8, and ``--batch-size`` if
    different; a batch size that raises, e.g. CUDA OOM under the cap, gets an ``error`` instead of its rates), then
    ``done`` with this process's peak RSS and torch's peak VRAM when torch is used."""
    def emit(line):
        out.write(json.dumps(line) + "\n")
        out.flush()

    _, first = _timed(predict, docs[:1])
    emit({"event": "first", "load_seconds": round(load_seconds, 3), "first_doc_seconds": round(first, 4)})
    for doc in docs[:WARMUP_DOCS]:
        _timed(predict, [doc])
    latencies = [_timed(predict, [doc])[1] for doc in docs]
    q = statistics.quantiles(latencies, n=20) if len(latencies) > 1 else latencies * 19
    emit({"event": "latency", "docs": len(docs), "median_ms": round(1000 * statistics.median(latencies), 2),
          "p95_ms": round(1000 * q[18], 2)})
    chars = sum(len(d["text"]) for d in docs)
    for size in sorted({COMMON_BATCH, batch_size}):
        line = {"event": "throughput", "batch_size": size, "common": size == COMMON_BATCH}
        try:
            seconds = sum(_timed(predict, docs[i:i + size])[1] for i in range(0, len(docs), size))
            line.update(docs_per_second=round(len(docs) / seconds, 3), chars_per_second=round(chars / seconds, 1))
        except Exception as error:  # noqa: BLE001  # e.g. CUDA OOM at this batch under the gpu_mem_gb cap: recorded
            line["error"] = _error(error)
            torch = sys.modules.get("torch")
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
        emit(line)
    done = {"event": "done", "peak_rss_mb": _peak_rss_mb()}
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        done["torch_peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    emit(done)


def serve(load, argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="{}", help="JSON object passed to load()")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--speed", action="store_true")
    args = parser.parse_args(argv)
    with open(args.input, encoding="utf-8") as f:
        docs = [json.loads(line) for line in f if line.strip()]
    started = time.perf_counter()
    predict = load(json.loads(args.config))
    load_seconds = time.perf_counter() - started
    with open(args.output + ".ready", "w") as ready:  # tells the runner the model loaded: doc timeouts start now
        ready.write(f"{load_seconds:.3f}\n")
    with open(args.output, "a", encoding="utf-8") as out:
        if args.speed:
            run_speed(predict, docs, out, args.batch_size, load_seconds)
        else:
            run_quality(predict, docs, out, max(1, args.batch_size))
