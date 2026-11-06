"""Timing and the hardware / software record every run writes."""

import os
import platform
import shutil
import subprocess
import time
from importlib import metadata
from pathlib import Path
from typing import Any

from benchmarks.score import dominant_script

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("SqueakyCleanText", "onnxruntime", "onnxruntime-gpu", "tokenizers", "lingua-language-detector",
            "huggingface-hub", "numpy", "regex", "ftfy", "emoji", "presidio_anonymizer", "pyarrow")
WARMUP_DOCS = 4


def run_batch(cleaner, texts: list[str], workers: int, chunk: int = 64):
    """Process *texts* in chunks with ``process_batch``; return ``(results, seconds)``."""
    results, seconds = [], 0.0
    for i in range(0, len(texts), chunk):
        started = time.perf_counter()
        results += cleaner.process_batch(texts[i:i + chunk], max_workers=workers)
        seconds += time.perf_counter() - started
    return results, seconds


def rates(docs: int, chars: int, seconds: float) -> dict[str, float | None]:
    return {"documents": docs, "chars": chars, "seconds": round(seconds, 3),
            "docs_per_second": round(docs / seconds, 3) if seconds else None,
            "chars_per_second": round(chars / seconds, 1) if seconds else None}


def measure(cleaner, records: list[dict], workers: int) -> dict[str, Any]:
    """docs/s and chars/s per script (and in total), after a short warmup."""
    texts = [r["text"] for r in records]
    cleaner.process_batch(texts[:WARMUP_DOCS], max_workers=workers)
    groups: dict[str, list[str]] = {}
    for record in records:
        groups.setdefault(f"{record['lang']} ({dominant_script(record['text'])})", []).append(record["text"])
    out, total_seconds = {}, 0.0
    for name, group in sorted(groups.items()):
        _, seconds = run_batch(cleaner, group, workers)
        total_seconds += seconds
        out[name] = rates(len(group), sum(map(len, group)), seconds)
    return {"total": rates(len(texts), sum(map(len, texts)), total_seconds), "by_script": out}


def _run(cmd: list[str]) -> str | None:
    if not shutil.which(cmd[0]):
        return None
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30).stdout.strip()  # noqa: S603
    except (OSError, subprocess.SubprocessError):
        return None


def environment() -> dict[str, Any]:
    """Software and git state for the result files. No hardware names: results are public."""
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    try:
        import onnxruntime

        providers = onnxruntime.get_available_providers()
    except ImportError:
        providers = []
    affinity = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    status = _run(["git", "-C", str(ROOT), "status", "--porcelain"])
    return {
        "git_commit": _run(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
        "git_dirty": bool(status) if status is not None else None,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpus_usable": affinity,
        "onnxruntime_providers": providers,
        "packages": versions,
    }
