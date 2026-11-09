"""Speed and memory per entry (PROTOCOL.md section 16).

    python -m benchmarks.compare.speed --entries sct-default sct-gliner --cpus 0-15 --threads 16 \\
        --out benchmarks/results/compare/2026-11-02-speed

For each repeat, entries run in a seeded random order, each in a fresh process of its venv
(``adapters/_protocol.py --speed``) on the FineWeb-2 speed sample:

- cold start: spawn -> first prediction done (imports, model load from the local cache, one doc);
- warm batch-1 latency: median and p95 after 20 warmup docs (GPU synchronised);
- throughput at the common batch 8 (and ``--batch-size`` when different), docs/s and chars/s;
- peak RSS: ``VmHWM`` of the adapter process from ``/proc`` (Linux), read while it runs; the
  adapter's own ``getrusage`` peak is kept too but on Linux it carries the parent's peak over
  fork + exec, so it never reads below the runner's size; peak VRAM: ``nvidia-smi`` polled for the
  adapter's PID only (never other users' processes), plus torch's own peak when torch is used;
- install size: the venv on disk (models are in the HF cache, shared, not counted);
- device per entry: the run's device, or cpu for an entry that only runs on CPU (``entries.py`` ``device``).

Median over repeats with min and max. CPU pinning and thread caps as in the runner.
"""

import argparse
import json
import random
import shutil
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from benchmarks.compare import entries as registry
from benchmarks.compare.runtime import (
    POLL_SECONDS,
    read_lines,
    write_json,
    add_entry_args,
    child_env,
    command,
    entry_device,
    device_env,
    entry_config,
    entry_python,
)
from benchmarks.datasets import get
from benchmarks.speed import environment

VRAM_POLL_SECONDS = 0.2


def vram_mb(pid: int) -> float | None:
    """GPU memory (MiB) used by *pid* now, from nvidia-smi's compute-app list; None without nvidia-smi or if it fails."""
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return None
    try:
        done = subprocess.run([smi, "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],  # noqa: S603
                              capture_output=True, text=True, timeout=30)
        if done.returncode != 0:
            return None
        return sum(float(used) for p, used in (line.split(", ") for line in done.stdout.strip().splitlines() if ", " in line)
                   if int(p) == pid)
    except (subprocess.TimeoutExpired, OSError, ValueError):  # the daemon poller thread would die on these
        return None


def vm_hwm_mb(pid: int) -> float | None:
    """Peak resident memory (MiB) of *pid* so far from ``/proc/<pid>/status``; None off Linux or once it is gone."""
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return round(int(line.split()[1]) / 1024, 1)
    except OSError:
        return None
    return None


def dir_mb(path: Path) -> float:
    return round(sum(f.stat().st_size for f in path.rglob("*") if f.is_file() and not f.is_symlink()) / 2**20, 1)


def _sample_vram(pid: int, stop: threading.Event, peak: dict[str, float | None]) -> None:
    while not stop.is_set():
        used = vram_mb(pid)
        if used is not None:
            peak["vram_mb"] = max(peak["vram_mb"] or 0.0, used)
        stop.wait(VRAM_POLL_SECONDS)


def _take_timings(lines: list[dict], result: dict[str, Any], started: float) -> None:
    for line in lines:
        if line["event"] == "first":
            result["cold_start_seconds"] = round(time.perf_counter() - started, 3)
        if line["event"] == "throughput":
            result.setdefault("throughput", []).append(line)
        else:
            result[line["event"]] = line


def measure(cmd: list[str], docs: list[dict], work: Path, env: dict, batch_size: int, timeout: float,
            poll_vram: bool) -> dict[str, Any]:
    """One fresh adapter process in speed mode; returns its timing lines plus cold start and VRAM peak."""
    work.mkdir(parents=True, exist_ok=True)
    inp, out, err = work / "speed-in.jsonl", work / "speed-out.jsonl", work / "speed-err.log"
    inp.write_text("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in docs), encoding="utf-8")
    out.write_bytes(b"")
    started = time.perf_counter()
    with err.open("w") as errf:  # the adapter's stderr stays in the cache (it may quote doc text): only the last line is kept
        proc = subprocess.Popen([*cmd, "--input", str(inp), "--output", str(out), "--speed",  # noqa: S603
                                 "--batch-size", str(batch_size)], stdout=subprocess.DEVNULL, stderr=errf, env=env)
    peak: dict[str, float | None] = {"vram_mb": None}
    stop = threading.Event()
    if poll_vram:
        threading.Thread(target=_sample_vram, args=(proc.pid, stop, peak), daemon=True).start()
    result: dict[str, Any] = {"cold_start_seconds": None, "peak_rss_mb_proc": None}
    offset = 0
    try:
        while True:
            lines, offset = read_lines(out, offset)
            _take_timings(lines, result, started)
            hwm = vm_hwm_mb(proc.pid)  # the last read before exit is the process peak
            if hwm is not None:
                result["peak_rss_mb_proc"] = max(result["peak_rss_mb_proc"] or 0.0, hwm)
            if proc.poll() is not None:
                _take_timings(read_lines(out, offset)[0], result, started)
                break
            if time.perf_counter() - started > timeout:
                proc.kill()
                proc.wait()
                result["error"] = f"timeout after {timeout:.0f}s"
                break
            time.sleep(POLL_SECONDS)
    finally:
        stop.set()
        inp.unlink(missing_ok=True)
    if proc.returncode not in (0, None) and "error" not in result:
        last = [line for line in err.read_text(errors="replace").splitlines() if line.strip()][-1:]
        result["error"] = f"crash: exit {proc.returncode}" + (f": {last[0].strip()[:300]}" if last else "")
    result["peak_vram_mb_nvidia_smi"] = peak["vram_mb"]
    return result


def _spread(values: list[float]) -> dict | None:
    values = [v for v in values if v is not None]
    return {"median": round(statistics.median(values), 4), "min": min(values), "max": max(values)} if values else None


def summarise(runs: list[dict]) -> dict[str, Any]:
    ok = [r for r in runs if "error" not in r and "done" in r]
    common = [next((t for t in r.get("throughput", []) if t["common"]), {}) for r in ok]
    return {
        "repeats": len(runs), "failed": [r["error"] for r in runs if "error" in r],
        "cold_start_seconds": _spread([r["cold_start_seconds"] for r in ok]),
        "load_seconds": _spread([r["first"]["load_seconds"] for r in ok]),
        "latency_median_ms": _spread([r["latency"]["median_ms"] for r in ok]),
        "latency_p95_ms": _spread([r["latency"]["p95_ms"] for r in ok]),
        "common_batch_docs_per_second": _spread([t.get("docs_per_second") for t in common]),
        "common_batch_chars_per_second": _spread([t.get("chars_per_second") for t in common]),
        "other_batch": [t for r in ok for t in r.get("throughput", []) if not t["common"]],
        "common_batch_errors": [t["error"] for t in common if "error" in t],
        "peak_rss_mb": _spread([r["peak_rss_mb_proc"] for r in ok]),
        "peak_rss_mb_getrusage": _spread([r["done"]["peak_rss_mb"] for r in ok]),
        "peak_vram_mb": _spread([r["peak_vram_mb_nvidia_smi"] for r in ok]),
        "torch_peak_vram_mb": _spread([r["done"].get("torch_peak_vram_mb") for r in ok]),
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    add_entry_args(parser)
    parser.add_argument("--n", type=int, default=100, help="FineWeb-2 speed docs per language")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8, help="entry's own best batch (8 = common only)")
    parser.add_argument("--timeout", type=float, default=3600.0)
    args = parser.parse_args(argv)
    docs = [{"id": r["id"], "text": r["text"], "lang": r["lang"]} for r in get("fineweb2").load(args.n, args.seed)]
    gpu, _ = device_env(args)
    tools = {name: registry.tool(name) for name in args.entries}
    commands, sizes, envs = {}, {}, {}
    for name, t in tools.items():
        py = entry_python(name, t, args)
        config = entry_config(t, args)
        commands[name] = command(py, t["adapter"], config, args.cpus)
        envs[name] = child_env(args.threads, config["device"], gpu, args.hf_hub_cache)
        pinned = any(item.startswith(f"{name}=") for item in args.python)
        sizes[name] = dir_mb(py.parent.parent) if not pinned and t["requirements"] else None
    runs: dict[str, list] = {name: [] for name in tools}
    rng = random.Random(args.seed)  # noqa: S311  # run order, not crypto
    for repeat in range(args.repeats):
        order = list(tools)
        rng.shuffle(order)
        for name in order:
            print(f"[speed] repeat {repeat + 1}/{args.repeats} {name}", file=sys.stderr, flush=True)
            runs[name].append(measure(commands[name], docs, args.cache / name / "speed", envs[name], args.batch_size,
                                      args.timeout, poll_vram=args.device == "cuda"))
    result = {"docs": len(docs), "chars": sum(len(d["text"]) for d in docs), "threads": args.threads, "cpus": args.cpus,
              "device": args.device, "gpu": gpu, "repeats": args.repeats, "seed": args.seed,
              "environment": environment(),
              "entries": {name: {**summarise(runs[name]), "install_mb": sizes[name],
                                 "device": entry_device(t["device"], args.device)} for name, t in tools.items()}}
    write_json(args.out / "speed.json", result)
    print(args.out / "speed.json")


if __name__ == "__main__":
    main()
