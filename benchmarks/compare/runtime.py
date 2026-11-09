"""Shared environments, commands and result files for comparison runs."""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from sct.utils.atomic_file import write_text_atomic

HERE = Path(__file__).resolve().parent
ADAPTERS = HERE / "adapters"
CACHE = HERE.parent / ".cache" / "compare"
PYTHON = "3.12"
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
POLL_SECONDS = 0.05

# CPU and GPU wheels own the same import package; reinstall the pinned GPU wheel last.
GPU_TWINS = {"onnxruntime-gpu": "onnxruntime"}


def gpu_twin_fixes(requirements_text: str) -> list[tuple[str, str]]:
    """(CPU package to uninstall, pinned GPU requirement to reinstall) for each GPU_TWINS build the spec pins."""
    lines = [line.strip() for line in requirements_text.splitlines() if line.strip() and not line.startswith("#")]
    return [(cpu, line) for gpu, cpu in GPU_TWINS.items() for line in lines
            if re.split(r"[\[=@ ]", line, maxsplit=1)[0].lower() == gpu]


def ensure_env(name: str, requirements: Path | None, envs_root: Path, python: str = PYTHON) -> Path:
    """Python of the entry's venv, built with uv when missing or when its spec changed.

    CUDA builds come from the spec itself as direct CUDA 12 wheel URLs (``envs/*.txt``): torch from the PyTorch
    cu126 index, onnxruntime-gpu from ORT's CUDA 12 feed, with ``GPU_TWINS`` resolved after the install.
    """
    if requirements is None:
        return Path(sys.executable)
    venv = envs_root / name
    py = venv / "bin" / "python"
    digest = hashlib.sha256(requirements.read_bytes() + python.encode()).hexdigest()
    stamp = venv / ".spec-sha256"
    if py.exists() and stamp.exists() and stamp.read_text().strip() == digest:
        return py
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is needed to build entry venvs")
    subprocess.run([uv, "venv", "--clear", "--python", python, str(venv)], check=True)  # noqa: S603
    subprocess.run([uv, "pip", "install", "--python", str(py), "-r", str(requirements)], check=True)  # noqa: S603
    for cpu, gpu in gpu_twin_fixes(requirements.read_text()):
        subprocess.run([uv, "pip", "uninstall", "--python", str(py), cpu], check=True)  # noqa: S603
        subprocess.run([uv, "pip", "install", "--python", str(py), "--reinstall", "--no-deps", gpu],  # noqa: S603
                       check=True)
    freeze = subprocess.run([uv, "pip", "freeze", "--python", str(py)], check=True,  # noqa: S603
                            capture_output=True, text=True, timeout=30).stdout
    write_text_atomic(venv / "freeze.txt", freeze)
    write_text_atomic(stamp, digest)
    return py


def freeze(py: Path) -> str:
    frozen = py.parent.parent / "freeze.txt"
    if frozen.exists():
        return frozen.read_text()
    uv = shutil.which("uv")
    cmd = [uv, "pip", "freeze", "--python", str(py)] if uv else [str(py), "-m", "pip", "freeze"]
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=30)  # noqa: S603
    except subprocess.TimeoutExpired:
        return ""
    return done.stdout if done.returncode == 0 else ""


def child_env(threads: int | None = None, device: str = "cpu", gpu: str | None = None,
              hf_hub_cache: str | None = None) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV")}
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYTHONUNBUFFERED"] = "1"
    if threads:
        env.update({k: str(threads) for k in THREAD_VARS})
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    elif gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    if hf_hub_cache:
        env["HF_HUB_CACHE"] = hf_hub_cache  # only the hub cache: HF_HOME (and its token) stays the user's
    return env


def entry_device(claimed: str, run_device: str) -> str:
    """Device an entry runs on: a CPU-only entry (``device`` "cpu" in entries.py) stays on cpu in a cuda run."""
    return run_device if run_device in claimed.split("|") else "cpu"


def command(py: Path, adapter: str, config: dict, cpus: str | None = None) -> list[str]:
    cmd = [str(py), str(ADAPTERS / adapter), "--config", json.dumps(config)]
    if cpus:
        taskset = shutil.which("taskset")
        if taskset is None:
            raise RuntimeError("--cpus needs taskset (Linux)")
        cmd = [taskset, "-c", cpus, *cmd]
    return cmd


def pick_gpu(min_free_gb: float = 6.0) -> str:
    """Index of the GPU with the most free memory; refuse when it has less than *min_free_gb*."""
    smi = shutil.which("nvidia-smi")
    if smi is None:
        raise RuntimeError("no nvidia-smi: use --device cpu")
    out = subprocess.run([smi, "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],  # noqa: S603
                         capture_output=True, text=True, check=True, timeout=30).stdout
    index, free = max((line.split(", ") for line in out.strip().splitlines()), key=lambda x: float(x[1]))
    if float(free) < min_free_gb * 1024:
        raise RuntimeError(f"GPU {index} has only {float(free) / 1024:.1f} GB free (< {min_free_gb})")
    return index


def read_lines(path: Path, offset: int) -> tuple[list[dict], int]:
    with path.open("rb") as f:
        f.seek(offset)
        data = f.read()
    end = data.rfind(b"\n") + 1
    return [json.loads(line) for line in data[:end].decode("utf-8").splitlines() if line.strip()], offset + end


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(path, json.dumps(data, indent=1, ensure_ascii=False, sort_keys=True) + "\n")


def add_entry_args(parser: argparse.ArgumentParser) -> None:
    """The arguments ``runner.py`` and ``speed.py`` share: which entries, where, on what hardware."""
    parser.add_argument("--entries", nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=CACHE, help="raw outputs (hold dataset text) and venvs")
    parser.add_argument("--python", action="append", default=[], metavar="ENTRY=PATH",
                        help="use this interpreter instead of building the entry's venv")
    parser.add_argument("--threads", type=int, help="thread caps (env + adapter config)")
    parser.add_argument("--cpus", help="taskset CPU list, e.g. 0-15")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--gpu", help="GPU index (default: the one with most free memory)")
    parser.add_argument("--gpu-mem-gb", type=float, default=5.0)
    parser.add_argument("--hf-hub-cache", help="HF_HUB_CACHE for the adapters")


def device_env(args: argparse.Namespace) -> tuple[str | None, dict[str, str]]:
    """The GPU the run uses (None on CPU) and the adapters' environment."""
    gpu = None if args.device == "cpu" else (args.gpu or pick_gpu(args.gpu_mem_gb + 1))
    return gpu, child_env(args.threads, args.device, gpu, args.hf_hub_cache)


def entry_python(name: str, tool: dict[str, Any], args: argparse.Namespace) -> Path:
    """The ``--python`` interpreter of the entry, else its venv (built when needed)."""
    pythons = dict(item.split("=", 1) for item in args.python)
    return Path(pythons[name]) if name in pythons else ensure_env(name, tool["requirements"], args.cache / "venvs")


def entry_config(tool: dict[str, Any], args: argparse.Namespace, overrides: dict | None = None) -> dict[str, Any]:
    return {**tool["adapter_config"], "threads": args.threads, "device": entry_device(tool["device"], args.device),
            "gpu_mem_gb": args.gpu_mem_gb, **(overrides or {})}
