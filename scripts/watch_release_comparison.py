"""Start the authorised two-version comparison after a published release includes the required merge.

Run under a user systemd timer. API failures retry on the next tick. Setup/benchmark failures stop for inspection.
"""

import argparse
from datetime import UTC, datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import urllib.request

REPOSITORY = "rhnfzl/SqueakyCleanText"
REQUIRED_MERGE = "20c88e5d92806ce5bd7aa7b8d69912dfbec8c51a"
DATASETS = ("openpii", "gretel", "nemotron", "redact", "iahlt", "bayanat", "thainer", "thai_pii",
            "naamapadam", "multiconer", "stockmark", "tab", "wili")
SPEED_REPEATS = 3
LEGACY_PYTHON = Path.home() / "sct-bench/venv-0.7.1/bin/python"


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def api(route):
    request = urllib.request.Request(f"https://api.github.com/{route}",
                                     headers={"User-Agent": "SCT-release-comparison"})
    with urllib.request.urlopen(request, timeout=90) as response:  # noqa: S310 - fixed HTTPS host
        return json.load(response)


def release_candidate(release, comparison, sha):
    if release.get("draft") or release.get("prerelease") or not release.get("published_at"):
        return None
    tag = release.get("tag_name", "")
    if not re.fullmatch(r"v0\.8\.\d+", tag):
        return None
    if comparison.get("status") not in ("ahead", "identical"):
        return None
    if not re.fullmatch(r"[a-f0-9]{40}", sha):
        raise ValueError("Release commit is not a full SHA")
    return {"tag": tag, "version": tag[1:], "sha": sha, "published_at": release["published_at"]}


def validate_results(out):
    results = {p.stem: json.loads(p.read_text()) for p in out.glob("*.json") if p.stem != "run"}
    errors = [f"{k}: {v['error']}" for k, v in results.items() if "error" in v]
    if not results or errors:
        raise ValueError("Missing/failed results: " + "; ".join(errors))
    return results


def compare_samples(left, right):
    if left.keys() != right.keys():
        raise ValueError("The two versions have different dataset lists")
    for name in left:
        if not left[name].get("sample_sha256") or left[name]["sample_sha256"] != right[name].get("sample_sha256"):
            raise ValueError(f"Different samples for {name}")


def run_command(command, log, *, cwd=None, env=None, timeout=36000):
    with log.open("a") as stream:
        command = [shutil.which(command[0]) or command[0], *command[1:]]
        subprocess.run(  # noqa: S603 - commands use fixed executables and validated release paths, no shell
            command, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=timeout)


def prepare(root, candidate):
    run = root / "runs" / f"{candidate['tag']}-{candidate['sha']}"
    run.mkdir(parents=True, exist_ok=True)
    archive = run / "release.tar.gz"
    source = run / "source"
    if not source.exists():
        request = urllib.request.Request(
            f"https://api.github.com/repos/{REPOSITORY}/tarball/{candidate['sha']}",
            headers={"User-Agent": "SCT-release-comparison"})
        with urllib.request.urlopen(request, timeout=180) as response, archive.open("wb") as target:  # noqa: S310 - fixed HTTPS host
            shutil.copyfileobj(response, target)
        stage = run / "extract"
        stage.mkdir(exist_ok=True)
        with tarfile.open(archive) as bundle:
            bundle.extractall(stage, filter="data")
        folders = list(stage.iterdir())
        if len(folders) != 1 or not folders[0].is_dir():
            raise ValueError("Unexpected release archive layout")
        folders[0].rename(source)
    harness = run / "harness"
    if not harness.exists():
        shutil.copytree(source / "benchmarks", harness / "benchmarks")
        support = harness / "support"
        support.mkdir()
        for name, relative in {"evaluation.py": "sct/evaluation.py", "atomic_file.py": "sct/utils/atomic_file.py",
                               "constants.py": "sct/utils/constants.py", "presets.py": "sct/presets.py"}.items():
            shutil.copy2(source / relative, support / name)
    python = run / "venv/bin/python"
    setup_log = run / "setup.log"
    if not python.exists():
        run_command(["uv", "venv", "--python", "/usr/bin/python3.12", str(run / "venv")], setup_log)
    if not (run / "setup-ok.json").exists():
        run_command(["uv", "pip", "install", "--python", str(python), f"{source}[fuzzy]", "pyarrow"], setup_log)
        version = subprocess.check_output(  # noqa: S603 - run-local interpreter and fixed Python code
            [str(python), "-c", "import sct; print(sct.__version__)"], text=True).strip()
        if version != candidate["version"]:
            raise ValueError(f"Tag {candidate['tag']} contains package version {version}")
        save(run / "setup-ok.json", candidate)
    elif json.loads((run / "setup-ok.json").read_text()) != candidate:
        raise ValueError("Cached setup identity differs from the release")
    manifest = {str(p.relative_to(harness)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(harness.rglob("*.py"))}
    save(run / "provenance.json", {**candidate, "required_merge": REQUIRED_MERGE, "harness_sha256": manifest,
                                  "launcher_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                                                      for name in ("watch_release_comparison.py",
                                                                   "release_comparison_version.py")},
                                  "legacy_runtime": str(LEGACY_PYTHON),
                                  "runtime_note": "Legacy runtime retained; package/runtime pins recorded per version.",
                                  "compatibility": "0.7.1 training uses the release preset's supported fields; CPU/thread, "
                                                   "batch keyword and language-result API shims only. The placement mirror "
                                                   "uses each version's cleanup. Evaluator is frozen from the release."})
    return run, harness, python


def execute(root, candidate, state):
    state.update(phase="preparing", release=candidate, step="installing the published release and freezing its harness")
    save(root / "state.json", state)
    run, harness, python = prepare(root, candidate)
    state.update(phase="running", release=candidate, started_at=datetime.now(UTC).isoformat())
    save(root / "state.json", state)
    wrapper = root / "release_comparison_version.py"
    env = {**os.environ, "HF_HUB_CACHE": str(Path.home() / "sct-bench/hub"), "PYTHONUNBUFFERED": "1",
           "TOKENIZERS_PARALLELISM": "false", "CUDA_VISIBLE_DEVICES": "", "no_proxy": "*"}
    versions = [("0.7.1", LEGACY_PYTHON), (candidate["version"], python)]
    for version, interpreter in versions:
        run_command(["uv", "pip", "freeze", "--python", str(interpreter)], run / f"{version}.freeze.txt")
        # Fail early on a loader/model/API error before spending hours on the full comparison.
        state["step"] = f"smoke {version}"
        save(root / "state.json", state)
        smoke = run / "smoke" / version
        run_command(["taskset", "-c", "0-63", str(interpreter), str(wrapper), str(harness),
                     "--sct-version", version,
                     "--suite", "pii", "--datasets", "openpii", "--n", "1", "--configs", "default", "training",
                     "--threads", "16", "--workers", "2", "--out", str(smoke)], run / f"smoke-{version}.log",
                    cwd=harness, env=env, timeout=1800)
        validate_results(smoke)
    outputs = []
    for version, interpreter in versions:
        out = run / "quality" / version
        state["step"] = f"quality {version}"
        save(root / "state.json", state)
        run_command(["taskset", "-c", "0-63", str(interpreter), str(wrapper), str(harness),
                     "--sct-version", version,
                     "--suite", "pii", "long", "lid", "--datasets", *DATASETS,
                     "--configs", "default", "training", "--threads", "16", "--workers", "2", "--out", str(out)],
                    run / f"quality-{version}.log", cwd=harness, env=env)
        results = validate_results(out)
        if set(results) != set(DATASETS):
            raise ValueError(f"Missing quality datasets for {version}")
        outputs.append(results)
    compare_samples(*outputs)
    speeds = []
    # Alternate version order between repeats to reduce systematic timing bias.
    for repeat in range(1, SPEED_REPEATS + 1):
        current = []
        for version, interpreter in versions if repeat % 2 else versions[::-1]:
            state["step"] = f"speed {repeat}/{SPEED_REPEATS}, {version}"
            save(root / "state.json", state)
            out = run / "speed" / str(repeat) / version
            run_command(["taskset", "-c", "0-63", str(interpreter), str(wrapper), str(harness),
                         "--sct-version", version,
                         "--suite", "speed", "--configs", "default", "training", "--threads", "32", "--workers", "2",
                         "--out", str(out)], run / f"speed-{repeat}-{version}.log", cwd=harness, env=env)
            current.append(validate_results(out))
        compare_samples(*current)
        speeds.append({version: json.loads((run / "speed" / str(repeat) / version / "fineweb2.json").read_text())
                       for version, _ in versions})
    save(run / "speed-repeats.json", speeds)
    state.update(phase="completed", step="comparison finished; charts and post need agent review",
                 finished_at=datetime.now(UTC).isoformat())
    save(root / "state.json", state)


def main(root):
    state_file = root / "state.json"
    state = json.loads(state_file.read_text()) if state_file.exists() else {"phase": "waiting"}
    if state["phase"] in {"preparing", "running"}:
        state.update(phase="failed", step="Previous benchmark stopped before completion; inspect its logs before retrying")
        save(state_file, state)
        return
    if state["phase"] in {"failed", "completed"}:
        return
    try:
        release = api(f"repos/{REPOSITORY}/releases/latest")
        tag = release.get("tag_name", "")
        candidate = None
        if re.fullmatch(r"v0\.8\.\d+", tag):
            commit = api(f"repos/{REPOSITORY}/commits/{tag}")
            if not re.fullmatch(r"[a-f0-9]{40}", commit["sha"]):
                raise ValueError("Resolved release commit is not a full SHA")
            comparison = api(f"repos/{REPOSITORY}/compare/{REQUIRED_MERGE}...{commit['sha']}")
            candidate = release_candidate(release, comparison, commit["sha"])
        state.update(checked_at=datetime.now(UTC).isoformat(), last_release=tag)
        if not candidate:
            state.update(phase="waiting", step="waiting for a published v0.8 release containing the required merge")
            save(state_file, state)
            return
        execute(root, candidate, state)
    except Exception as error:  # API failures may retry; setup and inference failures require inspection.
        state.update(phase="failed" if state.get("phase") in {"preparing", "running"} else "waiting",
                     step=f"{type(error).__name__}: {error}", checked_at=datetime.now(UTC).isoformat())
        save(state_file, state)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.home() / "sct-release-comparison")
    root = parser.parse_args().root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / "watcher.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(0) from None
        main(root)
