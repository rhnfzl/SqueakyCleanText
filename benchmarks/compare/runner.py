"""Run comparison entries over dataset splits, each in its own venv, and score them.

    python -m benchmarks.compare.runner --entries sct-default sct-gliner --datasets gretel_en redact \\
        --part dev --n 100 --out benchmarks/results/compare/2026-11-02-dev

Per entry: build (or reuse) ``<envs-root>/<entry>`` with ``uv`` from ``envs/<entry>.txt``,
check offsets on the hand-made multilingual fixtures, then run the adapter (contract in
``adapters/_protocol.py``) under a supervisor: per-doc timeout, one retry, then the doc is
recorded as failed with its reason; nothing is dropped silently. Raw adapter output holds
dataset text (``masked_text``), so it stays in ``--cache`` (gitignored); ``--out`` gets
scores, counts, failure kinds, the venv freeze and, only for ``scores-and-ids`` datasets,
ids and predicted spans.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from benchmarks.compare import entries as registry
from benchmarks.compare import scorer, split
from benchmarks.datasets import get
from benchmarks.labels import DATASET_LABELS
from benchmarks.speed import environment

HERE = Path(__file__).resolve().parent
ADAPTERS = HERE / "adapters"
CACHE = HERE.parent / ".cache" / "compare"
DOC_TIMEOUT = {"tab": 600.0}  # PROTOCOL.md section 15; others use --doc-timeout
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
POLL_SECONDS = 0.05
MAX_FAILED_STREAK = 3  # docs in a row failing with no output at all: the adapter cannot start

FIXTURE_EMAIL = "ana.lee@example.org"
# Hand-made offset fixtures (PROTOCOL.md section 6): each holds FIXTURE_EMAIL once; every
# entry supports EMAIL, so each must return exactly that span (code-point offsets).
OFFSET_FIXTURES = {
    "latin-combining": ("fr", "Café Ñoño, résumé naïf: write to ana.lee@example.org today."),
    "arabic": ("ar", "مرحبا، راسلني على ana.lee@example.org شكرا جزيلا"),
    "devanagari": ("hi", "नमस्ते, कृपया ana.lee@example.org पर ईमेल भेजें"),
    "thai": ("th", "สวัสดีครับ ติดต่อ ana.lee@example.org ได้เลยครับ"),
    "cjk": ("zh", "请发邮件到 ana.lee@example.org 谢谢。東京で会いましょう"),
    "emoji": ("en", "Hi \U0001F44B\U0001F3FD family \U0001F468‍\U0001F469‍\U0001F467 mail ana.lee@example.org \U0001F642"),
    "crlf": ("en", "Line one\r\nLine two\r\nMail: ana.lee@example.org\r\nEnd"),
}


def ensure_env(name: str, requirements: Path | None, envs_root: Path, python: str = registry.PYTHON) -> Path:
    """Python of the entry's venv, built with uv when missing or when its spec changed."""
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
    freeze = subprocess.run([uv, "pip", "freeze", "--python", str(py)], check=True,  # noqa: S603
                            capture_output=True, text=True).stdout
    (venv / "freeze.txt").write_text(freeze)
    stamp.write_text(digest)
    return py


def freeze(py: Path) -> str:
    frozen = py.parent.parent / "freeze.txt"
    if frozen.exists():
        return frozen.read_text()
    uv = shutil.which("uv")
    cmd = [uv, "pip", "freeze", "--python", str(py)] if uv else [str(py), "-m", "pip", "freeze"]
    done = subprocess.run(cmd, capture_output=True, text=True)  # noqa: S603
    return done.stdout


def child_env(threads: int | None = None, device: str = "cpu", gpu: str | None = None,
              hf_hub_cache: str | None = None) -> dict:
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
                         capture_output=True, text=True, check=True).stdout
    index, free = max((line.split(", ") for line in out.strip().splitlines()), key=lambda x: float(x[1]))
    if float(free) < min_free_gb * 1024:
        raise RuntimeError(f"GPU {index} has only {float(free) / 1024:.1f} GB free (< {min_free_gb})")
    return index


def _read_lines(path: Path, offset: int) -> tuple[list[dict], int]:
    with path.open("rb") as f:
        f.seek(offset)
        data = f.read()
    end = data.rfind(b"\n") + 1
    return [json.loads(line) for line in data[:end].decode("utf-8").splitlines() if line.strip()], offset + end


def supervise(cmd: list[str], docs: list[dict], work: Path, env: dict, doc_timeout: float, start_timeout: float,
              retries: int = 1, batch_size: int = 1) -> tuple[dict[str, dict], dict]:
    """Run the adapter until every doc has a line; restart after crashes and timeouts.

    A doc gets ``doc_timeout * batch_size`` seconds after the previous line or after the
    adapter's ``.ready`` marker (model loaded); before the marker a fresh process gets
    ``start_timeout`` more. When the process dies or hangs, the first unfinished doc is
    blamed: retried ``retries`` times, then given an ``error`` line.
    """
    work.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    tries: Counter = Counter()
    pending, attempts, streak = list(docs), 0, 0
    restarts: Counter = Counter()
    while pending:
        attempts += 1
        inp, out, err = work / f"in-{attempts}.jsonl", work / f"out-{attempts}.jsonl", work / f"err-{attempts}.log"
        inp.write_text("".join(json.dumps({"id": d["id"], "text": d["text"], "lang": d["lang"]}, ensure_ascii=False) + "\n"
                               for d in pending), encoding="utf-8")
        out.write_bytes(b"")
        Path(f"{out}.ready").unlink(missing_ok=True)  # attempt names restart per run: a stale marker skips the load allowance
        with err.open("w") as errf:
            proc = subprocess.Popen([*cmd, "--input", str(inp), "--output", str(out), "--batch-size", str(batch_size)],  # noqa: S603
                                    stdout=subprocess.DEVNULL, stderr=errf, env=env)
        expected = {d["id"] for d in pending}
        marker = Path(f"{out}.ready")
        offset, last, got, ready, reason = 0, time.monotonic(), 0, False, None

        def take(lines):
            nonlocal got
            for line in lines:
                if line.get("id") in expected:
                    results[line["id"]] = line
                    got += 1

        while True:
            lines, offset = _read_lines(out, offset)
            take(lines)
            if lines or (not ready and marker.exists()):
                ready, last = True, time.monotonic()
            code = proc.poll()
            if code is not None:
                take(_read_lines(out, offset)[0])
                reason = None if code == 0 else f"crash: exit {code}"
                break
            limit = doc_timeout * batch_size + (0 if ready else start_timeout)
            if time.monotonic() - last > limit:
                proc.kill()
                proc.wait()
                reason = f"timeout: no output for {limit:.0f}s"
                break
            time.sleep(POLL_SECONDS)
        inp.unlink()
        pending = [d for d in pending if d["id"] not in results]
        if not pending:
            break
        reason = reason or "crash: exited before every doc had output"
        restarts[reason.split(":")[0]] += 1
        streak = 0 if got else streak
        stuck = pending[0]
        tries[stuck["id"]] += 1
        if tries[stuck["id"]] > retries:
            results[stuck["id"]] = {"id": stuck["id"], "spans": [], "error": reason, "seconds": None}
            pending = pending[1:]
            streak += 1
            if streak >= MAX_FAILED_STREAK:
                for d in pending:
                    results[d["id"]] = {"id": d["id"], "spans": [], "error": f"aborted: {streak} docs in a row failed "
                                        f"without output ({reason})", "seconds": None}
                pending = []
    return results, {"processes": attempts, "restarts": dict(restarts)}


def cache_key(config: dict, adapter: str, frozen: str, records: list[dict]) -> str:
    """Fingerprint of everything that makes a raw output: adapter config and code, venv, input docs."""
    h = hashlib.sha256()
    for part in (json.dumps(config, sort_keys=True), (ADAPTERS / adapter).read_text(encoding="utf-8"),
                 (ADAPTERS / "_protocol.py").read_text(encoding="utf-8"), frozen):
        h.update(part.encode() + b"\0")
    for r in records:
        h.update(json.dumps([r["id"], r["lang"], r["text"]], ensure_ascii=False).encode() + b"\0")
    return h.hexdigest()


def curve_floor(config: dict, entry: dict) -> float | None:
    """Threshold the adapter already applied (config override, else the entry's numeric default)."""
    value = config.get("threshold", entry.get("threshold"))
    return float(value) if isinstance(value, (int, float)) else None


def check_offsets(results: dict[str, dict], labels: dict[str, str]) -> tuple[dict[str, str], dict[str, str]]:
    """(offset problems, detection errors): fixture name -> reason, for fixtures not returned exactly.

    An error line or an EMAIL span that cuts into the e-mail is an offset problem (the entry is
    skipped). No overlapping EMAIL span, or one that covers the whole e-mail plus extra text, is
    the model's detection error: recorded, not a skip.
    """
    problems, misses = {}, {}
    for name, (_, text) in OFFSET_FIXTURES.items():
        line = results.get(name, {})
        start = text.index(FIXTURE_EMAIL)
        end = start + len(FIXTURE_EMAIL)
        emails = [(s["start"], s["end"]) for s in line.get("spans", ()) if labels.get(s["label"]) == "EMAIL"]
        if name not in results or "error" in line:
            problems[name] = line.get("error", "no output")
        elif (start, end) in emails:
            continue
        else:
            got = [(s["start"], s["end"], s["label"]) for s in line.get("spans", ())]
            reason = f"want EMAIL {(start, end)}, got {got}"
            overlapping = [(a, b) for a, b in emails if a < end and b > start]
            if all(a <= start and b >= end for a, b in overlapping):
                misses[name] = reason
            else:
                problems[name] = reason
    return problems, misses


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    return value


def _write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _pick(docs: list, keep) -> list:
    return [x for x in docs if keep(x[2])]


def _agg(per_doc: dict, datasets, keep) -> dict | None:
    """Group macro over *datasets* using only the docs whose language passes *keep*; None when no doc does."""
    chosen = {d: [(lang, c) for _, _, lang, c in _pick(per_doc[d], keep)] for d in datasets}
    chosen = {d: v for d, v in chosen.items() if v}
    return scorer.aggregate(chosen) if chosen else None


def rule_views(per_doc: dict, baseline: str, views=("shared", "full"), samples: int = scorer.BOOTSTRAP_SAMPLES) -> dict:
    """Decision 21 + claimed languages (PROTOCOL.md sections 12, 17), from ``per_doc[entry][dataset][view]``.

    - ``own[entry][view][group]``: per contamination status (clean / unknown / excluded) the datasets and the
      group macro on the entry's claimed languages (``claimed``) and outside them (``outside_claim``); the
      clean, claimed cell (the headline) carries a 95% bootstrap CI (``ci``).
    - ``same_set[view][group][pool]``: datasets clean for every entry of the pool, languages all of them claim
      (``all``: every entry; ``others``: the non-SCT entries with a clean set in the group). Empty: none.
    - ``pairwise[entry][view][group]``: baseline minus entry, paired bootstrap, on the datasets clean for both
      (``clean``) and on those clean or unknown for both with at least one unknown (``unknown``), languages
      both claim.
    """
    own: dict = {}
    pairwise: dict = {}
    same_set: dict = {}
    entries = list(per_doc)
    for view in views:
        for group, members in scorer.GROUPS.items():
            present = [d for d in members if all(d in per_doc[e] for e in entries)]
            for e in entries:
                cells = own.setdefault(e, {}).setdefault(view, {})
                by_doc = {d: per_doc[e][d][view] for d in members if d in per_doc[e]}
                if not by_doc:
                    continue
                cell = {}
                mine = (lambda lang, e=e: registry.claims(e, lang))  # noqa: E731
                for status in ("clean", "unknown", "excluded"):
                    sets = [d for d in by_doc if registry.status(e, d) == status]
                    if not sets:
                        continue
                    cell[status] = {"datasets": sets,
                                    "claimed": _agg(by_doc, sets, mine),
                                    "outside_claim": _agg(by_doc, sets, lambda lang, mine=mine: not mine(lang))}
                    if status == "clean" and cell[status]["claimed"]:
                        pairs = {d: (v, v) for d in sets if (v := _pick(by_doc[d], mine))}
                        res = scorer.macro_bootstrap(pairs, samples=samples)["metrics"]
                        cell[status]["ci"] = {k: {"value": m["a"], "ci": m["a_ci"]} for k, m in res.items()}
                cells[group] = cell
            others = [e for e in entries if e != baseline and not e.startswith("sct")
                      and any(registry.status(e, d) == "clean" for d in present)]
            for pool, chosen in (("all", entries), ("others", others)):
                sets = [d for d in present if chosen and all(registry.status(e, d) == "clean" for e in chosen)]
                keep = (lambda lang, chosen=chosen: all(registry.claims(e, lang) for e in chosen))  # noqa: E731
                scores = {e: _agg({d: per_doc[e][d][view] for d in sets}, sets, keep) for e in chosen} if sets else {}
                same_set.setdefault(view, {}).setdefault(group, {})[pool] = {
                    "entries": chosen, "datasets": sets, "scores": {e: a for e, a in scores.items() if a}}
            base = per_doc.get(baseline, {})
            for e in entries:
                if e == baseline:
                    continue
                keep = (lambda lang, e=e: registry.claims(e, lang) and registry.claims(baseline, lang))  # noqa: E731
                both = [d for d in members if d in base and d in per_doc[e]]
                statuses = {d: {registry.status(e, d), registry.status(baseline, d)} for d in both}
                for kind, sets in (("clean", [d for d in both if statuses[d] == {"clean"}]),
                                   ("unknown", [d for d in both if statuses[d] in ({"unknown"}, {"clean", "unknown"})])):
                    pairs = {d: (a, b) for d in sets
                             if (a := _pick(base[d][view], keep)) and (b := _pick(per_doc[e][d][view], keep))}
                    if pairs:
                        pairwise.setdefault(e, {}).setdefault(view, {}).setdefault(group, {})[kind] = {
                            "datasets": list(pairs), **scorer.macro_bootstrap(pairs, samples=samples)}
    return {"own": own, "same_set": same_set, "pairwise": pairwise}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--entries", nargs="+", required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--part", choices=split.PARTS, required=True)
    parser.add_argument("--track", choices=registry.TRACKS, default="gold",
                        help="language track for every entry (PROTOCOL.md section 12; gold is the main track)")
    parser.add_argument("--n", type=int, help="docs per language (default: all of the part)")
    parser.add_argument("--seed", type=int, default=split.SEED)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=CACHE, help="raw outputs (hold dataset text) and venvs")
    parser.add_argument("--python", action="append", default=[], metavar="ENTRY=PATH",
                        help="use this interpreter instead of building the entry's venv")
    parser.add_argument("--config", action="append", default=[], metavar="ENTRY=JSON",
                        help="adapter config overrides (tuning on dev only; logged)")
    parser.add_argument("--tuned", type=Path, help="frozen tune.py configs (tuned track), applied under --config")
    parser.add_argument("--baseline", default="sct-default", help="paired bootstrap: baseline minus each entry")
    parser.add_argument("--doc-timeout", type=float, default=120.0)
    parser.add_argument("--start-timeout", type=float, default=1800.0, help="model download + load allowance")
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--threads", type=int, help="thread caps (env + adapter config)")
    parser.add_argument("--cpus", help="taskset CPU list, e.g. 0-15")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--gpu", help="GPU index (default: the one with most free memory)")
    parser.add_argument("--gpu-mem-gb", type=float, default=5.0)
    parser.add_argument("--hf-hub-cache", help="HF_HUB_CACHE for the adapters")
    parser.add_argument("--reuse", action="store_true", help="score cached raw outputs when their cache key matches")
    parser.add_argument("--skip-offset-check", action="store_true")
    parser.add_argument("--no-recover", action="store_true",
                        help="score without recovered dropped copies (scorer.recover_dropped): the round-1 mapping")
    args = parser.parse_args(argv)

    pythons = dict(item.split("=", 1) for item in args.python)
    tuned = json.loads(args.tuned.read_text())["entries"] if args.tuned else {}
    overrides = {k: v["config"] for k, v in tuned.items()}
    for k, v in (item.split("=", 1) for item in args.config):
        overrides[k] = {**overrides.get(k, {}), **json.loads(v)}
    tools = {name: registry.tool(name, args.track) for name in args.entries}
    for name, t in tools.items():
        if t["adapter"] is None:
            raise SystemExit(f"entry {name!r} has no adapter yet")
    gpu = None if args.device == "cpu" else (args.gpu or pick_gpu(args.gpu_mem_gb + 1))
    env = child_env(args.threads, args.device, gpu, args.hf_hub_cache)
    shared = scorer.shared_types([t for n, t in tools.items() if n not in registry.TEST_ENTRIES] or list(tools.values()),
                                 registry.SCT_TYPES)
    run_info = {"command": " ".join(["python -m benchmarks.compare.runner", *(argv or sys.argv[1:])]),
                "part": args.part, "track": args.track,
                "tuned": str(args.tuned) if args.tuned else None, "n_per_language": args.n, "seed": args.seed,
                "threads": args.threads,
                "cpus": args.cpus, "device": args.device, "gpu": gpu, "gpu_mem_gb": args.gpu_mem_gb,
                "batch_size": args.batch_size, "doc_timeout": args.doc_timeout, "retries": args.retries,
                "shared_types": list(shared), "environment": environment()}
    _write(args.out / "run.json", run_info)
    per_doc: dict[str, dict[str, list]] = {name: {} for name in tools}
    records_by_dataset = {}
    for name, t in tools.items():
        py = Path(pythons[name]) if name in pythons else ensure_env(name, t["requirements"], args.cache / "venvs")
        config = {**t["adapter_config"], "threads": args.threads, "device": args.device,
                  "gpu_mem_gb": args.gpu_mem_gb, **overrides.get(name, {})}
        cmd = command(py, t["adapter"], config, args.cpus)
        (args.out / "envs").mkdir(parents=True, exist_ok=True)
        frozen = freeze(py)
        (args.out / "envs" / f"{name}.freeze.txt").write_text(frozen)
        meta = {"entry": name, "entry_config": _jsonable({k: v for k, v in t.items() if k != "requirements"}),
                "adapter_config": config, "python": str(py)}
        if not args.skip_offset_check:
            fixtures = [{"id": k, "lang": lang, "text": text} for k, (lang, text) in OFFSET_FIXTURES.items()]
            res, _ = supervise(cmd, fixtures, args.cache / name / "offsets", env, args.doc_timeout, args.start_timeout)
            problems, misses = check_offsets(res, t["labels"])
            _write(args.out / name / "offset-check.json",
                   {"passed": not problems, "problems": problems, "detection_errors": misses})
            if problems:
                print(f"[{name}] offset check failed, skipped: {problems}", file=sys.stderr)
                continue
        for dataset in args.datasets:
            module = get(dataset)
            if dataset not in records_by_dataset:
                records_by_dataset[dataset] = split.load(dataset, args.part, args.n, args.seed)
            records = records_by_dataset[dataset]
            raw = args.cache / name / f"{dataset}-{args.part}.jsonl"
            stamp = raw.with_suffix(".key")
            key = cache_key(config, t["adapter"], frozen, records)
            outputs = {}
            if args.reuse and raw.exists() and stamp.exists() and stamp.read_text().strip() == key:
                outputs = {line["id"]: line for line in map(json.loads, raw.read_text(encoding="utf-8").splitlines())}
            started = time.perf_counter()
            if not {r["id"] for r in records} <= set(outputs):
                print(f"[{name}] {dataset} {args.part}: {len(records)} docs", file=sys.stderr, flush=True)
                outputs, process_info = supervise(cmd, records, args.cache / name / f"{dataset}-{args.part}", env,
                                                  DOC_TIMEOUT.get(dataset, args.doc_timeout), args.start_timeout,
                                                  args.retries, args.batch_size)
                raw.write_text("".join(json.dumps(outputs[r["id"]], ensure_ascii=False) + "\n" for r in records),
                               encoding="utf-8")
                stamp.write_text(key)
            else:
                process_info = {"reused": str(raw)}
            views = scorer.views_for(t["supports"], shared, registry.SCT_TYPES)
            result, docs = scorer.score_run(records, outputs, t["labels"], DATASET_LABELS[dataset], views,
                                            curve_floor=curve_floor(config, t),
                                            curve_by_filter=t.get("curve_by_filter", True),
                                            recover=t.get("recover_dropped", False) and not args.no_recover)
            per_doc[name][dataset] = docs
            ids = sorted(r["id"] for r in records)
            result.update(meta, dataset={"name": dataset, "source": module.HF_ID, "revision": module.REVISION,
                                         "licence": module.LICENCE, "publish": module.PUBLISH},
                          contamination=registry.contamination(name, dataset), process=process_info,
                          wall_seconds=round(time.perf_counter() - started, 1),
                          sample_id_sha256=hashlib.sha256("\n".join(ids).encode()).hexdigest(),
                          seconds_total=round(sum(o.get("seconds") or 0 for o in outputs.values()), 3))
            if module.PUBLISH == "scores-and-ids":
                result["failed_ids"] = sorted(r["id"] for r in records if "error" in outputs[r["id"]])
                (args.out / "predictions" / name).mkdir(parents=True, exist_ok=True)
                (args.out / "predictions" / name / f"{dataset}-{args.part}.jsonl").write_text("".join(
                    json.dumps({"id": r["id"], "start": s["start"], "end": s["end"], "label": s["label"],
                                "score": s.get("score")}) + "\n"
                    for r in records for s in outputs[r["id"]]["spans"]), encoding="utf-8")
            _write(args.out / name / f"{dataset}-{args.part}.json", result)
    summary = {"aggregate": {}, "paired": {}}
    for name, by_dataset in per_doc.items():
        summary["aggregate"][name] = {}
        for view in ("supported", "shared", "full"):
            agg = {}
            for group, members in scorer.GROUPS.items():
                chosen = {d: [(lang, c) for _, _, lang, c in by_dataset[d][view]] for d in members if d in by_dataset}
                if chosen:
                    agg[group] = scorer.aggregate(chosen)
            summary["aggregate"][name][view] = agg
    base = per_doc.get(args.baseline, {})
    for name, by_dataset in per_doc.items():
        if name == args.baseline:
            continue
        for dataset, docs in by_dataset.items():
            if dataset in base:
                summary["paired"].setdefault(name, {})[dataset] = scorer.paired_bootstrap(base[dataset]["shared"],
                                                                                          docs["shared"])
    summary.update(rule_views(per_doc, args.baseline))
    summary["paired_note"] = (f"a = {args.baseline}, b = entry, effective protection (failed docs unmasked); paired: "
                              "shared view per dataset, every cell; own / same_set / pairwise: decision 21 and the "
                              "claimed-language rule (PROTOCOL.md sections 12 and 17)")
    _write(args.out / "summary.json", summary)
    print(args.out / "summary.json")


if __name__ == "__main__":
    main()
