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
import ast
import hashlib
import json
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, NamedTuple

from benchmarks.compare import entries as registry
from benchmarks.compare import runtime, scorer, split
from benchmarks.compare.views import rule_views
from benchmarks.datasets import ROUND2, get
from benchmarks.labels import DATASET_LABELS, unknown_gold_labels
from benchmarks.speed import environment
from sct.utils.atomic_file import write_text_atomic

DOC_TIMEOUT = {"tab": 600.0, "redactionbench": 600.0}  # PROTOCOL.md section 15; others use --doc-timeout
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


def shared_view(datasets, tools: dict) -> tuple:
    """Types every comparison entry of the round supports (a run of test entries only: those entries)."""
    if set(datasets) & set(ROUND2) and set(datasets) - set(ROUND2):
        raise ValueError("a run mixes round-1 and round-2 datasets; run each round on its own (shared view per round)")
    pool = registry.ENTRIES if set(datasets) & set(ROUND2) else {
        k: v for k, v in registry.ENTRIES.items() if k not in registry.ROUND2_ENTRIES}
    compared = list(pool.values()) if set(tools) - set(registry.TEST_ENTRIES) else list(tools.values())
    return scorer.shared_types(compared, registry.SCT_TYPES)


def _take(lines: list[dict], expected: set[str], results: dict[str, dict]) -> int:
    fresh = [line for line in lines if line.get("id") in expected]
    results.update((line["id"], line) for line in fresh)
    return len(fresh)


def _watch(proc: subprocess.Popen, out: Path, expected: set[str], results: dict[str, dict], doc_limit: float,
           start_timeout: float) -> tuple[int, str | None]:
    """Collect *proc*'s lines until it exits or goes silent; returns (lines taken, why it stopped early)."""
    marker = Path(f"{out}.ready")
    offset, last, got, ready = 0, time.monotonic(), 0, False
    while True:
        lines, offset = runtime.read_lines(out, offset)
        got += _take(lines, expected, results)
        if lines or (not ready and marker.exists()):
            ready, last = True, time.monotonic()
        code = proc.poll()
        if code is not None:
            got += _take(runtime.read_lines(out, offset)[0], expected, results)
            return got, None if code == 0 else f"crash: exit {code}"
        limit = doc_limit + (0 if ready else start_timeout)
        if time.monotonic() - last > limit:
            proc.kill()
            proc.wait()
            return got, f"timeout: no output for {limit:.0f}s"
        time.sleep(runtime.POLL_SECONDS)


def _abort(docs: list[dict], streak: int, reason: str) -> dict[str, dict]:
    message = f"aborted: {streak} docs in a row failed without output ({reason})"
    return {d["id"]: {"id": d["id"], "spans": [], "error": message, "seconds": None} for d in docs}


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
        got, reason = _watch(proc, out, {d["id"] for d in pending}, results, doc_timeout * batch_size, start_timeout)
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
                results.update(_abort(pending, streak, reason))
                pending = []
    return results, {"processes": attempts, "restarts": dict(restarts)}


def _adapter_sources(adapter: str) -> list[tuple[str, str]]:
    """Checkout benchmark code; installed SCT is identified by the entry's environment freeze."""
    benchmark_root = runtime.ADAPTERS.parents[1]
    checkout = benchmark_root.parent
    pending = [runtime.ADAPTERS / adapter]
    sources: dict[Path, str] = {}
    while pending:
        path = pending.pop()
        if path in sources:
            continue
        source = path.read_text(encoding="utf-8")
        sources[path] = source
        for node in ast.walk(ast.parse(source)):
            modules = ([node.module, *(f"{node.module}.{alias.name}" for alias in node.names)]
                       if isinstance(node, ast.ImportFrom) else
                       [alias.name for alias in node.names] if isinstance(node, ast.Import) else [])
            for module in modules:
                if module == "benchmarks" or (module and module.startswith("benchmarks.")):
                    target = checkout.joinpath(*module.split("."))
                    dependency = target.with_suffix(".py")
                    if not dependency.is_file():
                        dependency = target / "__init__.py"
                    if dependency.is_file():
                        pending.append(dependency)
                        package = dependency.parent
                        while package.is_relative_to(benchmark_root):
                            initializer = package / "__init__.py"
                            if initializer.is_file():
                                pending.append(initializer)
                            package = package.parent
                elif module and "." not in module:
                    dependency = runtime.ADAPTERS / f"{module}.py"
                    if dependency.is_file():
                        pending.append(dependency)
    return sorted((path.relative_to(checkout).as_posix(), source) for path, source in sources.items())


def cache_key(config: dict, adapter: str, frozen: str, records: list[dict]) -> str:
    """Fingerprint of everything that makes a raw output: adapter config and code, venv, input docs."""
    h = hashlib.sha256()
    for part in (json.dumps(config, sort_keys=True), json.dumps(_adapter_sources(adapter)), frozen):
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
    skipped). No overlapping EMAIL span, one that covers the whole e-mail plus extra text, or one
    that starts on the e-mail and stops at its own ``.`` / ``@`` (a model tagging only the first
    subwords; a shifted or inclusive-end span cannot land there) is the model's detection error:
    recorded, not a skip.
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
            if all((a <= start and b >= end) or (a == start and b < end and text[b] in ".@") for a, b in overlapping):
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


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    runtime.add_entry_args(parser)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--part", choices=split.PARTS, required=True)
    parser.add_argument("--track", choices=registry.TRACKS, default="gold",
                        help="language track for every entry (PROTOCOL.md section 12; gold is the main track)")
    parser.add_argument("--n", type=int, help="docs per language (default: all of the part)")
    parser.add_argument("--seed", type=int, default=split.SEED)
    parser.add_argument("--config", action="append", default=[], metavar="ENTRY=JSON",
                        help="adapter config overrides (tuning on dev only; logged)")
    parser.add_argument("--tuned", type=Path, help="frozen tune.py configs (tuned track), applied under --config")
    parser.add_argument("--baseline", default="sct-default", help="paired bootstrap: baseline minus each entry")
    parser.add_argument("--doc-timeout", type=float, default=120.0)
    parser.add_argument("--start-timeout", type=float, default=1800.0, help="model download + load allowance")
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--reuse", action="store_true", help="score cached raw outputs when their cache key matches")
    parser.add_argument("--skip-offset-check", action="store_true")
    parser.add_argument("--no-recover", action="store_true",
                        help="score without recovered dropped copies (scorer.recover_dropped): the round-1 mapping")
    return parser.parse_args(argv)


def _config_overrides(args: argparse.Namespace) -> dict[str, dict]:
    tuned = json.loads(args.tuned.read_text())["entries"] if args.tuned else {}
    overrides = {k: v["config"] for k, v in tuned.items()}
    for k, v in (item.split("=", 1) for item in args.config):
        overrides[k] = {**overrides.get(k, {}), **json.loads(v)}
    return overrides


class _EntryRun(NamedTuple):
    """One entry made ready to run: what each of its datasets needs."""

    name: str
    tool: dict[str, Any]
    cmd: list[str]
    env: dict[str, str]
    config: dict[str, Any]
    frozen: str
    shared: tuple[str, ...]
    meta: dict[str, Any]


def _offsets_pass(run: _EntryRun, args: argparse.Namespace) -> bool:
    fixtures = [{"id": k, "lang": lang, "text": text} for k, (lang, text) in OFFSET_FIXTURES.items()]
    res, _ = supervise(run.cmd, fixtures, args.cache / run.name / "offsets", run.env, args.doc_timeout, args.start_timeout)
    problems, misses = check_offsets(res, run.tool["labels"])
    runtime.write_json(args.out / run.name / "offset-check.json",
           {"passed": not problems, "problems": problems, "detection_errors": misses})
    if problems:
        print(f"[{run.name}] offset check failed, skipped: {problems}", file=sys.stderr)
    return not problems


def _cached_outputs(raw: Path, stamp: Path, key: str) -> dict[str, dict]:
    if raw.exists() and stamp.exists() and stamp.read_text().strip() == key:
        return {line["id"]: line for line in map(json.loads, raw.read_text(encoding="utf-8").splitlines())}
    return {}


def _score_dataset(run: _EntryRun, dataset: str, records: list[dict], args: argparse.Namespace) -> dict[str, list]:
    """Run (or reuse) one entry on one dataset, write its result files, return its ``per_doc``."""
    name, tool = run.name, run.tool
    module = get(dataset)
    raw = args.cache / name / f"{dataset}-{args.part}.jsonl"
    stamp = raw.with_suffix(".key")
    key = cache_key(run.config, tool["adapter"], run.frozen, records)
    outputs = _cached_outputs(raw, stamp, key) if args.reuse else {}
    started = time.perf_counter()
    if {r["id"] for r in records} <= set(outputs):
        process_info = {"reused": str(raw)}
    else:
        print(f"[{name}] {dataset} {args.part}: {len(records)} docs", file=sys.stderr, flush=True)
        outputs, process_info = supervise(run.cmd, records, args.cache / name / f"{dataset}-{args.part}", run.env,
                                          DOC_TIMEOUT.get(dataset, args.doc_timeout), args.start_timeout,
                                          args.retries, args.batch_size)
        write_text_atomic(raw, "".join(json.dumps(outputs[r["id"]], ensure_ascii=False) + "\n" for r in records))
        write_text_atomic(stamp, key)
    views = scorer.views_for(tool["supports"], run.shared, registry.SCT_TYPES)
    result, docs = scorer.score_run(records, outputs, tool["labels"], DATASET_LABELS[dataset], views,
                                    curve_floor=curve_floor(run.config, tool),
                                    curve_by_filter=tool.get("curve_by_filter", True),
                                    recover=tool.get("recover_dropped", False) and not args.no_recover)
    if getattr(module, "DELETE_PREDICTIONS", False):
        raw.unlink(missing_ok=True)
        stamp.unlink(missing_ok=True)
        shutil.rmtree(args.cache / name / f"{dataset}-{args.part}", ignore_errors=True)
        process_info = {**process_info, "predictions_deleted": True}
    ids = sorted(r["id"] for r in records)
    result.update(run.meta, dataset={"name": dataset, "source": module.HF_ID, "revision": module.REVISION,
                                     "licence": module.LICENCE, "publish": module.PUBLISH},
                  contamination=registry.contamination(name, dataset), process=process_info,
                  wall_seconds=round(time.perf_counter() - started, 1),
                  sample_id_sha256=hashlib.sha256("\n".join(ids).encode()).hexdigest(),
                  seconds_total=round(sum(o.get("seconds") or 0 for o in outputs.values()), 3))
    if module.PUBLISH == "scores-and-ids":
        result["failed_ids"] = sorted(r["id"] for r in records if "error" in outputs[r["id"]])
        predictions = args.out / "predictions" / name / f"{dataset}-{args.part}.jsonl"
        predictions.parent.mkdir(parents=True, exist_ok=True)
        write_text_atomic(predictions, "".join(
            json.dumps({"id": r["id"], "start": s["start"], "end": s["end"], "label": s["label"],
                        "score": s.get("score")}) + "\n"
            for r in records for s in outputs[r["id"]]["spans"]))
    runtime.write_json(args.out / name / f"{dataset}-{args.part}.json", result)
    return docs


def _summary(per_doc: dict[str, dict[str, Any]], baseline: str) -> dict[str, Any]:
    summary: dict[str, Any] = {"aggregate": {}, "paired": {}}
    for name, by_dataset in per_doc.items():
        summary["aggregate"][name] = {}
        for view in ("supported", "shared", "full"):
            agg = {}
            for group, members in scorer.GROUPS.items():
                chosen = {d: [(lang, c) for _, _, lang, c in by_dataset[d][view]] for d in members if d in by_dataset}
                if chosen:
                    agg[group] = scorer.aggregate(chosen)
            summary["aggregate"][name][view] = agg
    base = per_doc.get(baseline, {})
    for name, by_dataset in per_doc.items():
        if name == baseline:
            continue
        for dataset, docs in by_dataset.items():
            if dataset in base:
                summary["paired"].setdefault(name, {})[dataset] = scorer.paired_bootstrap(base[dataset]["shared"],
                                                                                          docs["shared"])
    summary.update(rule_views(per_doc, baseline))
    summary["paired_note"] = (f"a = {baseline}, b = entry, effective protection (failed docs unmasked); paired: "
                              "shared view per dataset, every cell; own / same_set / pairwise: decision 21 and the "
                              "claimed-language rule (PROTOCOL.md sections 12 and 17)")
    return summary


def main(argv=None) -> None:
    args = _parse_args(argv)
    overrides = _config_overrides(args)
    tools = {name: registry.tool(name, args.track) for name in args.entries}
    for name, t in tools.items():
        if t["adapter"] is None:
            raise SystemExit(f"entry {name!r} has no adapter yet")
    gpu, _ = runtime.device_env(args)
    shared = shared_view(args.datasets, tools)
    run_info = {"command": " ".join(["python -m benchmarks.compare.runner", *(argv or sys.argv[1:])]),
                "part": args.part, "track": args.track,
                "tuned": str(args.tuned) if args.tuned else None, "n_per_language": args.n, "seed": args.seed,
                "threads": args.threads,
                "cpus": args.cpus, "device": args.device, "gpu": gpu, "gpu_mem_gb": args.gpu_mem_gb,
                "batch_size": args.batch_size, "doc_timeout": args.doc_timeout, "retries": args.retries,
                "shared_types": list(shared), "environment": environment()}
    runtime.write_json(args.out / "run.json", run_info)
    per_doc: dict[str, dict[str, Any]] = {name: {} for name in tools}
    records_by_dataset = {}
    for name, t in tools.items():
        py = runtime.entry_python(name, t, args)
        config = runtime.entry_config(t, args, overrides.get(name))
        cmd = runtime.command(py, t["adapter"], config, args.cpus)
        env = runtime.child_env(args.threads, config["device"], gpu, args.hf_hub_cache)
        (args.out / "envs").mkdir(parents=True, exist_ok=True)
        frozen = runtime.freeze(py)
        write_text_atomic(args.out / "envs" / f"{name}.freeze.txt", frozen)
        meta = {"entry": name, "entry_config": _jsonable({k: v for k, v in t.items() if k != "requirements"}),
                "adapter_config": config, "python": str(py)}
        run = _EntryRun(name, t, cmd, env, config, frozen, shared, meta)
        if not args.skip_offset_check and not _offsets_pass(run, args):
            continue
        for dataset in args.datasets:
            if dataset not in records_by_dataset:
                records_by_dataset[dataset] = split.load(dataset, args.part, args.n, args.seed)
                unknown = unknown_gold_labels(dataset, records_by_dataset[dataset])
                if unknown:
                    raise SystemExit(f"{dataset}: gold labels neither mapped nor out of scope: {unknown}; add each to "
                                     "labels.DATASET_LABELS or labels.DATASET_OUT_OF_SCOPE")
            per_doc[name][dataset] = _score_dataset(run, dataset, records_by_dataset[dataset], args)
    runtime.write_json(args.out / "summary.json", _summary(per_doc, args.baseline))
    print(args.out / "summary.json")


if __name__ == "__main__":
    main()
