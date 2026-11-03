"""Run SCT over the open benchmark datasets and write scores.

    python -m benchmarks.run --suite all --out benchmarks/results/0.8.0/

Writes one ``<dataset>.json`` per dataset (each with the run that made it: command,
seed, versions, hardware), ``run.json`` (the latest run) and ``summary.md`` (built
from every json in ``--out``). Results hold
scores, counts and, only where the dataset allows it, sample ids: never dataset text.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

from benchmarks import datasets
from benchmarks.labels import DATASET_LABELS
from benchmarks.score import inject_noise, masking_recall, score_lid, score_noise, score_pii
from benchmarks.speed import environment, measure, rates, run_batch
from scripts.benchmark_release import predicted_spans

ROOT = Path(__file__).resolve().parent
SUITES = {
    "pii": ("openpii", "gretel", "nemotron", "redact", "iahlt", "bayanat", "thainer", "thai_pii",
            "naamapadam", "multiconer", "stockmark", "gretel_en", "argilla_pii"),
    "long": ("tab",),
    "lid": ("wili",),
    "speed": ("fineweb2",),
    "clean": ("fineweb2",),
}
# Scoring needs each placeholder to map back to the value it replaced.
SCORING = {"replacement_mode": "reversible", "check_to_ascii_unicode": False}
DEFAULT_N = {"pii": 300, "long": None, "lid": 100, "speed": 100, "clean": 100}


def config_overrides(name: str) -> dict:
    from sct.presets import PRESETS

    if name == "default":
        return {}
    if name == "training":
        return dict(PRESETS["training"])
    if name == "ner-off":
        return {"check_ner_process": False}
    raise ValueError(f"unknown config {name!r}: default, training or ner-off")


class Cleaners:
    """One TextCleaner per (config, scoring) pair, built on first use."""

    def __init__(self, args):
        self.args, self.cache = args, {}

    def overrides(self, name: str, scoring: bool) -> dict:
        args = self.args
        cfg = {**config_overrides(name), **(SCORING if scoring else {}),
               "onnx_intra_op_threads": args.threads, "ner_device": args.device}
        if args.device == "cuda" and args.gpu_mem_gb:
            cfg["onnx_cuda_provider_options"] = {"gpu_mem_limit": int(args.gpu_mem_gb * 2**30)}
        return cfg

    def get(self, name: str, scoring: bool = True):
        key = (name, scoring)
        if key not in self.cache:
            from sct.config import TextCleanerConfig
            from sct.sct import TextCleaner

            self.cache[key] = TextCleaner(cfg=TextCleanerConfig(**self.overrides(name, scoring)))
        return self.cache[key]

    def models(self) -> dict:
        cleaner = next(iter(self.cache.values()), None)
        if cleaner is None:
            return {}
        cfg = cleaner.cfg
        return {"ner_backend": cfg.ner_backend, "ner_models": dict(cfg.ner_models or {}),
                "ner_model_revisions": dict(cfg.ner_model_revisions or {})}


def predict(cleaner, records: list[dict], workers: int) -> tuple[dict, dict, float]:
    """Return ``(predicted spans, outputs, seconds)`` keyed by record id."""
    results, seconds = run_batch(cleaner, [r["text"] for r in records], workers)
    predictions, outputs = {}, {}
    for record, result in zip(records, results, strict=True):
        anon_map = result.metadata.get("anon_map")
        originals = {e.placeholder: e.original for e in anon_map.entries} if anon_map else {}
        outputs[record["id"]] = result.lm_text
        predictions[record["id"]] = [
            {"start": s.start, "end": s.end, "label": s.label}
            for s in predicted_spans(record["text"], result.lm_text, originals)]
    return predictions, outputs, seconds


def run_dataset(name: str, suite: str, args, cleaners: Cleaners) -> dict:
    module = datasets.get(name)
    n = args.n if args.n is not None else DEFAULT_N[suite]
    started = time.perf_counter()
    records = module.load(n, args.seed)
    result = {
        "dataset": {"name": name, "source": module.HF_ID, "revision": module.REVISION,
                    "licence": module.LICENCE, "publish": module.PUBLISH},
        "suite": suite, "seed": args.seed, "n_per_language": n, "documents": len(records),
        "languages": sorted({r["lang"] for r in records}), "load_seconds": round(time.perf_counter() - started, 1),
        "configs": {},
    }
    if module.PUBLISH == "scores-and-ids":
        result["sample_ids"] = [r["id"] for r in records]
    chars = sum(len(r["text"]) for r in records)
    if suite in ("pii", "long"):
        for config in args.configs:
            predictions, outputs, seconds = predict(cleaners.get(config), records, args.workers)
            scores = score_pii(records, predictions, outputs, DATASET_LABELS[name])
            if suite == "long":
                scores["masking_recall"] = masking_recall(records, predictions)
            result["configs"][config] = {**scores, "time": rates(len(records), chars, seconds)}
    elif suite == "lid":
        from lingua import Language

        cleaner = cleaners.get("default", scoring=False)
        started = time.perf_counter()
        guesses = {}
        for record in records:
            lingua_name = cleaner._detect_language(record["text"])[0]  # the detector process() uses
            guesses[record["id"]] = (getattr(Language, lingua_name).iso_code_639_3.name.lower()
                                     if lingua_name else None)
        seconds = time.perf_counter() - started
        in_set = {lang.iso_code_639_3.name.lower() for lang in Language.all()}
        result["configs"]["default"] = {**score_lid(records, guesses, in_set), "time": rates(len(records), chars, seconds)}
    elif suite == "speed":
        for config in args.configs:
            result["configs"][config] = measure(cleaners.get(config, scoring=False), records, args.workers)
    elif suite == "clean":
        rng = random.Random(args.seed)  # noqa: S311  # noise placement, not crypto
        noisy, items = [], {}
        for record in records:
            text, items[record["id"]] = inject_noise(record["text"], rng)
            noisy.append({**record, "text": text})
        for config in args.configs:
            _, outputs, seconds = predict(cleaners.get(config, scoring=False), noisy, args.workers)
            result["configs"][config] = {"noise": score_noise(items, outputs),
                                         "time": rates(len(noisy), sum(len(r["text"]) for r in noisy), seconds)}
    return result


def _pct(value) -> str:
    return "-" if value is None else f"{100 * value:.1f}"


def summary(out: Path) -> str:
    """Markdown tables from every result json in *out*."""
    run = json.loads((out / "run.json").read_text(encoding="utf-8")) if (out / "run.json").exists() else {}
    env = run.get("environment", {})
    lines = [f"# SCT {run.get('sct_version', '?')} benchmark", "",
             f"Commit `{env.get('git_commit')}` (dirty: {env.get('git_dirty')}). {env.get('cpu_model')}, "
             f"{env.get('cpus_usable') or env.get('logical_cpus')} CPUs, GPUs: {', '.join(env.get('gpus', [])) or 'none'}.",
             f"Device `{run.get('device')}`, ONNX threads {run.get('threads')}, workers {run.get('workers')}, "
             f"seed {run.get('seed')}. Command: `{run.get('command')}`", "",
             "Percentages. Strict = exact span + type. Partial = overlap + type. Masked R = overlap, any type. "
             "Leak = gold PII value (or a 4+ char word of it) still in the output; lower is better.", ""]
    results = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(out.glob("*.json")) if p.stem != "run"}
    pii = {k: v for k, v in results.items() if v.get("suite") in ("pii", "long") and "configs" in v}
    if pii:
        lines += ["## PII", "", "| dataset | config | docs | strict F1 | partial P | partial R | masked R | leak | docs/s |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for name, res in pii.items():
            for config, c in res["configs"].items():
                lines.append(f"| {name} | {config} | {c['documents']} | {_pct(c['strict']['overall']['f1'])} | "
                             f"{_pct(c['partial']['overall']['precision'])} | {_pct(c['partial']['overall']['recall'])} | "
                             f"{_pct(c['partial']['overall'].get('masked_recall'))} | "
                             f"{_pct(c['leak']['overall']['leak_rate'])} | {c['time']['docs_per_second']} |")
        for name, res in pii.items():
            config = next(iter(res["configs"]))
            c = res["configs"][config]
            lines += ["", f"### {name} ({config}): by type and language", "",
                      "| | gold | partial R | masked R | partial P | strict F1 | leak |", "|---|---|---|---|---|---|---|"]
            for group in ("by_label", "by_language"):
                for key, m in c["partial"][group].items():
                    strict = c["strict"][group].get(key, {})
                    leak = c["leak"][group].get(key, {})
                    lines.append(f"| {key} | {m['gold']} | {_pct(m['recall'])} | {_pct(m.get('masked_recall'))} | "
                                 f"{_pct(m['precision'])} | "
                                 f"{_pct(strict.get('f1'))} | {_pct(leak.get('leak_rate'))} |")
            if "masking_recall" in c:
                mr = c["masking_recall"]
                lines += ["", f"TAB masking recall (any type, macro over {mr['annotators']} annotators): "
                          f"all {_pct(mr['all']['macro_recall'])}, direct {_pct(mr['direct']['macro_recall'])}, "
                          f"quasi {_pct(mr['quasi']['macro_recall'])}.",
                          "Partial recall by start offset: " + ", ".join(
                              f"{k}: {_pct(v['recall'])} ({v['gold']})" for k, v in c["by_position"].items())]
    for name, res in results.items():
        if res.get("suite") == "lid" and "configs" in res:
            c = res["configs"]["default"]
            lines += ["", f"## Language ID ({name})", "",
                      f"Accuracy {_pct(c['accuracy'])}, macro {_pct(c['macro_accuracy'])} over "
                      f"{c['in_set_documents']} in-set docs. Out-of-set docs given a language: "
                      f"{_pct(c['out_of_set']['false_positive_rate'])} of {c['out_of_set']['documents']}.", "",
                      "| script | docs | accuracy |", "|---|---|---|",
                      *(f"| {k} | {v['total']} | {_pct(v['accuracy'])} |" for k, v in c["by_script"].items())]
        if res.get("suite") == "speed" and "configs" in res:
            lines += ["", f"## Speed ({name})", "", "| config | script | docs | docs/s | chars/s |", "|---|---|---|---|---|"]
            for config, c in res["configs"].items():
                for key, m in [("all", c["total"]), *c["by_script"].items()]:
                    lines.append(f"| {config} | {key} | {m['documents']} | {m['docs_per_second']} | {m['chars_per_second']} |")
        if res.get("suite") == "clean" and "configs" in res:
            lines += ["", f"## Cleaning: injected noise removed ({name})", "", "| config | kind | injected | removed |",
                      "|---|---|---|---|"]
            for config, c in res["configs"].items():
                lines += [f"| {config} | {k} | {v['injected']} | {_pct(v['removal_rate'])} |" for k, v in c["noise"].items()]
    errors = {k: v["error"] for k, v in results.items() if "error" in v}
    if errors:
        lines += ["", "## Not run", "", *(f"- {k}: {v}" for k, v in errors.items())]
    return "\n".join(lines) + "\n"


def main(argv=None) -> None:
    import sct

    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--suite", nargs="+", default=["all"], choices=["all", *SUITES])
    parser.add_argument("--datasets", nargs="+", help="only these datasets of the chosen suites")
    parser.add_argument("--configs", nargs="+", default=["default", "training", "ner-off"],
                        choices=["default", "training", "ner-off"])
    parser.add_argument("--out", type=Path, help="default: benchmarks/results/<sct version>/")
    parser.add_argument("--sct-version", default=sct.__version__, help="must match the installed sct")
    parser.add_argument("--n", type=int, help="docs per language (default per suite: pii 300, lid 100, ...)")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--threads", type=int, default=4, help="ONNX intra-op threads")
    parser.add_argument("--workers", type=int, default=4, help="process_batch document workers")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--gpu-mem-gb", type=float, help="cap ONNX CUDA memory (gpu_mem_limit)")
    args = parser.parse_args(argv)
    if args.sct_version != sct.__version__:
        raise SystemExit(f"--sct-version {args.sct_version} but sct {sct.__version__} is installed")
    out = args.out or ROOT / "results" / sct.__version__
    out.mkdir(parents=True, exist_ok=True)
    suites = list(SUITES) if "all" in args.suite else args.suite
    cleaners = Cleaners(args)
    started = time.perf_counter()
    run = {
        "sct_version": sct.__version__, "command": " ".join(["python -m benchmarks.run", *(argv or sys.argv[1:])]),
        "suites": suites, "configs": args.configs, "seed": args.seed, "n": args.n, "threads": args.threads,
        "workers": args.workers, "device": args.device, "gpu_mem_gb": args.gpu_mem_gb,
        "scoring_overrides": SCORING, "environment": environment(),
    }
    for suite in suites:
        for name in SUITES[suite]:
            if args.datasets and name not in args.datasets:
                continue
            target = out / (f"{name}-clean.json" if suite == "clean" else f"{name}.json")
            print(f"[{suite}] {name} ...", file=sys.stderr, flush=True)
            try:
                result = run_dataset(name, suite, args, cleaners)
            except Exception as error:  # noqa: BLE001  # one missing dataset must not stop a long run
                result = {"suite": suite, "error": f"{type(error).__name__}: {error}"}
                print(f"  failed: {result['error']}", file=sys.stderr)
            result["run"] = {**run, "models": cleaners.models()}
            target.write_text(json.dumps(result, indent=1, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    run.update(models=cleaners.models(), wall_seconds=round(time.perf_counter() - started, 1))
    (out / "run.json").write_text(json.dumps(run, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    (out / "summary.md").write_text(summary(out), encoding="utf-8")
    print(out / "summary.md")


if __name__ == "__main__":
    main()
