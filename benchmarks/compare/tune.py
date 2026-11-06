"""Tuned track: run each entry's fixed dev grid, pick one configuration, freeze it (PROTOCOL.md section 3).

    python -m benchmarks.compare.tune run --entries gliner-nvidia --root R -- --device cuda --gpu 3
    python -m benchmarks.compare.tune select --root R --out benchmarks/compare/configs/tuned.json

``GRID`` is the pre-declared budget: per entry, adapter runs (config overrides) x thresholds
filtered from each run's scores, at most ``BUDGET`` candidates. ``run`` calls ``runner.main``
once per run on the dev part (``N_DEV`` docs per language) into ``<root>/<entry>.<run>``;
extra arguments after ``--`` go to the runner. ``select`` reads each run's ``tuning`` rows
(language macro, full taxonomy minus DATE, failed docs unmasked) and picks the candidate with
the highest masked F2 averaged over the privacy datasets; ties go to the higher masked
precision, then grid order. One configuration per entry for every dataset.
"""

import argparse
import json
import sys
from pathlib import Path

from benchmarks.compare import runner, scorer
from benchmarks.compare.entries import PHONE_REGIONS, PRESIDIO_MODELS, SPACY_LG, SPACY_XX  # noqa: F401

BUDGET = 12
N_DEV = 100
PRIVACY = (*scorer.GROUPS["privacy-synthetic"], *scorer.GROUPS["privacy-real"])
GLINER_T = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
FLOOR = 0.05  # scored runs decode at the curve floor; the grid is filtered from them

# Presidio language packs and phone regions live in entries.py (presidio-full uses them too).
# OPF's README describes the trade-off (less background stay + easier span entry/continuation =
# more recall), but ships only the all-zero default; b scales that one documented direction.
OPF_B = (-1.0, -0.5, 0.0, 0.5, 1.0, 2.0)


def opf_preset(b: float) -> dict:
    return {"biases": {"transition_bias_background_stay": -b, "transition_bias_background_to_start": b,
                       "transition_bias_inside_to_continue": b}}


GRID = {
    "sct-default": {"runs": {f"nct{t}": {"overrides": {"ner_confidence_threshold": t}}
                             for t in (0.5, 0.6, 0.7, 0.8, 0.85, 0.9)}},
    "sct-gliner": {"runs": {f"gt{t}": {"overrides": {"gliner_threshold": t}} for t in GLINER_T}},
    "presidio": {"runs": {"en": {}, "langpacks": {"language": "gold", "models": PRESIDIO_MODELS,
                                                  "phone_regions": list(PHONE_REGIONS)}},
                 "thresholds": (None, 0.1, 0.2, 0.3, 0.4, 0.5), "key": "score_threshold"},
    "gliner-nvidia": {"runs": {p: {"labels": p, "threshold": FLOOR} for p in ("common", "card")},
                      "thresholds": GLINER_T, "key": "threshold"},
    "gliner2-fastino": {"runs": {p: {"labels": p, "threshold": FLOOR} for p in ("gliner", "card")},
                        "thresholds": GLINER_T, "key": "threshold"},
    "opf": {"runs": {f"b{b}": opf_preset(b) for b in OPF_B}},
    "bardsai-v3": {"runs": {"vendor": {}}, "thresholds": (None, 0.3, 0.5, 0.7, 0.8, 0.9), "key": "threshold"},
    "datafog": {"runs": {"none": {"locales": None}, "gold": {"locales": "gold"}}},  # regex: every type is on already
}


def candidates(entry: str) -> list[tuple[str, float | None]]:
    grid = GRID[entry]
    out = [(run, t) for run in grid["runs"] for t in grid.get("thresholds", (None,))]
    if len(out) > BUDGET:
        raise ValueError(f"{entry}: {len(out)} candidates exceed the budget of {BUDGET}")
    return out


def frozen_config(entry: str, run: str, threshold: float | None) -> dict:
    """Adapter config overrides for the chosen candidate (a filtered threshold becomes the run's threshold)."""
    grid = GRID[entry]
    config = dict(grid["runs"][run])
    if "key" in grid:
        config.pop(grid["key"], None)
        if threshold is not None:
            config[grid["key"]] = threshold
    return config


def row_at(rows: list[dict], threshold: float | None) -> dict:
    match = [r for r in rows if r["threshold"] == threshold]
    if not match:
        raise ValueError(f"no tuning row at threshold {threshold}")
    return match[0]


def select(entry: str, root: Path, datasets=PRIVACY) -> dict:
    """Score every candidate of *entry* from its run results; return the frozen choice and the grid."""
    results = {}
    for run in GRID[entry]["runs"]:
        found = {d: json.loads(p.read_text()) for d in datasets
                 if (p := root / f"{entry}.{run}" / entry / f"{d}-dev.json").exists()}
        if not found:
            raise FileNotFoundError(f"no dev results for {entry}.{run} under {root}")
        results[run] = found
    covered = sorted(set.intersection(*(set(r) for r in results.values())))
    if any(set(r) != set(covered) for r in results.values()):
        raise ValueError(f"{entry}: runs cover different datasets")
    table = []
    for run, t in candidates(entry):
        per = {d: row_at(results[run][d]["tuning"], t) for d in covered}
        mean = {k: scorer._mean(per[d][k] for d in covered) for k in scorer.TUNING}
        table.append({"run": run, "threshold": t, "objective": mean["masked_f2"], "mean": mean,
                      "datasets": {d: {k: v for k, v in r.items() if k != "threshold"} for d, r in per.items()}})
    best = max(enumerate(table), key=lambda x: (x[1]["objective"] or -1, x[1]["mean"]["masked_precision"] or -1,
                                                -x[0]))[1]
    curves = {run: {d: res["tuning"] for d, res in by_d.items()} for run, by_d in results.items()}
    return {"config": frozen_config(entry, best["run"], best["threshold"]), "run": best["run"],
            "threshold": best["threshold"], "dev_objective": best["objective"], "dev_mean": best["mean"],
            "dev_datasets": best["datasets"], "datasets": covered, "candidates": [
                {k: c[k] for k in ("run", "threshold", "objective", "mean")} for c in table],
            "completion": {d: results[best["run"]][d]["completion_rate"] for d in covered}}, curves


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    passthrough = argv[argv.index("--") + 1:] if "--" in argv else []
    argv = argv[:argv.index("--")] if "--" in argv else argv
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("action", choices=("run", "select"))
    parser.add_argument("--entries", nargs="+", default=list(GRID))
    parser.add_argument("--datasets", nargs="+", default=list(PRIVACY))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--runs", nargs="+", help="only these run names (run action)")
    parser.add_argument("--out", type=Path, help="frozen configs JSON (select action)")
    args = parser.parse_args(argv)
    if args.action == "run":
        for entry in args.entries:
            candidates(entry)  # budget check before any compute
            for run, config in GRID[entry]["runs"].items():
                if args.runs and run not in args.runs:
                    continue
                print(f"[tune] {entry}.{run}", file=sys.stderr, flush=True)
                runner.main(["--entries", entry, "--datasets", *args.datasets, "--part", "dev", "--n", str(N_DEV),
                             "--out", str(args.root / f"{entry}.{run}"), "--config", f"{entry}={json.dumps(config)}",
                             *passthrough])
        return
    frozen, curves = {}, {}
    for entry in args.entries:
        frozen[entry], curves[entry] = select(entry, args.root, args.datasets)
    out = args.out or args.root / "tuned.json"
    runner._write(out, {"objective": "max mean over privacy datasets of language-macro masked F2 (full taxonomy, "
                                     "DATE out, failed docs unmasked), dev part", "budget": BUDGET, "n_dev": N_DEV,
                        "entries": frozen})
    runner._write(args.root / "curves.json", curves)
    print(out)


if __name__ == "__main__":
    main()
