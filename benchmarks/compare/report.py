"""Summary tables (markdown) from a comparison bundle: quality per track, completion, paired wins, speed.

    python -m benchmarks.compare.report benchmarks/results/compare-2026-11-02 > TABLES.md

The bundle holds ``<track>/summary.json`` and ``<track>/<entry>/<dataset>-<part>.json`` per
track (``default``, ``tuned``) and optionally ``speed-16t/speed.json`` and ``speed-1t/speed.json``.
Group scores are the macro over datasets of each dataset's language macro (PROTOCOL.md section 13),
effective protection (failed docs unmasked), DATE out.
"""

import argparse
import json
from pathlib import Path

from benchmarks.compare import scorer

TRACKS = ("default", "tuned")
QUALITY = (("masked_recall", "mR"), ("masked_precision", "mP"), ("masked_f2", "mF2"),
           ("doc_leak_rate", "doc leak"), ("partial_f1", "pF1"))
LOWER_IS_BETTER = {"doc_leak_rate", "char_survival", "overmask_rate"}


def _fmt(value) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def quality_table(summary: dict, view: str, group: str) -> list[str]:
    rows = []
    for entry, views in summary["aggregate"].items():
        agg = views.get(view, {}).get(group)
        if agg is None:
            continue
        ood = views[view].get(group + ":out-of-domain")
        rows.append((agg["macro"].get("masked_f2") or -1, entry, agg["macro"], ood["macro"] if ood else None,
                     len(ood["datasets"]) if ood else 0))
    head = "| entry | " + " | ".join(n for _, n in QUALITY) + " | mF2 out-of-domain (n sets) |"
    out = [head, "|" + "---|" * (len(QUALITY) + 2)]
    for _, entry, m, ood, n in sorted(rows, reverse=True):
        out.append(f"| {entry} | " + " | ".join(_fmt(m.get(k)) for k, _ in QUALITY)
                   + f" | {_fmt(ood.get('masked_f2')) if ood else 'n/a'} ({n}) |")
    return out


def completion_table(track_dir: Path, entries) -> list[str]:
    out = ["| entry | docs | failed | flagged datasets (< 99% complete) |", "|---|---|---|---|"]
    for entry_dir in (track_dir / e for e in entries):
        docs = failed = 0
        flagged = []
        for f in sorted(entry_dir.glob("*-*.json")):
            if f.name in ("offset-check.json", "part-run.json"):
                continue
            r = json.loads(f.read_text())
            docs += r["documents"]
            failed += r["documents"] - r["completed"]
            if r.get("flag_low_completion"):
                flagged.append(f"{r['dataset']['name']} {r['completion_rate']:.3f}")
        out.append(f"| {entry_dir.name} | {docs} | {failed} | {', '.join(flagged) or '-'} |")
    return out


def paired_table(summary: dict, metric: str = "masked_recall") -> list[str]:
    """Per entry: datasets where the baseline is significantly better / worse / not different."""
    out = [f"| entry | baseline better | entry better | no difference ({metric}, shared view) |", "|---|---|---|---|"]
    for entry, by_dataset in summary.get("paired", {}).items():
        better, worse, tie = [], [], []
        for dataset, res in sorted(by_dataset.items()):
            m = res["metrics"].get(metric)
            if m is None or not m["excludes_zero"]:
                tie.append(dataset)
                continue
            diff = m["a"] - m["b"]
            (better if (diff < 0 if metric in LOWER_IS_BETTER else diff > 0) else worse).append(dataset)
        out.append(f"| {entry} | {', '.join(better) or '-'} | {', '.join(worse) or '-'} | {len(tie)} |")
    return out


def dataset_table(track_dir: Path, entries, view: str = "full", metric: str = "masked_f2") -> list[str]:
    """Per dataset (rows, group order) x entry: the dataset's language macro on completed docs."""
    entries = list(entries)
    out = [f"| dataset | {' | '.join(entries)} |", "|---|" + "---|" * len(entries)]
    for dataset in (d for group in scorer.GROUPS.values() for d in group):
        files = [track_dir / e / f"{dataset}-test.json" for e in entries]
        if not all(f.exists() for f in files):
            continue
        cells = [json.loads(f.read_text())["views"][view]["compare"]["macro"].get(metric) for f in files]
        out.append(f"| {dataset} | {' | '.join(_fmt(c) for c in cells)} |")
    return out


def speed_table(speed: dict) -> list[str]:
    def med(x):
        return "fail" if x is None else f"{x['median']:.4g}"

    out = [f"device {speed['device']}, cpus {speed['cpus']}, threads {speed['threads']}, {speed['docs']} docs, "
           f"{speed['repeats']} repeats, median", "",
           "| entry | cold start s | latency median ms | p95 ms | batch-8 docs/s | chars/s | peak RSS MB | venv MB | failed |",
           "|---|---|---|---|---|---|---|---|---|"]
    for entry, s in sorted(speed["entries"].items(), key=lambda x: -((x[1]["common_batch_docs_per_second"] or {})
                                                                     .get("median") or 0)):
        out.append(f"| {entry} | {med(s['cold_start_seconds'])} | {med(s['latency_median_ms'])} | "
                   f"{med(s['latency_p95_ms'])} | {med(s['common_batch_docs_per_second'])} | "
                   f"{med(s['common_batch_chars_per_second'])} | {med(s['peak_rss_mb'])} | {s['install_mb']} | "
                   f"{len(s['failed'])} |")
    return out


def tables(bundle: Path) -> str:
    lines = []
    for track in TRACKS:
        path = bundle / track / "summary.json"
        if not path.exists():
            continue
        summary = json.loads(path.read_text())
        for view in ("shared", "full"):
            for group in scorer.GROUPS:
                lines += ["", f"## {track} track, {view} view, {group}", "", *quality_table(summary, view, group)]
        lines += ["", f"## {track} track, per dataset, full view, masked F2 (completed docs)", "",
                  *dataset_table(bundle / track, summary["aggregate"])]
        lines += ["", f"## {track} track, completion", "", *completion_table(bundle / track, summary["aggregate"])]
        lines += ["", f"## {track} track, paired bootstrap vs baseline", "", *paired_table(summary)]
    for name in ("speed-16t", "speed-1t"):
        path = bundle / name / "speed.json"
        if path.exists():
            lines += ["", f"## speed ({name})", "", *speed_table(json.loads(path.read_text()))]
    return "\n".join(lines).lstrip() + "\n"


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("bundle", type=Path)
    print(tables(parser.parse_args(argv).bundle), end="")


if __name__ == "__main__":
    main()
