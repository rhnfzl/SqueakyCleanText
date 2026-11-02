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

from benchmarks.compare import entries as registry
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
           f"{speed['repeats']} repeats, median; peak RSS from "
           + ("/proc VmHWM" if any("peak_rss_mb_getrusage" in s for s in speed["entries"].values()) else
              "getrusage (on Linux it never reads below the runner's own ~2 GB: values at that floor are an upper bound)"),
           "",
           "| entry | cold start s | latency median ms | p95 ms | batch-8 docs/s | chars/s | peak RSS MB | venv MB | failed |",
           "|---|---|---|---|---|---|---|---|---|"]
    for entry, s in sorted(speed["entries"].items(), key=lambda x: -((x[1]["common_batch_docs_per_second"] or {})
                                                                     .get("median") or 0)):
        out.append(f"| {entry} | {med(s['cold_start_seconds'])} | {med(s['latency_median_ms'])} | "
                   f"{med(s['latency_p95_ms'])} | {med(s['common_batch_docs_per_second'])} | "
                   f"{med(s['common_batch_chars_per_second'])} | {med(s['peak_rss_mb'])} | {s['install_mb']} | "
                   f"{len(s['failed'])} |")
    return out


# Feature-matrix facts for the headline (model cards and PyPI at the pinned versions, PROTOCOL.md section 2).
LANGUAGES = {"sct-default": "75 LID; NER en nl de es + multilingual", "sct-gliner": "75 LID; GLiNER model tagged en",
             "presidio": "en (default); spaCy packs when tuned", "gliner-nvidia": "en", "gliner2-fastino": "7 EU",
             "opf": "primarily en", "bardsai-v3": "24 EU", "datafog": "regex (US shapes; DE pack)"}
LICENCES = {"sct-default": "MIT", "sct-gliner": "MIT + Apache-2.0 model", "presidio": "MIT", "datafog": "MIT",
            "gliner-nvidia": "Apache-2.0 + NVIDIA Open Model", "gliner2-fastino": "Apache-2.0", "opf": "Apache-2.0",
            "bardsai-v3": "Apache-2.0"}
HEAD = (("masked_recall", "mR"), ("masked_precision", "mP"), ("strict_f1", "strict F1"),
        ("doc_leak_rate", "doc leak"), ("overmask_rate", "over-mask"))


def _speed(bundle: Path, entry: str) -> tuple[str, str]:
    """batch-8 docs/s (16 threads) and peak RSS MB (the /proc VmHWM rerun where there is one)."""
    docs = rss = None
    for name in ("speed-16t", "speed-16t-rss"):
        path = bundle / name / "speed.json"
        s = json.loads(path.read_text())["entries"].get(entry) if path.exists() else None
        if s:
            docs = docs or (s["common_batch_docs_per_second"] or {}).get("median")
            rss = (s["peak_rss_mb"] or {}).get("median") or rss
    return (f"{docs:,.0f}" if docs and docs >= 10 else f"{docs:.2f}" if docs else "n/a"), (f"{rss:,.0f} MB" if rss else "n/a")


def headline_table(summary: dict, bundle: Path, view: str, group: str, baseline: str = "sct-default") -> list[str]:
    """Group macro without in-domain (T/E) datasets: entry value and its difference to the baseline, 95% CI.

    Each pair uses the datasets in-domain for neither side, so the baseline's own score changes per row
    (shown as "SCT mR on same sets"). * = the difference CI excludes 0.
    """
    head = ("| entry | sets | " + " | ".join(f"{n} (entry - SCT, 95% CI)" for _, n in HEAD)
            + " | SCT mR on same sets | docs/s CPU | docs/s GPU | peak RAM / VRAM | languages | licence |")
    out = [head, "|" + "---|" * (len(HEAD) + 8)]
    pairs = summary.get("paired_macro", {})
    full = max((p[view][group] for p in pairs.values() if group in p.get(view, {})),
               key=lambda r: len(r["datasets"]), default=None)
    if full:  # the baseline alone, on every dataset of the group (it has no T/E cell)
        m = full["metrics"]
        cells = [f"{m[k]['a']:.3f} [{m[k]['a_ci'][0]:.3f}, {m[k]['a_ci'][1]:.3f}]" if k in m else "n/a" for k, _ in HEAD]
        out.append(f"| {baseline} | {len(full['datasets'])} | " + " | ".join(cells) + " | - | "
                   + " | ".join(_speed(bundle, baseline)[:1]) + " | not run | " + _speed(bundle, baseline)[1]
                   + f" / n/a | {LANGUAGES.get(baseline, '?')} | {LICENCES.get(baseline, '?')} |")
    rows = []
    for entry, p in pairs.items():
        r = p.get(view, {}).get(group)
        docs, rss = _speed(bundle, entry)
        tail = f" | {docs} | not run | {rss} / n/a | {LANGUAGES.get(entry, '?')} | {LICENCES.get(entry, '?')} |"
        if r is None:
            rows.append((-1, f"| {entry} | 0 | " + " | ".join("n/a" for _ in HEAD) + " | n/a" + tail))
            continue
        m = r["metrics"]
        cells = []
        for k, _ in HEAD:
            if k not in m:
                cells.append("n/a")
                continue
            lo, hi = m[k]["diff_ci"]  # a - b: flip to entry - SCT
            cells.append(f"{m[k]['b']:.3f} ({-hi:+.3f}, {-lo:+.3f}){'*' if m[k]['excludes_zero'] else ''}")
        sct = m.get("masked_recall", {}).get("a")
        rows.append((m.get("masked_recall", {}).get("b") or 0, f"| {entry} | {len(r['datasets'])} | "
                     + " | ".join(cells) + f" | {_fmt(sct)}" + tail))
    out += [row for _, row in sorted(rows, reverse=True)]
    return out


def language_table(track_dir: Path, entries, group: str, view: str = "full", metric: str = "masked_recall") -> list[str]:
    """Per language: mean over the group's datasets that have it of that dataset's language score (completed docs)."""
    entries = list(entries)
    cells: dict[str, dict[str, list]] = {}
    for dataset in scorer.GROUPS[group]:
        for e in entries:
            f = track_dir / e / f"{dataset}-test.json"
            if f.exists():
                for lang, m in json.loads(f.read_text())["views"][view]["compare"]["by_language"].items():
                    cells.setdefault(lang, {}).setdefault(e, []).append((m.get(metric), dataset))
    out = [f"| lang | datasets | {' | '.join(entries)} |", "|---|---|" + "---|" * len(entries)]
    for lang in sorted(cells):
        sets = sorted({d for v in cells[lang].values() for _, d in v})
        vals = [scorer._mean(v for v, _ in cells[lang].get(e, [])) for e in entries]
        out.append(f"| {lang} | {', '.join(sets)} | {' | '.join(_fmt(v) for v in vals)} |")
    return out


def contamination_table(entries) -> list[str]:
    """PROTOCOL.md section 17 flags; in-domain cells (T/E, left out of the headline) in bold."""
    datasets = [d for g in scorer.GROUPS.values() for d in g]
    out = [f"| entry | {' | '.join(datasets)} |", "|---|" + "---|" * len(datasets)]
    for e in entries:
        flags = [registry.contamination(e, d) for d in datasets]
        out.append(f"| {e} | " + " | ".join(f"**{f}**" if registry.in_domain(e, d) else f
                                             for f, d in zip(flags, datasets, strict=True)) + " |")
    return out


def tables(bundle: Path) -> str:
    lines = []
    for track in TRACKS:
        path = bundle / track / "summary.json"
        if not path.exists():
            continue
        summary = json.loads(path.read_text())
        for view in ("shared", "full"):
            for group in ("privacy-synthetic", "privacy-real"):
                if any(group in p.get(view, {}) for p in summary.get("paired_macro", {}).values()):
                    lines += ["", f"## {track} track, headline, {view} view, {group}, in-domain (T/E) sets out", "",
                              *headline_table(summary, bundle, view, group)]
        for group in scorer.GROUPS:
            lines += ["", f"## {track} track, per language, full view, masked recall, {group}", "",
                      *language_table(bundle / track, summary["aggregate"], group)]
        for view in ("shared", "full"):
            for group in scorer.GROUPS:
                lines += ["", f"## {track} track, {view} view, {group}", "", *quality_table(summary, view, group)]
        lines += ["", f"## {track} track, per dataset, full view, masked F2 (completed docs)", "",
                  *dataset_table(bundle / track, summary["aggregate"])]
        lines += ["", f"## {track} track, completion", "", *completion_table(bundle / track, summary["aggregate"])]
        lines += ["", f"## {track} track, paired bootstrap vs baseline", "", *paired_table(summary)]
    lines += ["", "## contamination (bold = in-domain, out of the headline)", "",
              *contamination_table(registry.ENTRIES)]
    for name in ("speed-16t", "speed-16t-rss", "speed-1t"):
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
