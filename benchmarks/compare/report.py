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
LOWER_IS_BETTER = {"doc_leak_rate", "char_survival", "overmask_rate"}


def _fmt(value) -> str:
    return "n/a" if value is None else f"{value:.3f}"


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


def paired_table(summary: dict, metric: str = "masked_recall", baseline: str = "sct-default") -> list[str]:
    """Per entry: datasets clean for both sides where the baseline is significantly better / worse / not different.

    Datasets excluded or unknown for either side (decision 21) are only counted, not compared.
    """
    out = [f"| entry | baseline better | entry better | no difference | not compared (excluded / unknown) "
           f"({metric}, shared view, all languages) |", "|---|---|---|---|---|"]
    for entry, by_dataset in summary.get("paired", {}).items():
        better, worse, tie, skipped = [], [], [], []
        for dataset, res in sorted(by_dataset.items()):
            if {registry.status(entry, dataset), registry.status(baseline, dataset)} != {"clean"}:
                skipped.append(dataset)
                continue
            m = res["metrics"].get(metric)
            if m is None or not m["excludes_zero"]:
                tie.append(dataset)
                continue
            diff = m["a"] - m["b"]
            (better if (diff < 0 if metric in LOWER_IS_BETTER else diff > 0) else worse).append(dataset)
        out.append(f"| {entry} | {', '.join(better) or '-'} | {', '.join(worse) or '-'} | {len(tie)} | {len(skipped)} |")
    return out


def _mark(value, entry: str, dataset: str) -> str:
    status = registry.status(entry, dataset)
    return f"({_fmt(value)})" if status == "excluded" else _fmt(value) + ("?" if status == "unknown" else "")


def dataset_table(track_dir: Path, entries, view: str = "full", metric: str = "masked_f2") -> list[str]:
    """Per dataset (rows, group order) x entry: the dataset's language macro on completed docs.

    Decision 21 markers: ``(x)`` = excluded cell (T/E/D, information only), ``x?`` = unknown training data.
    """
    entries = list(entries)
    out = [f"| dataset | {' | '.join(entries)} |", "|---|" + "---|" * len(entries)]
    for dataset in (d for group in scorer.GROUPS.values() for d in group):
        files = [track_dir / e / f"{dataset}-test.json" for e in entries]
        if not all(f.exists() for f in files):
            continue
        cells = [_mark(json.loads(f.read_text())["views"][view]["compare"]["macro"].get(metric), e, dataset)
                 for f, e in zip(files, entries, strict=True)]
        out.append(f"| {dataset} | {' | '.join(cells)} |")
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


LICENCES = {"sct-default": "MIT", "sct-gliner": "MIT + Apache-2.0 model", "presidio": "MIT", "datafog": "MIT",
            "gliner-nvidia": "Apache-2.0 + NVIDIA Open Model", "gliner2-fastino": "Apache-2.0", "opf": "Apache-2.0",
            "bardsai-v3": "Apache-2.0"}
HEAD = (("masked_recall", "mR"), ("masked_precision", "mP"), ("strict_f1", "strict F1"),
        ("doc_leak_rate", "doc leak"), ("overmask_rate", "over-mask"))


def claimed(entry: str) -> str:
    langs = registry.CLAIMED_LANGUAGES.get(entry)
    return "no claim (all)" if langs is None else "24 EU" if langs == registry.EU24 else " ".join(langs)


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


def _point(agg: dict | None, metric: str = "masked_recall") -> str:
    return "-" if not agg else f"{_fmt(agg['macro'].get(metric))} ({len(agg['datasets'])})"


def own_table(summary: dict, bundle: Path, view: str, group: str) -> list[str]:
    """Headline (decision 21): each entry on its own clean datasets, claimed languages, 95% bootstrap CI.

    Beside it, for information: clean sets outside the claimed languages, unknown-training-data sets
    (claimed languages), and excluded (T/E/D) sets, each as masked recall (number of datasets).
    """
    head = ("| entry | clean sets with claimed-language docs (all clean sets) | "
            + " | ".join(f"{n} [95% CI]" for _, n in HEAD)
            + " | mR clean, outside claimed langs | mR unknown (U) sets | (mR excluded sets, info only) | docs/s CPU"
            " | peak RAM | claimed languages | licence |")
    rows = []
    for entry, views in summary.get("own", {}).items():
        cell = views.get(view, {}).get(group)
        if cell is None:
            continue
        clean = cell.get("clean") or {}
        ci = clean.get("ci", {})
        cells = [f"{ci[k]['value']:.3f} [{ci[k]['ci'][0]:.3f}, {ci[k]['ci'][1]:.3f}]" if k in ci else "-" for k, _ in HEAD]
        docs, rss = _speed(bundle, entry)
        excluded = (cell.get("excluded") or {}).get("claimed")
        used = (clean.get("claimed") or {}).get("datasets", [])
        rows.append((ci.get("masked_recall", {}).get("value") or -1,
                     f"| {entry} | {', '.join(used) or 'none'} ({len(clean.get('datasets', []))}) | " + " | ".join(cells)
                     + f" | {_point(clean.get('outside_claim'))} | {_point((cell.get('unknown') or {}).get('claimed'))}"
                     + f" | {'(' + _point(excluded) + ')' if excluded else '-'} | {docs} | {rss}"
                     + f" | {claimed(entry)} | {LICENCES.get(entry, '?')} |"))
    return [head, "|" + "---|" * (len(HEAD) + 9), *(row for _, row in sorted(rows, reverse=True))]


def pairwise_table(summary: dict, view: str, group: str, kind: str = "clean", baseline: str = "sct-default") -> list[str]:
    """SCT vs each entry on the datasets *kind* for both (clean, or unknown), languages both claim.

    Cell: the entry's score, then the 95% paired bootstrap CI of (entry - SCT); * = the CI excludes 0.
    """
    head = ("| entry | sets | " + " | ".join(f"{n} (entry - SCT, 95% CI)" for _, n in HEAD)
            + " | SCT mR, same docs | SCT mP, same docs |")
    out = [head, "|" + "---|" * (len(HEAD) + 4)]
    rows = []
    for entry in summary.get("own", {}):
        if entry == baseline:
            continue
        r = summary.get("pairwise", {}).get(entry, {}).get(view, {}).get(group, {}).get(kind)
        if r is None:
            rows.append((-1, f"| {entry} | none | " + " | ".join("-" for _ in HEAD) + " | - | - |"))
            continue
        m = r["metrics"]
        cells = []
        for k, _ in HEAD:
            if k not in m:
                cells.append("n/a")
                continue
            lo, hi = m[k]["diff_ci"]  # a - b: flip to entry - SCT
            cells.append(f"{m[k]['b']:.3f} ({-hi:+.3f}, {-lo:+.3f}){'*' if m[k]['excludes_zero'] else ''}")
        sct = [_fmt(m.get(k, {}).get("a")) for k in ("masked_recall", "masked_precision")]
        rows.append((m.get("masked_recall", {}).get("b") or 0,
                     f"| {entry} | {', '.join(r['datasets'])} | " + " | ".join(cells) + f" | {' | '.join(sct)} |"))
    return out + [row for _, row in sorted(rows, reverse=True)]


def same_set_table(summary: dict, view: str, group: str, pool: str) -> list[str]:
    s = summary.get("same_set", {}).get(view, {}).get(group, {}).get(pool)
    if not s or not s["datasets"] or not s["scores"]:
        names = ", ".join(s["entries"]) if s else "-"
        return [f"No dataset in this group is clean for every one of: {names} (in languages all of them claim)."]
    used = sorted({d for a in s["scores"].values() for d in a["datasets"]})
    out = [f"Datasets: {', '.join(used)} (clean for all: {', '.join(s['datasets'])}); only docs in languages every "
           "entry claims; point values.", "",
           "| entry | " + " | ".join(n for _, n in HEAD) + " |", "|---|" + "---|" * len(HEAD)]
    rows = sorted(s["scores"].items(), key=lambda x: -(x[1]["macro"].get("masked_recall") or 0))
    return out + [f"| {e} | " + " | ".join(_fmt(a["macro"].get(k)) for k, _ in HEAD) + " |" for e, a in rows]


def language_table(track_dir: Path, entries, group: str, view: str = "full", metric: str = "masked_recall",
                   status: str = "clean") -> list[str]:
    """Per language: mean over the entry's *status* datasets of the group that have it (completed docs).

    ``°`` = a language outside the entry's claimed languages (shown apart from the claim, not a plain loss).
    """
    entries = list(entries)
    cells: dict[str, dict[str, list]] = {}
    for dataset in scorer.GROUPS[group]:
        for e in entries:
            f = track_dir / e / f"{dataset}-test.json"
            if f.exists() and registry.status(e, dataset) == status:
                for lang, m in json.loads(f.read_text())["views"][view]["compare"]["by_language"].items():
                    cells.setdefault(lang, {}).setdefault(e, []).append((m.get(metric), dataset))
    entries = [e for e in entries if any(e in v for v in cells.values())]
    if not entries:
        return [f"No {status} dataset in this group."]
    out = [f"| lang | datasets | {' | '.join(entries)} |", "|---|---|" + "---|" * len(entries)]
    for lang in sorted(cells):
        sets = sorted({d for v in cells[lang].values() for _, d in v})
        vals = [scorer._mean(v for v, _ in cells[lang][e]) if e in cells[lang] else None for e in entries]
        out.append(f"| {lang} | {', '.join(sets)} | " + " | ".join(
            "-" if v is None else _fmt(v) + ("" if registry.claims(e, lang) else "°")
            for v, e in zip(vals, entries, strict=True)) + " |")
    return out


def contamination_table(entries) -> list[str]:
    """PROTOCOL.md section 17 flags: bold = excluded (T/E/D), italic = unknown (U only), plain = clean."""
    datasets = [d for g in scorer.GROUPS.values() for d in g]
    style = {"excluded": "**{}**", "unknown": "_{}_", "clean": "{}"}
    out = [f"| entry | {' | '.join(datasets)} |", "|---|" + "---|" * len(datasets)]
    for e in entries:
        out.append(f"| {e} | " + " | ".join(style[registry.status(e, d)].format(registry.contamination(e, d))
                                             for d in datasets) + " |")
    return out


def recovery_table(bundle: Path, track: str, view: str = "full", metric: str = "masked_recall") -> list[str]:
    """SCT with and without ``scorer.recover_dropped`` (``norecover/<track>``: ``runner.py --no-recover``).

    Per SCT entry x dataset: *metric* (completed docs, language macro) and the copies recovered; excluded
    cells in parentheses, unknown with ``?``, as everywhere.
    """
    other = bundle / "norecover" / track
    if not other.exists():
        return []
    out = [f"| entry | dataset | {metric} with | without | doc leak with | without | copies recovered |",
           "|---|---|---|---|---|---|---|"]
    for entry_dir in sorted(p for p in other.iterdir() if p.is_dir() and (bundle / track / p.name).exists()):
        for dataset in (d for group in scorer.GROUPS.values() for d in group):
            files = [bundle / track / entry_dir.name / f"{dataset}-test.json", entry_dir / f"{dataset}-test.json"]
            if not all(f.exists() for f in files):
                continue
            new, old = (json.loads(f.read_text()) for f in files)
            m = [r["views"][view]["compare"]["macro"] for r in (new, old)]
            out.append(f"| {entry_dir.name} | {dataset} | {_mark(m[0].get(metric), entry_dir.name, dataset)} | "
                       f"{_mark(m[1].get(metric), entry_dir.name, dataset)} | {_fmt(m[0].get('doc_leak_rate'))} | "
                       f"{_fmt(m[1].get('doc_leak_rate'))} | {new.get('recovered_dropped_copies')} |")
    return out


def tables(bundle: Path) -> str:
    lines = []
    for track in TRACKS:
        path = bundle / track / "summary.json"
        if not path.exists():
            continue
        summary = json.loads(path.read_text())
        for view in ("full", "shared"):
            for group in scorer.GROUPS:
                if not any(group in v.get(view, {}) for v in summary.get("own", {}).values()):
                    continue
                lines += ["", f"## {track} track, {view} view, {group}: each entry on its own clean sets, claimed languages",
                          "", *own_table(summary, bundle, view, group)]
                for pool in ("all", "others"):
                    lines += ["", f"### same-set ({'every entry' if pool == 'all' else 'non-SCT entries with clean sets'})",
                              "", *same_set_table(summary, view, group, pool)]
                for kind in ("clean", "unknown"):
                    lines += ["", f"### SCT vs each entry, datasets {kind} for both, languages both claim", "",
                              *pairwise_table(summary, view, group, kind)]
        for group in scorer.GROUPS:
            for status in ("clean", "unknown"):
                lines += ["", f"## {track} track, per language, full view, masked recall, {group}, {status} sets "
                          "(° = outside the entry's claimed languages)", "",
                          *language_table(bundle / track, summary["aggregate"], group, status=status)]
        lines += ["", f"## {track} track, per dataset, full view, masked F2 (completed docs; (x) = excluded, "
                  "information only; x? = unknown training data)", "",
                  *dataset_table(bundle / track, summary["aggregate"])]
        lines += ["", f"## {track} track, completion", "", *completion_table(bundle / track, summary["aggregate"])]
        lines += ["", f"## {track} track, paired bootstrap vs baseline per dataset", "", *paired_table(summary)]
        recovery = recovery_table(bundle, track)
        if recovery:
            lines += ["", f"## {track} track, SCT with and without recovered dropped copies (full view)", "", *recovery]
    lines += ["", "## contamination (bold = excluded, italic = unknown, plain = clean)", "",
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
