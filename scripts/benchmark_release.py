#!/usr/bin/env python3
"""Release benchmark: exact-span correctness, leaks and speed of the default cleaner.

Run once per release on the same machine and compare the JSON files:

    python -m scripts.benchmark_release

Correctness runs the default config with ``replacement_mode="reversible"`` (each
placeholder maps back to the exact value it replaced, which gives the predicted
span) and ``check_to_ascii_unicode=False`` (output stays in the source script, so
leaks are visible). Speed runs
the plain default config, single process, after one warmup pass over all documents.
"""

import argparse
from collections import Counter
import hashlib
import json
import platform
import re
import subprocess
import time
from pathlib import Path

from sct.evaluation import EntitySpan, evaluate_pii, load_evaluation_jsonl

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "evaluation" / "release_benchmark.jsonl"
RESULTS = ROOT / "evaluation" / "results"
PLACEHOLDER_REGEX = re.compile(r"<([A-Z][A-Z_]*?)_\d+>")
NON_PII_LABELS = frozenset({"YEAR", "NUMBER"})
SPEED_REPEATS = 3
CORRECTNESS_OVERRIDES = {"replacement_mode": "reversible", "check_to_ascii_unicode": False}


def predicted_spans(source: str, output: str, originals: dict[str, str]) -> list[EntitySpan]:
    """Map each ``<TAG_n>`` placeholder in output order to the source span it replaced."""
    spans, cursor = [], 0
    for match in PLACEHOLDER_REGEX.finditer(output):
        value = originals.get(match.group(0), "")
        start = source.find(value, cursor) if value else -1
        if start < 0 and value:
            start = source.find(value)
        if start < 0:
            # The replaced value is not in the source (an earlier step changed it): never matches.
            spans.append(EntitySpan(match.group(1), -1, -1))
            continue
        spans.append(EntitySpan(match.group(1), start, start + len(value)))
        cursor = start + len(value)
    return spans


def find_leaks(example, output: str) -> tuple[list[dict], list[dict]]:
    """Return PII values still fully present, and values with a 3+ char piece left."""
    leaks, partial = [], []
    for span in example.expected:
        if span.label in NON_PII_LABELS:
            continue
        value = example.text[span.start:span.end]
        item = {"id": example.id, "label": span.label, "value": value}
        if value in output:
            leaks.append(item)
        elif any(len(token) >= 3 and token in output for token in re.findall(r"\w+", value)):
            partial.append(item)
    return leaks, partial


def _leak_summary(items: list[dict], total: int) -> dict:
    return {
        "count": len(items),
        "total_pii": total,
        "by_label": dict(sorted(Counter(item["label"] for item in items).items())),
        "items": items,
    }


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
            text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run(dataset: Path) -> dict:
    import sct
    from sct.config import TextCleanerConfig
    from sct.sct import TextCleaner

    examples = load_evaluation_jsonl(dataset)

    cleaner = TextCleaner(cfg=TextCleanerConfig(**CORRECTNESS_OVERRIDES))
    outputs, predictions, leaks, partial = {}, {}, [], []
    for example in examples:
        result = cleaner.process(example.text)
        output = result.lm_text
        originals = {
            entry.placeholder: entry.original
            for entry in result.metadata["anon_map"].entries
        }
        outputs[example.id] = output
        predictions[example.id] = tuple(predicted_spans(example.text, output, originals))
        doc_leaks, doc_partial = find_leaks(example, output)
        leaks += doc_leaks
        partial += doc_partial
    report = evaluate_pii(examples, predictions, model="default")
    pii_total = sum(
        span.label not in NON_PII_LABELS for e in examples for span in e.expected
    )
    del cleaner

    speed_cleaner = TextCleaner()
    for example in examples:
        speed_cleaner.process(example.text)
    durations = []
    for _ in range(SPEED_REPEATS):
        for example in examples:
            started_at = time.perf_counter()
            speed_cleaner.process(example.text)
            durations.append(time.perf_counter() - started_at)
    durations.sort()
    cfg = speed_cleaner.cfg

    return {
        "sct_version": sct.__version__,
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dataset": dataset.name,
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "documents": len(examples),
        "entities": sum(len(e.expected) for e in examples),
        "models": {
            "ner_backend": cfg.ner_backend,
            "ner_models": dict(cfg.ner_models or {}),
            "ner_model_revisions": dict(cfg.ner_model_revisions or {}),
        },
        "correctness": {
            "config": f"TextCleanerConfig(**{CORRECTNESS_OVERRIDES!r})",
            "report": report.to_dict(),
            "leaks": _leak_summary(leaks, pii_total),
            "partial_leaks": _leak_summary(partial, pii_total),
        },
        "speed": {
            "config": "TextCleanerConfig()",
            "warmup": "one pass over all documents",
            "repeats": SPEED_REPEATS,
            "timed_documents": len(durations),
            "seconds": round(sum(durations), 3),
            "docs_per_second": round(len(durations) / sum(durations), 3),
            "mean_ms": round(1000 * sum(durations) / len(durations), 3),
            "p95_ms": round(1000 * durations[max(0, -(-len(durations) * 95 // 100) - 1)], 3),
        },
        "outputs": outputs,
    }


def _row(name: str, metrics: dict) -> str:
    return (
        f"| {name} | {metrics['true_positives']} | {metrics['false_positives']} "
        f"| {metrics['false_negatives']} | {metrics['precision']:.2f} "
        f"| {metrics['recall']:.2f} | {metrics['f1']:.2f} |"
    )


def to_markdown(result: dict) -> str:
    report = result["correctness"]["report"]
    header = "| | TP | FP | FN | P | R | F1 |\n|---|---|---|---|---|---|---|"
    speed = result["speed"]
    leaks = result["correctness"]["leaks"]
    partial = result["correctness"]["partial_leaks"]
    lines = [
        f"# Release benchmark: SCT {result['sct_version']}",
        "",
        f"Commit `{result['git_commit']}`, {result['platform']}, Python {result['python']}.",
        f"Dataset `{result['dataset']}`: {result['documents']} documents, "
        f"{result['entities']} entities. Exact label + span match.",
        "",
        f"Speed (default config, single process): **{speed['docs_per_second']} docs/s**, "
        f"mean {speed['mean_ms']} ms, p95 {speed['p95_ms']} ms.",
        f"Leaks: **{leaks['count']} / {leaks['total_pii']}** PII values still in output "
        f"(partial: {partial['count']}).",
        "",
        "## By label",
        "",
        header,
        _row("**overall**", report["overall"]),
        *(_row(key, value) for key, value in report["by_label"].items()),
        "",
        "## By language",
        "",
        header,
        *(_row(key, value) for key, value in report["by_language"].items()),
        "",
        "## Leaks",
        "",
        *(f"- `{i['id']}` {i['label']}: {i['value']}" for i in leaks["items"]),
        *(f"- `{i['id']}` {i['label']} (partial): {i['value']}" for i in partial["items"]),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--output-dir", type=Path, default=RESULTS)
    args = parser.parse_args()
    result = run(args.dataset)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / f"benchmark-{result['sct_version']}"
    stem.with_name(stem.name + ".json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    stem.with_name(stem.name + ".md").write_text(to_markdown(result), encoding="utf-8")


if __name__ == "__main__":
    main()
