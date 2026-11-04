#!/usr/bin/env python3
"""Release benchmark: exact-span correctness, leaks and speed of the default cleaner.

Run once per release on the same machine and compare the JSON files:

    python -m scripts.benchmark_release

Batch throughput (``process_batch``) runs only with ``--batch N``:

    python -m scripts.benchmark_release --batch 256 --batch-size 8 --ner-batch-size 8

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
import sys
import time
from pathlib import Path
from typing import Any

from benchmarks.score import PLACEHOLDER, predicted_spans
import sct
from sct.config import TextCleanerConfig
from sct.evaluation import evaluate_pii, load_evaluation_jsonl
from sct.sct import TextCleaner
from sct.utils.atomic_file import write_text_atomic

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "evaluation" / "release_benchmark.jsonl"
RESULTS = ROOT / "evaluation" / "results"
NON_PII_LABELS = frozenset({"YEAR", "NUMBER"})
SPEED_REPEATS = 3
CORRECTNESS_OVERRIDES = {"replacement_mode": "reversible", "check_to_ascii_unicode": False}


def find_leaks(example, output: str) -> tuple[list[dict], list[dict]]:
    """Return PII values still fully present, and values with a piece left.

    A piece is a 3+ char word of the value still in the output, or text glued
    to a placeholder with 3+ letters or digits that is part of the value, so
    split values (``98765<NUMBER>``, ``<NUMBER>.1.10``) count.
    """
    leaks, partial = [], []
    visible = PLACEHOLDER.sub("\0", output)
    fragments = [
        fragment.rstrip(".,;:!?)'\"")
        for fragment in re.findall(r"[^\s\0]+(?=\0)|(?<=\0)[^\s\0]+", visible)
    ]
    for span in example.expected:
        if span.label in NON_PII_LABELS:
            continue
        value = example.text[span.start:span.end]
        item = {"id": example.id, "label": span.label, "value": value}
        if value in output:
            leaks.append(item)
        elif any(
            len(token) >= 3 and token in visible for token in re.findall(r"\w+", value)
        ) or any(
            len(re.findall(r"\w", fragment)) >= 3 and fragment in value
            for fragment in fragments
        ):
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
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,  # noqa: S607
            text=True, check=True, timeout=30,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def run_header(dataset: Path) -> dict[str, str | None]:
    """Run metadata every result file starts with."""
    return {
        "sct_version": sct.__version__,
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dataset": dataset.name,
    }


def dataset_header(dataset: Path, examples: list) -> dict[str, str | int | None]:
    """``run_header`` plus what pins a scored dataset."""
    return {
        **run_header(dataset),
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "documents": len(examples),
        "entities": sum(len(e.expected) for e in examples),
    }


def run(dataset: Path) -> dict[str, Any]:
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
        **dataset_header(dataset, examples),
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


def long_documents(texts: list[str], per_document: int = 8) -> list[str]:
    """Join every *per_document* consecutive texts into one document longer than a model window."""
    return [" ".join(texts[i:i + per_document]) for i in range(0, len(texts), per_document)]


def _peak_rss_mb() -> float | None:
    try:
        import resource  # noqa: PLC0415  # POSIX only
    except ImportError:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(peak / (1024 * 1024 if sys.platform == "darwin" else 1024), 1)  # bytes vs KiB


def run_batch(
    dataset: Path,
    documents: int,
    batch_size: int | None,
    ner_batch_size: int,
    intra_op_threads: int | None,
    repeats: int,
    long: bool,
    **cfg_overrides,
) -> dict[str, Any]:
    """Time ``process_batch`` over *documents* docs (the dataset cycled) after one warmup call."""
    texts = [example.text for example in load_evaluation_jsonl(dataset)]
    if long:
        texts = long_documents(texts)
    docs = [texts[i % len(texts)] for i in range(documents)]
    cleaner = TextCleaner(cfg=TextCleanerConfig(
        ner_batch_size=ner_batch_size, onnx_intra_op_threads=intra_op_threads, **cfg_overrides,
    ))
    cleaner.process_batch(docs, batch_size=batch_size)
    seconds = []
    for _ in range(repeats):
        started_at = time.perf_counter()
        results = cleaner.process_batch(docs, batch_size=batch_size)
        seconds.append(time.perf_counter() - started_at)
    output_hash = hashlib.sha256("\0".join(r.lm_text for r in results).encode()).hexdigest()
    return {
        **run_header(dataset),
        "long_documents": long,
        "settings": {
            "documents": documents,
            "batch_size": batch_size,
            "ner_batch_size": ner_batch_size,
            "onnx_intra_op_threads": intra_op_threads,
            "repeats": repeats,
        },
        "seconds": [round(value, 3) for value in seconds],
        "docs_per_second": round(documents / min(seconds), 3),
        "peak_rss_mb": _peak_rss_mb(),
        "output_sha256": output_hash,
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
    parser = argparse.ArgumentParser(description=(__doc__ or "Release benchmark").splitlines()[0])
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--output-dir", type=Path, default=RESULTS)
    batch = parser.add_argument_group("batch throughput (only with --batch)")
    batch.add_argument("--batch", type=int, metavar="N", help="time process_batch over N documents")
    batch.add_argument("--batch-size", type=int, metavar="B", help="process_batch workers")
    batch.add_argument("--ner-batch-size", type=int, default=8, metavar="K")
    batch.add_argument("--intra-op-threads", type=int, metavar="T")
    batch.add_argument("--repeats", type=int, default=3, metavar="R")
    batch.add_argument("--long", action="store_true", help="join documents into long ones")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.batch:
        result = run_batch(
            args.dataset, args.batch, args.batch_size, args.ner_batch_size,
            args.intra_op_threads, args.repeats, args.long,
        )
        name = (
            f"benchmark-{result['sct_version']}-batch{'-long' if args.long else ''}"
            f"-N{args.batch}-B{args.batch_size}-K{args.ner_batch_size}-T{args.intra_op_threads}.json"
        )
        write_text_atomic(args.output_dir / name, json.dumps(result, indent=2, sort_keys=True) + "\n")
        return
    result = run(args.dataset)
    stem = args.output_dir / f"benchmark-{result['sct_version']}"
    write_text_atomic(
        stem.with_name(stem.name + ".json"),
        json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    )
    write_text_atomic(stem.with_name(stem.name + ".md"), to_markdown(result))


if __name__ == "__main__":
    main()
