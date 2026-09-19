#!/usr/bin/env python3
"""Benchmark one pinned model candidate on reviewed PII examples."""

import argparse
import json
from pathlib import Path

from sct.evaluation import load_evaluation_jsonl
from sct.model_benchmark import candidate_config, run_model_benchmark
from sct.model_candidates import MODEL_CANDIDATES
from sct.sct import TextCleaner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--candidate",
        required=True,
        choices=tuple(item.key for item in MODEL_CANDIDATES),
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidate = next(
        item for item in MODEL_CANDIDATES if item.key == args.candidate
    )
    cleaner = TextCleaner(
        cfg=candidate_config(candidate),
        include_entities=True,
    )
    benchmark = run_model_benchmark(
        candidate,
        load_evaluation_jsonl(args.dataset),
        cleaner,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(benchmark.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
