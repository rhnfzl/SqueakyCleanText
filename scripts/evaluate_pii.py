#!/usr/bin/env python3
"""Score PII predictions against reviewed JSONL ground truth."""

import argparse
import json
from pathlib import Path

from sct.evaluation import (
    evaluate_pii,
    load_evaluation_jsonl,
    load_predictions_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate exact PII entity labels and character spans.",
    )
    parser.add_argument("--dataset", required=True, help="Reviewed JSONL dataset")
    parser.add_argument("--predictions", required=True, help="Prediction JSONL file")
    parser.add_argument("--model", required=True, help="Model identifier for the report")
    parser.add_argument("--output", required=True, help="Output JSON report")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate_pii(
        load_evaluation_jsonl(args.dataset),
        load_predictions_jsonl(args.predictions),
        model=args.model,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
