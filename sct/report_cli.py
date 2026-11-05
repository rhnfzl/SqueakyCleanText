"""``sct-report``: clean a JSONL/Parquet column and write a value-free corpus report."""

import argparse
import json
from pathlib import Path
from typing import Any
from collections.abc import Sequence

from sct.presets import PRESETS, preset_config
from sct.risk import assess_reidentification_risk
from sct.score import (
    ScoreConfig,
    build_data_card,
    minhash_metadata,
    render_data_card,
    score_records,
)
from sct.sct import TextCleaner
from sct.utils.atomic_file import write_text_atomic
from sct.utils.ner import ModelLoadError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sct-report")
    parser.add_argument("input", help="Input .jsonl or .parquet file")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--preset", default="training", choices=sorted(PRESETS))
    parser.add_argument("--no-ner", action="store_true")
    parser.add_argument("--quality-model", help="Directory from scripts/export_quality_head.py")
    parser.add_argument("--minhash", action="store_true")
    parser.add_argument(
        "--quasi-identifiers",
        help="Comma-separated columns for k-anonymity (only these are read)",
    )
    parser.add_argument("--k", type=int, default=5, help="k-anonymity threshold")
    parser.add_argument("--out", required=True, help="Report JSON path")
    parser.add_argument("--card", help="Also write a Markdown data card here")
    parser.add_argument("--scores", help="Also write per-row score columns as JSONL here")
    return parser


def _read_rows(path: Path, parser: argparse.ArgumentParser) -> list[dict[str, Any]]:
    if path.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError:
            parser.error("Parquet input needs pyarrow: pip install squeakycleantext[parquet]")
        try:
            return pq.read_table(path).to_pylist()
        except (OSError, ValueError) as exc:
            parser.error(f"cannot read {path.name}: {exc}")
    rows = []
    try:
        with path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except ValueError as exc:
                    parser.error(f"{path.name} line {number}: {exc}")
    except (OSError, ValueError) as exc:  # missing file or not UTF-8
        parser.error(f"cannot read {path.name}: {exc}")
    return rows


def _hashable(value):
    # Encode every value, so "[1]" vs [1] and True vs 1 stay distinct groups.
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def _column_texts(rows: list[dict[str, Any]], column: str, parser: argparse.ArgumentParser) -> list[str]:
    texts = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            parser.error(f"row {i}: not a JSON object")
        value = row.get(column)
        if value is not None and not isinstance(value, str):
            parser.error(f"row {i}: column {column!r} is not text")
        texts.append(value or "")
    return texts


def _k_anonymity(rows: list[dict[str, Any]], columns: str, k_threshold: int,
                 parser: argparse.ArgumentParser) -> dict[str, Any]:
    names = tuple(c.strip() for c in columns.split(",") if c.strip())
    # Only the named columns are read, so other columns never reach the report.
    try:
        records = [{c: _hashable(row[c]) for c in names} for row in rows]
        risk = assess_reidentification_risk(
            records, quasi_identifiers=names, k_threshold=k_threshold,
        )
    except KeyError as exc:
        parser.error(f"quasi-identifier column missing: {exc}")
    except ValueError as exc:
        parser.error(str(exc))
    return {
        "quasi_identifiers": list(risk.quasi_identifiers),
        "k_threshold": risk.k_threshold,
        "minimum_k": risk.minimum_k,
        "at_risk_records": risk.at_risk_records,
        "at_risk_fraction": risk.at_risk_fraction,
        "class_size_counts": {
            str(k): v for k, v in sorted(risk.class_size_counts.items())
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the report command."""
    parser = _parser()
    args = parser.parse_args(argv)
    rows = _read_rows(Path(args.input), parser)

    texts = _column_texts(rows, args.text_column, parser)

    config = ScoreConfig(
        quality=args.quality_model is not None,
        quality_model=args.quality_model,
        minhash=args.minhash,
    )
    # Fail before the corpus is cleaned, which can take far longer than this check.
    if args.quality_model and not Path(args.quality_model, "model.onnx").is_file():
        parser.error(f"no model.onnx in --quality-model folder {args.quality_model}")
    overrides = {"check_ner_process": False} if args.no_ner else {}
    try:
        cleaner = TextCleaner(cfg=preset_config(args.preset, **overrides))
        results = cleaner.process_batch(texts)
    except (ImportError, ModelLoadError, OSError, ValueError) as exc:
        parser.error(str(exc))
    try:
        scored = score_records(results, texts, config)
    except Exception as exc:  # onnxruntime and tokenizers errors derive directly from Exception
        parser.error(f"cannot score: {exc}")
    models = [m for result in results for m in result.signals["models"]]

    report: dict[str, Any] = {
        "input": Path(args.input).name,
        "text_column": args.text_column,
        "preset": args.preset,
        "ner": not args.no_ner,
        "summary": build_data_card(scored, models=models),
    }
    if config.quality:
        manifest = Path(args.quality_model) / "sct_quality.json"
        recorded = json.loads(manifest.read_text()) if manifest.is_file() else {}
        if not isinstance(recorded, dict):
            parser.error(f"{manifest.name} is not a JSON object")
        report["quality"] = {
            "model_dir": Path(args.quality_model).name,
            "input": "original text",
            **recorded,
        }
    if config.minhash:
        report["minhash"] = minhash_metadata(config)
    if args.quasi_identifiers:
        report["k_anonymity"] = _k_anonymity(rows, args.quasi_identifiers, args.k, parser)

    try:
        _write(args.out, json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        if args.card:
            _write(args.card, render_data_card(report["summary"]))
        if args.scores:
            _write(args.scores, "".join(
                json.dumps(row, ensure_ascii=False) + "\n" for row in scored
            ))
    except OSError as exc:
        parser.error(f"cannot write output: {exc}")
    return 0


def _write(path: str, text: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(target, text)


if __name__ == "__main__":
    raise SystemExit(main())
