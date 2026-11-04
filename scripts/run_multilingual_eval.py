#!/usr/bin/env python3
"""Score PII detectors on ``evaluation/multilingual_pii.jsonl`` per language and label.

Models (``--models``, default all):

- ``sct-default``: today's default ONNX ensemble with language detection, run like
  the release benchmark (reversible placeholders give the predicted spans).
- ``gliner:<hub id>``: a GLiNER model (``pip install gliner``), e.g.
  ``gliner:urchade/gliner_multi-v2.1`` or ``gliner:urchade/gliner_multi_pii-v1``.

Downloads several GB of models: run locally or on a GPU box, never in CI.

    python -m scripts.run_multilingual_eval

Writes ``evaluation/results/multilingual-<date>.json`` with one exact-span report
per model (``sct.evaluation.evaluate_pii``), the predictions and the model licences.
Only the dataset labels are scored; other predicted labels (NUMBER, DATE, ...) are dropped.
"""

import argparse
from datetime import date
import json
import time
from pathlib import Path

from benchmarks.score import predicted_spans
from scripts.benchmark_release import CORRECTNESS_OVERRIDES, dataset_header, write_atomic
from scripts.build_multilingual_eval import LABELS, OUTPUT as DATASET
from sct.config import TextCleanerConfig
from sct.evaluation import EntitySpan, evaluate_pii, load_evaluation_jsonl
from sct.sct import TextCleaner

RESULTS = DATASET.parent / "results"
DEFAULT_MODELS = ("sct-default", "gliner:urchade/gliner_multi-v2.1", "gliner:urchade/gliner_multi_pii-v1")
# GLiNER prompt -> dataset label.
GLINER_LABELS = {
    "person": "PERSON",
    "location": "LOCATION",
    "organization": "ORGANISATION",
    "email": "EMAIL",
    "phone number": "PHONE",
    "ip address": "IP",
}


def keep_labels(spans) -> tuple[EntitySpan, ...]:
    return tuple(span for span in spans if span.label in LABELS)


def predict_sct(examples) -> tuple[dict, dict]:
    cleaner = TextCleaner(cfg=TextCleanerConfig(**CORRECTNESS_OVERRIDES))
    predictions = {}
    for example in examples:
        result = cleaner.process(example.text)
        originals = {e.placeholder: e.original for e in result.metadata["anon_map"].entries}
        predictions[example.id] = keep_labels(predicted_spans(example.text, result.lm_text, originals))
    cfg = cleaner.cfg
    info = {"ner_models": dict(cfg.ner_models or {}), "config": f"TextCleanerConfig(**{CORRECTNESS_OVERRIDES!r})"}
    return predictions, info


def predict_gliner(examples, model_id: str, threshold: float) -> tuple[dict, dict]:
    from gliner import GLiNER

    model = GLiNER.from_pretrained(model_id)
    predictions = {}
    for example in examples:
        found = model.predict_entities(example.text, list(GLINER_LABELS), threshold=threshold)
        predictions[example.id] = tuple(
            EntitySpan(GLINER_LABELS[item["label"]], item["start"], item["end"]) for item in found
        )
    return predictions, {"threshold": threshold, "labels": GLINER_LABELS}


def hub_licence(model_id: str) -> str | None:
    try:
        from huggingface_hub import model_info

        return (model_info(model_id).card_data or {}).get("license")
    except Exception:  # noqa: BLE001  # offline or gated: record unknown
        return None


def run(dataset: Path, models: list[str], threshold: float) -> dict:
    examples = load_evaluation_jsonl(dataset)
    reports = {}
    for name in models:
        started = time.perf_counter()
        if name == "sct-default":
            predictions, info = predict_sct(examples)
        elif name.startswith("gliner:"):
            model_id = name.split(":", 1)[1]
            predictions, info = predict_gliner(examples, model_id, threshold)
            info["licence"] = hub_licence(model_id)
        else:
            raise SystemExit(f"unknown model {name!r}: use sct-default or gliner:<hub id>")
        reports[name] = {
            **info,
            "seconds": round(time.perf_counter() - started, 1),
            "report": evaluate_pii(examples, predictions, model=name).to_dict(),
            "predictions": {
                key: [span.__dict__ for span in spans] for key, spans in predictions.items()
            },
        }
    return {**dataset_header(dataset, examples), "models": reports}


def to_markdown(result: dict) -> str:
    """Recall / precision per language (rows) and model (columns)."""
    names = list(result["models"])
    languages = list(next(iter(result["models"].values()))["report"]["by_language"])
    head = "| language | " + " | ".join(f"{n} R / P" for n in names) + " |"
    lines = [head, "|---" * (len(names) + 1) + "|"]
    for group, keys in (("by_language", languages + ["overall"]), ("by_label", list(LABELS))):
        for key in keys:
            cells = []
            for name in names:
                report = result["models"][name]["report"]
                metrics = report["overall"] if key == "overall" else report[group].get(key)
                cells.append(f"{metrics['recall']:.2f} / {metrics['precision']:.2f}" if metrics else "-")
            lines.append(f"| {key} | " + " | ".join(cells) + " |")
        if group == "by_language":
            lines += ["", head.replace("language", "label"), lines[1]]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--threshold", type=float, default=0.5, help="GLiNER score threshold")
    parser.add_argument("--output", type=Path, default=RESULTS / f"multilingual-{date.today()}.json")
    args = parser.parse_args()
    result = run(args.dataset, args.models, args.threshold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(args.output, json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    print(to_markdown(result))


if __name__ == "__main__":
    main()
