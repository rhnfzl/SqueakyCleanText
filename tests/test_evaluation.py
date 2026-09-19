import json
import sys

import pytest

from sct.evaluation import EntitySpan, EvaluationExample, evaluate_pii
from scripts.evaluate_pii import main as evaluate_pii_main


def test_evaluate_pii_counts_exact_span_matches():
    example = EvaluationExample(
        id="email-1",
        text="Email Ana at ana@example.com.",
        language="en",
        domain="support",
        expected=(
            EntitySpan("PERSON", 6, 9),
            EntitySpan("EMAIL", 13, 28),
        ),
    )
    predictions = {
        "email-1": (
            EntitySpan("PERSON", 6, 9),
            EntitySpan("PHONE_NUMBER", 13, 28),
        ),
    }

    report = evaluate_pii([example], predictions, model="candidate")

    assert report.model == "candidate"
    assert report.overall.true_positives == 1
    assert report.overall.false_positives == 1
    assert report.overall.false_negatives == 1
    assert report.overall.precision == 0.5
    assert report.overall.recall == 0.5
    assert report.overall.f1 == 0.5


def test_evaluate_pii_reports_label_language_and_domain_slices():
    examples = [
        EvaluationExample(
            id="support-en",
            text="Ana uses ana@example.com.",
            language="en",
            domain="support",
            expected=(
                EntitySpan("PERSON", 0, 3),
                EntitySpan("EMAIL", 9, 24),
            ),
        ),
        EvaluationExample(
            id="legal-nl",
            text="Piet tekent.",
            language="nl",
            domain="legal",
            expected=(EntitySpan("PERSON", 0, 4),),
            slices=("long-document",),
        ),
    ]
    predictions = {
        "support-en": (EntitySpan("PERSON", 0, 3),),
        "legal-nl": (
            EntitySpan("PERSON", 0, 4),
            EntitySpan("EMAIL", 5, 11),
        ),
    }

    report = evaluate_pii(examples, predictions, model="candidate")

    assert report.by_label["PERSON"].recall == 1.0
    assert report.by_label["EMAIL"].false_positives == 1
    assert report.by_label["EMAIL"].false_negatives == 1
    assert report.by_language["en"].recall == 0.5
    assert report.by_language["nl"].precision == 0.5
    assert report.by_domain["support"].false_negatives == 1
    assert report.by_domain["legal"].false_positives == 1
    assert report.by_slice["long-document"].precision == 0.5


def test_evaluate_pii_cli_writes_json_report(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(json.dumps({
        "id": "email-1",
        "text": "Email ana@example.com.",
        "language": "en",
        "domain": "support",
        "entities": [{"label": "EMAIL", "start": 6, "end": 21}],
    }) + "\n")
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(json.dumps({
        "id": "email-1",
        "entities": [{"label": "EMAIL", "start": 6, "end": 21}],
    }) + "\n")
    output = tmp_path / "report.json"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_pii",
            "--dataset", str(dataset),
            "--predictions", str(predictions),
            "--model", "test-model",
            "--output", str(output),
        ],
    )
    evaluate_pii_main()

    report = json.loads(output.read_text())
    assert report["model"] == "test-model"
    assert report["overall"]["recall"] == 1.0
    assert report["by_label"]["EMAIL"]["true_positives"] == 1


def test_evaluate_pii_rejects_predictions_for_unknown_documents():
    example = EvaluationExample(
        id="known",
        text="No PII here.",
        language="en",
        domain="general",
        expected=(),
    )

    with pytest.raises(ValueError, match="unknown document IDs: unknown"):
        evaluate_pii(
            [example],
            {"unknown": (EntitySpan("PERSON", 0, 2),)},
            model="candidate",
        )
