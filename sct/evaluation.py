"""Ground-truth evaluation for PII entity detection."""

from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class EntitySpan:
    """One labeled entity in a document."""

    label: str
    start: int
    end: int


@dataclass(frozen=True)
class EvaluationExample:
    """A document with reviewed ground-truth entities."""

    id: str
    text: str
    language: str
    domain: str
    expected: tuple[EntitySpan, ...]
    slices: tuple[str, ...] = ()


@dataclass(frozen=True)
class EntityMetrics:
    """Exact-span entity detection metrics."""

    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True)
class EvaluationReport:
    """Evaluation results for one model."""

    model: str
    overall: EntityMetrics
    by_label: Mapping[str, EntityMetrics]
    by_language: Mapping[str, EntityMetrics]
    by_domain: Mapping[str, EntityMetrics]
    by_slice: Mapping[str, EntityMetrics]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation."""
        return asdict(self)


def _metrics(true_positives: int, false_positives: int, false_negatives: int) -> EntityMetrics:
    precision_denominator = true_positives + false_positives
    recall_denominator = true_positives + false_negatives
    precision = true_positives / precision_denominator if precision_denominator else 0.0
    recall = true_positives / recall_denominator if recall_denominator else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return EntityMetrics(
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def evaluate_pii(
    examples: Sequence[EvaluationExample],
    predictions: Mapping[str, Sequence[EntitySpan]],
    *,
    model: str,
) -> EvaluationReport:
    """Score predictions using exact label and character-span matches."""
    example_ids = {example.id for example in examples}
    unknown_ids = sorted(set(predictions) - example_ids)
    if unknown_ids:
        raise ValueError(
            f"predictions contain unknown document IDs: {', '.join(unknown_ids)}"
        )

    true_positives = 0
    false_positives = 0
    false_negatives = 0
    label_counts: dict[str, list[int]] = {}
    language_counts: dict[str, list[int]] = {}
    domain_counts: dict[str, list[int]] = {}
    slice_counts: dict[str, list[int]] = {}

    for example in examples:
        expected = Counter(example.expected)
        predicted = Counter(predictions.get(example.id, ()))
        matches = expected & predicted
        matched_count = sum(matches.values())
        example_counts = (
            matched_count,
            sum((predicted - matches).values()),
            sum((expected - matches).values()),
        )
        true_positives += example_counts[0]
        false_positives += example_counts[1]
        false_negatives += example_counts[2]

        for grouped_counts, key in (
            (language_counts, example.language),
            (domain_counts, example.domain),
        ):
            counts = grouped_counts.setdefault(key, [0, 0, 0])
            for index, value in enumerate(example_counts):
                counts[index] += value
        for slice_name in example.slices:
            counts = slice_counts.setdefault(slice_name, [0, 0, 0])
            for index, value in enumerate(example_counts):
                counts[index] += value

        labels = {span.label for span in expected}
        labels.update(span.label for span in predicted)
        for label in labels:
            expected_for_label = Counter({
                span: count for span, count in expected.items() if span.label == label
            })
            predicted_for_label = Counter({
                span: count for span, count in predicted.items() if span.label == label
            })
            matches_for_label = expected_for_label & predicted_for_label
            counts = label_counts.setdefault(label, [0, 0, 0])
            counts[0] += sum(matches_for_label.values())
            counts[1] += sum((predicted_for_label - matches_for_label).values())
            counts[2] += sum((expected_for_label - matches_for_label).values())

    return EvaluationReport(
        model=model,
        overall=_metrics(true_positives, false_positives, false_negatives),
        by_label={key: _metrics(*counts) for key, counts in sorted(label_counts.items())},
        by_language={
            key: _metrics(*counts) for key, counts in sorted(language_counts.items())
        },
        by_domain={key: _metrics(*counts) for key, counts in sorted(domain_counts.items())},
        by_slice={key: _metrics(*counts) for key, counts in sorted(slice_counts.items())},
    )


def _span_from_dict(data: Mapping[str, Any]) -> EntitySpan:
    span = EntitySpan(
        label=str(data["label"]),
        start=int(data["start"]),
        end=int(data["end"]),
    )
    if span.start < 0 or span.end <= span.start:
        raise ValueError(
            f"Invalid entity span: start={span.start}, end={span.end}"
        )
    return span


def load_evaluation_jsonl(path: str | Path) -> list[EvaluationExample]:
    """Load reviewed examples from a JSON Lines file."""
    examples = []
    seen_ids = set()
    with Path(path).open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            data = json.loads(line)
            example_id = str(data["id"])
            if example_id in seen_ids:
                raise ValueError(f"Duplicate evaluation ID: {example_id}")
            seen_ids.add(example_id)
            text = str(data["text"])
            expected = tuple(
                _span_from_dict(entity) for entity in data["entities"]
            )
            if any(span.end > len(text) for span in expected):
                raise ValueError(
                    f"Entity span exceeds text length for ID: {example_id}"
                )
            examples.append(EvaluationExample(
                id=example_id,
                text=text,
                language=str(data["language"]),
                domain=str(data["domain"]),
                expected=expected,
                slices=tuple(str(value) for value in data.get("slices", ())),
            ))
    return examples


def load_predictions_jsonl(path: str | Path) -> dict[str, tuple[EntitySpan, ...]]:
    """Load model predictions from a JSON Lines file."""
    predictions = {}
    with Path(path).open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            data = json.loads(line)
            prediction_id = str(data["id"])
            if prediction_id in predictions:
                raise ValueError(f"Duplicate prediction ID: {prediction_id}")
            predictions[prediction_id] = tuple(
                _span_from_dict(entity) for entity in data["entities"]
            )
    return predictions
