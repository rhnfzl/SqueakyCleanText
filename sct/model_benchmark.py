"""Ground-truth quality and latency benchmarks for model candidates."""

import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Sequence

from sct.evaluation import (
    EntitySpan,
    EvaluationExample,
    EvaluationReport,
    evaluate_pii,
)
from sct.config import PII_LABEL_MAP, PII_LABELS, TextCleanerConfig
from sct.model_candidates import ModelCandidate


@dataclass(frozen=True)
class LatencyMetrics:
    """Document-level inference latency."""

    mean_ms: float
    p95_ms: float
    documents_per_second: float


@dataclass(frozen=True)
class ModelBenchmark:
    """Quality and performance results for one pinned model."""

    candidate: ModelCandidate
    report: EvaluationReport
    latency: LatencyMetrics

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": asdict(self.candidate),
            "report": self.report.to_dict(),
            "latency": asdict(self.latency),
        }


@dataclass(frozen=True)
class AdoptionDecision:
    """Decision from the default model adoption gate."""

    accepted: bool
    reasons: tuple[str, ...]


def candidate_config(candidate: ModelCandidate) -> TextCleanerConfig:
    """Build a span-preserving configuration for one supported candidate."""
    common = {
        "check_fix_bad_unicode": False,
        "check_to_ascii_unicode": False,
        "check_replace_html": False,
        "check_replace_urls": False,
        "check_replace_emails": False,
        "check_replace_years": False,
        "check_replace_dates": False,
        "check_replace_phone_numbers": False,
        "check_replace_numbers": False,
        "check_replace_currency_symbols": False,
        "check_remove_isolated_letters": False,
        "check_remove_isolated_special_symbols": False,
        "check_remove_bracket_content": False,
        "check_remove_brace_content": False,
        "check_normalize_whitespace": False,
        "check_statistical_model_processing": False,
        "positional_tags": tuple(sorted({
            PII_LABEL_MAP.get(label, label.upper())
            for label in PII_LABELS
        })),
    }
    if candidate.backend == "gliner":
        return TextCleanerConfig(
            **common,
            ner_backend="gliner",
            gliner_model=candidate.model_id,
            gliner_revision=candidate.revision,
            gliner_variant=candidate.variant,
            gliner_labels=PII_LABELS,
            gliner_label_map=PII_LABEL_MAP,
            gliner_threshold=0.3,
            gliner_onnx=candidate.supports_onnx,
        )
    if candidate.backend == "torch":
        model_keys = ("ENGLISH", "MULTILINGUAL")
        return TextCleanerConfig(
            **common,
            ner_backend="torch",
            torch_ner_models={
                key: candidate.model_id for key in model_keys
            },
            torch_ner_model_revisions={
                key: candidate.revision for key in model_keys
            },
        )
    raise ValueError(
        f"Candidate {candidate.key!r} requires a new backend adapter"
    )


def assess_candidate(
    baseline: ModelBenchmark,
    candidate: ModelBenchmark,
    *,
    onnx_equivalent: bool,
    min_f1_gain: float = 0.01,
    max_p95_ratio: float = 1.5,
) -> AdoptionDecision:
    """Apply quality, recall, latency, and runtime parity requirements."""
    reasons = []
    if candidate.report.overall.recall < baseline.report.overall.recall:
        reasons.append("recall-regression")
    if (
        candidate.report.overall.f1
        < baseline.report.overall.f1 + min_f1_gain
    ):
        reasons.append("insufficient-f1-gain")
    if (
        candidate.latency.p95_ms
        > baseline.latency.p95_ms * max_p95_ratio
    ):
        reasons.append("latency-regression")
    if not onnx_equivalent:
        reasons.append("onnx-not-verified")
    return AdoptionDecision(not reasons, tuple(reasons))


def run_model_benchmark(
    candidate: ModelCandidate,
    examples: Sequence[EvaluationExample],
    cleaner: Any,
    *,
    clock: Callable[[], float] = time.perf_counter,
) -> ModelBenchmark:
    """Measure exact-span quality and per-document latency."""
    predictions = {}
    durations = []
    for example in examples:
        started_at = clock()
        result = cleaner.process(example.text)
        durations.append((clock() - started_at) * 1000)
        if not result.metadata or "findings" not in result.metadata:
            raise ValueError(
                "Benchmark cleaner must be created with include_entities=True"
            )
        findings = result.metadata["findings"]
        predictions[example.id] = tuple(
            EntitySpan(
                label=finding.entity.entity_type,
                start=finding.entity.start,
                end=finding.entity.end,
            )
            for finding in findings
        )

    report = evaluate_pii(examples, predictions, model=candidate.model_id)
    sorted_durations = sorted(durations)
    mean_ms = sum(durations) / len(durations) if durations else 0.0
    p95_index = max(0, math.ceil(len(sorted_durations) * 0.95) - 1)
    p95_ms = sorted_durations[p95_index] if sorted_durations else 0.0
    documents_per_second = 1000 / mean_ms if mean_ms else 0.0
    return ModelBenchmark(
        candidate=candidate,
        report=report,
        latency=LatencyMetrics(
            mean_ms=round(mean_ms, 3),
            p95_ms=round(p95_ms, 3),
            documents_per_second=round(documents_per_second, 3),
        ),
    )
