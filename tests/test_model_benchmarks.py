from sct.config import PII_LABEL_MAP, PII_LABELS
from sct.evaluation import (
    EntityMetrics,
    EntitySpan,
    EvaluationExample,
    EvaluationReport,
)
from sct.model_benchmark import (
    LatencyMetrics,
    ModelBenchmark,
    assess_candidate,
    candidate_config,
    run_model_benchmark,
)
from sct.model_candidates import MODEL_CANDIDATES
from sct.privacy import DetectedEntity, EntityFinding
from sct.utils.process_result import ProcessResult


def test_model_candidate_registry_pins_reviewed_models():
    candidates = {candidate.key: candidate for candidate in MODEL_CANDIDATES}

    assert candidates["gliner2-pii"].model_id == (
        "fastino/gliner2-privacy-filter-PII-multi"
    )
    assert candidates["gliner2-pii"].revision == (
        "c153999da5f4c509df4322b0c6a1baf3d2c284d7"
    )
    assert candidates["nvidia-gliner-pii"].license == "nvidia-open-model-license"
    assert candidates["gliner2-pii"].reported_span_f1 == 0.477
    assert candidates["nvidia-gliner-pii"].reported_span_f1 == 0.400
    assert candidates["gliner25-multi"].status == "adapter-required"


def test_model_benchmark_scores_findings_and_latency():
    class FakeCleaner:
        def process(self, text):
            entity = DetectedEntity("PERSON", 0.9, "Maria", 0, 5)
            return ProcessResult(
                text,
                None,
                "ENGLISH",
                metadata={
                    "findings": (
                        EntityFinding(
                            entity,
                            "placeholder",
                            "detector-threshold",
                        ),
                    ),
                },
            )

    candidate = MODEL_CANDIDATES[0]
    examples = (
        EvaluationExample(
            id="one",
            text="Maria arrived.",
            language="en",
            domain="support",
            expected=(EntitySpan("PERSON", 0, 5),),
        ),
    )
    times = iter((1.0, 1.01))

    benchmark = run_model_benchmark(
        candidate,
        examples,
        FakeCleaner(),
        clock=lambda: next(times),
    )

    assert benchmark.report.overall.f1 == 1.0
    assert benchmark.latency.mean_ms == 10.0
    assert benchmark.latency.documents_per_second == 100.0


def test_adoption_gate_rejects_recall_regression():
    def benchmark(f1, recall, p95):
        metrics = EntityMetrics(1, 0, 0, f1, recall, f1)
        return ModelBenchmark(
            candidate=MODEL_CANDIDATES[0],
            report=EvaluationReport(
                model="model",
                overall=metrics,
                by_label={},
                by_language={},
                by_domain={},
                by_slice={},
            ),
            latency=LatencyMetrics(p95, p95, 1000 / p95),
        )

    decision = assess_candidate(
        baseline=benchmark(0.80, 0.85, 10),
        candidate=benchmark(0.83, 0.84, 12),
        onnx_equivalent=True,
    )

    assert decision.accepted is False
    assert "recall-regression" in decision.reasons


def test_candidate_config_preserves_offsets_and_revision():
    candidate = {
        item.key: item for item in MODEL_CANDIDATES
    }["gliner2-pii"]

    config = candidate_config(candidate)

    assert config.gliner_model == candidate.model_id
    assert config.gliner_revision == candidate.revision
    assert config.gliner_variant == "gliner2"
    assert config.check_replace_emails is False
    assert config.check_normalize_whitespace is False
    assert set(config.positional_tags) == {
        PII_LABEL_MAP.get(label, label.upper())
        for label in PII_LABELS
    }
