"""Pinned model candidates for repeatable PII evaluation."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelCandidate:
    """One reviewed model checkpoint and its integration state."""

    key: str
    model_id: str
    revision: str
    backend: str
    variant: str
    languages: tuple[str, ...]
    license: str
    status: str
    supports_onnx: bool = False
    reported_benchmark: str | None = None
    reported_span_f1: float | None = None


MODEL_CANDIDATES = (
    ModelCandidate(
        key="current-pii",
        model_id="knowledgator/gliner-pii-base-v1.0",
        revision="61726e0ad791dcab3e29339bbec3ad42ded65641",
        backend="gliner",
        variant="gliner",
        languages=("en",),
        license="apache-2.0",
        status="baseline",
        supports_onnx=True,
    ),
    ModelCandidate(
        key="gliner2-pii",
        model_id="fastino/gliner2-privacy-filter-PII-multi",
        revision="c153999da5f4c509df4322b0c6a1baf3d2c284d7",
        backend="gliner",
        variant="gliner2",
        languages=("en", "fr", "es", "de", "it", "pt", "nl"),
        license="apache-2.0",
        status="benchmark-ready",
        reported_benchmark="SPY",
        reported_span_f1=0.477,
    ),
    ModelCandidate(
        key="nvidia-gliner-pii",
        model_id="nvidia/gliner-PII",
        revision="bd23e8ef4425fd04e34c5204ab49ffaa706eae79",
        backend="gliner",
        variant="gliner",
        languages=("en",),
        license="nvidia-open-model-license",
        status="benchmark-ready",
        reported_benchmark="SPY",
        reported_span_f1=0.400,
    ),
    ModelCandidate(
        key="mmbert-pii",
        model_id="llm-semantic-router/mmbert-pii-detector-merged",
        revision="4e9c41ddae1eab9006a2c9395a4336a264f2eed4",
        backend="torch",
        variant="token-classification",
        languages=("multilingual",),
        license="apache-2.0",
        status="benchmark-ready",
    ),
    ModelCandidate(
        key="gliner25-multi",
        model_id="fastino/gliner2.5-multi-v1",
        revision="235cf92d6d4318da9bfca0d08975c8fa7250d13b",
        backend="gliner2",
        variant="boundary",
        languages=("multilingual",),
        license="apache-2.0",
        status="adapter-required",
    ),
    ModelCandidate(
        key="gliformer-base",
        model_id="knowledgator/gliformer-base-v1",
        revision="590f9d3f577ea2f6d685aaec84b2d66b6db86b15",
        backend="gliformer",
        variant="base",
        languages=("en",),
        license="apache-2.0",
        status="adapter-required",
    ),
)
