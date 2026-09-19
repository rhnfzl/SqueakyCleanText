"""SqueakyCleanText - Text cleaning and preprocessing pipeline for ML/NLP."""

from sct.config import TextCleanerConfig, PII_LABELS, PII_LABEL_MAP
from sct.privacy import (
    DetectedEntity,
    EntityFinding,
    EntityRule,
    InMemoryTokenStore,
    PrivacyPolicy,
    ProcessingMetrics,
    TokenStore,
)
from sct.sct import TextCleaner
from sct.adapters import (
    DataFrameResult,
    PathFinding,
    ProcessedRAGChunk,
    RAGChunk,
    StructuredResult,
    process_dataframe,
    process_json,
    process_rag_chunks,
)
from sct.browser import build_browser_manifest, write_browser_manifest
from sct.model_benchmark import (
    AdoptionDecision,
    LatencyMetrics,
    ModelBenchmark,
    assess_candidate,
    candidate_config,
    run_model_benchmark,
)
from sct.model_candidates import MODEL_CANDIDATES, ModelCandidate
from sct.ocr import (
    ImageRedactionResult,
    OCRSpan,
    TesseractOCRProvider,
    redact_image,
    select_redaction_regions,
)
from sct.risk import (
    ReidentificationRiskReport,
    assess_reidentification_risk,
)
from sct.streaming import (
    BufferedStreamingRedactor,
    WindowedStreamingRedactor,
)
from sct.utils.anonymization_map import AnonymizationMap, MapEntry
from sct.utils.process_result import ProcessResult

__version__ = "0.7.0"
__all__ = [
    "TextCleaner", "TextCleanerConfig",
    "PII_LABELS", "PII_LABEL_MAP",
    "AnonymizationMap", "MapEntry", "ProcessResult",
    "DetectedEntity", "EntityFinding", "EntityRule",
    "PrivacyPolicy", "ProcessingMetrics",
    "TokenStore", "InMemoryTokenStore",
    "DataFrameResult", "PathFinding", "ProcessedRAGChunk",
    "RAGChunk", "StructuredResult",
    "process_dataframe", "process_json", "process_rag_chunks",
    "build_browser_manifest", "write_browser_manifest",
    "AdoptionDecision", "LatencyMetrics", "ModelBenchmark",
    "assess_candidate", "candidate_config", "run_model_benchmark",
    "MODEL_CANDIDATES", "ModelCandidate",
    "ImageRedactionResult", "OCRSpan", "TesseractOCRProvider",
    "redact_image", "select_redaction_regions",
    "ReidentificationRiskReport", "assess_reidentification_risk",
    "BufferedStreamingRedactor", "WindowedStreamingRedactor",
]
