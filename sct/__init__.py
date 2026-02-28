"""SqueakyCleanText - Text cleaning and preprocessing pipeline for ML/NLP."""

from sct.config import TextCleanerConfig, PII_LABELS, PII_LABEL_MAP
from sct.sct import TextCleaner
from sct.utils.anonymization_map import AnonymizationMap, MapEntry
from sct.utils.process_result import ProcessResult

__version__ = "0.6.1"
__all__ = [
    "TextCleaner", "TextCleanerConfig",
    "PII_LABELS", "PII_LABEL_MAP",
    "AnonymizationMap", "MapEntry", "ProcessResult",
]
