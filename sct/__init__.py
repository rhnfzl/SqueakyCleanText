"""SqueakyCleanText - Text cleaning and preprocessing pipeline for ML/NLP."""

from sct.config import TextCleanerConfig
from sct.sct import TextCleaner

__version__ = "0.5.0"
__all__ = ["TextCleaner", "TextCleanerConfig"]
