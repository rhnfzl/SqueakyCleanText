import warnings
import threading

from lingua import Language, LanguageDetectorBuilder

# Only suppress known lingua deprecation warnings, not all warnings globally
warnings.filterwarnings('ignore', module='lingua')

# ---- Supported languages (also used for language-to-NER-model mapping)
LANGUAGES = [Language.DUTCH, Language.ENGLISH, Language.GERMAN, Language.SPANISH]
LANGUAGE_NAME = [(language.name).lower() for language in LANGUAGES]

# Lazy language detector: built on first use, not at import time
_detector = None
_detector_lock = threading.Lock()


def _get_detector():
    global _detector
    if _detector is None:
        with _detector_lock:
            if _detector is None:
                _detector = LanguageDetectorBuilder.from_languages(*LANGUAGES).build()
    return _detector


class _LazyDetectorProxy:
    """Proxy that behaves like the detector but builds it lazily."""

    def __getattr__(self, name):
        return getattr(_get_detector(), name)


DETECTOR = _LazyDetectorProxy()
