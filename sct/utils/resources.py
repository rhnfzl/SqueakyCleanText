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

# ---------------------------------------------------------------------------
# Language extensibility utilities (used by TextCleanerConfig and TextCleaner)
# ---------------------------------------------------------------------------

DEFAULT_LANGUAGES = frozenset({'ENGLISH', 'DUTCH', 'GERMAN', 'SPANISH'})


def validate_language_name(name: str) -> None:
    """Validate that name corresponds to a lingua.Language member.

    Raises ValueError if invalid. Skips 'MULTILINGUAL' (not a Lingua language).
    """
    if name == 'MULTILINGUAL':
        return
    if not hasattr(Language, name):
        raise ValueError(
            f"Unknown language: '{name}'. Must be a lingua.Language member "
            f"(e.g. ENGLISH, POLISH, FRENCH). See lingua docs for full list."
        )


def build_detector(supported_languages: frozenset):
    """Build a Lingua detector for the given language set.

    Filters out 'MULTILINGUAL' (not a real Lingua language).
    Returns a Lingua LanguageDetector instance.
    """
    lingua_langs = []
    for name in supported_languages:
        if name == 'MULTILINGUAL':
            continue
        lingua_langs.append(getattr(Language, name))
    if not lingua_langs:
        lingua_langs = [getattr(Language, n) for n in DEFAULT_LANGUAGES]
    return LanguageDetectorBuilder.from_languages(*lingua_langs).build()


def language_to_iso(name: str) -> str:
    """Convert a Lingua language name to ISO 639-1 code.

    E.g. 'POLISH' -> 'pl', 'ENGLISH' -> 'en'.
    Falls back to first 2 chars lowercased if not a valid Lingua language.
    """
    if name == 'MULTILINGUAL':
        return 'mul'
    try:
        lang = getattr(Language, name)
        return lang.iso_code_639_1.name.lower()
    except AttributeError:
        return name[:2].lower()
