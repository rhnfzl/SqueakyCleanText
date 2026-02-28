import warnings
import threading

from lingua import Language, LanguageDetectorBuilder

# Only suppress known lingua deprecation warnings, not all warnings globally
warnings.filterwarnings('ignore', module='lingua')

# ---------------------------------------------------------------------------
# ISO code → canonical Lingua name reverse lookup tables (built once)
# ---------------------------------------------------------------------------
_ISO1_TO_LANG: dict[str, str] = {}  # 'en' → 'ENGLISH'
_ISO3_TO_LANG: dict[str, str] = {}  # 'eng' → 'ENGLISH'
for _lang in Language.all():
    _ISO1_TO_LANG[_lang.iso_code_639_1.name.lower()] = _lang.name
    _ISO3_TO_LANG[_lang.iso_code_639_3.name.lower()] = _lang.name


def resolve_language(value: str) -> str:
    """Resolve a language identifier to its canonical uppercase Lingua name.

    Accepts:
      - Full Lingua name: ``'ENGLISH'``, ``'english'``, ``'English'``
      - ISO 639-1 code: ``'en'``, ``'EN'``
      - ISO 639-3 code: ``'eng'``, ``'ENG'``

    Returns the canonical uppercase name (e.g. ``'ENGLISH'``).

    Raises:
        ValueError: If the value doesn't match any known language.
    """
    upper = value.strip().upper()
    if upper in ('MULTILINGUAL', 'MUL'):
        return 'MULTILINGUAL'
    if hasattr(Language, upper):
        return upper
    lower = value.strip().lower()
    if lower in _ISO1_TO_LANG:
        return _ISO1_TO_LANG[lower]
    if lower in _ISO3_TO_LANG:
        return _ISO3_TO_LANG[lower]
    raise ValueError(
        f"Unknown language: '{value}'. Use a Lingua name (ENGLISH), "
        f"ISO 639-1 code (en), or ISO 639-3 code (eng)."
    )


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


def build_detector(supported_languages: frozenset):
    """Build a Lingua detector for the given language set.

    Filters out 'MULTILINGUAL' (not a real Lingua language).
    Returns a Lingua LanguageDetector instance.
    """
    from sct.config import DEFAULT_LANGUAGES  # lazy: avoids circular import with config.py

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
