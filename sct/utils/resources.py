import os
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


DEFAULT_LANGUAGES = frozenset({'ENGLISH', 'DUTCH', 'GERMAN', 'SPANISH'})

_detector_lock = threading.Lock()
if hasattr(os, "register_at_fork"):
    # A child forked while another thread built a detector must not inherit the lock held.
    os.register_at_fork(after_in_child=lambda: globals().update(_detector_lock=threading.Lock()))


# ---------------------------------------------------------------------------
# Language extensibility utilities (used by TextCleanerConfig and TextCleaner)
# ---------------------------------------------------------------------------


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


_all_detector = None


def build_all_detector():
    """Lingua detector over all 75 languages (high accuracy), built once per process."""
    global _all_detector
    if _all_detector is None:
        with _detector_lock:
            if _all_detector is None:
                _all_detector = LanguageDetectorBuilder.from_all_languages().build()
    return _all_detector


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


def configured_month_names(cfg):
    """Month-name dict for cfg's date options, or None when all of them are at the default (DATE_REGEX).

    Shared by TextCleaner and the browser export so both build the same date regex.
    """
    custom = cfg.custom_month_names or {}
    if not (custom or cfg.date_builtin_month_languages is not None or not cfg.date_month_name_slash_year):
        return None
    from sct.utils import constants
    extra = {language_to_iso(lang): names for lang, names in custom.items()}
    return constants.build_month_names_dict(extra, cfg.date_builtin_month_languages)
