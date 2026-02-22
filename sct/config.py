"""
Configuration for the SqueakyCleanText pipeline.

New API (thread-safe, recommended):
    from sct.config import TextCleanerConfig
    cfg = TextCleanerConfig(check_ner_process=False)
    cleaner = TextCleaner(config=cfg)

Legacy API (backward compatible):
    from sct import config
    config.CHECK_NER_PROCESS = False
    cleaner = TextCleaner()
"""

import sys
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class TextCleanerConfig:
    """Thread-safe, immutable configuration for the text cleaning pipeline.

    Each TextCleaner instance stores its own config copy. Use
    ``dataclasses.replace()`` to create modified copies::

        import dataclasses
        new_cfg = dataclasses.replace(cfg, check_ner_process=False)
    """

    # Pipeline step toggles
    check_detect_language: bool = True
    check_fix_bad_unicode: bool = True
    check_to_ascii_unicode: bool = True
    check_replace_html: bool = True
    check_replace_urls: bool = True
    check_replace_emails: bool = True
    check_replace_years: bool = True
    check_replace_dates: bool = False
    check_fuzzy_replace_dates: bool = False
    check_replace_phone_numbers: bool = True
    check_replace_numbers: bool = True
    check_replace_currency_symbols: bool = True
    check_ner_process: bool = True
    check_remove_isolated_letters: bool = True
    check_remove_isolated_special_symbols: bool = True
    check_remove_bracket_content: bool = True
    check_remove_brace_content: bool = True
    check_normalize_whitespace: bool = True
    check_statistical_model_processing: bool = True
    check_casefold: bool = True
    check_smart_casefold: bool = False
    check_remove_stopwords: bool = True
    check_remove_punctuation: bool = True
    check_remove_stext_custom_stop_words: bool = True
    check_remove_emoji: bool = False

    # Fuzzy matching settings (requires rapidfuzz optional dependency)
    # Empirically calibrated: 85 catches 18/19 common misspellings (e.g.
    # "Feburary", "Septmber", "enro") while avoiding false positives from
    # common words.  Lower to 80 for maximum recall (catches "Agusto"→agosto
    # at 83.3 but risks rare FP like "jungle"→june at 80).  Raise to 87+ for
    # zero false-positive risk at the cost of missing some edge-case typos.
    fuzzy_date_score_cutoff: int = 85

    # Replacement tokens
    replace_with_url: str = "<URL>"
    replace_with_html: str = "<HTML>"
    replace_with_email: str = "<EMAIL>"
    replace_with_years: str = "<YEAR>"
    replace_with_dates: str = "<DATE>"
    replace_with_phone_numbers: str = "<PHONE>"
    replace_with_numbers: str = "<NUMBER>"
    replace_with_currency_symbols: Optional[str] = None

    # NER settings
    positional_tags: Tuple[str, ...] = ('PER', 'LOC', 'ORG', 'MISC')
    ner_confidence_threshold: float = 0.85
    language: Optional[str] = None

    # Order matters: English, Dutch, German, Spanish, Multilingual
    ner_models_list: Tuple[str, ...] = (
        "FacebookAI/xlm-roberta-large-finetuned-conll03-english",
        "FacebookAI/xlm-roberta-large-finetuned-conll02-dutch",
        "FacebookAI/xlm-roberta-large-finetuned-conll03-german",
        "FacebookAI/xlm-roberta-large-finetuned-conll02-spanish",
        "Babelscape/wikineural-multilingual-ner",
    )

    def __post_init__(self):
        # Convert mutable collections to immutable for frozen safety
        if isinstance(self.positional_tags, list):
            object.__setattr__(self, 'positional_tags', tuple(self.positional_tags))
        if isinstance(self.ner_models_list, list):
            object.__setattr__(self, 'ner_models_list', tuple(self.ner_models_list))


def _config_from_module_globals() -> TextCleanerConfig:
    """Build a TextCleanerConfig from the current module-level variable state.

    Supports backward compatibility with the legacy config pattern.
    """
    m = sys.modules[__name__]
    return TextCleanerConfig(
        check_detect_language=m.CHECK_DETECT_LANGUAGE,
        check_fix_bad_unicode=m.CHECK_FIX_BAD_UNICODE,
        check_to_ascii_unicode=m.CHECK_TO_ASCII_UNICODE,
        check_replace_html=m.CHECK_REPLACE_HTML,
        check_replace_urls=m.CHECK_REPLACE_URLS,
        check_replace_emails=m.CHECK_REPLACE_EMAILS,
        check_replace_years=m.CHECK_REPLACE_YEARS,
        check_replace_dates=getattr(m, 'CHECK_REPLACE_DATES', False),
        check_fuzzy_replace_dates=getattr(m, 'CHECK_FUZZY_REPLACE_DATES', False),
        fuzzy_date_score_cutoff=getattr(m, 'FUZZY_DATE_SCORE_CUTOFF', 85),
        check_replace_phone_numbers=m.CHECK_REPLACE_PHONE_NUMBERS,
        check_replace_numbers=m.CHECK_REPLACE_NUMBERS,
        check_replace_currency_symbols=m.CHECK_REPLACE_CURRENCY_SYMBOLS,
        check_ner_process=m.CHECK_NER_PROCESS,
        check_remove_isolated_letters=m.CHECK_REMOVE_ISOLATED_LETTERS,
        check_remove_isolated_special_symbols=m.CHECK_REMOVE_ISOLATED_SPECIAL_SYMBOLS,
        check_remove_bracket_content=getattr(m, 'CHECK_REMOVE_BRACKET_CONTENT', True),
        check_remove_brace_content=getattr(m, 'CHECK_REMOVE_BRACE_CONTENT', True),
        check_normalize_whitespace=m.CHECK_NORMALIZE_WHITESPACE,
        check_statistical_model_processing=m.CHECK_STATISTICAL_MODEL_PROCESSING,
        check_casefold=m.CHECK_CASEFOLD,
        check_smart_casefold=getattr(m, 'CHECK_SMART_CASEFOLD', False),
        check_remove_stopwords=m.CHECK_REMOVE_STOPWORDS,
        check_remove_punctuation=m.CHECK_REMOVE_PUNCTUATION,
        check_remove_stext_custom_stop_words=m.CHECK_REMOVE_STEXT_CUSTOM_STOP_WORDS,
        check_remove_emoji=getattr(m, 'CHECK_REMOVE_EMOJI', False),
        replace_with_url=m.REPLACE_WITH_URL,
        replace_with_html=m.REPLACE_WITH_HTML,
        replace_with_email=m.REPLACE_WITH_EMAIL,
        replace_with_years=m.REPLACE_WITH_YEARS,
        replace_with_dates=getattr(m, 'REPLACE_WITH_DATES', '<DATE>'),
        replace_with_phone_numbers=m.REPLACE_WITH_PHONE_NUMBERS,
        replace_with_numbers=m.REPLACE_WITH_NUMBERS,
        replace_with_currency_symbols=m.REPLACE_WITH_CURRENCY_SYMBOLS,
        positional_tags=tuple(m.POSITIONAL_TAGS),
        ner_confidence_threshold=m.NER_CONFIDENCE_THRESHOLD,
        language=m.LANGUAGE,
        ner_models_list=tuple(m.NER_MODELS_LIST),
    )


# ---------------------------------------------------------------------------
# Backward-compatible module-level variables
# Users can still do: config.CHECK_NER_PROCESS = False
# ---------------------------------------------------------------------------
CHECK_DETECT_LANGUAGE = True
CHECK_FIX_BAD_UNICODE = True
CHECK_TO_ASCII_UNICODE = True
CHECK_REPLACE_HTML = True
CHECK_REPLACE_URLS = True
CHECK_REPLACE_EMAILS = True
CHECK_REPLACE_YEARS = True
CHECK_REPLACE_DATES = False
CHECK_FUZZY_REPLACE_DATES = False
FUZZY_DATE_SCORE_CUTOFF = 85
CHECK_REPLACE_PHONE_NUMBERS = True
CHECK_REPLACE_NUMBERS = True
CHECK_REPLACE_CURRENCY_SYMBOLS = True
CHECK_NER_PROCESS = True
CHECK_REMOVE_ISOLATED_LETTERS = True
CHECK_REMOVE_ISOLATED_SPECIAL_SYMBOLS = True
CHECK_REMOVE_BRACKET_CONTENT = True
CHECK_REMOVE_BRACE_CONTENT = True
CHECK_NORMALIZE_WHITESPACE = True
CHECK_STATISTICAL_MODEL_PROCESSING = True
CHECK_CASEFOLD = True
CHECK_SMART_CASEFOLD = False
CHECK_REMOVE_STOPWORDS = True
CHECK_REMOVE_PUNCTUATION = True
CHECK_REMOVE_STEXT_CUSTOM_STOP_WORDS = True
CHECK_REMOVE_EMOJI = False

REPLACE_WITH_URL = "<URL>"
REPLACE_WITH_HTML = "<HTML>"
REPLACE_WITH_EMAIL = "<EMAIL>"
REPLACE_WITH_YEARS = "<YEAR>"
REPLACE_WITH_DATES = "<DATE>"
REPLACE_WITH_PHONE_NUMBERS = "<PHONE>"
REPLACE_WITH_NUMBERS = "<NUMBER>"
REPLACE_WITH_CURRENCY_SYMBOLS = None

POSITIONAL_TAGS = ['PER', 'LOC', 'ORG', 'MISC']
NER_CONFIDENCE_THRESHOLD = 0.85
LANGUAGE = None

# Order: English, Dutch, German, Spanish, Multilingual
NER_MODELS_LIST = [
    "FacebookAI/xlm-roberta-large-finetuned-conll03-english",
    "FacebookAI/xlm-roberta-large-finetuned-conll02-dutch",
    "FacebookAI/xlm-roberta-large-finetuned-conll03-german",
    "FacebookAI/xlm-roberta-large-finetuned-conll02-spanish",
    "Babelscape/wikineural-multilingual-ner",
]
