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
from types import MappingProxyType
from typing import Mapping, Optional, Tuple


LANG_KEYS = ('ENGLISH', 'DUTCH', 'GERMAN', 'SPANISH', 'MULTILINGUAL')
REQUIRED_NER_KEYS = frozenset({'ENGLISH', 'MULTILINGUAL'})
DEFAULT_LANGUAGES = frozenset({'ENGLISH', 'DUTCH', 'GERMAN', 'SPANISH'})

VALID_NER_BACKENDS = frozenset({'onnx', 'torch', 'gliner', 'ensemble_onnx', 'ensemble_torch'})

DEFAULT_NER_MODELS: dict[str, str] = {
    'ENGLISH': 'rhnfzl/xlm-roberta-large-conll03-english-onnx',
    'DUTCH': 'rhnfzl/xlm-roberta-large-conll02-dutch-onnx',
    'GERMAN': 'rhnfzl/xlm-roberta-large-conll03-german-onnx',
    'SPANISH': 'rhnfzl/xlm-roberta-large-conll02-spanish-onnx',
    'MULTILINGUAL': 'rhnfzl/wikineural-multilingual-ner-onnx',
}

DEFAULT_TORCH_NER_MODELS: dict[str, str] = {
    'ENGLISH': 'FacebookAI/xlm-roberta-large-finetuned-conll03-english',
    'DUTCH': 'FacebookAI/xlm-roberta-large-finetuned-conll02-dutch',
    'GERMAN': 'FacebookAI/xlm-roberta-large-finetuned-conll03-german',
    'SPANISH': 'FacebookAI/xlm-roberta-large-finetuned-conll02-spanish',
    'MULTILINGUAL': 'Babelscape/wikineural-multilingual-ner',
}


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

    # NER backend selection
    ner_backend: str = 'onnx'  # 'onnx' | 'torch' | 'gliner' | 'ensemble_onnx' | 'ensemble_torch'

    # Preferred: language-keyed dict of HuggingFace ONNX model repo IDs.
    # Models must have model.onnx, config.json, tokenizer.json on Hub.
    # ENGLISH and MULTILINGUAL are always required (loaded eagerly).
    # Missing keys are filled from DEFAULT_NER_MODELS.
    ner_models: Optional[Mapping[str, str]] = None

    # Deprecated: positional tuple (English, Dutch, German, Spanish, Multilingual).
    # Use ner_models dict instead. Kept for backward compatibility.
    ner_models_list: Tuple[str, ...] = (
        "rhnfzl/xlm-roberta-large-conll03-english-onnx",
        "rhnfzl/xlm-roberta-large-conll02-dutch-onnx",
        "rhnfzl/xlm-roberta-large-conll03-german-onnx",
        "rhnfzl/xlm-roberta-large-conll02-spanish-onnx",
        "rhnfzl/wikineural-multilingual-ner-onnx",
    )

    # Torch backend models (only used when ner_backend is 'torch' or 'ensemble_torch')
    torch_ner_models: Optional[Mapping[str, str]] = None

    # GLiNER settings (only used when ner_backend contains 'gliner' or 'ensemble')
    gliner_model: Optional[str] = None
    gliner_variant: str = 'gliner'  # 'gliner' or 'gliner2'
    gliner_labels: Tuple[str, ...] = ('person', 'organization', 'location')
    gliner_label_map: Optional[Mapping[str, str]] = None
    gliner_threshold: float = 0.4

    # Language extensibility — extend Lingua detection, stopwords, dates, NER
    extra_languages: Tuple[str, ...] = ()
    custom_stopwords: Optional[Mapping[str, frozenset]] = None
    custom_month_names: Optional[Mapping[str, Tuple[str, ...]]] = None

    # Computed at __post_init__ — union of all language sources
    supported_languages: frozenset = frozenset()

    def __post_init__(self):
        # Convert mutable collections to immutable for frozen safety
        if isinstance(self.positional_tags, list):
            object.__setattr__(self, 'positional_tags', tuple(self.positional_tags))
        if isinstance(self.ner_models_list, list):
            object.__setattr__(self, 'ner_models_list', tuple(self.ner_models_list))

        # --- NER backend validation ---
        if self.ner_backend not in VALID_NER_BACKENDS:
            raise ValueError(
                f"ner_backend must be one of {sorted(VALID_NER_BACKENDS)}, "
                f"got: {self.ner_backend!r}"
            )

        # GLiNER fields required for gliner/ensemble backends
        needs_gliner = self.ner_backend in ('gliner', 'ensemble_onnx', 'ensemble_torch')
        if needs_gliner:
            if not self.gliner_model:
                raise ValueError(
                    f"gliner_model is required when ner_backend='{self.ner_backend}'. "
                    f"Provide a HuggingFace model ID (e.g. 'urchade/gliner_large-v2.1')."
                )
            if self.gliner_variant not in ('gliner', 'gliner2'):
                raise ValueError(
                    f"gliner_variant must be 'gliner' or 'gliner2', "
                    f"got: {self.gliner_variant!r}"
                )

        # Freeze gliner_label_map
        if self.gliner_label_map is not None:
            object.__setattr__(self, 'gliner_label_map',
                               MappingProxyType(dict(self.gliner_label_map)))

        # Reconcile torch_ner_models
        if self.torch_ner_models is not None:
            missing = REQUIRED_NER_KEYS - set(self.torch_ner_models)
            if missing:
                raise ValueError(
                    f"torch_ner_models must include {sorted(REQUIRED_NER_KEYS)}, "
                    f"missing: {sorted(missing)}"
                )
            merged = {**DEFAULT_TORCH_NER_MODELS, **self.torch_ner_models}
            object.__setattr__(self, 'torch_ner_models', MappingProxyType(merged))
        elif self.ner_backend in ('torch', 'ensemble_torch'):
            object.__setattr__(self, 'torch_ner_models',
                               MappingProxyType(dict(DEFAULT_TORCH_NER_MODELS)))

        # Reconcile ner_models (dict, preferred) and ner_models_list (tuple, deprecated).
        # After this block, both fields are guaranteed populated.
        if self.ner_models is not None:
            # Dict API used — validate required keys, fill defaults, freeze
            missing = REQUIRED_NER_KEYS - set(self.ner_models)
            if missing:
                raise ValueError(
                    f"ner_models must include {sorted(REQUIRED_NER_KEYS)}, "
                    f"missing: {sorted(missing)}"
                )
            merged = {**DEFAULT_NER_MODELS, **self.ner_models}
            object.__setattr__(self, 'ner_models', MappingProxyType(merged))
            object.__setattr__(
                self, 'ner_models_list',
                tuple(merged[k] for k in LANG_KEYS),
            )
        else:
            # No dict provided — derive from ner_models_list (may be default or custom)
            models_dict = dict(zip(LANG_KEYS, self.ner_models_list))
            object.__setattr__(self, 'ner_models', MappingProxyType(models_dict))

        # --- Language validation and supported_languages computation ---
        from sct.utils.resources import validate_language_name

        # Coerce extra_languages to tuple
        if isinstance(self.extra_languages, list):
            object.__setattr__(self, 'extra_languages', tuple(self.extra_languages))

        # Validate extra_languages
        for lang in self.extra_languages:
            validate_language_name(lang)

        # Validate and freeze custom_stopwords
        if self.custom_stopwords:
            for lang in self.custom_stopwords:
                validate_language_name(lang)
            object.__setattr__(self, 'custom_stopwords',
                               MappingProxyType(dict(self.custom_stopwords)))

        # Validate and freeze custom_month_names
        if self.custom_month_names:
            for lang in self.custom_month_names:
                validate_language_name(lang)
            object.__setattr__(self, 'custom_month_names',
                               MappingProxyType({k: tuple(v) for k, v in self.custom_month_names.items()}))

        # Compute supported_languages: union of all sources
        langs = set(DEFAULT_LANGUAGES)
        langs.update(self.extra_languages)
        # Add ner_models keys (minus MULTILINGUAL — it's a model, not a spoken language)
        if self.ner_models:
            langs.update(k for k in self.ner_models if k != 'MULTILINGUAL')
        if self.custom_stopwords:
            langs.update(self.custom_stopwords.keys())
        if self.custom_month_names:
            langs.update(self.custom_month_names.keys())
        object.__setattr__(self, 'supported_languages', frozenset(langs))

        # Validate language pin against supported set
        if self.language:
            upper = self.language.upper()
            if upper not in self.supported_languages:
                raise ValueError(
                    f"language='{self.language}' not in supported languages: "
                    f"{sorted(self.supported_languages)}"
                )


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
        ner_models=dict(zip(LANG_KEYS, m.NER_MODELS_LIST)),
        extra_languages=(),
        custom_stopwords=None,
        custom_month_names=None,
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
    "rhnfzl/xlm-roberta-large-conll03-english-onnx",
    "rhnfzl/xlm-roberta-large-conll02-dutch-onnx",
    "rhnfzl/xlm-roberta-large-conll03-german-onnx",
    "rhnfzl/xlm-roberta-large-conll02-spanish-onnx",
    "rhnfzl/wikineural-multilingual-ner-onnx",
]
