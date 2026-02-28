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
from typing import Mapping, Optional, Tuple, Union


LANG_KEYS = ('ENGLISH', 'DUTCH', 'GERMAN', 'SPANISH', 'MULTILINGUAL')
REQUIRED_NER_KEYS = frozenset({'ENGLISH', 'MULTILINGUAL'})
DEFAULT_LANGUAGES = frozenset({'ENGLISH', 'DUTCH', 'GERMAN', 'SPANISH'})

VALID_NER_BACKENDS = frozenset({
    'onnx', 'torch', 'gliner', 'ensemble_onnx', 'ensemble_torch', 'presidio_gliner',
})

DEFAULT_NER_MODELS: dict[str, str] = {
    'ENGLISH': 'rhnfzl/xlm-roberta-large-conll03-english-onnx',
    'DUTCH': 'rhnfzl/xlm-roberta-large-conll02-dutch-onnx',
    'GERMAN': 'rhnfzl/xlm-roberta-large-conll03-german-onnx',
    'SPANISH': 'rhnfzl/xlm-roberta-large-conll02-spanish-onnx',
    # FR/PT/IT: no dedicated ONNX model available; map to MULTILINGUAL fallback
    'FRENCH': 'rhnfzl/wikineural-multilingual-ner-onnx',
    'PORTUGUESE': 'rhnfzl/wikineural-multilingual-ner-onnx',
    'ITALIAN': 'rhnfzl/wikineural-multilingual-ner-onnx',
    'MULTILINGUAL': 'rhnfzl/wikineural-multilingual-ner-onnx',
}

DEFAULT_TORCH_NER_MODELS: dict[str, str] = {
    'ENGLISH': 'FacebookAI/xlm-roberta-large-finetuned-conll03-english',
    'DUTCH': 'FacebookAI/xlm-roberta-large-finetuned-conll02-dutch',
    'GERMAN': 'FacebookAI/xlm-roberta-large-finetuned-conll03-german',
    'SPANISH': 'FacebookAI/xlm-roberta-large-finetuned-conll02-spanish',
    # FR/PT/IT: no dedicated Torch model available; map to MULTILINGUAL fallback
    'FRENCH': 'Babelscape/wikineural-multilingual-ner',
    'PORTUGUESE': 'Babelscape/wikineural-multilingual-ner',
    'ITALIAN': 'Babelscape/wikineural-multilingual-ner',
    'MULTILINGUAL': 'Babelscape/wikineural-multilingual-ner',
}

# NER ensemble: ordered model keys to run per input language.
# Keys must exist in the resolved ner_models dict.
# EN/NL/DE/ES get a 3-model ensemble (lang-specific + English CoNLL + Multilingual).
# All other languages (FR, PT, IT, etc.) fall back to NER_ENSEMBLE_DEFAULT_KEYS.
DEFAULT_NER_ENSEMBLE: dict[str, tuple] = {
    'ENGLISH': ('ENGLISH', 'MULTILINGUAL'),
    'DUTCH':   ('DUTCH',   'ENGLISH', 'MULTILINGUAL'),
    'GERMAN':  ('GERMAN',  'ENGLISH', 'MULTILINGUAL'),
    'SPANISH': ('SPANISH', 'ENGLISH', 'MULTILINGUAL'),
}
# Fallback for any language not in DEFAULT_NER_ENSEMBLE (FR, PT, IT, etc.)
NER_ENSEMBLE_DEFAULT_KEYS: tuple = ('MULTILINGUAL', 'ENGLISH')

# PII label categories (compatible with knowledgator/gliner-pii-base-v1.0)
PII_LABELS: tuple[str, ...] = (
    # Personal
    'name', 'first_name', 'last_name', 'date_of_birth', 'age',
    'gender', 'marital_status', 'nationality',
    # Contact
    'email', 'phone_number', 'ip_address', 'url',
    'street_address', 'city', 'state', 'country', 'zip_code',
    # Financial (PCI)
    'credit_card_number', 'cvv', 'expiration_date',
    'social_security_number', 'bank_account_number', 'routing_number', 'tax_id',
    # Healthcare (PHI)
    'medical_condition', 'drug_name', 'dosage', 'blood_type',
    'medical_code', 'healthcare_number',
    # Identity
    'passport_number', 'driver_license_number', 'username', 'password',
    'vehicle_identification_number',
    # Employment
    'company_name', 'job_title', 'employee_id', 'salary',
    # Digital
    'mac_address', 'imei', 'device_id', 'api_key', 'access_token',
    # Relationship
    'spouse_name', 'child_name', 'emergency_contact',
    # Geographic
    'latitude', 'longitude', 'gps_coordinates',
    # Temporal
    'date', 'time', 'timestamp',
    # Misc
    'account_number', 'license_plate', 'serial_number',
    'insurance_policy_number', 'beneficiary',
)

PII_LABEL_MAP: dict[str, str] = {
    'name': 'PER', 'first_name': 'PER', 'last_name': 'PER',
    'spouse_name': 'PER', 'child_name': 'PER', 'emergency_contact': 'PER',
    'company_name': 'ORG', 'court_name': 'ORG', 'school_name': 'ORG',
    'city': 'LOC', 'state': 'LOC', 'country': 'LOC',
}

PII_DEFAULT_MODEL = 'knowledgator/gliner-pii-base-v1.0'
PII_DEFAULT_THRESHOLD = 0.3  # Recall > precision for PII

VALID_NER_MODES = frozenset({'standard', 'pii'})
VALID_REPLACEMENT_MODES = frozenset({'placeholder', 'synthetic', 'reversible'})

# Optional ModernBERT NER models (available when exported via export_onnx_models.py)
MODERNBERT_NER_MODELS: dict[str, str] = {
    'ENGLISH': 'rhnfzl/modernbert-base-ner-conll03-english-onnx',
    # Multilingual: not yet available — WikiNEuRaL remains the best option
    'MULTILINGUAL': 'rhnfzl/wikineural-multilingual-ner-onnx',
}

# GLiClass document-level classification defaults
GLICLASS_DEFAULT_MODEL = 'knowledgator/gliclass-edge-v3.0'      # 32.7M params, 131MB
GLICLASS_ONNX_MODEL = 'cnmoro/gliclass-edge-v3.0-onnx'         # Community ONNX conversion

# Placeholder constants for future multilingual NER models (blocked: models not yet published)
OTTER_NER_MODELS: dict[str, str] = {}   # arXiv:2601.06347 — checkpoints not on HF Hub
MMBERT_NER_MODELS: dict[str, str] = {}  # GLiNER bug #332 blocks mmBERT as backbone


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
    ner_batch_size: int = 8
    language: Optional[Union[str, Tuple[str, ...]]] = None

    # NER backend selection
    ner_backend: str = 'onnx'  # 'onnx' | 'torch' | 'gliner' | 'ensemble_onnx' | 'ensemble_torch' | 'presidio_gliner'

    # NER mode: 'standard' uses configured models, 'pii' auto-configures GLiNER for PII detection
    ner_mode: str = 'standard'
    # Replacement mode: 'placeholder' uses <TAG> tokens, 'synthetic' uses Faker-generated values
    replacement_mode: str = 'placeholder'

    # Preferred: language-keyed dict of HuggingFace ONNX model repo IDs.
    # Models must have model.onnx, config.json, tokenizer.json on Hub.
    # ENGLISH and MULTILINGUAL are always required (loaded eagerly).
    # Missing keys are filled from DEFAULT_NER_MODELS.
    ner_models: Optional[Mapping[str, str]] = None

    # Ensemble routing: maps each input language to an ordered tuple of model keys
    # to run. Keys must exist in the resolved ner_models dict.
    # If None, DEFAULT_NER_ENSEMBLE is used (with NER_ENSEMBLE_DEFAULT_KEYS fallback).
    ner_ensemble: Optional[Mapping[str, Tuple[str, ...]]] = None

    # Fallback keys for any language not listed in ner_ensemble (e.g. FR, PT, IT).
    # If None, NER_ENSEMBLE_DEFAULT_KEYS is used. Pass an empty tuple to skip
    # ensemble inference for unmapped languages.
    ner_ensemble_default_keys: Optional[Tuple[str, ...]] = None

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

    # Entity description labels (ZERONER-style): maps labels to natural-language descriptions.
    # When provided, descriptions are sent to GLiNER for inference, results mapped back to labels.
    # Example: {'person': "a person's full legal name", 'location': "a geographical place name"}
    gliner_label_descriptions: Optional[Mapping[str, str]] = None

    # Load GLiNER model in ONNX mode (uses pre-built ONNX weights from HuggingFace Hub).
    # Only works with uni-encoder models. Auto-set when ner_mode='pii' + ner_backend='onnx'.
    gliner_onnx: bool = False

    # GLiClass document-level pre-classification (zero-shot)
    check_classify_document: bool = False
    gliclass_model: Optional[str] = None
    gliclass_labels: Tuple[str, ...] = ()
    gliclass_threshold: float = 0.5
    gliclass_classification_type: str = 'single-label'
    gliclass_onnx: bool = False

    # Plugin: user-provided pipeline steps, each callable (text: str) -> str.
    # Appended after all built-in steps in _init_pipeline().
    custom_pipeline_steps: Tuple = ()

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
        if isinstance(self.custom_pipeline_steps, list):
            object.__setattr__(self, 'custom_pipeline_steps', tuple(self.custom_pipeline_steps))
        for step in self.custom_pipeline_steps:
            if not callable(step):
                raise ValueError(
                    f"custom_pipeline_steps must contain callables, got: {type(step)!r}"
                )

        # --- NER mode + replacement mode validation ---
        if self.ner_mode not in VALID_NER_MODES:
            raise ValueError(
                f"ner_mode must be one of {sorted(VALID_NER_MODES)}, got: {self.ner_mode!r}"
            )
        if self.replacement_mode not in VALID_REPLACEMENT_MODES:
            raise ValueError(
                f"replacement_mode must be one of {sorted(VALID_REPLACEMENT_MODES)}, "
                f"got: {self.replacement_mode!r}"
            )

        # PII mode auto-configuration: sets GLiNER defaults for PII detection.
        # User-provided values take priority (checked via default comparisons).
        if self.ner_mode == 'pii':
            if self.ner_backend == 'onnx':
                object.__setattr__(self, 'ner_backend', 'gliner')
                # Use ONNX-loaded GLiNER when available (pre-built ONNX on Hub)
                if not self.gliner_onnx:
                    object.__setattr__(self, 'gliner_onnx', True)
            if not self.gliner_model:
                object.__setattr__(self, 'gliner_model', PII_DEFAULT_MODEL)
            if self.gliner_labels == ('person', 'organization', 'location'):
                object.__setattr__(self, 'gliner_labels', PII_LABELS)
            if self.gliner_threshold == 0.4:
                object.__setattr__(self, 'gliner_threshold', PII_DEFAULT_THRESHOLD)
            if self.gliner_label_map is None:
                object.__setattr__(self, 'gliner_label_map', PII_LABEL_MAP)
            # Expand positional_tags to include all PII label tags
            pii_tags = set(self.positional_tags)
            for label in self.gliner_labels:
                pii_tags.add(PII_LABEL_MAP.get(label, label.upper()))
            object.__setattr__(self, 'positional_tags', tuple(sorted(pii_tags)))

        # --- NER backend validation ---
        if self.ner_backend not in VALID_NER_BACKENDS:
            raise ValueError(
                f"ner_backend must be one of {sorted(VALID_NER_BACKENDS)}, "
                f"got: {self.ner_backend!r}"
            )

        # ner_batch_size must be positive
        if self.ner_batch_size <= 0:
            raise ValueError(
                f"ner_batch_size must be >= 1, got {self.ner_batch_size}"
            )

        # GLiNER fields required for gliner/ensemble backends
        needs_gliner = self.ner_backend in (
            'gliner', 'ensemble_onnx', 'ensemble_torch', 'presidio_gliner',
        )
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

        # Freeze gliner_label_descriptions
        if self.gliner_label_descriptions is not None:
            object.__setattr__(self, 'gliner_label_descriptions',
                               MappingProxyType(dict(self.gliner_label_descriptions)))

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
            import warnings
            warnings.warn(
                "ner_models_list is deprecated. Use ner_models dict instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            models_dict = dict(zip(LANG_KEYS, self.ner_models_list))
            object.__setattr__(self, 'ner_models', MappingProxyType(models_dict))

        # --- Language validation and supported_languages computation ---
        from sct.utils.resources import resolve_language

        # Resolve language (str or tuple of str) via ISO code / name lookup
        if self.language is not None:
            if isinstance(self.language, str):
                object.__setattr__(self, 'language', resolve_language(self.language))
            elif isinstance(self.language, (list, tuple)):
                resolved = tuple(resolve_language(lang) for lang in self.language)
                object.__setattr__(self, 'language', resolved)

        # Coerce extra_languages to tuple and resolve each entry
        if isinstance(self.extra_languages, list):
            object.__setattr__(self, 'extra_languages', tuple(self.extra_languages))
        if self.extra_languages:
            resolved_extra = tuple(resolve_language(lang) for lang in self.extra_languages)
            object.__setattr__(self, 'extra_languages', resolved_extra)

        # Validate and freeze custom_stopwords (keys must be valid language names)
        if self.custom_stopwords:
            resolved_sw = {resolve_language(k): v for k, v in self.custom_stopwords.items()}
            object.__setattr__(self, 'custom_stopwords',
                               MappingProxyType(resolved_sw))

        # Validate and freeze custom_month_names
        if self.custom_month_names:
            resolved_mn = {resolve_language(k): tuple(v) for k, v in self.custom_month_names.items()}
            object.__setattr__(self, 'custom_month_names',
                               MappingProxyType(resolved_mn))

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
        # Add languages from tuple-of-languages config
        if isinstance(self.language, tuple):
            langs.update(self.language)
        object.__setattr__(self, 'supported_languages', frozenset(langs))

        # Validate language pin/restriction against supported set
        if self.language is not None:
            if isinstance(self.language, str):
                if self.language not in self.supported_languages:
                    raise ValueError(
                        f"language='{self.language}' not in supported languages: "
                        f"{sorted(self.supported_languages)}"
                    )
            elif isinstance(self.language, tuple):
                for lang in self.language:
                    if lang not in self.supported_languages:
                        raise ValueError(
                            f"language='{lang}' (from tuple) not in supported languages: "
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
        ner_mode=getattr(m, 'NER_MODE', 'standard'),
        replacement_mode=getattr(m, 'REPLACEMENT_MODE', 'placeholder'),
        gliner_onnx=getattr(m, 'GLINER_ONNX', False),
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
NER_MODE = 'standard'
REPLACEMENT_MODE = 'placeholder'
GLINER_ONNX = False
LANGUAGE = None

# Order: English, Dutch, German, Spanish, Multilingual
NER_MODELS_LIST = [
    "rhnfzl/xlm-roberta-large-conll03-english-onnx",
    "rhnfzl/xlm-roberta-large-conll02-dutch-onnx",
    "rhnfzl/xlm-roberta-large-conll03-german-onnx",
    "rhnfzl/xlm-roberta-large-conll02-spanish-onnx",
    "rhnfzl/wikineural-multilingual-ner-onnx",
]
