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
from datetime import date
from types import MappingProxyType
from typing import Any, Mapping, Optional, Tuple, Union


LANG_KEYS = ('ENGLISH', 'DUTCH', 'GERMAN', 'SPANISH', 'MULTILINGUAL')
REQUIRED_NER_KEYS = frozenset({'ENGLISH', 'MULTILINGUAL'})
DEFAULT_LANGUAGES = frozenset({'ENGLISH', 'DUTCH', 'GERMAN', 'SPANISH'})

VALID_NER_BACKENDS = frozenset({
    'onnx', 'torch', 'gliner', 'ensemble_onnx', 'ensemble_torch', 'presidio_gliner',
})
GLINER_BACKENDS = frozenset({'gliner', 'ensemble_onnx', 'ensemble_torch', 'presidio_gliner'})

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

DEFAULT_NER_REVISIONS: dict[str, str] = {
    'ENGLISH': '2c10955af9538d6ff8a68316c5dbfcd0b968761e',
    'DUTCH': '381e34ba4c222bc221e9fd02832f52a179ed840d',
    'GERMAN': 'fe4aecfa6dc376244ef37da37e6df3b1b0911d71',
    'SPANISH': 'ef52a7f08172203c763d51fe36660d36985bb388',
    'FRENCH': '74191432b21c956b5b04b4778c0e92e569b1f80d',
    'PORTUGUESE': '74191432b21c956b5b04b4778c0e92e569b1f80d',
    'ITALIAN': '74191432b21c956b5b04b4778c0e92e569b1f80d',
    'MULTILINGUAL': '74191432b21c956b5b04b4778c0e92e569b1f80d',
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

DEFAULT_TORCH_NER_REVISIONS: dict[str, str] = {
    'ENGLISH': '18f95e9924f3f452df09cc90945073906ef18f1e',
    'DUTCH': '630d5d48d08071704d9a2719b045082019b6ac12',
    'GERMAN': '1fbcc7a00a69ce5ab754623154a8e9cc6ba868e2',
    'SPANISH': 'a7c5f08c766adcbd6c22f343145fa65d38b3c1d5',
    'FRENCH': 'bed6ee7a45d2827b6c90a4fd7983f0241ae0a5c1',
    'PORTUGUESE': 'bed6ee7a45d2827b6c90a4fd7983f0241ae0a5c1',
    'ITALIAN': 'bed6ee7a45d2827b6c90a4fd7983f0241ae0a5c1',
    'MULTILINGUAL': 'bed6ee7a45d2827b6c90a4fd7983f0241ae0a5c1',
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
    # One name per type on every surface: the regex steps emit <PHONE> / <IP> too.
    'phone_number': 'PHONE', 'ip_address': 'IP',
}

# legacy_type_names -> the label map PII mode derives (identity marks it as derived).
_PII_MAPS = {legacy: MappingProxyType({k: v for k, v in PII_LABEL_MAP.items()
                                       if not (legacy and v in ('PHONE', 'IP'))})
             for legacy in (False, True)}

PII_DEFAULT_MODEL = 'knowledgator/gliner-pii-base-v1.0'
PII_DEFAULT_REVISION = '61726e0ad791dcab3e29339bbec3ad42ded65641'
PII_DEFAULT_THRESHOLD = 0.3  # Recall > precision for PII

VALID_NER_MODES = frozenset({'standard', 'pii'})
VALID_REPLACEMENT_MODES = frozenset({'placeholder', 'synthetic', 'reversible'})
VALID_JSON_KEY_CLEANING = frozenset({'pii', 'pii_ner', 'full', 'off'})

# Optional ModernBERT NER models (available when exported via export_onnx_models.py)
MODERNBERT_NER_MODELS: dict[str, str] = {
    'ENGLISH': 'rhnfzl/modernbert-base-ner-conll03-english-onnx',
    # Multilingual: not yet available — WikiNEuRaL remains the best option
    'MULTILINGUAL': 'rhnfzl/wikineural-multilingual-ner-onnx',
}

# GLiClass document-level classification defaults
GLICLASS_DEFAULT_MODEL = 'knowledgator/gliclass-edge-v3.0'      # 32.7M params, 131MB
GLICLASS_DEFAULT_REVISION = 'df03993a2ed98e5e4a0d2dd7efbbd105abe874cf'

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
    check_to_ascii_unicode: bool = False  # True = unidecode transliteration (0.7: 日本語 -> Ri Ben Yu)
    check_replace_html: bool = True
    check_replace_urls: bool = True
    check_replace_emails: bool = True
    check_replace_years: bool = True
    check_replace_dates: bool = True
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
    ner_confidence_threshold: Optional[float] = None  # None = effective_ner_confidence_threshold
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
    ner_model_revisions: Optional[Mapping[str, str]] = None

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
    torch_ner_model_revisions: Optional[Mapping[str, str]] = None

    # GLiNER settings (only used when ner_backend contains 'gliner' or 'ensemble')
    gliner_model: Optional[str] = None
    gliner_revision: Optional[str] = None
    gliner_variant: str = 'gliner'  # 'gliner' or 'gliner2'
    gliner_labels: Tuple[str, ...] = ('person', 'organization', 'location')
    gliner_label_map: Optional[Mapping[str, str]] = None
    gliner_threshold: Optional[float] = None  # None = effective_gliner_threshold

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
    gliclass_revision: Optional[str] = None
    gliclass_labels: Tuple[str, ...] = ()
    gliclass_threshold: float = 0.5
    gliclass_classification_type: str = 'single-label'
    gliclass_onnx: bool = False

    # Plugin: user-provided pipeline steps, each callable (text: str) -> str.
    # Appended to the pre-NER pipeline in _init_pipeline(), so they run before NER.
    custom_pipeline_steps: Tuple = ()

    # Language extensibility — extend Lingua detection, stopwords, dates, NER
    extra_languages: Tuple[str, ...] = ()
    custom_stopwords: Optional[Mapping[str, frozenset]] = None
    custom_month_names: Optional[Mapping[str, Tuple[str, ...]]] = None

    # Computed at __post_init__ — union of all language sources
    supported_languages: frozenset = frozenset()

    # Added in 0.7.1; kept last so existing positional arguments keep their meaning.
    check_replace_ips: bool = True
    replace_ips_public_only: bool = False  # True keeps private/loopback/reserved IPs
    replace_with_ips: str = "<IP>"
    replace_numbers_in_words: bool = False  # True also replaces digits glued to Latin/Greek/Cyrillic letters (GPT4)

    # Added in 0.8.0; kept last so existing positional arguments keep their meaning.
    onnx_intra_op_threads: Optional[int] = None  # None = ONNX Runtime default (all cores)
    onnx_inter_op_threads: Optional[int] = None  # None = ONNX Runtime default
    ner_chunk_overlap: int = 0  # tokens of neighbouring text each NER window adds per side; 0 = 0.7.1 chunks
    detect_all_languages: bool = True  # False = 0.7 detector over supported_languages only
    language_confidence_warning: float = 0.0  # below this confidence, warn language_low_confidence; 0.5 = early 0.8.0
    check_decode_unicode_escapes: bool = True  # False keeps literal \n, \u00e9 escapes (code, JSON)
    ner_device: Optional[str] = None  # None = auto (CUDA when available), 'cpu', 'cuda'
    cuda_device_id: int = 0  # GPU index for ONNX, torch and GLiNER when on CUDA
    onnx_cuda_provider_options: Optional[Mapping[str, Any]] = None  # extra CUDAExecutionProvider options
    # process_json keys: 'pii' = regex PII steps only (NER tags lone words like 'message' out of context),
    # 'pii_ner' = PII steps plus NER, 'full' = whole config (0.7), 'off' = raw
    json_key_cleaning: str = 'pii'
    synthetic_seed: int = 42  # base seed for replacement_mode='synthetic'
    # True: fakes seeded per document from synthetic_seed + its text (differ across documents,
    # same document -> same fakes); False = 0.7 single Faker stream per thread
    synthetic_per_document: bool = True
    # Warn language_low_confidence when the top language leads the runner-up by less than this
    # (ambiguous detection); 0 turns it off. Absolute confidence is low for many correct texts.
    language_margin_warning: float = 0.15
    # Part of check_fix_bad_unicode. False keeps full-width CJK punctuation (，！：) and half-width
    # katakana, converting only full-width ASCII letters/digits; True = ftfy's full width fix (0.7)
    fix_character_width: bool = False
    # True: phone and IP types use their pre-0.8.0 names PHONE_NUMBER / IP_ADDRESS in reversible
    # placeholders, metadata, signals and PII-mode NER tags; False = PHONE / IP everywhere
    legacy_type_names: bool = False
    # WordPiece NER: widen an entity to its whole word when the added text is at most this many
    # times the tagged span ("John Do" + "##e"); None = always (whole unspaced Thai/kana clauses
    # too), 0 = never (0.7)
    ner_subword_widen: Optional[float] = 1.0
    # Part of check_fix_bad_unicode. True folds math-alphanumeric (𝐞𝐱), circled (ⓙ) and full-width
    # URL punctuation (：／？) inside ASCII runs and drops invisible Cf characters (ZWSP, soft hyphen)
    # inside them, so styled emails/URLs/phones are replaced without
    # check_to_ascii_unicode; False = earlier 0.8.0 (full-width letters/digits and ＠．－＿＋ only)
    fold_styled_characters: bool = True
    # WordPiece NER widening stops where the Unicode script of the letters changes, so a Latin
    # name glued to kana or Thai ("David Millerは") keeps its own span; False = widen across scripts
    ner_subword_widen_same_script: bool = True
    # Faker locale for replacement_mode='synthetic': None = follow the detected document language
    # (Dutch text gets Dutch names, en_US when Faker has no locale for it); a string forces that
    # locale ('en_US' = 0.7)
    synthetic_locale: Optional[str] = None
    # synthetic/reversible: NER reads the same text as placeholder mode (<EMAIL>, <PHONE>...) and
    # the fakes / indexed placeholders go in after NER, so every mode finds the same entities;
    # False = 0.7 order (NER reads the fakes / <EMAIL_0>)
    ner_sees_tags: bool = True
    # synthetic_locale=None: Faker locale for documents whose detection gets the
    # language_low_confidence warning; None = follow the detected language anyway (earlier 0.8.0)
    synthetic_low_confidence_locale: Optional[str] = 'en_US'
    # synthetic: drop a fake's final '.' when a '.' follows the entity ('B.Eng..' -> 'B.Eng.');
    # False = keep both (earlier 0.8.0)
    synthetic_collapse_period: bool = True
    # synthetic: ISO date that date fakes (DATE, DATE_OF_BIRTH) end on, so a document gets the
    # same dates on any day; None = today, from the clock (earlier 0.8.0)
    synthetic_reference_date: Optional[str] = '2026-01-01'
    # NER ensemble vote per span: 'max' keeps a span when its best model score passes
    # ner_confidence_threshold, so a weak second model never hides a strong hit; 'mean' averages
    # the model scores (0.8.0)
    ner_ensemble_vote: str = 'max'
    # mask every other case-sensitive whole-word copy of a masked PER/ORG/LOC value in the same
    # document (the value comes from the model's own span); False = 0.8.0 output
    ner_propagate_masked: bool = True
    # score a span needs before its value is propagated (0..1); None = the effective NER cut.
    # Raising it spreads fewer wrong tags at the cost of more visible repeats
    ner_propagate_min_score: Optional[float] = None
    # IBANs (mod-97 checksum) and payment cards (Luhn checksum) become one <IBAN> / <CARD> each,
    # before the phone step; False = 0.8.0 (the phone/number steps take pieces of them)
    check_replace_iban_cards: bool = True
    replace_with_iban: str = "<IBAN>"
    replace_with_cards: str = "<CARD>"
    # Identifier tokens: 6+ uppercase letters and digits with 2+ digits and a letter (RBCON68T08).
    # Off by default: the shape also takes product and reference codes (RTX4090, COVID19)
    check_replace_identifiers: bool = False
    replace_with_identifiers: str = "<ID>"
    # Bracket/brace removal only takes innermost spans of at most this many characters and
    # whitespace-separated words, so prose clauses stay; 0 = no limit (both 0 = 0.8.0 matching)
    bracket_content_max_chars: int = 32
    bracket_content_max_words: int = 3
    # A {...} that parses as a JSON object (json.raw_decode, nested ones included) is never removed
    keep_json_braces: bool = True
    # A month name, "/" and a 2- or 4-digit year is one <DATE> (October/55, Oct/2019)
    date_month_name_slash_year: bool = True
    # ISO codes of the built-in month data dates use (en nl de es fr pt it th); None = all
    date_builtin_month_languages: Optional[Tuple[str, ...]] = None
    # HTML rule reads only tag shapes (<b>, </a>, <br/>, <!-- -->); False = 0.8.0 "<.*?>" (eats "<a@b.com>")
    html_tag_strict: bool = True

    def __post_init__(self):
        if self.synthetic_reference_date is not None:
            date.fromisoformat(self.synthetic_reference_date)  # ValueError on a bad date
        if self.gliclass_onnx:
            raise ValueError(
                "GLiClass ONNX is not supported by the gliclass package. "
                "Use gliclass_onnx=False."
            )
        if (
            self.gliclass_model in (None, GLICLASS_DEFAULT_MODEL)
            and self.gliclass_revision is None
        ):
            object.__setattr__(
                self,
                'gliclass_revision',
                GLICLASS_DEFAULT_REVISION,
            )

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
        if self.json_key_cleaning not in VALID_JSON_KEY_CLEANING:
            raise ValueError(
                f"json_key_cleaning must be one of {sorted(VALID_JSON_KEY_CLEANING)}, "
                f"got: {self.json_key_cleaning!r}"
            )
        if (
            self.replacement_mode == 'reversible'
            and self.check_fuzzy_replace_dates
        ):
            raise ValueError(
                "reversible mode does not support fuzzy date replacement"
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
            if (
                self.gliner_model == PII_DEFAULT_MODEL
                and self.gliner_revision is None
            ):
                object.__setattr__(
                    self,
                    'gliner_revision',
                    PII_DEFAULT_REVISION,
                )
            if self.gliner_labels == ('person', 'organization', 'location'):
                object.__setattr__(self, 'gliner_labels', PII_LABELS)
            pii_tags = set(self.positional_tags)
            # Derived maps are shared objects, so a dataclasses.replace() clone re-derives them
            # (and drops the other naming's phone/IP tags) when legacy_type_names changes.
            if self.gliner_label_map is _PII_MAPS[True] or self.gliner_label_map is _PII_MAPS[False]:
                pii_tags -= {'PHONE', 'IP', 'PHONE_NUMBER', 'IP_ADDRESS'}
                object.__setattr__(self, 'gliner_label_map', None)
            if self.gliner_label_map is None:
                object.__setattr__(self, 'gliner_label_map', _PII_MAPS[self.legacy_type_names])
            # Expand positional_tags to include all PII label tags
            for label in self.gliner_labels:
                pii_tags.add(self.gliner_label_map.get(label, label.upper()))
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
        for name in ('onnx_intra_op_threads', 'onnx_inter_op_threads'):
            value = getattr(self, name)
            if value is not None and value < 1:
                raise ValueError(f"{name} must be None or >= 1, got {value}")
        if self.ner_chunk_overlap < 0:
            raise ValueError(f"ner_chunk_overlap must be >= 0, got {self.ner_chunk_overlap}")
        if self.ner_device not in (None, 'cpu', 'cuda'):
            raise ValueError(f"ner_device must be None, 'cpu' or 'cuda', got {self.ner_device!r}")
        if self.cuda_device_id < 0:
            raise ValueError(f"cuda_device_id must be >= 0, got {self.cuda_device_id}")
        if self.onnx_cuda_provider_options is not None:
            object.__setattr__(self, 'onnx_cuda_provider_options',
                               MappingProxyType(dict(self.onnx_cuda_provider_options)))
        for name in ('ner_confidence_threshold', 'gliner_threshold'):
            value = getattr(self, name)
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be None or in [0, 1], got {value}")
        if self.ner_ensemble_vote not in ('max', 'mean'):
            raise ValueError(f"ner_ensemble_vote must be 'max' or 'mean', got {self.ner_ensemble_vote!r}")
        if self.ner_propagate_min_score is not None and not 0.0 <= self.ner_propagate_min_score <= 1.0:
            raise ValueError(f"ner_propagate_min_score must be None or in [0, 1], got {self.ner_propagate_min_score}")
        if self.ner_subword_widen is not None and self.ner_subword_widen < 0:
            raise ValueError(f"ner_subword_widen must be None or >= 0, got {self.ner_subword_widen}")
        if not 0.0 <= self.language_confidence_warning <= 1.0:
            raise ValueError(
                "language_confidence_warning must be in [0, 1], "
                f"got {self.language_confidence_warning}"
            )
        if not 0.0 <= self.language_margin_warning <= 1.0:
            raise ValueError(
                f"language_margin_warning must be in [0, 1], got {self.language_margin_warning}"
            )

        # GLiNER fields required for gliner/ensemble backends
        needs_gliner = self.ner_backend in GLINER_BACKENDS
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
        if self.gliner_label_map is not None and not any(self.gliner_label_map is m for m in _PII_MAPS.values()):
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
        if self.torch_ner_models is not None:
            torch_revisions = {
                key: revision
                for key, revision in DEFAULT_TORCH_NER_REVISIONS.items()
                if self.torch_ner_models.get(key) == DEFAULT_TORCH_NER_MODELS.get(key)
            }
            if self.torch_ner_model_revisions is not None:
                torch_revisions.update(self.torch_ner_model_revisions)
            object.__setattr__(
                self,
                'torch_ner_model_revisions',
                MappingProxyType(torch_revisions),
            )

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
            # A map of exactly LANG_KEYS matching ner_models_list is a normalized
            # legacy/default config (replace(), copy): keep it exact. Decided by
            # content only, so a dict and a mapping proxy give the same config.
            derived = (
                set(self.ner_models) == set(LANG_KEYS)
                and tuple(self.ner_models[k] for k in LANG_KEYS) == self.ner_models_list
            )
            merged = (
                dict(self.ner_models) if derived
                else {**DEFAULT_NER_MODELS, **self.ner_models}
            )
            object.__setattr__(self, 'ner_models', MappingProxyType(merged))
            object.__setattr__(
                self, 'ner_models_list',
                tuple(merged[k] for k in LANG_KEYS),
            )
        else:
            # No dict provided — derive from ner_models_list (may be default or custom)
            default_models_list = tuple(DEFAULT_NER_MODELS[k] for k in LANG_KEYS)
            if self.ner_models_list != default_models_list:
                import warnings
                warnings.warn(
                    "ner_models_list is deprecated. Use ner_models dict instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            models_dict = dict(zip(LANG_KEYS, self.ner_models_list))
            object.__setattr__(self, 'ner_models', MappingProxyType(models_dict))

        pinned_revisions = {
            key: revision
            for key, revision in DEFAULT_NER_REVISIONS.items()
            if self.ner_models.get(key) == DEFAULT_NER_MODELS.get(key)
        }
        if self.ner_model_revisions is not None:
            pinned_revisions.update(self.ner_model_revisions)
        object.__setattr__(
            self,
            'ner_model_revisions',
            MappingProxyType(pinned_revisions),
        )

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
            resolved_sw = {resolve_language(k): frozenset(v) for k, v in self.custom_stopwords.items()}
            object.__setattr__(self, 'custom_stopwords',
                               MappingProxyType(resolved_sw))

        if self.date_builtin_month_languages is not None:
            from sct.utils.constants import MONTH_NAMES_MULTILINGUAL
            codes = tuple(self.date_builtin_month_languages)
            unknown = set(codes) - set().union(*MONTH_NAMES_MULTILINGUAL.values())
            if isinstance(self.date_builtin_month_languages, str) or unknown:
                raise ValueError(f"date_builtin_month_languages must be a tuple of built-in month ISO codes, "
                                 f"got {self.date_builtin_month_languages!r}")
            object.__setattr__(self, 'date_builtin_month_languages', codes)

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
        # Any Lingua language is a valid pin (resolve_language raised above otherwise);
        # supported_languages only means "has stopwords/dates/NER key".
        object.__setattr__(self, 'supported_languages', frozenset(langs))

    @property
    def effective_gliner_threshold(self) -> float:
        """GLiNER score cut: explicit value, else 0.3 in PII mode, else 0.4."""
        if self.gliner_threshold is not None:
            return self.gliner_threshold
        return PII_DEFAULT_THRESHOLD if self.ner_mode == 'pii' else 0.4

    @property
    def effective_ner_confidence_threshold(self) -> float:
        """Final NER filter: explicit value, else the GLiNER cut in PII mode, else 0.85."""
        if self.ner_confidence_threshold is not None:
            return self.ner_confidence_threshold
        # PII mode: the final ensemble filter must not drop what GLiNER kept.
        return self.effective_gliner_threshold if self.ner_mode == 'pii' else 0.85

    @classmethod
    def from_preset(cls, name: str = "training", **overrides) -> "TextCleanerConfig":
        """Build a config from a named preset (see ``sct.presets.PRESETS``)."""
        from sct.presets import preset_config
        return preset_config(name, **overrides)

    def __getstate__(self):
        # Restore the exact normalized state; never re-run __post_init__.
        state = dict(self.__dict__)
        frozen = [k for k, v in state.items() if isinstance(v, MappingProxyType)]
        for key in frozen:
            state[key] = dict(state[key])
        # derived only in PII mode (re-shared on load); a map kept after a replace() to another mode is data
        if self.ner_mode == 'pii' and any(self.gliner_label_map is m for m in _PII_MAPS.values()):
            state['gliner_label_map'] = None
            frozen.remove('gliner_label_map')
        return state, frozen

    def __setstate__(self, packed):
        state, frozen = packed
        for key, value in state.items():
            if key in frozen:
                value = MappingProxyType(value)
            object.__setattr__(self, key, value)
        if self.ner_mode == 'pii' and self.gliner_label_map is None:
            object.__setattr__(self, 'gliner_label_map', _PII_MAPS[self.legacy_type_names])


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
        check_replace_ips=getattr(m, 'CHECK_REPLACE_IPS', True),
        replace_ips_public_only=getattr(m, 'REPLACE_IPS_PUBLIC_ONLY', False),
        check_replace_years=m.CHECK_REPLACE_YEARS,
        check_replace_dates=getattr(m, 'CHECK_REPLACE_DATES', True),
        check_fuzzy_replace_dates=getattr(m, 'CHECK_FUZZY_REPLACE_DATES', False),
        fuzzy_date_score_cutoff=getattr(m, 'FUZZY_DATE_SCORE_CUTOFF', 85),
        check_replace_phone_numbers=m.CHECK_REPLACE_PHONE_NUMBERS,
        check_replace_numbers=m.CHECK_REPLACE_NUMBERS,
        replace_numbers_in_words=getattr(m, 'REPLACE_NUMBERS_IN_WORDS', False),
        onnx_intra_op_threads=getattr(m, 'ONNX_INTRA_OP_THREADS', None),
        onnx_inter_op_threads=getattr(m, 'ONNX_INTER_OP_THREADS', None),
        ner_batch_size=getattr(m, 'NER_BATCH_SIZE', 8),
        ner_chunk_overlap=getattr(m, 'NER_CHUNK_OVERLAP', 0),
        detect_all_languages=getattr(m, 'DETECT_ALL_LANGUAGES', True),
        language_confidence_warning=getattr(m, 'LANGUAGE_CONFIDENCE_WARNING', 0.0),
        check_decode_unicode_escapes=getattr(m, 'CHECK_DECODE_UNICODE_ESCAPES', True),
        ner_device=getattr(m, 'NER_DEVICE', None),
        cuda_device_id=getattr(m, 'CUDA_DEVICE_ID', 0),
        onnx_cuda_provider_options=getattr(m, 'ONNX_CUDA_PROVIDER_OPTIONS', None),
        json_key_cleaning=getattr(m, 'JSON_KEY_CLEANING', 'pii'),
        synthetic_seed=getattr(m, 'SYNTHETIC_SEED', 42),
        synthetic_per_document=getattr(m, 'SYNTHETIC_PER_DOCUMENT', True),
        language_margin_warning=getattr(m, 'LANGUAGE_MARGIN_WARNING', 0.15),
        fix_character_width=getattr(m, 'FIX_CHARACTER_WIDTH', False),
        legacy_type_names=getattr(m, 'LEGACY_TYPE_NAMES', False),
        ner_subword_widen=getattr(m, 'NER_SUBWORD_WIDEN', 1.0),
        fold_styled_characters=getattr(m, 'FOLD_STYLED_CHARACTERS', True),
        ner_subword_widen_same_script=getattr(m, 'NER_SUBWORD_WIDEN_SAME_SCRIPT', True),
        synthetic_locale=getattr(m, 'SYNTHETIC_LOCALE', None),
        ner_sees_tags=getattr(m, 'NER_SEES_TAGS', True),
        synthetic_low_confidence_locale=getattr(m, 'SYNTHETIC_LOW_CONFIDENCE_LOCALE', 'en_US'),
        synthetic_collapse_period=getattr(m, 'SYNTHETIC_COLLAPSE_PERIOD', True),
        synthetic_reference_date=getattr(m, 'SYNTHETIC_REFERENCE_DATE', '2026-01-01'),
        ner_ensemble_vote=getattr(m, 'NER_ENSEMBLE_VOTE', 'max'),
        ner_propagate_masked=getattr(m, 'NER_PROPAGATE_MASKED', True),
        ner_propagate_min_score=getattr(m, 'NER_PROPAGATE_MIN_SCORE', None),
        check_replace_iban_cards=getattr(m, 'CHECK_REPLACE_IBAN_CARDS', True),
        replace_with_iban=getattr(m, 'REPLACE_WITH_IBAN', '<IBAN>'),
        replace_with_cards=getattr(m, 'REPLACE_WITH_CARDS', '<CARD>'),
        check_replace_identifiers=getattr(m, 'CHECK_REPLACE_IDENTIFIERS', False),
        replace_with_identifiers=getattr(m, 'REPLACE_WITH_IDENTIFIERS', '<ID>'),
        bracket_content_max_chars=getattr(m, 'BRACKET_CONTENT_MAX_CHARS', 32),
        bracket_content_max_words=getattr(m, 'BRACKET_CONTENT_MAX_WORDS', 3),
        keep_json_braces=getattr(m, 'KEEP_JSON_BRACES', True),
        date_month_name_slash_year=getattr(m, 'DATE_MONTH_NAME_SLASH_YEAR', True),
        date_builtin_month_languages=getattr(m, 'DATE_BUILTIN_MONTH_LANGUAGES', None),
        html_tag_strict=getattr(m, 'HTML_TAG_STRICT', True),
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
        replace_with_ips=getattr(m, 'REPLACE_WITH_IPS', '<IP>'),
        replace_with_years=m.REPLACE_WITH_YEARS,
        replace_with_dates=getattr(m, 'REPLACE_WITH_DATES', '<DATE>'),
        replace_with_phone_numbers=m.REPLACE_WITH_PHONE_NUMBERS,
        replace_with_numbers=m.REPLACE_WITH_NUMBERS,
        replace_with_currency_symbols=m.REPLACE_WITH_CURRENCY_SYMBOLS,
        positional_tags=tuple(m.POSITIONAL_TAGS),
        ner_confidence_threshold=getattr(m, 'NER_CONFIDENCE_THRESHOLD', None),
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
CHECK_TO_ASCII_UNICODE = False
CHECK_REPLACE_HTML = True
CHECK_REPLACE_URLS = True
CHECK_REPLACE_EMAILS = True
CHECK_REPLACE_IPS = True
REPLACE_IPS_PUBLIC_ONLY = False
CHECK_REPLACE_YEARS = True
CHECK_REPLACE_DATES = True
CHECK_FUZZY_REPLACE_DATES = False
FUZZY_DATE_SCORE_CUTOFF = 85
CHECK_REPLACE_PHONE_NUMBERS = True
CHECK_REPLACE_NUMBERS = True
REPLACE_NUMBERS_IN_WORDS = False
ONNX_INTRA_OP_THREADS = None
ONNX_INTER_OP_THREADS = None
NER_BATCH_SIZE = 8
NER_CHUNK_OVERLAP = 0
DETECT_ALL_LANGUAGES = True
LANGUAGE_CONFIDENCE_WARNING = 0.0
CHECK_DECODE_UNICODE_ESCAPES = True
NER_DEVICE = None
CUDA_DEVICE_ID = 0
ONNX_CUDA_PROVIDER_OPTIONS = None
JSON_KEY_CLEANING = 'pii'
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
REPLACE_WITH_IPS = "<IP>"
REPLACE_WITH_YEARS = "<YEAR>"
REPLACE_WITH_DATES = "<DATE>"
REPLACE_WITH_PHONE_NUMBERS = "<PHONE>"
REPLACE_WITH_NUMBERS = "<NUMBER>"
REPLACE_WITH_CURRENCY_SYMBOLS = None

POSITIONAL_TAGS = ['PER', 'LOC', 'ORG', 'MISC']
NER_CONFIDENCE_THRESHOLD = None  # None = 0.85, or the GLiNER threshold in PII mode
NER_MODE = 'standard'
REPLACEMENT_MODE = 'placeholder'
SYNTHETIC_SEED = 42
SYNTHETIC_PER_DOCUMENT = True
LANGUAGE_MARGIN_WARNING = 0.15
FIX_CHARACTER_WIDTH = False
LEGACY_TYPE_NAMES = False
NER_SUBWORD_WIDEN = 1.0
FOLD_STYLED_CHARACTERS = True
NER_SUBWORD_WIDEN_SAME_SCRIPT = True
SYNTHETIC_LOCALE = None
NER_SEES_TAGS = True
SYNTHETIC_LOW_CONFIDENCE_LOCALE = 'en_US'
SYNTHETIC_COLLAPSE_PERIOD = True
SYNTHETIC_REFERENCE_DATE = '2026-01-01'
NER_ENSEMBLE_VOTE = 'max'
NER_PROPAGATE_MASKED = True
NER_PROPAGATE_MIN_SCORE = None
CHECK_REPLACE_IBAN_CARDS = True
REPLACE_WITH_IBAN = "<IBAN>"
REPLACE_WITH_CARDS = "<CARD>"
CHECK_REPLACE_IDENTIFIERS = False
REPLACE_WITH_IDENTIFIERS = "<ID>"
BRACKET_CONTENT_MAX_CHARS = 32
BRACKET_CONTENT_MAX_WORDS = 3
KEEP_JSON_BRACES = True
DATE_MONTH_NAME_SLASH_YEAR = True
DATE_BUILTIN_MONTH_LANGUAGES = None
HTML_TAG_STRICT = True
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
