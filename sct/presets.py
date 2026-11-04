"""Named config presets.

``"training"`` (the adapter default) only removes PII and repairs broken
unicode: URLs, emails, phones, IPs and NER entities become placeholders;
numbers, dates, HTML, brackets, case, punctuation, line breaks, indentation
and escapes in code stay as written. ``"sct"`` is the library default config.
"""

from types import MappingProxyType
from typing import Any
from collections.abc import Mapping

from sct.config import TextCleanerConfig

_TRAINING = {
    # On (library defaults, listed so the preset reads on its own)
    'check_replace_urls': True,
    'check_replace_emails': True,
    'check_replace_phone_numbers': True,
    'check_replace_ips': True,
    'check_replace_iban_cards': True,
    'check_ner_process': True,
    'check_fix_bad_unicode': True,
    'check_detect_language': True,
    # Off
    'check_to_ascii_unicode': False,
    'check_replace_numbers': False,
    'check_replace_identifiers': False,  # takes product/reference codes too; not measured on training data
    'check_replace_years': False,
    'check_replace_dates': False,
    'check_fuzzy_replace_dates': False,
    'check_remove_isolated_letters': False,
    'check_replace_html': False,
    'check_decode_unicode_escapes': False,
    'check_remove_bracket_content': False,
    'check_remove_brace_content': False,
    'check_remove_isolated_special_symbols': False,
    'check_replace_currency_symbols': False,
    'check_normalize_whitespace': False,
    'check_casefold': False,
    'check_smart_casefold': False,
    'check_remove_stopwords': False,
    'check_remove_punctuation': False,
    'check_statistical_model_processing': False,
}

PRESETS: Mapping[str, Mapping[str, Any]] = MappingProxyType({
    'training': MappingProxyType(_TRAINING),
    'sct': MappingProxyType({}),
})


def preset_config(name: str = "training", **overrides) -> TextCleanerConfig:
    """Return the ``name`` preset as a config; ``overrides`` are any config fields."""
    if name not in PRESETS:
        raise ValueError(f"unknown preset {name!r}; valid presets: {sorted(PRESETS)}")
    return TextCleanerConfig(**{**PRESETS[name], **overrides})
