"""One name per PII type on every surface: tag, reversible placeholder, metadata, signals."""

import re

import pytest

from sct import TextCleaner, TextCleanerConfig, config
from sct.signals import public_view

TEXT = "Call +1-202-555-0147 from 8.8.8.8 or mail ann@example.com"


def _cleaner(**overrides):
    overrides.setdefault("check_ner_process", False)
    overrides.setdefault("check_detect_language", False)
    overrides.setdefault("check_statistical_model_processing", False)
    return TextCleaner(cfg=TextCleanerConfig(**overrides))


def test_tag_reversible_and_metadata_share_one_name():
    tagged = _cleaner().process(TEXT)
    reversible = _cleaner(replacement_mode="reversible").process(TEXT)

    tags = set(re.findall(r"<([A-Z]+)>", tagged.lm_text))
    placeholders = set(re.findall(r"<([A-Z]+)_\d+>", reversible.lm_text))
    assert tags == placeholders == {"PHONE", "IP", "EMAIL"}
    assert set(tagged.metadata["regex_counts"]) == tags
    assert set(reversible.metadata["regex_counts"]) == tags
    assert set(tagged.signals["pii_types"]) == tags
    assert {e.entity_type for e in reversible.metadata["anon_map"].entries} == tags
    assert reversible.metadata["anon_map"].deanonymize(reversible.lm_text) == TEXT


def test_legacy_type_names_restore_old_names_and_round_trip():
    result = _cleaner(replacement_mode="reversible", legacy_type_names=True).process(TEXT)

    assert "<PHONE_NUMBER_0>" in result.lm_text and "<IP_ADDRESS_0>" in result.lm_text
    assert set(result.metadata["regex_counts"]) == {"PHONE_NUMBER", "IP_ADDRESS", "EMAIL"}
    assert set(public_view(result)["pii_types"]) == {"PHONE_NUMBER", "IP_ADDRESS", "EMAIL"}
    assert result.metadata["anon_map"].deanonymize(result.lm_text) == TEXT
    # Tag text is set by replace_with_* and does not change.
    assert "<PHONE>" in _cleaner(legacy_type_names=True).process(TEXT).lm_text


def test_legacy_type_names_module_global():
    old = config.LEGACY_TYPE_NAMES
    try:
        config.LEGACY_TYPE_NAMES = True
        assert config._config_from_module_globals().legacy_type_names is True
    finally:
        config.LEGACY_TYPE_NAMES = old


def test_pii_mode_ner_tags_use_canonical_names():
    cfg = TextCleanerConfig(check_ner_process=False, ner_mode="pii")
    assert cfg.gliner_label_map["phone_number"] == "PHONE"
    assert cfg.gliner_label_map["ip_address"] == "IP"
    assert {"PHONE", "IP"} <= set(cfg.positional_tags)

    legacy = TextCleanerConfig(check_ner_process=False, ner_mode="pii", legacy_type_names=True)
    assert "phone_number" not in legacy.gliner_label_map
    assert {"PHONE_NUMBER", "IP_ADDRESS"} <= set(legacy.positional_tags)


def test_pii_mode_positional_tags_follow_custom_label_map():
    # Old code expanded positional_tags from the global map, so a custom target was filtered out.
    cfg = TextCleanerConfig(
        check_ner_process=False, ner_mode="pii", gliner_label_map={"phone_number": "TEL"},
    )
    assert "TEL" in cfg.positional_tags


def test_synthetic_mode_fakes_phone_and_ip_by_type():
    pytest.importorskip("faker")
    out = _cleaner(replacement_mode="synthetic").process("from 8.8.8.8").lm_text
    # A name-like fallback would have letters; ipv4 fakes are dotted digits.
    assert re.fullmatch(r"from \d+\.\d+\.\d+\.\d+", out), out
