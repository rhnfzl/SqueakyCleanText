"""K4: the training preset, the escape-decoding flag and the threshold sentinels."""

import pickle
from pathlib import Path

import pytest

from sct import TextCleaner, TextCleanerConfig, config, preset_config
from sct.presets import PRESETS
from tests.conftest import TEST_NER_MODELS

GOLDEN = Path(__file__).parent / "golden" / "training"
CASES = ("prose_en", "prose_de_umlaut", "code_python", "markup_html_md",
         "indented_yaml", "non_latin_hi_zh_ar")


@pytest.fixture(scope="module")
def cleaner():
    return TextCleaner(cfg=preset_config(check_ner_process=False))


@pytest.mark.parametrize("case", CASES)
def test_training_golden(cleaner, case):
    result = cleaner.process((GOLDEN / f"{case}.txt").read_text(encoding="utf-8"))
    assert result.lm_text == (GOLDEN / f"{case}.expected.txt").read_text(encoding="utf-8")
    assert result.stat_text is None


def test_training_keeps_text_shape_and_replaces_pii(cleaner):
    out = {c: cleaner.process((GOLDEN / f"{c}.txt").read_text(encoding="utf-8")).lm_text
           for c in CASES}
    assert "München" in out["prose_de_umlaut"] and "Straße" in out["prose_de_umlaut"]
    assert '"Caf\\u00e9 says:\\n"' in out["code_python"]
    assert "\n    if len(name) > 3:\n" in out["code_python"]
    assert "\n\n\nADMIN_EMAIL" in out["code_python"]
    assert "  tags: [alpha, beta]\n\nretries: 3" in out["indented_yaml"]
    assert "[1]" in out["prose_en"] and "{see notes}" in out["prose_en"]
    assert "In 2019" in out["prose_en"] and "42 bugs" in out["prose_en"]
    assert '<a href="<URL>">docs</a>' in out["markup_html_md"]
    for token in ("<EMAIL>", "<PHONE>", "<URL>", "<IP>"):
        assert token in out["prose_en"]
    assert out["non_latin_hi_zh_ar"] == (GOLDEN / "non_latin_hi_zh_ar.txt").read_text(encoding="utf-8")


def test_training_preset_ner_replaces_person():
    cleaner = TextCleaner(cfg=preset_config(ner_models=TEST_NER_MODELS, language="ENGLISH"))
    result = cleaner.process("Yesterday Barack Obama visited Paris.\n  Indented line.")
    assert "Obama" not in result.lm_text and "<PERSON>" in result.lm_text
    assert result.lm_text.endswith("\n  Indented line.")


def test_preset_overrides_and_unknown_name():
    assert preset_config("training", check_replace_numbers=True).check_replace_numbers
    assert TextCleanerConfig.from_preset(check_casefold=True).check_casefold
    assert not TextCleanerConfig.from_preset().check_casefold
    assert preset_config("sct") == TextCleanerConfig()
    assert PRESETS["training"]["check_decode_unicode_escapes"] is False
    with pytest.raises(ValueError, match="training"):
        preset_config("nope")


def test_decode_unicode_escapes_flag():
    text = "Caf\\u00e9\\n ok"
    off = TextCleaner(cfg=preset_config("sct", check_ner_process=False,
                                        check_decode_unicode_escapes=False))
    on = TextCleaner(cfg=preset_config("sct", check_ner_process=False))
    assert "\\u00e9" in off.process(text).lm_text
    assert "\\u00e9" not in on.process(text).lm_text


def test_decode_unicode_escapes_legacy_global(monkeypatch):
    assert config._config_from_module_globals().check_decode_unicode_escapes
    monkeypatch.setattr(config, "CHECK_DECODE_UNICODE_ESCAPES", False)
    assert not config._config_from_module_globals().check_decode_unicode_escapes


def test_threshold_sentinels():
    # Old code rewrote an explicit 0.85 / 0.4 because they equalled the defaults.
    assert TextCleanerConfig(
        ner_mode="pii", ner_confidence_threshold=0.85).effective_ner_confidence_threshold == 0.85
    pii = TextCleanerConfig(ner_mode="pii")
    assert pii.ner_confidence_threshold is None and pii.gliner_threshold is None
    assert pii.effective_ner_confidence_threshold == pii.effective_gliner_threshold == 0.3
    assert TextCleanerConfig(ner_mode="pii", gliner_threshold=0.4).effective_gliner_threshold == 0.4
    assert TextCleanerConfig(
        ner_mode="pii", gliner_threshold=0.4).effective_ner_confidence_threshold == 0.4
    std = TextCleanerConfig(check_ner_process=False)
    assert (std.effective_ner_confidence_threshold, std.effective_gliner_threshold) == (0.85, 0.4)
    clone = pickle.loads(pickle.dumps(pii))  # noqa: S301 - trusted round trip
    assert clone.ner_confidence_threshold is None and clone.gliner_threshold is None
    assert config._config_from_module_globals().ner_confidence_threshold is None
    for bad in ({"ner_confidence_threshold": 1.5}, {"gliner_threshold": -0.1}):
        with pytest.raises(ValueError, match="in \\[0, 1\\]"):
            TextCleanerConfig(check_ner_process=False, **bad)
