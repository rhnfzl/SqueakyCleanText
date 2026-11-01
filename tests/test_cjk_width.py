"""ftfy's width fix no longer mixes CJK punctuation; full-width PII is still replaced."""

import pytest

from sct import TextCleaner, TextCleanerConfig
from sct.utils.normtext import NormaliseText

ZH = "你好，世界！我的电话是０６１２３４５６７８，邮箱ｊｏｈｎ＿ｄｏｅ＠ｅｘａｍｐｌｅ．ｃｏｍ。"


def _cleaner(**overrides):
    return TextCleaner(cfg=TextCleanerConfig(
        check_ner_process=False, check_detect_language=False,
        check_statistical_model_processing=False, check_to_ascii_unicode=False, **overrides))


def test_default_keeps_cjk_punctuation_and_katakana_width():
    norm = NormaliseText()
    out = norm.fix_bad_unicode("你好，世界。ｶﾀｶﾅ：（括弧）！")
    assert out == "你好，世界。ｶﾀｶﾅ：（括弧）！"
    assert norm.fix_bad_unicode(ZH) == (
        "你好，世界！我的电话是0612345678，邮箱john_doe@example.com。")


def test_default_converts_fullwidth_phone_joiners():
    norm = NormaliseText()
    assert norm.fix_bad_unicode("电话：＋３１　６－１２３４　５６７８。") == "电话：+31 6-1234 5678。"


def test_fix_character_width_true_is_ftfy_full_fix():
    out = NormaliseText().fix_bad_unicode("你好，世界。ｶﾀｶﾅ！", fix_character_width=True)
    assert out == "你好,世界。カタカナ!"


@pytest.mark.parametrize("legacy", [False, True])
def test_fullwidth_phone_and_email_replaced_either_way(legacy):
    out = _cleaner(fix_character_width=legacy).process(ZH).lm_text
    assert "<PHONE>" in out and "<EMAIL>" in out
    assert not any(ch in out for ch in "０６１２３４５６７８ｊｏｈｅｘ@")
    assert ("，" in out) is not legacy


def test_legacy_module_global():
    from sct import config
    old = config.FIX_CHARACTER_WIDTH
    try:
        config.FIX_CHARACTER_WIDTH = True
        assert config._config_from_module_globals().fix_character_width is True
    finally:
        config.FIX_CHARACTER_WIDTH = old
