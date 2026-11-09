"""Styled/compatibility letters and digits (𝐞 ⓙ ｈ) no longer hide PII when to_ascii is off."""

import pytest

from sct import TextCleaner, TextCleanerConfig
from sct.presets import preset_config
from sct.utils.normtext import NormaliseText

# (text, PII value as written, its ASCII form)
CASES = [
    ("mail john@𝐞𝐱𝐚𝐦𝐩𝐥𝐞.com now", "john@𝐞𝐱𝐚𝐦𝐩𝐥𝐞.com", "john@example.com"),
    ("mail ⓙⓞⓗⓝ@example.com now", "ⓙⓞⓗⓝ@example.com", "john@example.com"),
    ("see ｈｔｔｐｓ：／／ｅｘａｍｐｌｅ．ｃｏｍ／ａ now", "ｅｘａｍｐｌｅ", "example.com"),
    ("see https://𝐞𝐱𝐚𝐦𝐩𝐥𝐞.com/a now", "𝐞𝐱𝐚𝐦𝐩𝐥𝐞", "example.com"),
    ("call 𝟎𝟔 𝟏𝟐𝟑𝟒 𝟓𝟔𝟕𝟖 now", "𝟏𝟐𝟑𝟒", "1234 5678"),
    ("call ⓪⑥①②③④⑤⑥⑦⑧ now", "⓪⑥①②", "0612345678"),
    ("host 𝟏𝟗𝟐.𝟏𝟔𝟖.𝟎.𝟏 now", "𝟏𝟗𝟐", "192.168.0.1"),
    ("on ２０２４／０１／１５ now", "２０２４", "2024/01/15"),
]


def _cfg(mode):
    if mode == "training":
        return preset_config("training", check_ner_process=False)
    return TextCleanerConfig(check_ner_process=False, check_to_ascii_unicode=False,
                             replacement_mode=mode)


@pytest.mark.parametrize("mode", ["placeholder", "synthetic", "reversible", "training"])
@pytest.mark.parametrize("text,styled,ascii_", CASES)
def test_styled_pii_replaced(mode, text, styled, ascii_):
    if mode == "training" and "２０２４" in text:
        pytest.skip("training preset keeps dates")
    out = TextCleaner(cfg=_cfg(mode)).process(text).lm_text
    assert styled not in out and ascii_ not in out, out


def test_off_keeps_earlier_behaviour():
    # Folding off keeps the styled letters; the email itself is still found, since
    # host names take letters beyond the BMP.
    assert "𝐞𝐱𝐚𝐦𝐩𝐥𝐞" in NormaliseText().fix_bad_unicode(CASES[0][0], fold_styled_characters=False)


def test_prose_keeps_its_characters():
    norm = NormaliseText()
    for text in ("① 項目を確認。② 次へ", "面積 25 m²、H₂O、nº 5", "価格：（税込）！", "重さ５㎏"):
        assert norm.fix_bad_unicode(text) == text.replace("５", "5")
