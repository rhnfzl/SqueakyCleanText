"""Dates glued to unspaced scripts, CJK/Korean unit dates, Han-numeral dates and styled month-name dates
(fuzz date shapes): redacted in every mode and in the stream, with fold_styled_characters as the switch for styled forms."""


import pytest

from sct import TextCleaner
from sct.presets import preset_config
from sct.streaming import WindowedStreamingRedactor
from sct.utils import constants
from tests.helpers import assert_near_linear

GLUED_AND_UNITS = [
    ("生日是", "15 January 1990", "谢谢"),
    ("เกิด", "15 January 1990", "ครับ"),
    ("生日是", "1990年1月15日", "谢谢"),
    ("生日是", "１９９０年１月１５日", "谢谢"),
    ("生日是", "1990/1/15", "谢谢"),
    ("생일은 ", "1990년 1월 15일", " 입니다"),
    ("생일 ", "1990. 1. 15.", " 입니다"),
    ("生日是", "一九九〇年一月十五日", "谢谢"),
    ("生日是", "二〇〇一年十二月三十一日", "谢谢"),
    ("Born ", "26 décembre 1990", " here."),
    ("في", "15 January 1990", " هنا"),
]
STYLED = [("Born ", "⑮ January 1990", " here."), ("Born ", "January ⑮, 1990", " here."),
          ("Born ", "26 ⓓéⓒⓔⓜⓑⓡⓔ 1990", " here."), ("생일 ", "1990﹒ 1﹒ 15﹒", " 입니다"),
          ("생일은 ", "1990년 6월⁠ 1일", " 입니다")]


def _cleaner(mode="placeholder", **overrides):
    return TextCleaner(cfg=preset_config("training", check_ner_process=False, check_replace_dates=True,
                                         replacement_mode=mode, **overrides))


def _stream(cleaner, text):
    stream = WindowedStreamingRedactor(cleaner, max_buffer_chars=48, overlap_chars=24)
    return "".join(stream.feed(ch) for ch in text) + stream.finish()


@pytest.mark.parametrize("before, date, after", GLUED_AND_UNITS + STYLED)
def test_date_is_redacted_in_every_path(before, date, after):
    text = "Notes for the file. " * 3 + before + date + after
    for mode in ("placeholder", "reversible", "synthetic"):
        out = _cleaner(mode).process(text).lm_text
        assert date not in out and out.startswith("Notes") and out.endswith(after), (mode, out)
    cleaner = _cleaner()
    assert _stream(cleaner, text) == cleaner.process(text).lm_text == text.replace(date, "<DATE>")


@pytest.mark.parametrize("before, date, after", STYLED[:3])
def test_styled_dates_follow_fold_styled_characters(before, date, after):
    text = before + date + after
    assert _cleaner(fold_styled_characters=False).process(text).lm_text == text


@pytest.mark.parametrize("text", [
    "abc15 January 1990", "15 January 1990abc", "Version 2024.1.5 out", "In 1990. 1. Item", "1990年1月 only",
    "1月15日 only", "三十年一月", "令和6年1月15日", "ⓐ list", "① 2024年",
])
def test_not_a_full_date(text):
    out = _cleaner().process(text).lm_text
    assert "<DATE>" not in out


def test_buddhist_era_years_and_custom_thai_months():
    thai = {"THAI": ("มกราคม", "กุมภาพันธ์")}
    cleaner = _cleaner(custom_month_names=thai)
    for text in ("เกิด 15 มกราคม 2533 ครับ", "เกิด15 มกราคม 2533ครับ", "เกิด 15/01/2533 ครับ"):
        assert "2533" not in cleaner.process(text).lm_text


@pytest.mark.parametrize("unit, n", [
    ("1990年1月", 5000), ("1990年 1 月 1 ", 5000), ("1990. 1. ", 5000), ("一九九〇年十", 5000), ("〇", 20000),
    ("⑮ January ", 5000), ("1990년 1월 ", 5000), ("January 1 ", 5000), ("ⓐé", 5000), ("월⁠ ", 5000),
])
def test_new_date_patterns_are_linear(unit, n):
    regex, cleaner = constants.build_date_regex(styled=True), _cleaner()

    def scan(text):
        list(regex.finditer(text))
        list(constants.FUZZY_DATE_CANDIDATE_REGEX.finditer(text))
        cleaner.process(text)

    assert_near_linear(scan, lambda size: unit * size, n)
