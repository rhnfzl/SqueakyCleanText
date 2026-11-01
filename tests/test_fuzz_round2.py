"""Fuzz round 2 (research/0.8.0-fuzz-round2.md): one regression per fixed root cause."""

import pytest

from sct import TextCleaner
from sct.presets import preset_config
from sct.utils.normtext import fold_styled


def _clean(text, **overrides):
    return TextCleaner(cfg=preset_config("training", check_ner_process=False, **overrides)).process(text).lm_text


@pytest.mark.parametrize("text, secret", [
    # a lone circled number after two or more digit groups is the phone's last group
    ("call 093 6㉒ ⑰ ㉝ now", "622"),
    ("call 06 73 86 88 ㊺ now", "738688"),
    ("call ၀၆ ၅၂ ၅၆ ၀၆ ㊶ now", "၅၂"),
    ("call ​૦⑤② ㊲⑧ ㉞ ⑯​ now", "378"),
])
def test_lone_circled_number_after_digit_groups_folds(text, secret):
    assert secret not in _clean(text).replace(" ", "")


@pytest.mark.parametrize("text", ["① 2024年", "見て 2024 ①", "item 12 ③"])
def test_lone_circled_list_marker_beside_one_number_stays(text):
    assert fold_styled(text) == text


def test_trailing_circled_number_stays_with_fold_off():
    assert "㊺" in _clean("call 06 73 86 88 ㊺ now", fold_styled_characters=False)
