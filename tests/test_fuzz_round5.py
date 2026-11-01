"""Fuzz round 5 (research/0.8.0-fuzz-round5.md): one regression per fixed root cause."""

import pytest

from sct import TextCleaner
from sct.presets import preset_config
from sct.streaming import WindowedStreamingRedactor
from sct.utils.normtext import fold_styled


def _cleaner(**overrides):
    return TextCleaner(cfg=preset_config("training", check_ner_process=False, **overrides))


# An invisible line-break hint (Cf: ZWSP, word joiner, soft hyphen) inside a value hid it from every regex.
@pytest.mark.parametrize("text, secret", [
    ("call 0671­427605 now", "0671"),
    ("call 06​ 71 42 76 05 now", "714276"),
    ("host 192.168​.1.1 ok", "192.168"),
    ("mail jan.de​vries@example.com ok", "jan.de"),
    ("mail jan@ex⁠ample.com ok", "jan@"),
    ("mail jan@例え​．jp ok", "jan@"),
    ("see https://exa­mple.com/a?id=12345 ok", "exa"),
])
def test_invisible_hint_inside_value_folds(text, secret):
    assert secret not in _cleaner().process(text).lm_text.replace(" ", "")


@pytest.mark.parametrize("text", [
    "ดู​สวัสดี",  # Thai word break
    "نامه‌ای",  # Persian ZWNJ
    "👨‍👩",  # emoji ZWJ sequence
])
def test_invisible_between_other_letters_stays(text):
    assert fold_styled(text) == text


def test_invisible_hint_stays_with_fold_off():
    assert "­" in _cleaner(fold_styled_characters=False).process("call 0671­427605 now").lm_text


# The stream's split guard did not see "06​ 12 ..." as one phone and cut it between windows.
@pytest.mark.parametrize("text", [
    "see you call 06​ 12 34 56 78 now ok bye",
    "see you call 06 12­ 34 56 78 now ok bye",
])
def test_stream_does_not_cut_phone_at_invisible_hint(text):
    cleaner = _cleaner()
    stream = WindowedStreamingRedactor(cleaner, max_buffer_chars=30, overlap_chars=8)
    out = "".join(stream.feed(ch) for ch in text) + stream.finish()
    assert out == cleaner.process(text).lm_text == "see you call <PHONE> now ok bye"
