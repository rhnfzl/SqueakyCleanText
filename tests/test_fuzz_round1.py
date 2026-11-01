"""Fuzz round 1 (research/0.8.0-fuzz-round1.md): one regression per fixed root cause."""

import pytest

from sct import TextCleaner
from sct.presets import preset_config
from sct.streaming import WindowedStreamingRedactor


def _clean(text, **overrides):
    return TextCleaner(cfg=preset_config("training", check_ner_process=False, **overrides)).process(text).lm_text


@pytest.mark.parametrize("text, secret", [
    # small-form contact punctuation (﹫ ﹒ ﹕ ﹖ ﹦ ﹣ ﹢) folds like the full-width forms
    ("mail mx1eaq﹫zx9wnzm﹒de now", "mx1eaq"),
    ("see http﹕//悛搖鯰卆哯﹒net/9jmz﹖user﹦zt0mkdjuj8c now", "zt0mkdjuj8c"),
    ("host 214﹒44﹒189﹒208 now", "189"),
    ("call 018﹣2896 ext now", "2896"),
    # font plus (﬩, decomposition <font> +) folds as a phone prefix and email separator
    ("call ﬩91 21774 41978 now", "21774"),
    ("mail षीबनझ﬩iax4n@k6bb.net now", "षीबनझ"),
])
def test_small_and_font_contact_punctuation_folds(text, secret):
    assert secret not in _clean(text)


def test_scheme_with_mixed_width_slashes_folds():
    assert "yuek6" not in _clean("see ｈtｔp：/／8cmca．net／yuek6？email=x now")


def test_full_stop_after_a_word_ending_in_a_mark_folds():
    # ै is a combining mark: the word before ． still ends there
    assert "g0up" not in _clean("see https://ळौछनै．de/g0up?token=w3gyy now")


def test_zero_width_joiner_inside_email_local_part():
    assert "घप" not in _clean("mail प्‍ॄघप@czgc0.org now")


@pytest.mark.parametrize("text, secret", [
    ("host ⑩⑤．⑨．②3⑥．③6 now", "236"),  # circled beside full-width punctuation
    ("host ⑭．⑩꤈．㉑꤆．꤆꤁ now", "꤆꤁"),  # circled beside a native digit; ． between native digits
    ("call 06 ⑬ 01 68 36 now", "68 36"),  # lone circled number between digit groups
])
def test_circled_digits_inside_numbers_fold(text, secret):
    assert secret not in _clean(text)


def test_plus_forms_before_native_digits():
    assert "᥋᥌" not in _clean("call ﹢᥏᥇ ᥆᥋᥌8᥏ 903᥋6 now")


def test_url_glued_after_thai_letters():
    assert "iyfmyxg1fn" not in _clean("ไปที่เว็บhttps://ldp3.example.nl/pl5wj?id=iyfmyxg1fn ครับ")


def test_trunk_phone_three_three_two_two():
    assert "566" not in _clean("bel 018 566 00 17 nu")


def test_url_path_ends_at_cjk_punctuation():
    out = _clean("見 https://example.de/a?q=zz（備考）！06 08 66 90 80 です")
    assert "（備考）" in out and "66 90" not in out


def test_stream_split_matches_whole_document_for_phone_glued_to_cjk():
    cleaner = TextCleaner(cfg=preset_config("training", check_ner_process=False))
    text = "ok 鸯搊緘薦袹 握！+31 6 2702 1527（櫿菲襋咈醩靭胰鐑聈螮頁湪土諵櫿菲襋咈醩靭胰鐑） ok ok"
    stream = WindowedStreamingRedactor(cleaner, max_buffer_chars=48, overlap_chars=12)
    out = "".join(stream.feed(text[i:i + 5]) for i in range(0, len(text), 5)) + stream.finish()
    assert out == cleaner.process(text).lm_text
