"""Phone numbers and IPs in native digits, circled digits, styled brackets and typeset separators."""

import pickle
import re

import pytest

from sct import TextCleaner, TextCleanerConfig
from sct.adapters import process_json
from sct.presets import preset_config
from sct.streaming import WindowedStreamingRedactor
from sct.utils.normtext import fold_styled


FAST = preset_config("training", check_ner_process=False)


# --- grouped national phone numbers are whole matches without number replacement ---
@pytest.mark.parametrize("text", [
    "Call 699 123 456 tomorrow.", "Call 912 345 678 tomorrow.", "Call 010-6552 1234 tomorrow.",
    "Ring 020 7946 0958 now.", "Call 06 12 34 56 78 tomorrow.",
])
def test_training_masks_grouped_national_phones(text):
    out = TextCleaner(cfg=FAST).process(text).lm_text
    assert "<PHONE>" in out and not re.search(r"\d", out), out


@pytest.mark.parametrize("text", [
    "On 01-02-2023 we met.", "In 2019 we had 42 bugs.", "Order 123 456 shipped.", "Paid 1 234 567 890 won.",
])
def test_training_leaves_short_digit_groups(text):
    assert "<PHONE>" not in TextCleaner(cfg=FAST).process(text).lm_text


MODES = ["placeholder", "synthetic", "reversible"]


def _clean_training(text, mode="placeholder", **kw):
    cfg = TextCleanerConfig.from_preset("training", check_ner_process=False, replacement_mode=mode, **kw)
    return TextCleaner(cfg).process(text).lm_text


# --- 3 / 4: native-digit phones and IPs ---
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("text", ["٠٦ ١٢ ٣٤ ٥٦ ٧٨", "٩٨٧٦٥ ٤٣٢١٠", "٠٩٨٧٦٥-٤٣٢١٠", "٠١٠-٦٥٥٢ ١٢٣٤",
                                  "०६ १२ ३४ ५६ ७८", "١٩٢．١٦٨．١．١", "1.2.3.4١"])
def test_native_digit_phone_and_ip_fully_replaced(text, mode):
    out = _clean_training(f"رقم {text} هنا", mode)
    assert not any(c.isdigit() and not c.isascii() for c in out) and "1.2.3.4" not in out, out


def test_native_digit_ip_public_only_check():
    out = _clean_training("ip ١٩٢．١٦٨．١．١ and ٨．٨．٨．٨", "placeholder", replace_ips_public_only=True)
    # ． between digits of any script folds like between ASCII digits; the private IP stays
    assert out == "ip ١٩٢.١٦٨.١.١ and <IP>"


def test_circled_number_between_typeset_dashes_folds():
    out = _clean_training("Born 2026‒⑩–02 ok", check_replace_dates=True)
    assert out == "Born <DATE> ok"


def _cleaner(mode="placeholder", **kw):
    return TextCleaner(TextCleanerConfig.from_preset("training", check_ner_process=False, replacement_mode=mode, **kw))


def _stream_safe(cleaner, text, secret):
    for pad in range(0, 30):
        stream = WindowedStreamingRedactor(cleaner, max_buffer_chars=80, overlap_chars=16)  # holds the URL
        out = stream.feed("x " * pad + text) + stream.finish()
        assert secret not in out, (pad, out)


# --- full-width and small area-code parentheses read as "(" ")" in the phone view ---
PHONES = ["Call +49（30）123456 today.", "Call +49﹙30﹚123456 today.", "Call +80（⑫）80847098 today.",
          "Call +47​(20)1123556 today."]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("text", PHONES)
def test_styled_paren_phone_masked(text, mode):
    out = _cleaner(mode).process(text).lm_text
    assert "123456" not in out and "8084" not in out and "1123556" not in out, out
    assert out.startswith("Call ") and out.endswith(" today."), out


def test_styled_paren_phone_placeholder_json_stream_pickle():
    cleaner = _cleaner()
    assert cleaner.process(PHONES[0]).lm_text == "Call <PHONE> today."
    assert "123456" not in str(process_json(cleaner, {"note": PHONES[0]}).value)
    cfg = pickle.loads(pickle.dumps(cleaner.cfg))  # noqa: S301  # our own config, round-tripped
    assert TextCleaner(cfg).process(PHONES[0]).lm_text == "Call <PHONE> today."
    _stream_safe(cleaner, PHONES[0], "123456")


def test_prose_bracket_glued_after_phone():
    # a CJK prose bracket right after the number does not cut it; prose keeps its full-width brackets
    out = _cleaner().process("電話+49 30 21513405（自宅）です").lm_text
    assert out == "電話<PHONE>（自宅）です", out
    assert _cleaner().process("中文（括号）①").lm_text == "中文（括号）①"


@pytest.mark.parametrize("text, secret", [
    ("host ⑩⑤．⑨．②3⑥．③6 now", "236"),  # circled beside full-width punctuation
    ("host ⑭．⑩꤈．㉑꤆．꤆꤁ now", "꤆꤁"),  # circled beside a native digit; ． between native digits
    ("call 06 ⑬ 01 68 36 now", "68 36"),  # lone circled number between digit groups
])
def test_circled_digits_inside_numbers_fold(text, secret):
    assert secret not in _clean_training(text)


def test_plus_forms_before_native_digits():
    assert "᥋᥌" not in _clean_training("call ﹢᥏᥇ ᥆᥋᥌8᥏ 903᥋6 now")


def test_trunk_phone_three_three_two_two():
    assert "566" not in _clean_training("bel 018 566 00 17 nu")


@pytest.mark.parametrize("text, secret", [
    # a lone circled number after two or more digit groups is the phone's last group
    ("call 093 6㉒ ⑰ ㉝ now", "622"),
    ("call 06 73 86 88 ㊺ now", "738688"),
    ("call ၀၆ ၅၂ ၅၆ ၀၆ ㊶ now", "၅၂"),
    ("call ​૦⑤② ㊲⑧ ㉞ ⑯​ now", "378"),
])
def test_lone_circled_number_after_digit_groups_folds(text, secret):
    assert secret not in _clean_training(text).replace(" ", "")


@pytest.mark.parametrize("text", ["① 2024年", "見て 2024 ①", "item 12 ③"])
def test_lone_circled_list_marker_beside_one_number_stays(text):
    assert fold_styled(text) == text


def test_trailing_circled_number_stays_with_fold_off():
    assert "㊺" in _clean_training("call 06 73 86 88 ㊺ now", fold_styled_characters=False)


def _preset_cleaner(preset="training", **overrides):
    return TextCleaner(cfg=preset_config(preset, check_ner_process=False, **overrides))


# Phone groups set off by another space (Zs) or dash (Pd), as word processors and PDFs write them,
# were not one phone: the phone view only read the ASCII and full-width forms.
@pytest.mark.parametrize("text", [
    "call 06\u00a012\u00a034\u00a056\u00a078 now",  # no-break space
    "call 06\u202f12\u202f34\u202f56\u202f78 now",  # narrow no-break space
    "call +31\u20096\u20091234\u20095678 now",  # thin space
    "call 020\u20077946\u20070958 now",  # figure space
    "call 06\u20111234\u20115678 now",  # non-breaking hyphen
    "call (061)\u00a0123\u20134567 now",  # en dash
])
def test_typeset_separators_inside_phone(text):
    assert _preset_cleaner().process(text).lm_text == "call <PHONE> now"


def test_typeset_separators_stay_with_fold_off():
    text = "call 06\u00a012\u00a034\u00a056\u00a078 now"
    assert _preset_cleaner(fold_styled_characters=False).process(text).lm_text == text


# A lone circled number between groups set off by a typeset space, and an invisible hint before a
# typeset dash, did not fold as they do with an ASCII space or hyphen.
@pytest.mark.parametrize("text, folded", [
    ("06\u202f98\u202f\u3256\u202f03\u202f84", "06\u202f98\u202f26\u202f03\u202f84"),
    ("06\u00a098\u00a003\u00a084\u00a0\u3256", "06\u00a098\u00a003\u00a084\u00a026"),
    ("131\u2060\u20107237", "131\u20107237"),
])
def test_fold_reads_typeset_separators(text, folded):
    assert fold_styled(text) == folded
