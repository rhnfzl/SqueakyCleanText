"""Codex 0.8.0 full review, round 25: ids 1, 2."""

import pickle

import pytest

from sct import TextCleaner, TextCleanerConfig
from sct.adapters import process_json
from sct.streaming import WindowedStreamingRedactor

MODES = ["placeholder", "synthetic", "reversible"]


def _cleaner(mode="placeholder", **kw):
    return TextCleaner(TextCleanerConfig.from_preset("training", check_ner_process=False, replacement_mode=mode, **kw))


def _stream_safe(cleaner, text, secret):
    for pad in range(0, 30):
        stream = WindowedStreamingRedactor(cleaner, max_buffer_chars=80, overlap_chars=16)  # holds the URL
        out = stream.feed("x " * pad + text) + stream.finish()
        assert secret not in out, (pad, out)


# --- 1: full-width and small area-code parentheses read as "(" ")" in the phone view ---

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


# --- 2: a full-width @ after punctuation in URL user info folds with the authority ---

URLS = ["https://alice:pass!＠example.com/?token=SecretABC", "see https://alice:p4ss!＠example.com/x?token=SecretABC now",
        "see https://alice:p4ss！＠example.com/x?token=SecretABC now"]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("text", URLS)
def test_userinfo_styled_at_after_punct_masked(text, mode):
    out = _cleaner(mode).process(text).lm_text
    assert "SecretABC" not in out and "pass" not in out and "p4ss" not in out, out


def test_userinfo_styled_at_json_and_stream():
    cleaner = _cleaner()
    assert cleaner.process(URLS[1]).lm_text == "see <URL> now"
    assert "SecretABC" not in str(process_json(cleaner, {"note": URLS[0]}).value)
    _stream_safe(cleaner, URLS[1], "SecretABC")
