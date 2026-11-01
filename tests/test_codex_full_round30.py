"""Codex 0.8.0 full review, round 30: ids 1 and 2."""

import pytest

from sct import TextCleaner
from sct.presets import preset_config
from sct.streaming import WindowedStreamingRedactor

B = chr(92)  # a literal backslash, as JSON-escaped text carries it


def _stream(cleaner, text, chunk):
    stream = WindowedStreamingRedactor(cleaner)
    size = chunk or len(text)
    return "".join(stream.feed(text[i:i + size]) for i in range(0, len(text), size)) + stream.finish()


# --- 1: the streaming split guard decodes backslash escapes, as the cleaner's escape step ---

@pytest.mark.parametrize("chunk", [1024, 4096, None])
@pytest.mark.parametrize("value, n, token", [
    ("06 12" + B + "u0033" + B + "u0034 56 78", 19, "<PHONE>"),
    ("06 12" + B + "u200b 34 56 78", 20, "<PHONE>"),
    ("06 12" + B + "u00a034 56 78", 20, "<PHONE>"),
    ("06 12 " + B + "x334 56 78", 20, "<PHONE>"),
    ("15 f" + B + "u00e9vrier 1990", 20, "<DATE>"),
])
def test_stream_keeps_escaped_token_whole(chunk, value, n, token):
    cleaner = TextCleaner(cfg=preset_config(
        "training", check_ner_process=False, check_replace_dates=True, check_decode_unicode_escapes=True))
    text = "Introduction. " * 300 + "Value " + value + " here. " + "Afterwards. " * n
    assert f"Value {token} here." in cleaner.process(text).lm_text
    assert f"Value {token} here." in _stream(cleaner, text, chunk)


def test_stream_keeps_escapes_raw_when_decoding_is_off():
    cleaner = TextCleaner(cfg=preset_config("training", check_ner_process=False))
    text = "Introduction. " * 300 + "Path C:" + B + "u0033 " + B + "x41 done. " + "Afterwards. " * 20
    assert _stream(cleaner, text, None) == cleaner.process(text).lm_text


# --- 2: with fuzzy dates on, the guard keeps a misspelled-month date in one window ---

@pytest.mark.parametrize("chunk", [1024, 4096, None])
@pytest.mark.parametrize("value", ["15 Janury 1990", "Janury 15, 1990"])
def test_stream_keeps_fuzzy_date_whole(chunk, value):
    pytest.importorskip("rapidfuzz")
    cleaner = TextCleaner(cfg=preset_config(
        "training", check_ner_process=False, check_replace_dates=True, check_fuzzy_replace_dates=True,
        language="en"))
    text = "Introduction. " * 300 + "Value " + value + " here. " + "Afterwards. " * 20
    assert "Value <DATE> here." in cleaner.process(text).lm_text
    assert "Value <DATE> here." in _stream(cleaner, text, chunk)
