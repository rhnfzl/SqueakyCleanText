"""Codex 0.8.0 full review, round 29: ids 1 and 2."""

from dataclasses import replace

import pytest

from sct import TextCleaner, TextCleanerConfig
from sct.presets import preset_config
from sct.streaming import WindowedStreamingRedactor

FAST = preset_config("training", check_ner_process=False)
DEFAULT = TextCleanerConfig(check_ner_process=False)


# --- 1: the streaming split guard composes a combining mark with its letter, as the cleaner's NFC ---

@pytest.mark.parametrize("chunk", [64, 1024, 4096, None])
def test_stream_keeps_decomposed_month_date_whole(chunk):
    cleaner = TextCleaner(cfg=preset_config("training", check_ner_process=False, check_replace_dates=True))
    text = "Introduction. " * 300 + "Born 15 février 1990. " + "Afterwards. " * 21
    stream = WindowedStreamingRedactor(cleaner)
    size = chunk or len(text)
    out = "".join(stream.feed(text[i:i + size]) for i in range(0, len(text), size)) + stream.finish()
    assert "Born <DATE>." in cleaner.process(text).lm_text
    assert "Born <DATE>." in out and "1990" not in out


# --- 2: a run of closing brackets followed by more URL text stays in the URL ---

@pytest.mark.parametrize("cfg", [FAST, DEFAULT], ids=["training", "default"])
@pytest.mark.parametrize("mode", ["placeholder", "reversible", "synthetic"])
@pytest.mark.parametrize("text, expected", [
    ("Open https://portal.example.com/(S(lit3py55t21z5v55vlm25s55))/orderform.aspx?email=alice%40example.com now.",
     "Open <URL> now."),
    ("See https://x.example.com/a/((b))/c?email=alice%40b.com now.", "See <URL> now."),
    ("Go https://x.example.com/?q=f(a(b))&email=alice%40b.com now.", "Go <URL> now."),
    ("Read (https://x.example.com/a(b)) now.", "Read (<URL>)) now."),
])
def test_closing_bracket_run_stays_in_url(cfg, mode, text, expected):
    cleaner = TextCleaner(cfg=replace(cfg, replacement_mode=mode))
    out = cleaner.process(text).lm_text
    assert "alice" not in out and "lit3py" not in out
    if cfg is FAST and mode == "placeholder":
        assert out == expected
