"""Codex 0.8.0 full review, round 28: id 1."""

import pytest

from sct import TextCleaner, TextCleanerConfig
from sct.presets import preset_config

FAST = preset_config("training", check_ner_process=False)
DEFAULT = TextCleanerConfig(check_ner_process=False)


# --- 1: a ) ] } before a file extension stays in the URL, so the query after it is masked ---

@pytest.mark.parametrize("cfg", [FAST, DEFAULT], ids=["training", "default"])
@pytest.mark.parametrize("text, expected", [
    ("Download https://files.example.com/report(1).pdf?email=alice%40example.com now.",
     "Download <URL> now."),
    ("https://en.wikipedia.org/wiki/Foo_(bar).html?token=abc123 x", "<URL> x"),
    ("https://x.com/?f=[a].b&email=bob%40x.org x", "<URL> x"),
    ("see (https://x.com/a). ok", "see (<URL>). ok"),
    ("Link: https://x.com/x_(y).", "Link: <URL>)."),
])
def test_closing_bracket_before_extension_stays_in_url(cfg, text, expected):
    cleaner = TextCleaner(cfg=cfg)
    out = cleaner.process(text).lm_text
    assert "alice" not in out and "bob" not in out and "abc123" not in out
    if cfg is FAST:
        assert out == expected
