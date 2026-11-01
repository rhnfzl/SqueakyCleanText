"""Codex 0.8.0 full review, round 37: id 1."""

import time

import pytest

from sct import TextCleaner, TextCleanerConfig
from sct.streaming import WindowedStreamingRedactor

# --- 1: HTML entities decode on their own, so a stream of markup splits in a few normalise calls ---


def _counting_cleaner(**overrides):
    cleaner = TextCleaner(cfg=TextCleanerConfig.from_preset("training", check_ner_process=False, **overrides))
    calls, normalise = [0], cleaner.normalise

    def counted(text):
        calls[0] += 1
        return normalise(text)

    cleaner.normalise = counted
    return cleaner, calls


@pytest.mark.parametrize("repeats", [400, 800])
def test_stream_of_markup_with_entities_splits(repeats):
    cleaner, calls = _counting_cleaner()
    text = "intro <b> " + "word &amp; " * repeats + "x" * 252
    stream = WindowedStreamingRedactor(cleaner)
    started = time.perf_counter()
    first = stream.feed(text)
    assert time.perf_counter() - started < 1.0
    assert calls[0] < 20  # was 1605 / 3205 (one prefix and suffix normalise per rejected space)
    assert first.startswith("intro <b> word & word")
    assert first + stream.feed(" ") + stream.finish() == cleaner.process(text + " ").lm_text


def test_entity_encoded_email_in_markup_is_replaced():
    text = "<p>Mail john&#64;example.com now</p>"
    out = TextCleaner(cfg=TextCleanerConfig(check_ner_process=False)).process(text).lm_text
    assert "john" not in out and "<EMAIL>" in out
