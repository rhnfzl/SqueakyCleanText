"""Codex 0.8.0 full review, round 24: ids 1, 2."""

import re

import pytest

from sct import TextCleaner, TextCleanerConfig
from sct.adapters import process_json
from sct.streaming import WindowedStreamingRedactor
from tests.conftest import TEST_NER_MODELS

MODES = ["placeholder", "synthetic", "reversible"]


def _cleaner(mode="placeholder", **kw):
    return TextCleaner(TextCleanerConfig.from_preset("training", check_ner_process=False, replacement_mode=mode, **kw))


# --- 1: a layout hint (ZWSP, word joiner) inside a Unicode URL host drops, as in a mail domain ---

HINTED = ["See https://例​え.jp/?token=SecretABC now", "See https://例⁠え.jp/?token=SecretABC now",
          "See www.例​え.jp/?token=SecretABC now"]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("text", HINTED)
def test_hinted_unicode_host_url_masked(text, mode):
    out = _cleaner(mode).process(text).lm_text
    assert "SecretABC" not in out and "例" not in out, out


def test_hinted_unicode_host_json_and_stream():
    cleaner = _cleaner()
    assert cleaner.process(HINTED[0]).lm_text == "See <URL> now"
    assert "SecretABC" not in str(process_json(cleaner, {"note": HINTED[0]}).value)
    for pad in range(0, 30):
        stream = WindowedStreamingRedactor(cleaner, max_buffer_chars=40, overlap_chars=16)
        out = stream.feed("x " * pad + HINTED[0]) + stream.finish()
        assert "SecretABC" not in out, (pad, out)


def test_join_controls_stay_in_url_host():
    # ZWNJ spells the word (Persian), so it is part of the host
    text = "https://می‌خواهم.ir/?t=SecretABC"
    assert "SecretABC" not in _cleaner().process(text).lm_text


# --- 2: a custom marker that is also source text keeps protection for its emitted occurrences ---

SOURCE = "The EMAIL field is mandatory. John Smith at ann@example.com and Jane Doe at bob@example.org, in London."


@pytest.fixture(scope="module")
def marker_cleaners():
    return {mode: TextCleaner(TextCleanerConfig.from_preset(
        "training", language="en", ner_models=TEST_NER_MODELS, replacement_mode=mode, replace_with_email="EMAIL"))
        for mode in MODES}


def test_literal_custom_marker_reversible_round_trip(marker_cleaners):
    cleaner = marker_cleaners["reversible"]
    for result in [cleaner.process(SOURCE), *cleaner.process_batch([SOURCE, SOURCE])]:
        assert "MA<" not in result.lm_text, result.lm_text
        assert result.metadata["anon_map"].deanonymize(result.lm_text) == SOURCE


@pytest.mark.parametrize("mode", ["placeholder", "synthetic"])
def test_literal_custom_marker_emitted_marker_not_split(marker_cleaners, mode):
    out = marker_cleaners[mode].process(SOURCE).lm_text
    assert not re.search(r"[a-z>]MA[A-Z<]", out), out
    assert "ann@example.com" not in out and "bob@example.org" not in out


def test_source_marker_spans_exempt_only_those_occurrences():
    from tests.helpers import FakeRunDetector, word_ner
    detector = FakeRunDetector()
    detector.pattern = re.compile(r"[A-Z]{2,}")
    ner = word_ner(0, detector)
    text = "<LONDON> and <LONDON>"
    result = ner.ner_process_detailed(
        text, positional_tags=("PER",), ner_confidence_threshold=0.5, source_markers=frozenset({(0, 8)}))
    assert result.text == "<<PERSON>> and <LONDON>"
