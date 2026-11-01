"""Codex 0.8.0 full review, round 22: ids 1-2."""

import pytest

from sct import TextCleaner, TextCleanerConfig
from sct.streaming import WindowedStreamingRedactor

MODES = ["placeholder", "synthetic", "reversible"]


def _clean(text, mode="placeholder", **kw):
    cfg = TextCleanerConfig.from_preset("training", check_ner_process=False, replacement_mode=mode, **kw)
    return TextCleaner(cfg).process(text).lm_text


# --- 1: an IPv4 host followed by a Unicode label is a host name, the whole URL is replaced ---

URLS = ["http://10.0.0.1.école.fr/a?token=SecretABC", "http://10.0.0.1.例子.中国/a?token=SecretABC",
        "http://10.0.0.1.𝒆𝒄𝒐𝒍𝒆.fr/a?token=SecretABC"]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("url", URLS)
def test_ipv4_prefixed_unicode_host_url_replaced(url, mode):
    out = _clean(f"see {url} x", mode)
    assert "SecretABC" not in out and "token" not in out


def test_ipv4_url_before_sentence_end_kept():
    assert _clean("see http://10.0.0.1. Next") == "see <URL>. Next"


# --- 2: a window split never lands inside a configured date ---

DATES = ["15 January 1990", "January 15, 1990", "15 de enero de 1990"]


@pytest.mark.parametrize("date", DATES)
def test_window_split_keeps_date_whole(date):
    cleaner = TextCleaner(TextCleanerConfig.from_preset(
        "training", check_ner_process=False, check_replace_dates=True,
        check_replace_numbers=False, check_replace_years=False))
    for pad in range(0, 30):
        stream = WindowedStreamingRedactor(cleaner, max_buffer_chars=40, overlap_chars=16)
        out = stream.feed("x " * pad + f"The patient was born on {date}. Call tomorrow.") + stream.finish()
        assert "1990" not in out and "<DATE>" in out, (pad, out)


@pytest.mark.parametrize("mode", MODES)
def test_ipv4_prefixed_native_digit_label_url_replaced(mode):
    out = _clean("see https://119.14.229.1௨3.pvbzn8iy.de/mu2g?q=SecretABC x", mode)
    assert "SecretABC" not in out
