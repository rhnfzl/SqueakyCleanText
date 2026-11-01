"""Codex 0.8.0 full review, round 32: ids 1 and 2."""

import random
import sys
import threading

import pytest

from sct import TextCleaner
from sct.presets import preset_config
from sct.streaming import WindowedStreamingRedactor
from sct.utils.synthetic import SyntheticReplacer


# --- 1: the stream guard drops terminal escapes and controls as the cleaner's ftfy step does ---

@pytest.mark.parametrize("date", [
    "\x1b[32m15 février 1990\x1b[0m", "15 \x1b[1mfévrier 1990", "15\x00 février 1990", "15 février￼ 1990",
])
def test_stream_keeps_date_with_terminal_escape_or_control_whole(date):
    cleaner = TextCleaner(cfg=preset_config("training", check_ner_process=False, check_replace_dates=True))
    text = "Introduction. " * 300 + f"Value {date} here. " + "Afterwards. " * 20
    stream = WindowedStreamingRedactor(cleaner)
    out = stream.feed(text) + stream.finish()
    assert "Value <DATE> here." in cleaner.process(text).lm_text
    assert "Value <DATE> here." in out
    assert "1990" not in out


# --- 2: a seeded locale fake does not use the shared global random, so other threads cannot shift it ---

@pytest.mark.parametrize("locale, group", [("th_TH", "SOCIAL_SECURITY_NUMBER"), ("uk_UA", "TAX_ID"),
                                           ("uk_UA", "LICENSE_PLATE")])
def test_locale_fake_ignores_concurrent_global_random(locale, group):
    replacer = SyntheticReplacer(locale=locale, reference_date="2026-01-01")

    def fake():
        replacer.reset_cache("same document")
        return replacer.get_replacement(group, "x")

    stop = threading.Event()
    noise = threading.Thread(target=lambda: [random.random() for _ in iter(stop.is_set, True)])  # noqa: S311
    interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    noise.start()
    try:
        values = {fake() for _ in range(300)}
    finally:
        stop.set()
        noise.join()
        sys.setswitchinterval(interval)
    assert values == {fake()}
