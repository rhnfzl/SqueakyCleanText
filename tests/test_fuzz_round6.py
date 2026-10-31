"""Fuzz round 6 (research/0.8.0-fuzz-round6.md): one regression per fixed root cause."""

import pytest

from sct import TextCleaner
from sct.presets import preset_config
from sct.streaming import WindowedStreamingRedactor
from sct.utils.normtext import fold_styled


def _cleaner(preset="training", **overrides):
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
    assert _cleaner().process(text).lm_text == "call <PHONE> now"


def test_typeset_separators_stay_with_fold_off():
    text = "call 06\u00a012\u00a034\u00a056\u00a078 now"
    assert _cleaner(fold_styled_characters=False).process(text).lm_text == text


# A lone circled number between groups set off by a typeset space, and an invisible hint before a
# typeset dash, did not fold as they do with an ASCII space or hyphen.
@pytest.mark.parametrize("text, folded", [
    ("06\u202f98\u202f\u3256\u202f03\u202f84", "06\u202f98\u202f26\u202f03\u202f84"),
    ("06\u00a098\u00a003\u00a084\u00a0\u3256", "06\u00a098\u00a003\u00a084\u00a026"),
    ("131\u2060\u20107237", "131\u20107237"),
])
def test_fold_reads_typeset_separators(text, folded):
    assert fold_styled(text) == folded


# The stream kept a no-break space it split at, while the whole text's whitespace step made it " ".
def test_stream_separator_follows_whitespace_step():
    cleaner = _cleaner("sct")
    text = "alpha beta\u202fgamma delta\u00a0epsilon zeta\u2009eta theta iota"
    stream = WindowedStreamingRedactor(cleaner, max_buffer_chars=20, overlap_chars=6)
    out = "".join(stream.feed(ch) for ch in text) + stream.finish()
    assert out == cleaner.process(text).lm_text
