"""The stream split guard reads the cleaner's own normalised text (TextCleaner.normalise), not a copy."""

import random

import pytest

from sct import TextCleaner
from sct.presets import preset_config
from sct.streaming import WindowedStreamingRedactor

# Steps the old guard copy did not model: emoji removal, ASCII folding, grouped numbers, CRLF line breaks.
CASES = [
    ("training", {"check_remove_emoji": True}, "Call 06 12\U0001F642 34 56 78 now."),
    ("training", {"check_remove_emoji": True}, "Call 06 \U0001F64212 34 56 78 now."),
    ("training", {"check_to_ascii_unicode": True}, "Call 06 12 〇〇 56 78 now."),
    ("sct", {"fold_styled_characters": False}, "Total 5 379 euros paid."),
    ("training", {}, "Call 06 12\r\n34 56 78 now."),
]


@pytest.mark.parametrize("preset, overrides, value", CASES)
def test_stream_equals_whole_at_every_split(preset, overrides, value):
    cleaner = TextCleaner(cfg=preset_config(preset, check_ner_process=False, **overrides))
    text = "Intro words here. " * 3 + value + " tail tail tail"
    whole = cleaner.process(text).lm_text
    for cut, character in enumerate(text):
        if not character.isspace():
            continue
        # the cutoff lands on this whitespace character
        stream = WindowedStreamingRedactor(cleaner, max_buffer_chars=len(text) - 1, overlap_chars=len(text) - cut)
        assert stream.feed(text) + stream.finish() == whole, cut


def test_guard_view_is_the_cleaner_normalise():
    cleaner = TextCleaner(cfg=preset_config("training", check_ner_process=False))
    calls = []
    normalise = cleaner.normalise
    cleaner.normalise = lambda text: calls.append(text) or normalise(text)
    text = "Intro words here. " * 30 + "Call ０６ １２ ３４ ５６ ７８ now."
    stream = WindowedStreamingRedactor(cleaner, max_buffer_chars=200, overlap_chars=40)
    stream.feed(text)
    assert text[:len(calls[0])] == calls[0] and len(calls) >= 2


def test_crlf_text_without_spaces_still_splits():
    cleaner = TextCleaner(cfg=preset_config("training", check_ner_process=False))
    text = "\r\n".join(f"row{i},val{i}" for i in range(600))
    stream = WindowedStreamingRedactor(cleaner, max_buffer_chars=256, overlap_chars=32)
    out = "".join(stream.feed(text[i:i + 100]) for i in range(0, len(text), 100)) + stream.finish()
    assert out == cleaner.process(text).lm_text


def test_cut_escape_at_buffer_end_does_not_hide_a_date():
    # fuzz seed 3128: a feed ending inside "\\U00002467" stopped the escape decoding of the earlier date
    cleaner = TextCleaner(cfg=preset_config("sct", check_ner_process=False))
    text = ("جؿب. غسغي؟ ةل؟ قحت ؼنؽز ػػسز ظوؾب. ①②①.1𝟜３.1⑥①.１8０\u200b ػثؿ ثوطى؟ شةرل خظؼت عػزع. ني ةدسة "
            "غزثى. 6 ｓeｐ \\x31\\u003992 طصق ثمظ طشمع طذحز. دـ ثسطط. "
            "①\\U00002467\\U00000038.④②.\\u2461②0.2③8 ةعيك ذقشت. زةن.")
    rng = random.Random(3128 * 7919 + 160)  # noqa: S311  # the fuzz harness's chunking
    stream = WindowedStreamingRedactor(cleaner, max_buffer_chars=160, overlap_chars=48)
    out, start = [], 0
    while start < len(text):
        size = rng.randint(1, 20)
        out.append(stream.feed(text[start:start + size]))
        start += size
    assert "".join(out) + stream.finish() == cleaner.process(text).lm_text


def test_split_jump_maps_expanding_characters_back_to_the_source():
    # Codex round 34: "⑫" normalises to "12", so the view distance to the phone start overshot in the source
    # and the jump skipped the safe split after "Call" (BufferError)
    cleaner = TextCleaner(cfg=preset_config("training", check_ner_process=False))
    text = "Call 06 ⑫㉞ ⑤⑥⑦⑧ now please."
    stream = WindowedStreamingRedactor(cleaner, max_buffer_chars=26, overlap_chars=17)
    assert stream.feed(text) + stream.finish() == cleaner.process(text).lm_text == "Call <PHONE> now please."
