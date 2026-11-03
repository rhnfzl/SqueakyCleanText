"""WindowedStreamingRedactor never shows a value the whole document masks, at any window size it accepts.

Leak classes (research/streaming-leak-fuzz.md): a rule that reads past its match (a trunk-0 phone needs its
9th digit, "05 234 567 1"), a phone the cleaner finds only once an earlier step masked its neighbour (an IP or
card next to "117 235 991"), and the numbers guard missing those earlier steps too.
"""
import random
import re

import pytest

from sct import TextCleaner, TextCleanerConfig
from sct.presets import preset_config
from sct.streaming import WindowedStreamingRedactor

_CLEANERS = {}


def _cleaner(preset):
    if preset not in _CLEANERS:
        cfg = TextCleanerConfig(check_ner_process=False) if preset == "default" \
            else preset_config(preset, check_ner_process=False)
        _CLEANERS[preset] = TextCleaner(cfg=cfg)
    return _CLEANERS[preset]


def _stream(cleaner, text, buf, overlap, chunks):
    """Stream *text* in chunks of the given sizes (cycled); None when the window raises BufferError."""
    redactor = WindowedStreamingRedactor(cleaner, max_buffer_chars=buf, overlap_chars=overlap)
    out, i, k = [], 0, 0
    try:
        while i < len(text):
            size = chunks[k % len(chunks)]
            out.append(redactor.feed(text[i:i + size]))
            i, k = i + size, k + 1
        out.append(redactor.finish())
    except BufferError:
        return None
    return "".join(out)


@pytest.mark.parametrize("preset, text", [
    ("default", "05 234 567 1 x"),  # trunk-0 phone needs a 9th digit after its match
    ("default", "+1 2 3 4 5 6 7 8 x"),  # international: same lookahead
    ("training", "ok 490 872 276 1.2.3.4 ok"),  # 3-3-3 phone once the IP after it is masked
    ("training", "1.2.3.4 117 235 991 ok"),
    ("training", "4111111111111111 117 235 991 ok"),  # ... once the card before it is masked
    ("sct", "ok 01 023 59 9 90.217.217.177 ok"),  # numbers guard: digit run ends at the masked IP
])
def test_no_window_leaks_a_value_the_whole_text_masks(preset, text):
    cleaner = _cleaner(preset)
    whole = cleaner.process(text).lm_text.split()
    for chunks in ([1], [3], [2, 5]):
        for buf in range(6, len(text) + 2):
            for overlap in range(1, buf):
                out = _stream(cleaner, text, buf, overlap, chunks)
                assert out is None or out.split() == whole, (buf, overlap, chunks, out)


def _digits(r, n, first="0123456789"):
    return r.choice(first) + "".join(r.choice("0123456789") for _ in range(n - 1))


def _value(r):
    kind = r.randrange(9)
    if kind == 0:  # 8- and 9-digit trunk-0 numbers in random groups
        s, out = "0" + _digits(r, r.randint(7, 8), "123456789"), []
        while s:
            k = r.choice([1, 2, 3])
            out, s = out + [s[:k]], s[k:]
        return r.choice(" .-").join(out)
    if kind == 1:
        return " ".join(_digits(r, 3, "123456789") for _ in range(3))
    if kind == 2:
        return "+" + " ".join(_digits(r, r.randint(1, 3), "123456789") for _ in range(r.randint(3, 6)))
    if kind == 3:
        return ".".join(str(r.randint(1, 254)) for _ in range(4))
    if kind == 4:
        return r.choice(["4111111111111111", "5500 0000 0000 0004"])
    if kind == 5:
        return r.choice(["NL91 ABNA 0417 1643 00", f"{r.randint(1, 28)} October {r.randint(1950, 2030)}"])
    if kind == 6:
        return f"{r.randint(1, 28):02d}-{r.randint(1, 12):02d}-{r.randint(1950, 2030)}"
    if kind == 7:
        return str(r.randint(0, 99))
    return r.choice(["ok", "call", "ref", "note", "x"])


_TOKEN = re.compile(r"\d+|[^\W\d_]{3,}")
_TAG = re.compile(r"<[A-Z_]+>")


@pytest.mark.parametrize("seed", range(4))
def test_seeded_fuzz_overlap_longer_than_every_value(seed):
    """With an overlap longer than every value, no split may show a token the whole document masked."""
    r = random.Random(seed)  # noqa: S311 - seeded test data
    for _ in range(60):
        cleaner = _cleaner(r.choice(["training", "sct", "default"]))
        text = " ".join(_value(r) for _ in range(r.randint(4, 12)))
        overlap = r.choice([24, 32])
        buf = overlap + r.choice([4, 8, 16, 32])
        out = _stream(cleaner, text, buf, overlap, [r.choice([1, 2, 3, 5, 8])])
        if out is None:
            continue
        whole = cleaner.process(text).lm_text

        def tokens(s):
            return set(_TOKEN.findall(_TAG.sub(" ", s)))
        assert tokens(out) <= tokens(whole), (text, buf, overlap, out, whole)
