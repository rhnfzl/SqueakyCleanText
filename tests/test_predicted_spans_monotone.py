"""predicted_spans only adds placements to b665956's on SCT-like text, so cells scored with it stay valid.

The text is synthetic: regex values (one placeholder per occurrence, true source span known) are replaced
on the raw text, SCT's own cleanup runs, then names are masked on the cleaned text as NER would.
"""
import random
import re
import unicodedata
from bisect import bisect_left, bisect_right

import pytest

from benchmarks.score import (
    GENERIC_MARKER, _LABEL_ALIASES as LABEL_ALIASES, _NUMBERED_PLACEHOLDER as PLACEHOLDER_REGEX,
    WHITESPACE_RUN, _collapse_whitespace, _find_value,
    _nfkc_view, _sct_cleanup, predicted_spans,
)
from sct.config import TextCleanerConfig
from sct.evaluation import EntitySpan
from sct.utils.special import ProcessSpecialSymbols


def b665956_predicted_spans(source, output, originals, view=None, cfg=None):
    """predicted_spans as of commit b665956, frozen: the round 2 cells for the typed sets were scored with it."""
    base, base_offsets = view or _nfkc_view(source)
    views = (_collapse_whitespace(*_sct_cleanup(base, base_offsets, cfg)), _collapse_whitespace(base, base_offsets))
    view, offsets = views[0]
    shown = {m.group(0) for m in PLACEHOLDER_REGEX.finditer(output)}
    swallowed = {}
    for placeholder, original in originals.items():
        tag = PLACEHOLDER_REGEX.fullmatch(placeholder)
        if tag and placeholder not in shown:
            swallowed.setdefault(tag.group(1), []).append(original)
    output = WHITESPACE_RUN.sub(" ", unicodedata.normalize("NFKC", output))
    spans, cursor, previous_end = [], 0, 0
    for match in PLACEHOLDER_REGEX.finditer(output):
        context = output[previous_end:match.start()]
        previous_end = match.end()
        low = context_at = cursor
        for cut in range(len(context)):
            found = view.find(context[cut:], bisect_left(offsets, cursor))
            if found >= 0:
                context_at, cursor = offsets[found], offsets[found + len(context) - cut]
                break
        label = LABEL_ALIASES.get(match.group(1), match.group(1))
        value = GENERIC_MARKER.sub(
            lambda m: swallowed[m.group(1)].pop(0) if swallowed.get(m.group(1)) else m.group(0),
            originals.get(match.group(0), ""))
        value = WHITESPACE_RUN.sub(" ", unicodedata.normalize("NFKC", value)).strip()
        found = (_find_value(value, views, cursor) or _find_value(value, views, low, context_at)) if value else None
        if found is None:
            spans.append(EntitySpan(label, -1, -1))
            continue
        cursor = base_offsets[bisect_right(base_offsets, found[1])]
        spans.append(EntitySpan(label, found[0], cursor))
    return spans


WORDS = ["met", "the", "and", "call", "at", "on", "with", "de", "wrote", "to", "Dr", "said", "x", "B", "A", "ł",
         ",", ".", ":", "-", "*", '"', "'", ";", "(", ")", "&", "/", ">", "<b>", "</td>", "%", "--", "!", "e.g.",
         "[1]", "[note here]", "(see p. 3)", "{x}", "[a much longer remark that is over the char limit]",
         '{"k": 1}', "[two words]", "U.S.", "3.5", "2024"]
# Markup and shell tokens as on the long RedactionBench docs.
MARKUP = ['" />', "/>", '"', "[", "]", "[x", "y]", "$", "<HTML>", "<B>", "\\[", "\\]", '" "', "-f"]
NAMES = ["Ann Lee", "Jan Kowalski", "Ann B Lee", "Bob", "Ann", "Anna (*) Smith", "John[1] Smith", "Lee",
         "René", "Jan  Kowalski", "Hans {x} Meier", "Tom * Hardy", "O'Neil", "Jean-Luc"]
REGEX = {"EMAIL": ["ann@x.io", "jan.k@mail.pl", "bob@y.org"], "PHONE": ["06 - 1234", "555 1234", "+31 6 1234 5678"],
         "URL": ["https://x.io/a", "www.y.org"], "IP_ADDRESS": ["8.8.8.8"]}
SEPS = [" ", " ", " ", "  ", "\n", "\t ", "\n\n  ", ""]


def _sct_clean(text, cfg):
    special = ProcessSpecialSymbols()
    text = special.remove_isolated_letters(text)
    text = special.remove_isolated_special_symbols(
        text, cfg.check_remove_bracket_content, cfg.check_remove_brace_content,
        cfg.bracket_content_max_chars, cfg.bracket_content_max_words, cfg.keep_json_braces)
    return re.sub(r"\s+", " ", text)


def _doc(rng, cfg, words):
    """(source, output, originals, truth): truth maps each regex placeholder to its source span."""
    source, pre, originals, truth, names, number = "", "", {}, {}, [], {}
    for _ in range(rng.randint(3, 40)):
        r = rng.random()
        if r < 0.15:
            label = rng.choice(list(REGEX))
            value = rng.choice(REGEX[label])
            left, right = ("", "") if rng.random() < 0.8 else (
                rng.choice(["[mail ", "(", '"', "[", "{"]), rng.choice(["]", ")", '"', " ]", "}"]))
            source, pre = source + left, pre + left
            if rng.random() < 0.85:
                placeholder = f"<{label}_{number.setdefault(label, 0)}>"
                number[label] += 1
                originals[placeholder] = value
                truth[placeholder] = (LABEL_ALIASES.get(label, label), len(source), len(source) + len(value))
                pre += placeholder
            else:
                pre += value
            source, pre = source + value + right, pre + right
        else:
            token = rng.choice(NAMES) if r < 0.35 else rng.choice(words)
            if r < 0.35:
                names.append(token)
            source, pre = source + token, pre + token
        sep = rng.choice(SEPS)
        source, pre = source + sep, pre + sep
    cleaned, output, pos = _sct_clean(pre, cfg), "", 0
    for name in names:
        value = _sct_clean(name, cfg).strip()
        at = cleaned.find(value, pos)
        if not value or at < 0 or rng.random() < 0.25:
            continue  # NER missed this mention: the output keeps it
        placeholder = f"<PERSON_{number.setdefault('PERSON', 0)}>"
        number["PERSON"] += 1
        originals[placeholder] = value
        output, pos = output + cleaned[pos:at] + placeholder, at + len(value)
    return source, output + cleaned[pos:], originals, truth


@pytest.mark.parametrize("words", [WORDS, WORDS + MARKUP], ids=["plain", "markup"])
def test_predicted_spans_keep_every_b665956_placement(words):
    rng = random.Random(7)  # noqa: S311 - reproducible test data
    configs = (TextCleanerConfig(), TextCleanerConfig(bracket_content_max_chars=0, bracket_content_max_words=0))
    for _ in range(1500):
        cfg = configs[rng.random() < 0.2]
        source, output, originals, truth = _doc(rng, cfg, words)
        tags = [m.group(0) for m in PLACEHOLDER_REGEX.finditer(output)]
        old = b665956_predicted_spans(source, output, originals, cfg=cfg)
        new = predicted_spans(source, output, originals, cfg=cfg)
        for tag, before, after in zip(tags, old, new):
            if before.start >= 0 and before != after:
                # Plain text: no placement changes. Markup: only a regex value b665956 put at the wrong
                # place may move, and only to its true span.
                assert words is not WORDS and tag in truth, (source, output)
                assert (before.label, before.start, before.end) != truth[tag] == (after.label, after.start, after.end)
