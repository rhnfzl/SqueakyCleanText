"""NER span post-processing: plain functions over text and span dicts, no model state."""

import re
import unicodedata
from collections.abc import Callable, Collection, Iterable
from typing import Any

import regex

# Entity group to Presidio entity type mapping
ENTITY_TYPE_MAP = {
    'PER': 'PERSON',
    'LOC': 'LOCATION',
    'ORG': 'ORGANISATION',
    'MISC': 'MISC',
}

# Propagated groups: the token models' short tags and the long labels the default gliner backend emits.
_PROPAGATED_GROUPS = frozenset({'PER', 'ORG', 'LOC', 'PERSON', 'ORGANIZATION', 'LOCATION'})

# Ends a LOCATION merge gap: a line break or a sentence end (format only, no words).
_GAP_BREAK = re.compile(r"[\n\r\u2028\u2029]|[.!?;:]\s|[。！？]")


def _letter_scripts(text: str) -> list[str]:
    """Script of each letter, from its Unicode name after NFKC ("LATIN", "HIRAGANA", "THAI").

    Marks, digits, punctuation and modifier letters (ー, ๆ) belong to no script here.
    """
    return [unicodedata.name(c, ' ').split(' ', 1)[0]
            for c in unicodedata.normalize('NFKC', text)
            if unicodedata.category(c) in ('Lu', 'Ll', 'Lt', 'Lo')]


def _joins_word(char: str, script: str | None) -> bool:
    if unicodedata.category(char)[0] not in 'LM' or unicodedata.east_asian_width(char) in 'WF':
        return False
    return not script or set(_letter_scripts(char)) <= {script}


def _word_bounds(text: str, start: int, end: int, scripts: list[str] | None) -> tuple[int, int]:
    new_start, new_end = start, end
    if _joins_word(text[start], None):
        while new_start > 0 and _joins_word(text[new_start - 1], scripts[0] if scripts else None):
            new_start -= 1
    if _joins_word(text[end - 1], None):
        while new_end < len(text) and _joins_word(text[new_end], scripts[-1] if scripts else None):
            new_end += 1
    return new_start, new_end


def snap_word_edges(text: str, start: int, end: int, ratio: float | None, same_script: bool) -> tuple[int, int]:
    """Move span edges that fall inside a word out to the word edges ("Salut" + "e").

    A word is a run of Unicode letters and marks; wide (CJK) characters never join, since
    those scripts write no spaces. With *same_script* an added letter must share the script
    of the span's edge letter. When the added text is over *ratio* times the span (an
    unspaced Thai clause), the span stays as tagged; *ratio* None = no cap.
    """
    if start >= end:
        return start, end
    new_start, new_end = _word_bounds(text, start, end, _letter_scripts(text[start:end]) if same_script else None)
    if ratio is not None and (new_end - new_start) - (end - start) > ratio * (end - start):
        return start, end
    return new_start, new_end


def _union(text: str, last: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
    best = last if last['score'] >= item['score'] else item
    start, end = last['start'], max(last['end'], item['end'])
    return {**best, 'start': start, 'end': end, 'word': text[start:end], 'key': f"{start}:{end}"}


def merge_overlaps(text: str, filtered_data: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge overlapping spans into their union, labelled by the highest score.

    Same rule as ``PrivacyPolicy.transform``, so no part of a nested or
    partly overlapping entity stays visible.
    """
    merged: list[dict[str, Any]] = []
    for item in sorted(filtered_data, key=lambda x: (x['start'], -x['end'])):
        if merged and item['start'] < merged[-1]['end']:
            merged[-1] = _union(text, merged[-1], item)
        else:
            merged.append(item)
    return merged


def merge_location_gaps(text: str, spans: Iterable[dict[str, Any]], markers: Collection[tuple[int, int]],
                        max_gap: int) -> list[dict[str, Any]]:
    """Join neighbouring LOCATION spans into one, so an address is one span.

    Format rule only: the gap between them may hold at most *max_gap* characters outside
    placeholders (``<NUMBER>`` in ``IL <NUMBER>,`` is not counted and is taken in), no line
    break and no sentence end (``.!?;:`` before whitespace, or ``。！？``).
    """
    merged: list[dict[str, Any]] = []
    for item in sorted(spans, key=lambda x: (x['start'], -x['end'])):
        last = merged[-1] if merged else None
        if last and {last['entity_group'], item['entity_group']} <= {'LOC', 'LOCATION'}:
            a, b = last['end'], item['start']
            gap = text[a:b]
            visible = sum(1 for i in range(a, b) if not any(s <= i < e for s, e in markers))
            if visible <= max_gap and not _GAP_BREAK.search(gap):
                merged[-1] = _union(text, last, item)
                continue
        merged.append(item)
    return merged


def _spreadable_values(text: str, spans: list[dict[str, Any]], min_score: float | None) -> dict[str, dict[str, Any]]:
    """Value -> its best-scored span, for the spans ``propagate`` may spread."""
    sources: dict[str, dict[str, Any]] = {}
    for item in spans:
        value = text[item['start']:item['end']]
        if (item['entity_group'] in _PROPAGATED_GROUPS and (min_score is None or item['score'] >= min_score)
                and len(value) >= 2 and any(ch.isalpha() for ch in value)
                and (value not in sources or item['score'] > sources[value]['score'])):
            sources[value] = item
    return sources


def propagate(text: str, spans: list[dict[str, Any]], markers: Iterable[tuple[int, int]],
              min_score: float | None) -> list[dict[str, Any]]:
    """Add a span at every other whole-word, case-sensitive copy of a masked person, organisation or location.

    The values come from the models' own spans; a span below *min_score*,
    shorter than 2 characters or without a letter is not spread. Copies that overlap a
    span or a marker are skipped, longest value first, so added spans stay disjoint.
    """
    sources = _spreadable_values(text, spans, min_score)
    taken = [(s['start'], s['end']) for s in spans] + list(markers)
    added = []
    for value in sorted(sources, key=len, reverse=True):
        # \p{M}: a combining mark (Hindi ा, Thai ิ) continues the word, though stdlib \w excludes it.
        for m in regex.finditer(rf"(?<![\w\p{{M}}]){regex.escape(value)}(?![\w\p{{M}}])", text):
            a, b = m.span()
            if not any(x < b and a < y for x, y in taken):
                taken.append((a, b))
                added.append({**sources[value], 'start': a, 'end': b, 'word': value, 'key': f"{a}:{b}"})
    return sorted(spans + added, key=lambda x: x['start']) if added else spans


def drop_allowed(allow: regex.Pattern | None, text: str, items: list[Any],
                 span: Callable[[Any], tuple[int, int]]) -> list[Any]:
    """*items* without those equal to or inside a custom_allow_terms occurrence (applied last)."""
    if allow is None or not items:
        return items
    allowed = [m.span() for m in allow.finditer(text)]
    return [item for item in items
            if not any(a <= span(item)[0] and span(item)[1] <= b for a, b in allowed)]
