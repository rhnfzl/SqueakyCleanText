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


def fit_to_markers(start: int, end: int, markers: list[tuple[int, int]]) -> tuple[int, int] | None:
    """The span to replace for (*start*, *end*): None inside a marker; one that cuts a marker ("of <EMAIL")
    takes it whole, so no fragment ("<EM", ">") is left beside the replacement in any mode."""
    if any(a <= start and end <= b for a, b in markers):
        return None
    for a, b in markers:
        if a < end and start < b:
            start, end = min(start, a), max(end, b)
    return start, end


def drop_allowed(allow: regex.Pattern | None, text: str, items: list[Any],
                 span: Callable[[Any], tuple[int, int]]) -> list[Any]:
    """*items* without those equal to or inside a custom_allow_terms occurrence (applied last)."""
    if allow is None or not items:
        return items
    allowed = [m.span() for m in allow.finditer(text)]
    return [item for item in items
            if not any(a <= span(item)[0] and span(item)[1] <= b for a, b in allowed)]


def finalize_spans(ner, text, windows, window_results, ner_confidence_threshold, source_markers=None,
                   *, protect_markers=True):
    """Apply the shared window, vote and span decisions before rendering."""
    spans: list[dict[str, Any]] = []
    foreign: list[dict[str, Any]] = []
    for (_, win_start, offsets, own_start, own_end), ner_results in zip(windows, window_results):
        for item in ner_results:
            start = win_start + offsets[item['start']]
            end = win_start + offsets[item['end']]
            if ner.cfg.ner_snap_word_edges:
                start, end = snap_word_edges(
                    text, start, end, ner.cfg.ner_subword_widen, ner.cfg.ner_subword_widen_same_script)
            span = {**item, 'start': start, 'end': end,
                    'word': text[start:end], 'key': f"{start}:{end}"}
            (spans if own_start <= start < own_end else foreign).append(span)
    # Windows repeat context and WordPiece widening can give one model two labels for a
    # span; neither is another model: one vote per model and span, its best score.
    best: dict[tuple, dict[str, Any]] = {}
    for span in spans + foreign:
        vote = (span.get('model'), span['key'])
        if vote not in best or span['score'] > best[vote]['score']:
            best[vote] = span
    owned = {(s.get('model'), s['key']) for s in spans}
    foreign = [span for vote, span in best.items() if vote not in owned]
    spans = [span for vote, span in best.items() if vote in owned]
    if foreign:
        spans = ner.vote_entities(spans, ner_confidence_threshold)
        # A neighbour's copy counts only where an eligible owned span covering it is concealed.
        covers = [
            o for o in spans
            if not protect_markers or ner._privacy_policy is None
            or ner._privacy_policy.evaluate((ner._entity(o),))[0].action in ('placeholder', 'redact')
        ]
        spans += [
            span for span in foreign
            if not any(o['start'] <= span['start'] and span['end'] <= o['end'] for o in covers)
        ]

    ensemble_results = ner.vote_entities(spans, ner_confidence_threshold)
    markers = ner._marker_spans(text, source_markers) if protect_markers else []
    if protect_markers and ner._synthetic_replacer:  # None outside synthetic mode
        # Fakes the regex steps inserted (emails, phones...) are protected like placeholders.
        for fake in set(ner._synthetic_replacer._get_cache().values()):
            markers += [m.span() for m in re.finditer(re.escape(fake), text)]
    fitted = []
    for item in ensemble_results:
        fit = fit_to_markers(item['start'], item['end'], markers)
        if fit == (item['start'], item['end']):
            fitted.append(item)
        elif fit is not None:
            fitted.append({**item, 'start': fit[0], 'end': fit[1], 'word': text[fit[0]:fit[1]]})
    ensemble_results = fitted
    if ner.cfg.ner_propagate_masked:
        ensemble_results = propagate(text, ensemble_results, markers, ner.cfg.ner_propagate_min_score)
    if ner.cfg.ner_merge_location_gap:
        ensemble_results = merge_location_gaps(
            text, ensemble_results, markers, ner.cfg.ner_merge_location_gap)
    ensemble_results = drop_allowed(ner._allow_regex, text, ensemble_results, lambda item: (item['start'], item['end']))
    return ensemble_results


def presidio_results(ner, text, language, source_markers=None, *, protect_markers=True):
    """Use Presidio decisions unchanged, independently of rendering."""
    from sct.utils.resources import language_to_iso

    analyzer_language = language_to_iso(language) if language else 'en'
    # Detection covers 75 languages; Presidio only analyzes those its NLP engine has.
    supported = getattr(ner._analyzer, 'supported_languages', None)
    if supported is not None and analyzer_language not in supported:
        analyzer_language = 'en'
    analyzer_results = ner._analyzer.analyze(
        text=text,
        language=analyzer_language,
    )
    markers = ner._marker_spans(text, source_markers) if protect_markers else []
    kept = []
    for result in analyzer_results:
        span = fit_to_markers(result.start, result.end, markers)
        if span is not None:
            result.start, result.end = span
            kept.append(result)
    analyzer_results = kept
    # Analyzer names (PHONE_NUMBER, a label's upper case) -> the configured type names,
    # then aliases (PER/LOC/ORG) to canonical names, as the other backends do.
    analyzer_results = drop_allowed(ner._allow_regex, text, analyzer_results, lambda r: (r.start, r.end))
    for result in analyzer_results:
        entity_type = ner._presidio_types.get(result.entity_type, result.entity_type)
        result.entity_type = ENTITY_TYPE_MAP.get(entity_type, entity_type)
    return analyzer_results
