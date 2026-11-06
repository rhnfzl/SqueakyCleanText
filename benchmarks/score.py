"""Scorers. They see only spans, outputs and counts; results never hold dataset text.

PII (per language and per SCT type):

- **strict**: exact label + span match, via ``sct.evaluation.evaluate_pii``.
- **partial**: a gold span is found when a predicted span of the same type overlaps
  it; a prediction is correct when it overlaps a gold span of the same type.
  Recall and precision use these two counts (not a one-to-one matching).
  ``masked_recall`` is the type-agnostic recall: a gold span counts when any kept
  prediction overlaps it, whatever its type (NUMBER masked as PHONE is not a leak).
- **leak**: a gold PII value is *fully* leaked when it is still in the output, and
  *partly* leaked when a 4+ char word of it is still outside the placeholders.
  Words are runs of letters, digits and combining marks, so Brahmic words are not cut
  at vowel signs. Scripts written without spaces (Thai) give one long "word", so their
  partial leaks are undercounted.
  ``leak_rate`` counts both. It is an upper bound: a word that also occurs elsewhere
  in the document counts as a leak. Both sides are NFKC-normalised. DATE is reported
  per type but not in the total or per language.
- **by_position**: partial recall by the gold span's start offset in 1000-char
  buckets, so recall lost after the first model window shows up.

Predictions only count when their (aliased) type is one the dataset scores, and a
prediction that only overlaps out-of-scope gold is ignored.
"""

import re
from bisect import bisect_left, bisect_right
import unicodedata
from collections import Counter, defaultdict
from typing import Any

import regex

from sct.evaluation import EntitySpan, EvaluationExample, evaluate_pii
from sct.utils.constants import LEGACY_TYPE_NAMES

from benchmarks.labels import DATE, PRED_ALIASES, normalise_gold

# What scoring needs from TextCleaner: each placeholder must map back to the value it replaced.
SCORING = {"replacement_mode": "reversible", "check_to_ascii_unicode": False}
LEAK_WORD = regex.compile(r"[\w\p{M}]+")  # \w alone splits Devanagari/Tamil at vowel signs
# Same characters as sct.utils.ner.PLACEHOLDER_TOKEN. A local copy: adapter venvs pin an sct release.
PLACEHOLDER = re.compile(r"<[A-Z][A-Z0-9_]*>")
_NUMBERED_PLACEHOLDER = re.compile(r"<([A-Z][A-Z0-9_]*?)_\d+>")
# Pre-0.8.0 reversible type names (legacy_type_names=True) -> dataset label.
_LABEL_ALIASES = {old: new for new, old in LEGACY_TYPE_NAMES.items()}
LEAK_MIN_CHARS = 4
LEAK_EXCLUDED = frozenset({DATE})
POSITION_BUCKET = 1000
POSITION_BUCKETS = 10
WHITESPACE_RUN = re.compile(r"\s+")
GENERIC_MARKER = re.compile(r"<([A-Z][A-Z0-9_]*)>")


def _collapse_whitespace(view: str, offsets: list[int]) -> tuple[str, list[int]]:
    """*view* with each whitespace run as one space (SCT's normalize_whitespace), offsets kept per char."""
    chars, kept = [], []
    for ch, offset in zip(view, offsets):
        if ch.isspace():
            if chars and chars[-1] == " ":
                continue
            ch = " "
        chars.append(ch)
        kept.append(offset)
    return "".join(chars), kept + offsets[len(view):]


def _drop(view: str, offsets: list[int], pattern: re.Pattern, removed=lambda m: True,
          pieces: list[tuple[int, int]] | None = None) -> tuple[str, list[int]]:
    """*view* without the *pattern* matches SCT's ``pattern.sub`` deletes (*removed*), offsets kept per char.

    *pieces* are ranges searched on their own, as SCT removes brackets only outside JSON objects.
    """
    gone = set()
    for start, end in pieces or [(0, len(view))]:
        for match in pattern.finditer(view, start, end):
            if removed(match):
                gone.update(range(*match.span()))
    kept = [i for i in range(len(view)) if i not in gone]
    return "".join(view[i] for i in kept), [offsets[i] for i in kept] + offsets[len(view):]


def _sct_cleanup(view: str, offsets: list[int], cfg) -> tuple[str, list[int]]:
    """*view* without what SCT's pre-NER cleanup removes under *cfg*: isolated letters, bracket and
    brace content, isolated symbols and quotes, with the regexes and order of sct/utils/special.py."""
    from sct.utils import constants, special

    if cfg.check_remove_isolated_letters:
        view, offsets = _drop(view, offsets, constants.ISOLATED_LETTERS_REGEX)
    if not cfg.check_remove_isolated_special_symbols:
        return view, offsets
    max_chars, max_words = cfg.bracket_content_max_chars, cfg.bracket_content_max_words
    limited = max_chars > 0 or max_words > 0

    def short(match):  # special._remove_bracketed: with a limit, only short innermost content goes
        inner = match.group()[1:-1]
        return not ((0 < max_chars < len(inner)) or (0 < max_words < len(inner.split())))

    for on, whole, inner in ((cfg.check_remove_bracket_content, constants.BRACKET_CONTENT_REGEX,
                              constants.BRACKET_INNER_REGEX),
                             (cfg.check_remove_brace_content, constants.BRACE_CONTENT_REGEX,
                              constants.BRACE_INNER_REGEX)):
        if on:
            edges = [0, *(e for span in (special._json_object_spans(view) if cfg.keep_json_braces else [])
                          for e in span), len(view)]
            view, offsets = _drop(view, offsets, inner if limited else whole,
                                  short if limited else (lambda m: True), list(zip(edges[::2], edges[1::2])))
    view, offsets = _drop(view, offsets, constants.ISOLATED_SPECIAL_SYMBOLS_REGEX)
    return _drop(view, offsets, constants.ISOLATED_QUOTES_REGEX, lambda m: special._drop_if_isolated(m) == "")


def _find_value(value: str, views, low: int, high: int | None = None,
                earliest: bool = False) -> tuple[int, int] | None:
    """Source offsets of the first and last char of the first *value* starting in source range
    [*low*, *high*), tried in each of *views* in turn (*earliest*: the earliest hit of all views); None if absent."""
    hits = []
    for text, offsets in views:
        at = text.find(value, bisect_left(offsets, low))
        if at >= 0 and (high is None or offsets[at] < high):
            hits.append((offsets[at], offsets[at + len(value) - 1]))
            if not earliest:
                break
    return min(hits) if hits else None


def _align(context: str, text: str, offsets: list[int], cursor: int, cuts: int) -> tuple[int, int, int] | None:
    """(cut, source start, source end) of the longest suffix ``context[cut:]`` (cut < *cuts*) found in
    *text* after source offset *cursor*; None if none is."""
    start = bisect_left(offsets, cursor)
    # The cursor sits inside a whitespace run the view kept as an earlier space: that space is still
    # the gap after the previous value, so the context may start on it.
    if 0 < start < len(offsets) and offsets[start] != cursor and text[start - 1] == " ":
        start -= 1
    for cut in range(cuts):
        at = text.find(context[cut:], start)
        if at >= 0:
            return cut, offsets[at], offsets[at + len(context) - cut]
    return None


def _locate(value: str, views, cursor: int, low: int, context_at: int,
            earliest: bool = False) -> tuple[int, int] | None:
    """*value* (whitespace collapsed) after *cursor*, else in [*low*, *context_at*): a context found too
    late moved the cursor past the value, which still lies after the previous one and before that
    context, never on text the output kept (an unmasked copy)."""
    value = WHITESPACE_RUN.sub(" ", unicodedata.normalize("NFKC", value)).strip()
    return (_find_value(value, views, cursor, earliest=earliest)
            or _find_value(value, views, low, context_at, earliest)) if value else None


def _recover(context: str, value: str, views, low: int, output: str, spans) -> tuple[int, int] | None:
    """*value* right after the whole output *context*, at the last such place before source offset *low*:
    an earlier span credited to a later copy moved the cursor past this one. Only when the output shows
    ``context + value`` nowhere (every source copy of the pair was masked) and no earlier span holds it."""
    value = WHITESPACE_RUN.sub(" ", unicodedata.normalize("NFKC", value)).strip()
    if not value or context + value in output:
        return None
    for text, offsets in views:
        end = bisect_left(offsets, low) + len(context) + len(value) - 1  # the value starts before *low*
        if (at := text.rfind(context + value, 0, end)) >= 0:
            start = offsets[at + len(context)]
            if any(s.start <= start < s.end for s in spans):
                return None
            return start, offsets[at + len(context) + len(value) - 1]
    return None


def _nfkc_view(text: str) -> tuple[str, list[int]]:
    """NFKC form of *text*, plus the source offset of each of its characters.

    Each base character is normalised together with its combining marks, so every
    piece of the view maps back to one exact source range.
    shortcut: Hangul jamo sequences compose across clusters and fall back to -1 spans; fine
    unless a dataset ships decomposed Hangul.
    """
    pieces, offsets, start = [], [], 0
    for i in range(1, len(text) + 1):
        if i < len(text) and unicodedata.combining(text[i]):
            continue
        piece = unicodedata.normalize("NFKC", text[start:i])
        pieces.append(piece)
        offsets.extend([start] * len(piece))
        start = i
    offsets.append(len(text))
    return "".join(pieces), offsets


def predicted_spans(source: str, output: str, originals: dict[str, str],
                    view: tuple[str, list[int]] | None = None, cfg=None) -> list[EntitySpan]:
    """Map each ``<TAG_n>`` placeholder in output order to the source span it replaced.

    SCT may have folded the text (full-width to ASCII) before replacing it, so the
    search runs on the NFKC form of the source and the found range is mapped back
    to source offsets. *view* replaces that NFKC view with another ``(text, source offset
    of each char + [len(source)])`` pair, e.g. one that also decodes entities as SCT does.
    Whitespace runs compare as one space. NER values are searched first in the view without
    what SCT's pre-NER cleanup under *cfg* (default: ``TextCleanerConfig()``) removed ("Ann B Lee"
    holds "Ann Lee"), then in the view itself, where values regex steps took before that cleanup are.
    """
    if cfg is None:
        from sct.config import TextCleanerConfig

        cfg = TextCleanerConfig()
    base, base_offsets = view or _nfkc_view(source)
    views = (_collapse_whitespace(*_sct_cleanup(base, base_offsets, cfg)), _collapse_whitespace(base, base_offsets))
    (clean, clean_offsets), (plain, plain_offsets) = views
    shown = {m.group(0) for m in _NUMBERED_PLACEHOLDER.finditer(output)}
    swallowed: dict[str, list[str]] = {}  # values whose placeholder a wider NER span took in, by tag
    for placeholder, original in originals.items():
        tag = _NUMBERED_PLACEHOLDER.fullmatch(placeholder)
        if tag and placeholder not in shown:
            swallowed.setdefault(tag.group(1), []).append(original)
    output = WHITESPACE_RUN.sub(" ", unicodedata.normalize("NFKC", output))
    spans, cursor, previous_end = [], 0, 0  # cursor: source offset
    for match in _NUMBERED_PLACEHOLDER.finditer(output):
        # Skip past the unchanged output text before this placeholder, so an
        # untouched earlier copy of the value ("Ann met <PERSON_0>") is not reused.
        context = output[previous_end:match.start()]
        previous_end = match.end()
        low = context_at = cursor
        # A cleanup step may have changed the context ("Ann x met" -> "Ann met"):
        # align on its longest suffix still found in the source.
        hit = _align(context, clean, clean_offsets, cursor, len(context))
        flipped, longer_at = False, None  # flipped: the value is searched in both views, earliest hit wins
        if hit and hit[0]:
            # SCT's own cleanup may have kept text the mirror drops (brackets around a value a regex step
            # replaced first): a longer suffix in the uncleaned view wins if it ends earlier, so it never
            # moves the cursor further than the cleaned view would.
            plain_hit = _align(context, plain, plain_offsets, cursor, hit[0])
            if plain_hit and plain_hit[2] < hit[2]:
                hit, flipped = plain_hit, True
            elif plain_hit and plain_hit[2] == hit[2]:
                longer_at = plain_hit[1]  # same end, earlier start: the fallback window stops there
        if hit:
            _, context_at, cursor = hit
            # A context that starts on the space before the cursor: the value lies in that collapsed
            # run (only whitespace or text the mirror dropped), not before it.
            if context_at < low:
                context_at = cursor
            elif longer_at is not None:
                context_at = max(low, longer_at)
        label = _LABEL_ALIASES.get(match.group(1), match.group(1))
        raw = originals.get(match.group(0), "")
        found = _locate(raw, views, cursor, low, context_at, flipped)
        # A "<TAG>" is literal markup ("<HTML>") or a regex step's marker a swallowed entry fills: the
        # reading found earlier wins; swallowed entries are used up only when the marker reading does.
        used: Counter = Counter()

        def fill(m):
            queue = swallowed.get(m.group(1), [])
            if used[m.group(1)] < len(queue):
                used[m.group(1)] += 1
                return queue[used[m.group(1)] - 1]
            return m.group(0)

        expanded = GENERIC_MARKER.sub(fill, raw)
        if expanded != raw:
            alt = _locate(expanded, views, cursor, low, context_at, flipped)
            if found is None or (alt and alt[0] < found[0]):
                found = alt
                for tag, n in used.items():
                    del swallowed[tag][:n]
        if found is None and any(ch.isalnum() for ch in context):  # a bare space or symbol anchors nothing
            found = _recover(context, raw, views, low, output, spans)
        if found is None:
            # The replaced value is not in the source (an earlier step changed it): never matches.
            spans.append(EntitySpan(label, -1, -1))
            continue
        # A piece of the view can be several chars of one source char: end after that char.
        cursor = base_offsets[bisect_right(base_offsets, found[1])]
        spans.append(EntitySpan(label, found[0], cursor))
    return spans


def overlaps(a: dict, b: dict) -> bool:
    return a["start"] < b["end"] and b["start"] < a["end"]


def prepare(record: dict, predicted: list[dict], mapping: dict[str, str]) -> tuple[list[dict], list[dict]]:
    """Return ``(gold, predictions)`` restricted to the dataset's scored SCT types."""
    gold, out_of_scope = normalise_gold(record["text"], record["spans"], mapping)
    scope = set(mapping.values())
    kept = []
    for span in predicted:
        span = {**span, "label": PRED_ALIASES.get(span["label"], span["label"])}
        if span["label"] not in scope:
            continue
        if any(overlaps(span, o) for o in out_of_scope) and not any(overlaps(span, g) for g in gold):
            continue
        kept.append(span)
    return gold, kept


def leak(value: str, output: str) -> str | None:
    """``"full"`` if *value* is still in *output*, ``"partial"`` if a long word of it is, else None."""
    # NFKC on both sides: SCT folds full-width text, so raw "Ａｌｉｃｅ" shows up as "Alice".
    value = unicodedata.normalize("NFKC", value)
    visible = unicodedata.normalize("NFKC", PLACEHOLDER.sub("\0", output))  # placeholder names are not leaks
    if value in visible:
        return "full"
    if any(len(word) >= LEAK_MIN_CHARS and word in visible for word in LEAK_WORD.findall(value)):
        return "partial"
    return None


def _ratio(part: int, whole: int) -> float:
    return round(part / whole, 4) if whole else 0.0


def _partial_metrics(c: Counter) -> dict:
    p, r = _ratio(c["pred_correct"], c["pred"]), _ratio(c["gold_found"], c["gold"])
    counts = {k: c[k] for k in ("gold", "gold_found", "gold_masked", "pred", "pred_correct")}
    return {**counts, "precision": p, "recall": r, "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0,
            "masked_recall": _ratio(c["gold_masked"], c["gold"])}


def _leak_metrics(c: Counter) -> dict:
    return {**{k: c[k] for k in ("total", "full", "partial")}, "leak_rate": _ratio(c["full"] + c["partial"], c["total"]),
            "full_leak_rate": _ratio(c["full"], c["total"])}


def score_pii(records: list[dict], predictions: dict[str, list[dict]], outputs: dict[str, str],
              mapping: dict[str, str]) -> dict[str, Any]:
    """Score one dataset under one config. *predictions* and *outputs* are keyed by record id."""
    examples, strict_preds = [], {}
    partial: dict[str, dict[str, Counter]] = {"overall": defaultdict(Counter), "by_label": defaultdict(Counter),
                                              "by_language": defaultdict(Counter)}
    leaks: dict[str, dict[str, Counter]] = {"overall": defaultdict(Counter), "by_label": defaultdict(Counter),
                                            "by_language": defaultdict(Counter)}
    position: dict[str, Counter] = defaultdict(Counter)
    out_of_scope = 0
    for record in records:
        gold, preds = prepare(record, predictions[record["id"]], mapping)
        out_of_scope += len(record["spans"]) - len(gold)
        lang, output, text = record["lang"], outputs[record["id"]], record["text"]
        examples.append(EvaluationExample(
            id=record["id"], text=text, language=lang, domain="all",
            expected=tuple(EntitySpan(g["label"], g["start"], g["end"]) for g in gold)))
        strict_preds[record["id"]] = tuple(EntitySpan(p["label"], p["start"], p["end"]) for p in preds)
        for g in gold:
            found = any(p["label"] == g["label"] and overlaps(p, g) for p in preds)
            masked = any(overlaps(p, g) for p in preds)
            for group, key in (("overall", "all"), ("by_label", g["label"]), ("by_language", lang)):
                partial[group][key]["gold"] += 1
                partial[group][key]["gold_found"] += found
                partial[group][key]["gold_masked"] += masked
            bucket = min(g["start"] // POSITION_BUCKET, POSITION_BUCKETS - 1)
            name = f"{bucket * POSITION_BUCKET}+" if bucket == POSITION_BUCKETS - 1 else (
                f"{bucket * POSITION_BUCKET}-{(bucket + 1) * POSITION_BUCKET}")
            position[name]["gold"] += 1
            position[name]["gold_found"] += found
            kind = leak(text[g["start"]:g["end"]], output)
            groups = [("by_label", g["label"])]
            if g["label"] not in LEAK_EXCLUDED:
                groups += [("by_language", lang), ("overall", "all")]
            for group, key in groups:
                leaks[group][key]["total"] += 1
                if kind:
                    leaks[group][key][kind] += 1
        for p in preds:
            correct = any(g["label"] == p["label"] and overlaps(p, g) for g in gold)
            for group, key in (("overall", "all"), ("by_label", p["label"]), ("by_language", lang)):
                partial[group][key]["pred"] += 1
                partial[group][key]["pred_correct"] += correct
    strict = evaluate_pii(examples, strict_preds, model="sct").to_dict()
    return {
        "documents": len(records),
        "chars": sum(len(r["text"]) for r in records),
        "gold_spans": sum(len(e.expected) for e in examples),
        "out_of_scope_gold_spans": out_of_scope,
        "strict": {k: strict[k] for k in ("overall", "by_label", "by_language")},
        "partial": {
            "overall": _partial_metrics(partial["overall"]["all"]),
            **{g: {k: _partial_metrics(v) for k, v in sorted(partial[g].items())} for g in ("by_label", "by_language")},
        },
        "leak": {
            "overall": _leak_metrics(leaks["overall"]["all"]),
            **{g: {k: _leak_metrics(v) for k, v in sorted(leaks[g].items())} for g in ("by_label", "by_language")},
        },
        "by_position": {k: {**v, "recall": _ratio(v["gold_found"], v["gold"])}
                        for k, v in sorted(position.items(), key=lambda kv: int(kv[0].split("-")[0].rstrip("+")))},
    }


def masking_recall(records: list[dict], predictions: dict[str, list[dict]]) -> dict[str, Any]:
    """TAB-style recall: share of DIRECT / QUASI mentions whose letters and digits are all inside predicted spans.

    Type does not matter (masking is what counts); a leftover ``/`` or space reveals nothing. Recall is computed per annotator
    over all their documents, then averaged over annotators (``macro``); ``micro``
    pools every annotated mention.
    """
    per: dict[str, Counter] = defaultdict(Counter)
    for record in records:
        masked = bytearray(len(record["text"]))
        for p in predictions[record["id"]]:
            if p["start"] >= 0:
                masked[p["start"]:p["end"]] = b"\1" * (p["end"] - p["start"])
        for annotator, mentions in record.get("annotators", {}).items():
            for m in mentions:
                if m["identifier"] not in ("DIRECT", "QUASI"):
                    continue
                covered = all(masked[i] or not record["text"][i].isalnum() for i in range(m["start"], m["end"]))
                for key in ("all", m["identifier"].lower()):
                    per[annotator][f"{key}_total"] += 1
                    per[annotator][f"{key}_masked"] += covered
    result = {}
    for key in ("all", "direct", "quasi"):
        rates = [_ratio(c[f"{key}_masked"], c[f"{key}_total"]) for c in per.values() if c[f"{key}_total"]]
        total = sum(c[f"{key}_total"] for c in per.values())
        result[key] = {
            "macro_recall": round(sum(rates) / len(rates), 4) if rates else 0.0,
            "micro_recall": _ratio(sum(c[f"{key}_masked"] for c in per.values()), total),
            "mentions": total,
        }
    result["annotators"] = len(per)
    return result


def dominant_script(text: str) -> str:
    """Most common Unicode script word (LATIN, CYRILLIC, CJK, ...) among the letters of *text*."""
    counts = Counter(unicodedata.name(ch, "UNKNOWN").split()[0] for ch in text if ch.isalpha())
    return counts.most_common(1)[0][0] if counts else "NONE"


def score_lid(records: list[dict], predicted: dict[str, str | None], in_set: set[str]) -> dict[str, Any]:
    """Language ID: accuracy per language and script for in-set languages; any answer for out-of-set text."""
    by_lang: dict[str, Counter] = defaultdict(Counter)
    by_script: dict[str, Counter] = defaultdict(Counter)
    confusions: Counter = Counter()
    out_total = out_answered = 0
    for record in records:
        gold, guess = record["lang"], predicted[record["id"]]
        if gold not in in_set:
            out_total += 1
            out_answered += guess is not None
            continue
        ok = guess == gold
        for table, key in ((by_lang, gold), (by_script, dominant_script(record["text"]))):
            table[key]["total"] += 1
            table[key]["correct"] += ok
            table[key]["unknown"] += guess is None
        if not ok:
            confusions[f"{gold}->{guess}"] += 1
    total = sum(c["total"] for c in by_lang.values())
    correct = sum(c["correct"] for c in by_lang.values())
    accuracies = [_ratio(c["correct"], c["total"]) for c in by_lang.values()]
    return {
        "in_set_documents": total,
        "accuracy": _ratio(correct, total),
        "macro_accuracy": round(sum(accuracies) / len(accuracies), 4) if accuracies else 0.0,
        "by_language": {k: {**v, "accuracy": _ratio(v["correct"], v["total"])} for k, v in sorted(by_lang.items())},
        "by_script": {k: {**v, "accuracy": _ratio(v["correct"], v["total"])} for k, v in sorted(by_script.items())},
        "top_confusions": dict(confusions.most_common(25)),
        "out_of_set": {"documents": out_total, "answered": out_answered,
                       "false_positive_rate": _ratio(out_answered, out_total)},
    }


NOISE_EMOJI = ("\U0001F642", "\U0001F680", "\u2764\ufe0f", "\U0001F44D\U0001F3FD")
NOISE_HTML = ('<div class="note">', "<br/>", "<span style=\"color:red\">", "</p>")


def inject_noise(text: str, rng) -> tuple[str, list[dict]]:
    """Insert one HTML tag, one emoji, one URL and one mojibake word at random word gaps.

    Mojibake is a word with non-ASCII letters, UTF-8 bytes read as Latin-1 (``café`` ->
    ``cafÃ©``); a document with no such word gets ``café`` as the word.
    """
    words = text.split(" ")
    accented = [w for w in words if any(ord(c) > 127 and c.isalpha() for c in w)]
    original = rng.choice(accented) if accented else "café"
    items = [
        {"kind": "html", "value": rng.choice(NOISE_HTML)},
        {"kind": "emoji", "value": rng.choice(NOISE_EMOJI)},
        {"kind": "url", "value": f"https://example.com/p/{rng.randrange(10**6)}?ref=bench"},
        {"kind": "mojibake", "value": original.encode("utf-8").decode("latin-1")},
    ]
    for item in items:
        words.insert(rng.randrange(len(words) + 1), item["value"])
    return " ".join(words), items


def score_noise(items: dict[str, list[dict]], outputs: dict[str, str]) -> dict[str, dict[str, float]]:
    """Share of injected noise items (by kind) no longer in the output."""
    counts: dict[str, Counter] = defaultdict(Counter)
    for doc_id, doc_items in items.items():
        for item in doc_items:
            counts[item["kind"]]["injected"] += 1
            counts[item["kind"]]["removed"] += item["value"] not in outputs[doc_id]
    return {k: {**v, "removal_rate": _ratio(v["removed"], v["injected"])} for k, v in sorted(counts.items())}

