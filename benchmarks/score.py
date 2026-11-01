"""Scorers. They see only spans, outputs and counts; results never hold dataset text.

PII (per language and per SCT type):

- **strict**: exact label + span match, via ``sct.evaluation.evaluate_pii``.
- **partial**: a gold span is found when a predicted span of the same type overlaps
  it; a prediction is correct when it overlaps a gold span of the same type.
  Recall and precision use these two counts (not a one-to-one matching).
- **leak**: a gold PII value is *fully* leaked when it is still in the output, and
  *partly* leaked when a 4+ char word of it is still outside the placeholders.
  ``leak_rate`` counts both. It is an upper bound: a word that also occurs elsewhere
  in the document counts as a leak. DATE is reported per type but not in the total.
- **by_position**: partial recall by the gold span's start offset in 1000-char
  buckets, so recall lost after the first model window shows up.

Predictions only count when their (aliased) type is one the dataset scores, and a
prediction that only overlaps out-of-scope gold is ignored.
"""

import re
import unicodedata
from collections import Counter, defaultdict

from sct.evaluation import EntitySpan, EvaluationExample, evaluate_pii

from benchmarks.labels import DATE, PRED_ALIASES, normalise_gold

PLACEHOLDER = re.compile(r"<[A-Z][A-Z_]*(?:_\d+)?>")
LEAK_MIN_CHARS = 4
LEAK_EXCLUDED = frozenset({DATE})
POSITION_BUCKET = 1000
POSITION_BUCKETS = 10


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
    visible = PLACEHOLDER.sub("\0", output)  # placeholder names ("PERSON") are not leaks
    if value in visible:
        return "full"
    if any(len(word) >= LEAK_MIN_CHARS and word in visible for word in re.findall(r"\w+", value)):
        return "partial"
    return None


def _ratio(part: int, whole: int) -> float:
    return round(part / whole, 4) if whole else 0.0


def _partial_metrics(c: Counter) -> dict:
    p, r = _ratio(c["pred_correct"], c["pred"]), _ratio(c["gold_found"], c["gold"])
    counts = {k: c[k] for k in ("gold", "gold_found", "pred", "pred_correct")}
    return {**counts, "precision": p, "recall": r, "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0}


def _leak_metrics(c: Counter) -> dict:
    return {**{k: c[k] for k in ("total", "full", "partial")}, "leak_rate": _ratio(c["full"] + c["partial"], c["total"]),
            "full_leak_rate": _ratio(c["full"], c["total"])}


def score_pii(records: list[dict], predictions: dict[str, list[dict]], outputs: dict[str, str],
              mapping: dict[str, str]) -> dict:
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
            for group, key in (("overall", "all"), ("by_label", g["label"]), ("by_language", lang)):
                partial[group][key]["gold"] += 1
                partial[group][key]["gold_found"] += found
            bucket = min(g["start"] // POSITION_BUCKET, POSITION_BUCKETS - 1)
            name = f"{bucket * POSITION_BUCKET}+" if bucket == POSITION_BUCKETS - 1 else (
                f"{bucket * POSITION_BUCKET}-{(bucket + 1) * POSITION_BUCKET}")
            position[name]["gold"] += 1
            position[name]["gold_found"] += found
            kind = leak(text[g["start"]:g["end"]], output)
            groups = [("by_label", g["label"]), ("by_language", lang)]
            if g["label"] not in LEAK_EXCLUDED:
                groups.append(("overall", "all"))
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


def masking_recall(records: list[dict], predictions: dict[str, list[dict]]) -> dict:
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


def score_lid(records: list[dict], predicted: dict[str, str | None], in_set: set[str]) -> dict:
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


def score_noise(items: dict[str, list[dict]], outputs: dict[str, str]) -> dict:
    """Share of injected noise items (by kind) no longer in the output."""
    counts: dict[str, Counter] = defaultdict(Counter)
    for doc_id, doc_items in items.items():
        for item in doc_items:
            counts[item["kind"]]["injected"] += 1
            counts[item["kind"]]["removed"] += item["value"] not in outputs[doc_id]
    return {k: {**v, "removal_rate": _ratio(v["removed"], v["injected"])} for k, v in sorted(counts.items())}

