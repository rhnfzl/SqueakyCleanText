"""Comparison scoring on top of ``benchmarks.score`` (PROTOCOL.md sections 4-10, 13-15).

Every tool's output goes through the same steps: map tool labels to SCT types (unmapped =
out of scope, counted), optional score threshold, drop empty spans, merge same-type
overlapping spans into their union. The common mask then replaces each merged region with
``<TYPE>`` (the detection track); a tool's own ``masked_text`` is only scored for leaks
(the native track).

Per document, ``doc_stats`` returns additive counts (no text), so the same counts give the
per-language, macro and micro scores, the threshold curves and the paired bootstrap.
DATE is out of every total (``with_date`` keeps it). A metric with an empty denominator is
None (N/A), never 0.

- strict: exact span + type, one-to-one (multiset).
- partial: same type, >= 1 char overlap, one-to-one: pairs taken by largest overlap, then
  earliest start. ``score_pii`` (many-to-one partial, word leaks, by_position) is still
  reported next to it.
- masking set (one per doc, type-agnostic): every mapped prediction except those that only
  touch out-of-view gold. Masked recall, char scores and leaks all read it, so a gold PERSON
  masked as NUMBER counts as masked everywhere, never masked in one metric and leaked in another.
- masked recall: gold mentions overlapped by any prediction of the masking set.
- masked precision / masked F2 (the dev tuning objective, PROTOCOL.md section 3): masking-set
  predictions that overlap any in-view gold / all masking-set predictions; F2 of that and masked recall.
- char: type-agnostic per-character P/R/F1/F2 over in-scope gold characters (masking set).
- F scores come from counts (``(1+b2)TP / (b2*gold + pred)``): 0 when gold or predictions exist
  but nothing matches, None only when both are empty.
- leaks, per gold mention on the common mask: ``exposed`` (a letter or digit of it is not
  masked; also the doc leak rule), ``complete_exposure`` (no char of it masked),
  ``char_survival`` (unmasked letters/digits / letters/digits), ``exact_visible`` (the NFKC
  value is still in the output anywhere), ``repeated_exposed`` (the value occurs more than once
  in the doc and an occurrence keeps a letter/digit).
- over-masking: masked chars outside every gold span (any label, in scope or not) / non-gold
  chars; masked chars / gold chars; clean-doc false positives (docs with no gold span at all
  that get any prediction).
- utility: corpus chrF (char n-grams 1-6, beta 2, whitespace removed) of the common-masked
  text against the gold-masked text.
"""

import math
from collections import Counter, defaultdict

import numpy as np

from benchmarks.labels import DATE, normalise_gold
from benchmarks.score import leak, masking_recall, overlaps, prepare, score_pii

THRESHOLDS = tuple(round(0.05 * i, 2) for i in range(1, 20))
CHRF_ORDER = 6
CHRF_BETA = 2
BOOTSTRAP_SAMPLES = 1000
SEED = 13
TUNING = ("masked_f2", "masked_recall", "masked_precision", "char_f2", "doc_leak_rate", "overmask_rate")
HEADLINE = ("masked_recall", "masked_precision", "partial_f1", "strict_f1", "char_f2", "char_survival", "doc_leak_rate",
            "overmask_rate")
# PROTOCOL.md sections 1 and 13: never pooled across these groups.
GROUPS = {
    "privacy-synthetic": ("openpii", "gretel", "gretel_en", "nemotron", "argilla_pii", "thai_pii", "bayanat"),
    "privacy-real": ("tab", "redact"),
    "ner": ("iahlt", "thainer", "naamapadam", "multiconer", "stockmark"),
}


def map_spans(spans: list[dict], labels: dict[str, str], threshold: float | None = None) -> tuple[list[dict], int]:
    """Tool spans -> SCT-typed, thresholded, cleaned spans; also the count of out-of-scope tool labels."""
    mapped, out_of_scope = [], 0
    for s in spans:
        label = labels.get(s["label"])
        if label is None:
            out_of_scope += 1
        elif s["end"] > s["start"] and (threshold is None or s.get("score") is None or s["score"] >= threshold):
            mapped.append({"start": s["start"], "end": s["end"], "label": label, "score": s.get("score")})
    return merge_same_type(mapped), out_of_scope


def merge_same_type(spans: list[dict]) -> list[dict]:
    """Union of overlapping spans of one type (score = max); spans of different types both stay."""
    merged: list[dict] = []
    for s in sorted(spans, key=lambda s: (s["label"], s["start"], s["end"])):
        last = merged[-1] if merged else None
        if last and last["label"] == s["label"] and s["start"] < last["end"]:
            last["end"] = max(last["end"], s["end"])
            if s["score"] is not None:
                last["score"] = s["score"] if last["score"] is None else max(last["score"], s["score"])
        else:
            merged.append(dict(s))
    return sorted(merged, key=lambda s: (s["start"], s["end"]))


def recover_dropped(text: str, masked_text: str, spans: list[dict]) -> tuple[list[dict], int]:
    """Add the source copies of a masked value that the tool's own output no longer holds anywhere.

    SCT spans come from placeholders found in its output; a placeholder a later cleaning step removed
    (the link text of ``[a@b.se](mailto:a@b.se)`` becomes ``<EMAIL_0>`` and is then dropped) leaves its
    source copy unmapped, so the copy scored as leaked although no character of it is visible. A copy is
    added only when neither the value nor any 3+ char word of it is in the output (``leak`` finds nothing,
    so no copy can still be visible, even in part) and it is a whole token (no letter or digit glued on).
    Returns the merged spans and the number of copies added.
    """
    extra = []
    for value, label in {(text[s["start"]:s["end"]], s["label"]) for s in spans}:
        if len(value.strip()) < 4 or leak(value, masked_text) or leak(" ".join(value.split()), masked_text):
            continue
        at = text.find(value)
        while at >= 0:
            end = at + len(value)
            if not (at and text[at - 1].isalnum()) and not (end < len(text) and text[end].isalnum()):
                extra.append({"start": at, "end": end, "label": label, "score": None})
            at = text.find(value, at + 1)
    if not extra:
        return spans, 0
    before = covered(len(text), spans)
    added = sum(1 for e in extra if not all(before[e["start"]:e["end"]]))
    return merge_same_type(spans + extra), added


def covered(length: int, spans: list[dict]) -> bytearray:
    flags = bytearray(length)
    for s in spans:
        if 0 <= s["start"] < s["end"]:
            flags[s["start"]:s["end"]] = b"\1" * (min(s["end"], length) - s["start"])
    return flags


def mask_text(text: str, spans: list[dict]) -> str:
    """Common mask: each region covered by spans (any type) becomes ``<TYPE>`` of its first span."""
    regions: list[list] = []
    for s in sorted(spans, key=lambda s: (s["start"], -s["end"])):
        if regions and s["start"] < regions[-1][1]:
            regions[-1][1] = max(regions[-1][1], s["end"])
        else:
            regions.append([s["start"], s["end"], s["label"]])
    parts, cursor = [], 0
    for start, end, label in regions:
        parts += [text[cursor:start], f"<{label}>"]
        cursor = end
    parts.append(text[cursor:])
    return "".join(parts)


def _ngrams(text: str, n: int) -> Counter:
    text = "".join(text.split())
    return Counter(text[i:i + n] for i in range(len(text) - n + 1))


def _one_to_one(gold: list[dict], preds: list[dict]) -> int:
    pairs = sorted(((min(g["end"], p["end"]) - max(g["start"], p["start"]), p["start"], gi, pi)
                    for gi, g in enumerate(gold) for pi, p in enumerate(preds)
                    if g["label"] == p["label"] and overlaps(g, p)), key=lambda x: (-x[0], x[1]))
    used_g, used_p = set(), set()
    for _, _, gi, pi in pairs:
        if gi not in used_g and pi not in used_p:
            used_g.add(gi)
            used_p.add(pi)
    return len(used_g)


def doc_stats(record: dict, spans: list[dict], mapping: dict[str, str], unplaced: int = 0,
              utility: bool = True) -> Counter:
    """Additive counts for one doc. *spans* are mapped + cleaned; *mapping* is the view's dataset map."""
    text = record["text"]
    gold, kept = prepare(record, spans, mapping)
    _, out_of_view = normalise_gold(text, record["spans"], mapping)
    masking = [p for p in spans if any(overlaps(p, g) for g in gold) or not any(overlaps(p, o) for o in out_of_view)]
    c = Counter(docs=1, gold=len(gold), pred=len(kept) + unplaced)
    c["strict_tp"] = sum((Counter((g["label"], g["start"], g["end"]) for g in gold)
                          & Counter((p["label"], p["start"], p["end"]) for p in kept)).values())
    c["partial_tp"] = _one_to_one(gold, kept)
    c["gold_masked"] = sum(any(overlaps(p, g) for p in masking) for g in gold)
    c["masked_pred"] = len(masking)
    c["masked_pred_hit"] = sum(any(overlaps(p, g) for g in gold) for p in masking)
    gold_chars, pred_chars = covered(len(text), gold), covered(len(text), masking)
    c["char_gold"], c["char_pred"] = sum(gold_chars), sum(pred_chars)
    c["char_tp"] = sum(1 for a, b in zip(gold_chars, pred_chars) if a and b)
    masked = covered(len(text), spans)
    visible = mask_text(text, spans)
    exposed_doc = False
    for g in gold:
        alnum = [i for i in range(g["start"], g["end"]) if text[i].isalnum()]
        survived = sum(1 for i in alnum if not masked[i])
        value = text[g["start"]:g["end"]]
        c["mentions"] += 1
        c["alnum"] += len(alnum)
        c["alnum_survived"] += survived
        c["exposed"] += survived > 0
        c["complete_exposure"] += not any(masked[g["start"]:g["end"]])
        c["exact_visible"] += leak(value, visible) == "full"
        starts, at = [], text.find(value)
        while at >= 0:
            starts.append(at)
            at = text.find(value, at + 1)
        if len(starts) > 1:
            c["repeated"] += 1
            c["repeated_exposed"] += any(not masked[i] and text[i].isalnum()
                                         for o in starts for i in range(o, o + len(value)))
        exposed_doc |= survived > 0
    c["docs_with_gold"] += bool(gold)
    c["docs_exposed"] += exposed_doc
    all_gold = covered(len(text), record["spans"])
    c["chars"] = len(text)
    c["gold_chars_all"] = sum(all_gold)
    c["masked_chars"] = sum(masked)
    c["nongold_masked"] = sum(1 for m, g in zip(masked, all_gold) if m and not g)
    c["clean_docs"] += not record["spans"]
    c["clean_docs_fp"] += not record["spans"] and bool(spans)
    if utility:
        reference = mask_text(text, gold)
        for n in range(1, CHRF_ORDER + 1):
            hyp, ref = _ngrams(visible, n), _ngrams(reference, n)
            c[f"chrf{n}_match"] += sum((hyp & ref).values())
            c[f"chrf{n}_hyp"] += sum(hyp.values())
            c[f"chrf{n}_ref"] += sum(ref.values())
    return c


def _r(part, whole):
    return part / whole if whole else None


def _round(value):
    return None if value is None else round(value, 4)


def _f(p, r, beta=1.0):
    if p is None or r is None:
        return None
    b2 = beta * beta
    return (1 + b2) * p * r / (b2 * p + r) if p + r else 0.0


def _fc(tp, pred, gold, beta=1.0):
    """F-beta from counts: 0 when only one side is empty (a full miss is a 0, not N/A)."""
    b2 = beta * beta
    return (1 + b2) * tp / (b2 * gold + pred) if gold or pred else None


def metrics(c: Counter) -> dict:
    sp, sr = _r(c["strict_tp"], c["pred"]), _r(c["strict_tp"], c["gold"])
    pp, pr = _r(c["partial_tp"], c["pred"]), _r(c["partial_tp"], c["gold"])
    cp, cr = _r(c["char_tp"], c["char_pred"]), _r(c["char_tp"], c["char_gold"])
    out = {
        "docs": c["docs"], "gold": c["gold"], "pred": c["pred"],
        "strict_precision": sp, "strict_recall": sr, "strict_f1": _fc(c["strict_tp"], c["pred"], c["gold"]),
        "partial_precision": pp, "partial_recall": pr, "partial_f1": _fc(c["partial_tp"], c["pred"], c["gold"]),
        "masked_recall": _r(c["gold_masked"], c["gold"]),
        "masked_precision": _r(c["masked_pred_hit"], c["masked_pred"]),
        "masked_f2": _f(_r(c["masked_pred_hit"], c["masked_pred"]) or 0.0, _r(c["gold_masked"], c["gold"]) or 0.0, 2)
        if c["gold"] or c["masked_pred"] else None,
        "char_precision": cp, "char_recall": cr, "char_f1": _fc(c["char_tp"], c["char_pred"], c["char_gold"]),
        "char_f2": _fc(c["char_tp"], c["char_pred"], c["char_gold"], 2),
        "char_survival": _r(c["alnum_survived"], c["alnum"]),
        "mention_exposure_rate": _r(c["exposed"], c["mentions"]),
        "complete_exposure_rate": _r(c["complete_exposure"], c["mentions"]),
        "exact_visible_rate": _r(c["exact_visible"], c["mentions"]),
        "repeated_values": c["repeated"], "repeated_exposed_rate": _r(c["repeated_exposed"], c["repeated"]),
        "doc_leak_rate": _r(c["docs_exposed"], c["docs_with_gold"]),
        "overmask_rate": _r(c["nongold_masked"], c["chars"] - c["gold_chars_all"]),
        "mask_to_gold_ratio": _r(c["masked_chars"], c["gold_chars_all"]),
        "clean_docs": c["clean_docs"], "clean_doc_fp_rate": _r(c["clean_docs_fp"], c["clean_docs"]),
    }
    orders = [n for n in range(1, CHRF_ORDER + 1) if c[f"chrf{n}_hyp"] and c[f"chrf{n}_ref"]]
    if orders:
        p = sum(c[f"chrf{n}_match"] / c[f"chrf{n}_hyp"] for n in orders) / len(orders)
        r = sum(c[f"chrf{n}_match"] / c[f"chrf{n}_ref"] for n in orders) / len(orders)
        out["chrf"] = _f(p, r, CHRF_BETA)
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in out.items()}


def total(counters) -> Counter:
    out: Counter = Counter()
    for c in counters:
        out.update(c)
    return out


def _mean(values) -> float | None:
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 4) if values else None


def macro(scores: list[dict]) -> dict:
    """Mean of each metric over *scores* (None skipped)."""
    keys = {k for s in scores for k, v in s.items() if isinstance(v, float) or v is None}
    return {k: _mean(s.get(k) for s in scores) for k in sorted(keys)}


def summarise(docs: list[tuple[str, Counter]]) -> dict:
    """One dataset: per-language metrics, macro over languages, micro over docs."""
    by_lang: dict[str, list] = defaultdict(list)
    for lang, c in docs:
        by_lang[lang].append(c)
    per_lang = {lang: metrics(total(cs)) for lang, cs in sorted(by_lang.items())}
    return {"micro": metrics(total(c for _, c in docs)), "macro": macro(list(per_lang.values())),
            "by_language": per_lang}


def aggregate(datasets: dict[str, list[tuple[str, Counter]]]) -> dict:
    """Across datasets: macro = mean of each dataset's language-macro; micro = pooled counts."""
    return {"datasets": sorted(datasets),
            "macro": macro([summarise(docs)["macro"] for docs in datasets.values()]),
            "micro": metrics(total(c for docs in datasets.values() for _, c in docs))}


def curve(records: list[dict], outputs: dict[str, dict], labels: dict[str, str], mapping: dict[str, str],
          floor: float | None = None) -> list[dict]:
    """Threshold sweep by filtering one run (completed docs only); [] when the tool gives no scores.

    *floor* is the threshold the adapter already applied: points below it were never decoded,
    so they are left out (a full curve needs a dev run with the adapter threshold at 0.05).
    """
    if not any(s.get("score") is not None for o in outputs.values() for s in o["spans"]):
        return []
    rows = []
    for t in THRESHOLDS:
        if floor is not None and t < floor:
            continue
        c = total(doc_stats(r, map_spans(outputs[r["id"]]["spans"], labels, t)[0], mapping,
                            outputs[r["id"]].get("unplaced", 0), utility=False)
                  for r in records if "error" not in outputs[r["id"]])
        m = metrics(c)
        rows.append({"threshold": t, **{k: m[k] for k in ("masked_recall", "partial_precision", "partial_recall",
                                                           "char_survival", "doc_leak_rate", "overmask_rate")}})
    return rows


def tuning_rows(records: list[dict], outputs: dict[str, dict], labels: dict[str, str], mapping: dict[str, str],
                thresholds=(None,)) -> list[dict]:
    """Dev tuning metrics per threshold: language macro over ALL docs, failed docs unmasked.

    ``None`` is the run as decoded (no filter). Used by ``tune.py`` to pick one configuration.
    """
    rows = []
    for t in thresholds:
        docs = [(r["lang"], doc_stats(r, map_spans([] if "error" in outputs[r["id"]] else outputs[r["id"]]["spans"],
                                                   labels, t)[0], mapping, outputs[r["id"]].get("unplaced", 0),
                                      utility=False)) for r in records]
        m = summarise(docs)["macro"]
        rows.append({"threshold": t, **{k: m.get(k) for k in TUNING}})
    return rows


def operating_point(rows: list[dict], target_recall: float) -> float | None:
    """Highest threshold whose masked recall reaches *target_recall* (dev only), else None."""
    ok = [row["threshold"] for row in rows if (row["masked_recall"] or 0) >= target_recall]
    return max(ok) if ok else None


def failure_kind(error: str) -> str:
    return error.split(":", 1)[0].strip()[:60]


def native_leak(records: list[dict], outputs: dict[str, dict], mapping: dict[str, str]) -> dict | None:
    """Native track: leaks in the tool's own ``masked_text`` (no offsets, so leak measures only)."""
    c: Counter = Counter()
    for r in records:
        out = outputs[r["id"]]
        if "masked_text" not in out or "error" in out:
            continue
        gold, _ = prepare(r, [], {k: v for k, v in mapping.items() if v != DATE})
        c["docs"] += 1
        for g in gold:
            kind = leak(r["text"][g["start"]:g["end"]], out["masked_text"])
            c["mentions"] += 1
            c[kind or "none"] += 1
    if not c["docs"]:
        return None
    return {"docs": c["docs"], "mentions": c["mentions"], "full_leak_rate": _round(_r(c["full"], c["mentions"])),
            "leak_rate": _round(_r(c["full"] + c["partial"], c["mentions"]))}


def score_run(records: list[dict], outputs: dict[str, dict], labels: dict[str, str], mapping: dict[str, str],
              views: dict[str, tuple], curve_view: str = "shared", curve_floor: float | None = None,
              curve_by_filter: bool = True, recover: bool = False) -> tuple[dict, dict]:
    """Score one entry on one dataset part. Returns ``(result, per_doc)``.

    *outputs* maps every record id to its adapter line. *views* maps a view name to its SCT
    types. ``per_doc[view]`` is ``[(id, group, lang, Counter)]`` over ALL docs with failed docs
    scored as unmasked (effective protection): the input of ``paired_bootstrap``.
    *curve_floor* is the threshold the adapter ran at; ``curve_by_filter=False`` (scores that
    are not the decoder's operating point, e.g. OPF bias presets) reports no curve. *recover* adds the
    copies the tool removed from its own output (``recover_dropped``; SCT entries).
    """
    failed = {r["id"] for r in records if "error" in outputs[r["id"]]}
    completed = [r for r in records if r["id"] not in failed]
    mapped, out_of_scope, recovered = {}, 0, 0
    for r in records:
        spans, dropped = map_spans([] if r["id"] in failed else outputs[r["id"]]["spans"], labels)
        if recover and r["id"] not in failed and "masked_text" in outputs[r["id"]]:
            spans, added = recover_dropped(r["text"], outputs[r["id"]]["masked_text"], spans)
            recovered += added
        mapped[r["id"]], out_of_scope = spans, out_of_scope + dropped
    result = {
        "documents": len(records), "completed": len(completed), "completion_rate": _round(_r(len(completed), len(records))),
        "flag_low_completion": len(completed) < 0.99 * len(records),
        "failure_kinds": dict(Counter(failure_kind(outputs[i]["error"]) for i in failed)),
        "out_of_scope_tool_spans": out_of_scope,
        "unplaced_spans": sum(outputs[r["id"]].get("unplaced", 0) for r in records),
        "recovered_dropped_copies": recovered if recover else None,
        "views": {},
    }
    per_doc = {}
    for view, types in views.items():
        with_date = {k: v for k, v in mapping.items() if v in types}
        view_map = {k: v for k, v in with_date.items() if v != DATE}
        docs = [(r, doc_stats(r, mapped[r["id"]], view_map, outputs[r["id"]].get("unplaced", 0)))
                for r in records]
        per_doc[view] = [(r["id"], str(r.get("group", r["id"])), r["lang"], c) for r, c in docs]
        done = [(r["lang"], c) for r, c in docs if r["id"] not in failed]
        result["views"][view] = {
            "types": sorted(types),
            "compare": summarise(done),
            "effective": metrics(total(c for _, c in docs)),
            "with_date": metrics(total(doc_stats(r, mapped[r["id"]], with_date, outputs[r["id"]].get("unplaced", 0),
                                                 utility=False) for r in completed)),
            "pii": score_pii(completed, {r["id"]: mapped[r["id"]] for r in completed},
                             {r["id"]: mask_text(r["text"], mapped[r["id"]]) for r in completed}, view_map),
        }
    if curve_view in views:
        view_map = {k: v for k, v in mapping.items() if v in views[curve_view] and v != DATE}
        result["curve"] = curve(completed, outputs, labels, view_map, curve_floor) if curve_by_filter else None
    if "full" in views:  # tuning objective: full taxonomy minus DATE, so a tool is tuned on everything it must mask
        view_map = {k: v for k, v in mapping.items() if v in views["full"] and v != DATE}
        scored = curve_by_filter and any(s.get("score") is not None for r in completed for s in outputs[r["id"]]["spans"])
        grid = [None, *(t for t in THRESHOLDS if curve_floor is None or t >= curve_floor)] if scored else [None]
        result["tuning"] = tuning_rows(records, outputs, labels, view_map, grid)
    if any(r.get("annotators") for r in records):  # TAB: DIRECT/QUASI full-coverage rule, every annotator
        result["tab_masking"] = masking_recall(records, mapped)  # failed docs have no spans: unmasked
    result["native"] = native_leak(records, outputs, mapping)
    return result, per_doc


def mcnemar(a: list[bool], b: list[bool]) -> dict:
    """Exact two-sided McNemar test on paired per-doc flags."""
    only_a = sum(1 for x, y in zip(a, b, strict=True) if x and not y)
    only_b = sum(1 for x, y in zip(a, b, strict=True) if y and not x)
    n = only_a + only_b
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, k) for k in range(min(only_a, only_b) + 1)) / 2**n)
    return {"only_a": only_a, "only_b": only_b, "p_value": round(p, 6)}


def paired_bootstrap(a: list[tuple], b: list[tuple], names=HEADLINE, samples: int = BOOTSTRAP_SAMPLES,
                     seed: int = SEED) -> dict:
    """Paired bootstrap over groups (template or doc) for entry A minus entry B on the same docs.

    Each draw recomputes the language macro (the headline), not pooled micro counts.

    *a* and *b* are ``per_doc`` lists from ``score_run`` for the same records in the same order.
    A difference counts only when its 95% CI excludes 0 (``excludes_zero``); the metric decides
    whether higher or lower is better.
    """
    out = macro_bootstrap({"": (a, b)}, names, samples, seed)
    out["doc_leak_mcnemar"] = mcnemar([x[3]["docs_exposed"] > 0 for x in a], [x[3]["docs_exposed"] > 0 for x in b])
    return out


def macro_bootstrap(pairs: dict[str, tuple[list, list]], names=HEADLINE, samples: int = BOOTSTRAP_SAMPLES,
                    seed: int = SEED) -> dict:
    """Paired bootstrap of A minus B on the macro over datasets, as ``aggregate`` scores a group.

    *pairs* maps a dataset to its ``(a, b)`` ``per_doc`` lists. Each draw resamples groups within
    every dataset independently (datasets are separate samples) and recomputes each dataset's
    language macro, then the mean over datasets. One dataset gives the per-dataset bootstrap.
    """
    prepared = []
    for a, b in pairs.values():
        if [x[0] for x in a] != [x[0] for x in b]:
            raise ValueError("paired bootstrap needs the same docs in the same order")
        keys = sorted(set().union(*(x[3] for x in a), *(x[3] for x in b)))
        groups = {g: i for i, g in enumerate(dict.fromkeys(x[1] for x in a))}
        langs = {lang: i for i, lang in enumerate(sorted({x[2] for x in a}))}
        mats = []
        for side in (a, b):
            m = np.zeros((len(langs), len(groups), len(keys)))  # per language, so each draw is a language macro
            for _, group, lang, c in side:
                m[langs[lang], groups[group]] += [c[k] for k in keys]
            mats.append(m)
        prepared.append((keys, len(groups), mats))

    def score(side, weights):
        return macro([macro([metrics(Counter(dict(zip(keys, (w @ per_lang).tolist())))) for per_lang in mats[side]])
                      for (keys, _, mats), w in zip(prepared, weights, strict=True)])

    rng = np.random.default_rng(seed)
    draws: dict[str, list] = {n: [] for n in names}
    for _ in range(samples):
        weights = [np.bincount(rng.integers(0, n, n), minlength=n) for _, n, _ in prepared]
        ma, mb = score(0, weights), score(1, weights)
        for n in names:
            if ma.get(n) is not None and mb.get(n) is not None:
                draws[n].append((ma[n], mb[n], ma[n] - mb[n]))
    ones = [np.ones(n) for _, n, _ in prepared]
    point_a, point_b = score(0, ones), score(1, ones)
    out = {"groups": sum(n for _, n, _ in prepared), "samples": samples, "seed": seed, "metrics": {}}
    for n, rows in draws.items():
        if not rows:
            continue
        arr = np.array(rows)
        lo, hi = np.percentile(arr, [2.5, 97.5], axis=0)
        out["metrics"][n] = {"a": point_a.get(n), "b": point_b.get(n),
                             "a_ci": [round(lo[0], 4), round(hi[0], 4)], "b_ci": [round(lo[1], 4), round(hi[1], 4)],
                             "diff_ci": [round(lo[2], 4), round(hi[2], 4)], "excludes_zero": bool(lo[2] > 0 or hi[2] < 0)}
    return out


def views_for(supports: tuple, shared: tuple, all_types: tuple) -> dict[str, tuple]:
    return {"supported": tuple(supports), "shared": tuple(shared), "full": tuple(all_types)}


def shared_types(entries: list[dict], all_types: tuple) -> tuple:
    """Types every entry supports (the headline view)."""
    return tuple(t for t in all_types if all(t in e["supports"] for e in entries))
