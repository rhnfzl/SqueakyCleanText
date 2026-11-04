"""The contamination and claimed-language views of a run's per-doc scores (decision 21, PROTOCOL.md sections 12, 17)."""

from typing import Any

from benchmarks.compare import entries as registry
from benchmarks.compare import scorer


def _pick(docs: list, keep) -> list:
    return [x for x in docs if keep(x[2])]


def _agg(per_doc: dict, datasets, keep) -> dict | None:
    """Group macro over *datasets* using only the docs whose language passes *keep*; None when no doc does."""
    chosen = {d: [(lang, c) for _, _, lang, c in _pick(per_doc[d], keep)] for d in datasets}
    chosen = {d: v for d, v in chosen.items() if v}
    return scorer.aggregate(chosen) if chosen else None


def _own_cell(entry: str, by_doc: dict, samples: int) -> dict[str, dict]:
    """Per contamination status: datasets, group macro inside and outside the claim, CI on the clean, claimed cell."""
    def mine(lang: str) -> bool:
        return registry.claims(entry, lang)

    def outside(lang: str) -> bool:
        return not mine(lang)

    cell = {}
    for status in ("clean", "unknown", "excluded"):
        sets = [d for d in by_doc if registry.status(entry, d) == status]
        if not sets:
            continue
        cell[status] = {"datasets": sets, "claimed": _agg(by_doc, sets, mine), "outside_claim": _agg(by_doc, sets, outside)}
        if status == "clean" and cell[status]["claimed"]:
            pairs = {d: (v, v) for d in sets if (v := _pick(by_doc[d], mine))}
            res = scorer.macro_bootstrap(pairs, samples=samples)["metrics"]
            cell[status]["ci"] = {k: {"value": m["a"], "ci": m["a_ci"]} for k, m in res.items()}
    return cell


def _same_set_pool(per_doc: dict, view: str, present: list[str], chosen: list[str]) -> dict[str, Any]:
    """Datasets clean for every entry of *chosen*, on the languages all of them claim."""
    def keep(lang: str) -> bool:
        return all(registry.claims(e, lang) for e in chosen)

    sets = [d for d in present if chosen and all(registry.status(e, d) == "clean" for e in chosen)]
    scores = {e: _agg({d: per_doc[e][d][view] for d in sets}, sets, keep) for e in chosen} if sets else {}
    return {"entries": chosen, "datasets": sets, "scores": {e: a for e, a in scores.items() if a}}


def _pairwise_cells(per_doc: dict, baseline: str, entry: str, view: str, members: tuple, samples: int) -> dict[str, dict]:
    """Baseline minus *entry*, paired bootstrap, per kind of dataset (clean / unknown for both), languages both claim."""
    def keep(lang: str) -> bool:
        return registry.claims(entry, lang) and registry.claims(baseline, lang)

    base = per_doc.get(baseline, {})
    both = [d for d in members if d in base and d in per_doc[entry]]
    statuses = {d: {registry.status(entry, d), registry.status(baseline, d)} for d in both}
    cells = {}
    for kind, sets in (("clean", [d for d in both if statuses[d] == {"clean"}]),
                       ("unknown", [d for d in both if statuses[d] in ({"unknown"}, {"clean", "unknown"})])):
        pairs = {d: (a, b) for d in sets
                 if (a := _pick(base[d][view], keep)) and (b := _pick(per_doc[entry][d][view], keep))}
        if pairs:
            cells[kind] = {"datasets": list(pairs), **scorer.macro_bootstrap(pairs, samples=samples)}
    return cells


def rule_views(per_doc: dict, baseline: str, views=("shared", "full"),
               samples: int = scorer.BOOTSTRAP_SAMPLES) -> dict[str, dict]:
    """Decision 21 + claimed languages (PROTOCOL.md sections 12, 17), from ``per_doc[entry][dataset][view]``.

    - ``own[entry][view][group]``: per contamination status (clean / unknown / excluded) the datasets and the
      group macro on the entry's claimed languages (``claimed``) and outside them (``outside_claim``); the
      clean, claimed cell (the headline) carries a 95% bootstrap CI (``ci``).
    - ``same_set[view][group][pool]``: datasets clean for every entry of the pool, languages all of them claim
      (``all``: every entry; ``others``: the non-SCT entries with a clean set in the group). Empty: none.
    - ``pairwise[entry][view][group]``: baseline minus entry, paired bootstrap, on the datasets clean for both
      (``clean``) and on those clean or unknown for both with at least one unknown (``unknown``), languages
      both claim.
    """
    own: dict = {}
    pairwise: dict = {}
    same_set: dict = {}
    entries = list(per_doc)
    for view in views:
        for group, members in scorer.GROUPS.items():
            present = [d for d in members if all(d in per_doc[e] for e in entries)]
            for e in entries:
                cells = own.setdefault(e, {}).setdefault(view, {})
                by_doc = {d: per_doc[e][d][view] for d in members if d in per_doc[e]}
                if by_doc:
                    cells[group] = _own_cell(e, by_doc, samples)
            others = [e for e in entries if e != baseline and not e.startswith("sct")
                      and any(registry.status(e, d) == "clean" for d in present)]
            for pool, chosen in (("all", entries), ("others", others)):
                same_set.setdefault(view, {}).setdefault(group, {})[pool] = _same_set_pool(per_doc, view, present, chosen)
            for e in entries:
                if e == baseline:
                    continue
                cells = _pairwise_cells(per_doc, baseline, e, view, members, samples)
                if cells:
                    pairwise.setdefault(e, {}).setdefault(view, {}).setdefault(group, {}).update(cells)
    return {"own": own, "same_set": same_set, "pairwise": pairwise}
