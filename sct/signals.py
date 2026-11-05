"""Value-free public view of a ``ProcessResult`` (``result.signals``).

Counts, shares, type names, language and model provenance only: never entity
text and never offsets, so the view is safe to log, store or return to a model.
"""

from typing import Any, TypedDict, cast
from collections.abc import Iterable

from sct.utils.process_result import ProcessResult


class Signals(TypedDict):
    """The value-free public view; every key is present for every input."""

    language: str | None
    language_confidence: float | None
    language_source: str
    chars: int
    pii_count: int
    pii_chars: int
    pii_share: float
    pii_types: dict[str, int]
    regex_counts: dict[str, dict[str, int]]
    entity_counts: dict[str, dict[str, int]]
    sensitive: bool
    warnings: list[str]
    models: list[dict[str, Any]]


PUBLIC_VIEW_KEYS: tuple[str, ...] = tuple(Signals.__annotations__)
# Regex counts that are not PII; every other key (the built-in PII types and the user labels of
# custom_deny_terms / custom_patterns) counts as PII.
NON_PII_REGEX_TYPES = frozenset({"HTML", "DATE", "YEAR", "NUMBER", "CURRENCY"})
WARNING_CODES = frozenset({
    "language_unknown", "language_fallback", "language_low_confidence",
    "ner_fallback_models", "ner_off", "cuda_fallback", "mps_fallback", "empty_input",
    "custom_pattern_timeout", "findings_not_collected",
})


def is_sensitive_result(result: ProcessResult) -> bool:
    """Return whether processing found an entity or made a regex replacement.

    The one sensitivity rule: ``public_view`` and OCR region selection both use it.
    """
    metadata = result.metadata or {}
    return bool(
        metadata.get("regex_counts")
        or metadata.get("entity_counts")
        or metadata.get("findings")
    )


def public_view(result: ProcessResult) -> Signals:
    """The value-free view of one result; same keys for every input."""
    metadata = result.metadata or {}
    chars = metadata.get("source_chars", 0)
    regex_counts = {k: dict(v) for k, v in sorted(metadata.get("regex_counts", {}).items())}
    entity_counts = {k: dict(v) for k, v in sorted(metadata.get("entity_counts", {}).items())}
    pii_types: dict[str, int] = {}
    pii_chars = 0
    pii_items = [(k, v) for k, v in regex_counts.items() if k not in NON_PII_REGEX_TYPES]
    for name, entry in pii_items + list(entity_counts.items()):
        pii_types[name] = pii_types.get(name, 0) + entry["count"]
        pii_chars += entry["chars"]
    provenance = metadata.get("provenance") or {}
    warnings = set(metadata.get("warnings", ()))
    if not provenance.get("backend"):
        warnings.add("ner_off")
    if chars == 0:
        warnings.add("empty_input")
    return {
        "language": result.language,
        "language_confidence": metadata.get("language_confidence"),
        "language_source": metadata.get("language_source", "none"),
        "chars": chars,
        "pii_count": sum(pii_types.values()),
        "pii_chars": pii_chars,
        "pii_share": min(1.0, pii_chars / chars) if chars else 0.0,
        "pii_types": dict(sorted(pii_types.items())),
        "regex_counts": regex_counts,
        "entity_counts": entity_counts,
        "sensitive": is_sensitive_result(result),
        "warnings": sorted(warnings),
        "models": [
            {k: model.get(k) for k in ("key", "id", "revision", "provider")}
            for model in provenance.get("models", ())
        ],
    }


def merge_views(views: Iterable[Signals]) -> Signals:
    """Merge leaf views (e.g. every string of a JSON value) into one view; ``ner_off`` stays only if every leaf has it."""
    leaves: list[Any] = list(views)
    if not leaves:
        return public_view(ProcessResult("", None, None))
    # Start from the longest leaf.
    merged: dict[str, Any] = dict(max(leaves, key=lambda v: v["chars"]))
    lang_chars: dict[Any, int] = {}
    for view in leaves:
        lang_chars[view["language"]] = lang_chars.get(view["language"], 0) + view["chars"]
    language = max(lang_chars, key=lang_chars.__getitem__)
    # Source follows the longest leaf of the chosen language, not of the whole value.
    merged["language_source"] = max(
        (v for v in leaves if v["language"] == language), key=lambda v: v["chars"]
    )["language_source"]
    same = [v for v in leaves if v["language"] == language and v["language_confidence"] is not None]
    weight = sum(v["chars"] for v in same)
    merged["language"] = language
    merged["language_confidence"] = (
        sum(v["language_confidence"] * v["chars"] for v in same) / weight if weight else None
    )
    for key in ("chars", "pii_count", "pii_chars"):
        merged[key] = sum(v[key] for v in leaves)
    for key in ("regex_counts", "entity_counts"):
        totals: dict[str, dict[str, int]] = {}
        for view in leaves:
            for name, entry in view[key].items():
                slot = totals.setdefault(name, {"count": 0, "chars": 0})
                slot["count"] += entry["count"]
                slot["chars"] += entry["chars"]
        merged[key] = dict(sorted(totals.items()))
    types: dict[str, int] = {}
    for view in leaves:
        for name, count in view["pii_types"].items():
            types[name] = types.get(name, 0) + count
    merged["pii_types"] = dict(sorted(types.items()))
    merged["pii_share"] = min(1.0, merged["pii_chars"] / merged["chars"]) if merged["chars"] else 0.0
    merged["sensitive"] = any(v["sensitive"] for v in leaves)
    warnings = {w for v in leaves for w in v["warnings"]}
    # A JSON key view runs with NER off by design, so one such leaf says nothing about the values.
    if not all("ner_off" in v["warnings"] for v in leaves):
        warnings.discard("ner_off")
    merged["warnings"] = sorted(warnings)
    merged["models"] = list({m["key"]: m for v in leaves for m in v["models"]}.values())
    return cast(Signals, merged)
