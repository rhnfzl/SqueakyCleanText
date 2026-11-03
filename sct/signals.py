"""Value-free public view of a ``ProcessResult`` (``result.signals``).

Counts, shares, type names, language and model provenance only: never entity
text and never offsets, so the view is safe to log, store or return to a model.
"""

from typing import Any, Iterable

PUBLIC_VIEW_KEYS: tuple[str, ...] = (
    "language", "language_confidence", "language_source", "chars",
    "pii_count", "pii_chars", "pii_share", "pii_types",
    "regex_counts", "entity_counts", "sensitive", "warnings", "models",
)
# IP_ADDRESS / PHONE_NUMBER: the names under legacy_type_names=True.
PII_REGEX_TYPES = frozenset({"EMAIL", "URL", "IP", "PHONE", "IP_ADDRESS", "PHONE_NUMBER", "IBAN", "CARD", "ID"})
# Regex counts that are not PII; every other key (the built-in PII types above and the user labels of
# custom_deny_terms / custom_patterns) counts as PII.
NON_PII_REGEX_TYPES = frozenset({"HTML", "DATE", "YEAR", "NUMBER", "CURRENCY"})
WARNING_CODES = frozenset({
    "language_unknown", "language_fallback", "language_low_confidence",
    "ner_fallback_models", "ner_off", "cuda_fallback", "mps_fallback", "empty_input",
})


def public_view(result) -> dict[str, Any]:
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
        "sensitive": bool(regex_counts or entity_counts),
        "warnings": sorted(warnings),
        "models": [
            {k: model.get(k) for k in ("key", "id", "revision", "provider")}
            for model in provenance.get("models", ())
        ],
    }


class _Empty:
    language = None
    metadata: dict = {}


def merge_views(views: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Merge leaf views (e.g. every string of a JSON value) into one view."""
    views = list(views)
    if not views:
        return public_view(_Empty())
    # Start from the longest leaf.
    merged = dict(max(views, key=lambda v: v["chars"]))
    lang_chars: dict[Any, int] = {}
    for view in views:
        lang_chars[view["language"]] = lang_chars.get(view["language"], 0) + view["chars"]
    language = max(lang_chars, key=lang_chars.get)
    # Source follows the longest leaf of the chosen language, not of the whole value.
    merged["language_source"] = max(
        (v for v in views if v["language"] == language), key=lambda v: v["chars"]
    )["language_source"]
    same = [v for v in views if v["language"] == language and v["language_confidence"] is not None]
    weight = sum(v["chars"] for v in same)
    merged["language"] = language
    merged["language_confidence"] = (
        sum(v["language_confidence"] * v["chars"] for v in same) / weight if weight else None
    )
    for key in ("chars", "pii_count", "pii_chars"):
        merged[key] = sum(v[key] for v in views)
    for key in ("regex_counts", "entity_counts"):
        totals: dict[str, dict[str, int]] = {}
        for view in views:
            for name, entry in view[key].items():
                slot = totals.setdefault(name, {"count": 0, "chars": 0})
                slot["count"] += entry["count"]
                slot["chars"] += entry["chars"]
        merged[key] = dict(sorted(totals.items()))
    types: dict[str, int] = {}
    for view in views:
        for name, count in view["pii_types"].items():
            types[name] = types.get(name, 0) + count
    merged["pii_types"] = dict(sorted(types.items()))
    merged["pii_share"] = min(1.0, merged["pii_chars"] / merged["chars"]) if merged["chars"] else 0.0
    merged["sensitive"] = any(v["sensitive"] for v in views)
    merged["warnings"] = sorted({w for v in views for w in v["warnings"]})
    merged["models"] = list({m["key"]: m for v in views for m in v["models"]}.values())
    return merged
