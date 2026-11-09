"""Value-free public view of a ``ProcessResult`` (``result.signals``).

Counts, shares, type names, language and model provenance only: never entity
text and never offsets, so the view is safe to log, store or return to a model.
"""

from typing import Any, Literal, NotRequired, TypedDict, cast
from collections.abc import Iterable

from sct.utils.process_result import ProcessResult


class CoverageV2(TypedDict):
    """Bounds on source characters covered by detected PII, not a privacy guarantee."""

    version: int
    quality: Literal['exact', 'covering', 'unavailable']
    chars_lower: int
    chars_upper: int
    share_lower: float
    share_upper: float


def _coverage(quality, lower: int, upper: int, chars: int) -> CoverageV2:
    return {'version': 2, 'quality': quality, 'chars_lower': lower, 'chars_upper': upper,
            'share_lower': min(1.0, lower / chars) if chars else 0.0,
            'share_upper': min(1.0, upper / chars) if chars else 0.0}


def _union_chars(ranges: list[tuple[int, int]]) -> int:
    end = total = 0
    for start, stop in sorted(ranges):
        total += max(0, stop - max(start, end))
        end = max(end, stop)
    return total


class Signals(TypedDict):
    """The value-free public view; every key is present for every input."""

    language: str | None
    language_confidence: float | None
    language_source: str
    chars: int
    pii_count: int
    pii_chars: int | None
    pii_share: float | None
    pii_types: dict[str, int]
    regex_counts: dict[str, dict[str, int]]
    entity_counts: dict[str, dict[str, int]]
    sensitive: bool
    warnings: list[str]
    models: list[dict[str, Any]]
    version: NotRequired[int]
    counts_v2: NotRequired[dict[str, int]]
    provenance_v2: NotRequired[dict[str, Any]]
    coverage_v2: NotRequired[CoverageV2]
    language_margin: NotRequired[float | None]
    language_reason: NotRequired[str]
    ner_skipped: NotRequired[bool]


PUBLIC_VIEW_KEYS: tuple[str, ...] = tuple(key for key in Signals.__annotations__
                                         if key not in ('version', 'counts_v2', 'provenance_v2', 'coverage_v2',
                                                        'language_margin', 'language_reason', 'ner_skipped'))
# Regex counts that are not PII; every other key (the built-in PII types and the user labels of
# custom_deny_terms / custom_patterns) counts as PII.
NON_PII_REGEX_TYPES = frozenset({"HTML", "DATE", "YEAR", "NUMBER", "CURRENCY"})
WARNING_CODES = frozenset({
    "language_unknown", "language_fallback", "language_low_confidence",
    "ner_fallback_models", "ner_off", "cuda_fallback", "mps_fallback", "empty_input",
    "custom_pattern_timeout", "findings_not_collected",
    "source_coverage_covering", "source_coverage_unavailable",
    "language_abstained", "ner_skipped_unknown_language",
})


def is_sensitive_result(result: ProcessResult) -> bool:
    """Return whether the result holds concealed PII: ``signals["sensitive"]``, the ``public_view`` rule."""
    return public_view(result)["sensitive"]


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
    pii_count = sum(pii_types.values())
    view: Signals = {
        "language": result.language,
        "language_confidence": metadata.get("language_confidence"),
        "language_source": metadata.get("language_source", "none"),
        "chars": chars,
        "pii_count": pii_count,
        "pii_chars": pii_chars,
        "pii_share": min(1.0, pii_chars / chars) if chars else 0.0,
        "pii_types": dict(sorted(pii_types.items())),
        "regex_counts": regex_counts,
        "entity_counts": entity_counts,
        "sensitive": pii_count > 0,
        "warnings": sorted(warnings),
        "models": [
            {k: model.get(k) for k in ("key", "id", "revision", "provider")}
            for model in provenance.get("models", ())
        ],
    }
    if 'language_margin' in metadata:
        view['language_margin'] = metadata['language_margin']
    if 'language_reason' in metadata:
        view['language_reason'] = metadata['language_reason']
    if 'ner_skipped' in metadata:
        view['ner_skipped'] = metadata['ner_skipped']

    if 'counts_v2' in metadata:
        counts = dict(metadata['counts_v2'])
        findings = metadata.get('findings_v2', ())
        types: dict[str, int] = {}
        exact_ranges: list[tuple[int, int]] = []
        ranges: list[tuple[int, int]] = []
        quality = 'exact'
        for hit in findings:
            if hit.get('builtin_regex_origin', False) and hit['entity_type'] in NON_PII_REGEX_TYPES:
                continue
            types[hit['entity_type']] = types.get(hit['entity_type'], 0) + 1
            projected = [(span['start'], span['end']) for span in hit['source_ranges']]
            ranges.extend(projected)
            if hit['mapping_quality'] == 'exact':
                exact_ranges.extend(projected)
            elif hit['mapping_quality'] == 'unavailable':
                quality = 'unavailable'
            elif quality == 'exact':
                quality = 'covering'
        lower = _union_chars(exact_ranges)
        upper = chars if quality == 'unavailable' else _union_chars(ranges)
        coverage = _coverage(quality, lower, upper, chars)
        view['version'] = 2
        view['counts_v2'] = counts
        view['provenance_v2'] = dict(metadata.get('provenance_v2', {}))
        view['pii_count'] = sum(types.values())
        view['pii_types'] = dict(sorted(types.items()))
        view['coverage_v2'] = coverage
        view['pii_chars'] = lower if quality == 'exact' else None
        view['pii_share'] = coverage['share_lower'] if quality == 'exact' else None
        if quality != 'exact':
            view['warnings'] = sorted({*view['warnings'], f'source_coverage_{quality}'})
        view['sensitive'] = bool(types)
        if any(detector.get('id') == 'builtin_ner' for detector in view['provenance_v2'].get('detectors', ())):
            view['warnings'] = [warning for warning in view['warnings'] if warning != 'ner_off']
    return view


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
    language_view = max(
        (v for v in leaves if v["language"] == language), key=lambda v: v["chars"]
    )
    merged["language_source"] = language_view["language_source"]
    for key in ('language_margin', 'language_reason'):
        merged.pop(key, None)
        if key in language_view:
            merged[key] = language_view[key]
    if any(v.get('ner_skipped') for v in leaves):
        merged['ner_skipped'] = True
    same = [v for v in leaves if v["language"] == language and v["language_confidence"] is not None]
    weight = sum(v["chars"] for v in same)
    merged["language"] = language
    merged["language_confidence"] = (
        sum(v["language_confidence"] * v["chars"] for v in same) / weight if weight else None
    )
    for key in ("chars", "pii_count", "pii_chars"):
        merged[key] = (None if any(v[key] is None for v in leaves) else sum(v[key] for v in leaves))
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
    merged["pii_share"] = (None if merged['pii_chars'] is None else
                           min(1.0, merged["pii_chars"] / merged["chars"]) if merged["chars"] else 0.0)
    merged["sensitive"] = any(v["sensitive"] for v in leaves)
    warnings = {w for v in leaves for w in v["warnings"]}
    # A JSON key view runs with NER off by design, so one such leaf says nothing about the values.
    if not all("ner_off" in v["warnings"] for v in leaves):
        warnings.discard("ner_off")
    merged["warnings"] = sorted(warnings)
    merged["models"] = list({m["key"]: m for v in leaves for m in v["models"]}.values())
    unified = [view for view in leaves if view.get('version') == 2]
    if unified:
        merged['version'] = 2
        bounds = [view['coverage_v2'] if 'coverage_v2' in view else
                  _coverage('unavailable', 0, view['chars'], view['chars']) if view['pii_count'] else
                  _coverage('exact', 0, 0, view['chars']) for view in leaves]
        qualities = {bound['quality'] for bound in bounds}
        quality = 'unavailable' if 'unavailable' in qualities else 'covering' if 'covering' in qualities else 'exact'
        merged['coverage_v2'] = _coverage(quality, sum(bound['chars_lower'] for bound in bounds),
                                          sum(bound['chars_upper'] for bound in bounds), merged['chars'])
        merged['pii_chars'] = merged['coverage_v2']['chars_lower'] if quality == 'exact' else None
        merged['pii_share'] = merged['coverage_v2']['share_lower'] if quality == 'exact' else None
        if quality != 'exact':
            merged['warnings'] = sorted({*merged['warnings'], f'source_coverage_{quality}'})
        merged['counts_v2'] = {'version': 2, **{
            name: sum(view['counts_v2'][name] for view in unified)
            for name in ('detector_hits', 'resolved_replacements')}}
        # A structured view can contain several policies, so keep each identity explicitly.
        identities = []
        for view in unified:
            identity = view.get('provenance_v2', {})
            if identity not in identities:
                identities.append(identity)
        merged['provenance_v2'] = {'version': 2, 'leaves': identities}
    return cast(Signals, merged)
