"""Corpus-level re-identification risk measurements."""

from collections import Counter
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class ReidentificationRiskReport:
    """K-anonymity summary without source quasi-identifier values."""

    record_count: int
    quasi_identifiers: tuple[str, ...]
    k_threshold: int
    minimum_k: int
    at_risk_records: int
    at_risk_fraction: float
    class_size_counts: Mapping[int, int]


def assess_reidentification_risk(
    records: Sequence[Mapping[str, Any]],
    *,
    quasi_identifiers: tuple[str, ...],
    k_threshold: int = 5,
) -> ReidentificationRiskReport:
    """Measure k-anonymity for selected quasi-identifiers."""
    if not quasi_identifiers:
        raise ValueError("quasi_identifiers must not be empty")
    if k_threshold < 2:
        raise ValueError("k_threshold must be >= 2")

    combinations = Counter()
    for record in records:
        missing = set(quasi_identifiers).difference(record)
        if missing:
            raise KeyError(f"Missing quasi-identifiers: {sorted(missing)}")
        combinations[tuple(record[key] for key in quasi_identifiers)] += 1

    class_size_counts = Counter(combinations.values())
    at_risk_records = sum(
        class_size * class_count
        for class_size, class_count in class_size_counts.items()
        if class_size < k_threshold
    )
    record_count = len(records)
    return ReidentificationRiskReport(
        record_count=record_count,
        quasi_identifiers=quasi_identifiers,
        k_threshold=k_threshold,
        minimum_k=min(combinations.values(), default=0),
        at_risk_records=at_risk_records,
        at_risk_fraction=(
            at_risk_records / record_count if record_count else 0.0
        ),
        class_size_counts=MappingProxyType(dict(class_size_counts)),
    )
