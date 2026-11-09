"""Private corpus references and value-free selection decisions."""
from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CorpusRecord:
    id: str = field(repr=False)
    text: str = field(repr=False)
    metadata: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        if not isinstance(self.id, str) or not self.id:
            raise ValueError('record id must be a nonempty string')
        if not isinstance(self.text, str):
            raise ValueError('record text must be a string')


@dataclass(frozen=True)
class SelectionConfig:
    quality_min: float | None = None
    dedup_mode: str = 'report'
    dedup_basis: str | None = None
    similarity_threshold: float = .9
    keeper: str = 'min_stable_id'

    def __post_init__(self):
        if self.dedup_mode not in ('report', 'apply'):
            raise ValueError('dedup_mode must be report or apply')
        if self.dedup_basis not in (None, 'pre_pii', 'pii_normalized'):
            raise ValueError('invalid dedup_basis')
        if self.dedup_mode == 'apply' and self.dedup_basis is None:
            raise ValueError('apply requires an explicit dedup_basis')
        if not isfinite(self.similarity_threshold) or not 0 < self.similarity_threshold <= 1:
            raise ValueError('similarity_threshold must be in (0, 1]')
        if self.quality_min is not None and (not isfinite(self.quality_min) or not 0 <= self.quality_min <= 1):
            raise ValueError('quality_min must be in [0, 1]')
        if self.keeper != 'min_stable_id':
            raise ValueError('keeper must be min_stable_id')

    @property
    def basis(self) -> str:
        return self.dedup_basis or 'pii_normalized'


@dataclass(frozen=True)
class RecordAnalysis:
    id: str = field(repr=False)
    scores: Mapping[str, Any]
    signature_metadata: Mapping[str, Any]
    signature: tuple[int, ...] | None = field(repr=False)
    output_ref: Path = field(repr=False)
    source_digest: str = field(repr=False)
    basis_digest: str = field(repr=False)
    ownership_token: str = field(default="", repr=False)


@dataclass(frozen=True)
class SelectionDecision:
    id: str = field(repr=False)
    keeper_id: str = field(repr=False)
    quality_pass: bool
    duplicate: bool
    basis_collision: bool
    kept: bool
    reasons: tuple[str, ...]
