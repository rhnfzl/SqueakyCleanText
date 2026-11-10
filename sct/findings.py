"""Value-free findings use explicit Unicode text stages."""

from collections.abc import Hashable
from dataclasses import dataclass, field
import math
from typing import Literal

MappingQuality = Literal["exact", "covering", "unavailable"]


def _offset(value: int) -> None:
    if type(value) is not int or value < 0:
        raise ValueError("Offsets must be nonnegative Unicode code-point integers")


@dataclass(frozen=True)
class TextRange:
    start: int
    end: int

    def __post_init__(self) -> None:
        _offset(self.start)
        _offset(self.end)
        if self.end <= self.start:
            raise ValueError("A text range must be nonempty and ordered")


def _mapping(ranges: tuple[TextRange, ...], quality: MappingQuality) -> None:
    if not isinstance(ranges, tuple) or any(not isinstance(item, TextRange) for item in ranges):
        raise ValueError("Source ranges must be a tuple of TextRange values")
    if quality not in ("exact", "covering", "unavailable"):
        raise ValueError("Unknown mapping quality")
    if bool(ranges) != (quality != "unavailable"):
        raise ValueError("Available mappings require source ranges; unavailable mappings forbid them")
    if any(left.end > right.start for left, right in zip(ranges, ranges[1:])):
        raise ValueError("Source ranges must be ordered and nonoverlapping")


@dataclass(frozen=True)
class Detection:
    entity_type: str
    start: int
    end: int
    detector_id: str
    score: float | None
    coordinate_space: str
    source_ranges: tuple[TextRange, ...] = ()
    mapping_quality: MappingQuality = "unavailable"

    def __post_init__(self) -> None:
        TextRange(self.start, self.end)
        for value in (self.entity_type, self.detector_id, self.coordinate_space):
            if not isinstance(value, str) or not value.strip():
                raise ValueError("Finding identifiers must be nonempty strings")
        if self.score is not None and (
            type(self.score) not in (int, float) or not 0 <= self.score <= 1 or not math.isfinite(self.score)
        ):
            raise ValueError("Scores must be finite numbers in [0, 1] or None")
        _mapping(self.source_ranges, self.mapping_quality)

    def validate(self, text: str, *, coordinate_space: str) -> None:
        """Reject findings from another stage or outside the supplied stage text."""
        if self.coordinate_space != coordinate_space or self.end > len(text):
            raise ValueError("Finding does not belong to the supplied text stage")


@dataclass(frozen=True)
class DetectionContext:
    document_id: str | None = field(default=None, repr=False)
    path: tuple[Hashable, ...] = field(default=(), repr=False)
    declared_entity_types: tuple[str, ...] = field(default=(), repr=False)
    schema_id: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.path, tuple):
            raise ValueError("Context paths must be immutable tuples")
        try:
            hash(self.path)
        except TypeError:
            raise ValueError("Context path components must be hashable") from None
        if not isinstance(self.declared_entity_types, tuple) or any(
            not isinstance(value, str) or not value.strip() for value in self.declared_entity_types
        ):
            raise ValueError("Declared entity types must be a tuple of nonempty strings")
        if any(value is not None and not isinstance(value, str) for value in (self.document_id, self.schema_id)):
            raise ValueError("Context identifiers must be strings or None")

    def to_dict(self) -> dict[str, object]:
        """Default output excludes caller context, whose identifiers may contain PII."""
        return {}
