"""Project proven edits to original Unicode coordinates without retaining public text."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from sct.findings import MappingQuality, TextRange, _mapping, _offset


@dataclass(frozen=True)
class Projection:
    ranges: tuple[TextRange, ...] = ()
    mapping_quality: MappingQuality = "unavailable"

    def __post_init__(self) -> None:
        _mapping(self.ranges, self.mapping_quality)


class SourceMap:
    """Immutable edit history; private text is excluded from default output."""

    __slots__ = ("_text", "_origins", "_isolation_monotone")
    _text: str
    _origins: tuple[Projection, ...]
    _isolation_monotone: bool | None

    def __init__(self, text: str, origins: tuple[Projection, ...]) -> None:
        if not isinstance(text, str) or len(text) != len(origins):
            raise ValueError("Each stage code point requires one provenance entry")
        if not isinstance(origins, tuple) or any(not isinstance(origin, Projection) for origin in origins):
            raise ValueError("Provenance must be an immutable tuple of projections")
        object.__setattr__(self, "_text", text)
        object.__setattr__(self, "_origins", origins)
        object.__setattr__(self, "_isolation_monotone", None)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("SourceMap is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("SourceMap is immutable")

    @classmethod
    def identity(cls, text: str) -> SourceMap:
        return cls(text, tuple(Projection((TextRange(i, i + 1),), "exact") for i in range(len(text))))

    def _validate(self, start: int, end: int, *, empty: bool = False) -> None:
        _offset(start)
        _offset(end)
        if start > end or (start == end and not empty) or end > len(self._text):
            raise ValueError("Range lies outside the current text stage or is empty")

    def project(self, start: int, end: int) -> Projection:
        self._validate(start, end)
        origins = self._origins[start:end]
        if any(origin.mapping_quality == "unavailable" for origin in origins):
            return Projection()
        ranges = sorted({item for origin in origins for item in origin.ranges}, key=lambda item: (item.start, item.end))
        merged: list[TextRange] = []
        for item in ranges:
            if merged and item.start <= merged[-1].end:
                merged[-1] = TextRange(merged[-1].start, max(merged[-1].end, item.end))
            else:
                merged.append(item)
        quality: MappingQuality = "exact" if all(origin.mapping_quality == "exact" for origin in origins) else "covering"
        return Projection(tuple(merged), quality)

    def _is_isolated(self, start: int, end: int) -> bool:
        """Check source-cover isolation with one cached linear proof and constant-time queries."""
        self._validate(start, end)
        if self._isolation_monotone is None:
            previous_start = previous_end = 0
            monotone = True
            for origin in self._origins:
                if (origin.mapping_quality == "unavailable" or origin.ranges[0].start < previous_start
                        or origin.ranges[-1].end < previous_end):
                    monotone = False
                    break
                previous_start, previous_end = origin.ranges[0].start, origin.ranges[-1].end
            object.__setattr__(self, "_isolation_monotone", monotone)
        if not self._isolation_monotone:
            return False
        # Monotone covers bound every earlier/later origin, including disjoint source ranges.
        first = self._origins[start].ranges[0].start
        last = self._origins[end - 1].ranges[-1].end
        return ((start == 0 or self._origins[start - 1].ranges[-1].end <= first)
                and (end == len(self._origins) or self._origins[end].ranges[0].start >= last))

    def replace(self, start: int, end: int, replacement: str) -> SourceMap:
        self._validate(start, end, empty=True)
        if not isinstance(replacement, str):
            raise ValueError("Replacement must be a string")
        if self._text[start:end] == replacement:
            return self
        projection = self.project(start, end) if start != end else Projection()
        if projection.mapping_quality != "unavailable":
            projection = Projection(projection.ranges, "covering")
        return SourceMap(
            self._text[:start] + replacement + self._text[end:],
            self._origins[:start] + (projection,) * len(replacement) + self._origins[end:],
        )

    def transform(self, callback: Callable[[str], str]) -> SourceMap:
        """An opaque callback cannot prove which source positions survived."""
        text = callback(self._text)
        if not isinstance(text, str):
            raise ValueError("A text transform must return a string")
        return SourceMap(text, (Projection(),) * len(text))

    def to_dict(self) -> dict[str, int]:
        return {"stage_length": len(self._text)}
