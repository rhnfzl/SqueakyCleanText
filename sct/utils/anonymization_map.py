"""Reversible anonymization: indexed placeholders + round-trip deanonymization.

When ``replacement_mode='reversible'``, entities are replaced with indexed
placeholders (``<PERSON_0>``, ``<LOCATION_1>``) and an ``AnonymizationMap``
records the mapping so the original text can be restored.
"""

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4


@dataclass
class MapEntry:
    """A single entity replacement record."""

    placeholder: str
    original: str = field(repr=False)
    entity_type: str
    # Offsets index the text the replacing stage saw: for NER entries that is
    # the whole post-regex document (not the raw input, not a chunk).
    start: int
    end: int


@dataclass
class AnonymizationMap:
    """Stores entity replacement mappings for round-trip anonymization.

    ``TextCleaner.process()`` creates a separate map for each document.
    """

    session_id: str = field(default_factory=lambda: str(uuid4()))
    entries: list[MapEntry] = field(default_factory=list)

    _counter: dict[str, int] = field(default_factory=dict, repr=False)
    reserved: set[str] = field(default_factory=set, repr=False)

    def __post_init__(self) -> None:
        for entry in self.entries:
            self._advance_past(entry)

    def _advance_past(self, entry: MapEntry) -> None:
        prefix = f"<{entry.entity_type}_"
        if entry.placeholder.startswith(prefix) and entry.placeholder.endswith(">"):
            index = entry.placeholder[len(prefix):-1]
            if index.isdecimal():
                self._counter[entry.entity_type] = max(self._counter.get(entry.entity_type, 0), int(index) + 1)

    def _allocation_snapshot(self) -> dict[str, int]:
        return self._counter.copy()

    def _restore_allocation(self, counters: dict[str, int]) -> None:
        if any(counters.get(kind, 0) < minimum for kind, minimum in self._counter.items()):
            raise ValueError("counter precedes restored entries")
        self._counter = counters.copy()

    def __repr__(self):
        return f"AnonymizationMap(entries={len(self.entries)})"

    def next_placeholder(self, entity_type: str) -> str:
        """Generate the next indexed placeholder for *entity_type*.

        Example: first PERSON → ``<PERSON_0>``, second → ``<PERSON_1>``.
        """
        idx = self._counter.get(entity_type, 0)
        while f"<{entity_type}_{idx}>" in self.reserved:
            idx += 1
        self._counter[entity_type] = idx + 1
        return f"<{entity_type}_{idx}>"

    def add(self, placeholder: str, original: str, entity_type: str,
            start: int, end: int) -> None:
        """Record a replacement entry."""
        entry = MapEntry(
            placeholder=placeholder,
            original=original,
            entity_type=entity_type,
            start=start,
            end=end,
        )
        self._advance_past(entry)
        self.entries.append(entry)

    def _record_replacement(self, original: str, entity_type: str, start: int, end: int) -> str:
        placeholder = self.next_placeholder(entity_type)
        self.add(placeholder, original, entity_type, start, end)
        return placeholder

    def deanonymize(self, text: str) -> str:
        """Restore original values by replacing placeholders.

        Sorts longest-placeholder-first to avoid partial-match collisions
        (e.g. ``<PERSON_10>`` must be replaced before ``<PERSON_1>``).
        """
        for entry in sorted(self.entries, key=lambda e: len(e.placeholder), reverse=True):
            text = text.replace(entry.placeholder, entry.original)
        return text

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {
            'session_id': self.session_id,
            'entries': [
                {
                    'placeholder': e.placeholder,
                    'original': e.original,
                    'entity_type': e.entity_type,
                    'start': e.start,
                    'end': e.end,
                }
                for e in self.entries
            ],
            'reserved': sorted(self.reserved),
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'AnonymizationMap':
        """Deserialize from a dict (inverse of ``to_dict()``).

        New placeholders continue after the restored ones. Older payloads have no
        ``reserved``, so it starts empty.
        """
        return cls(
            session_id=data['session_id'],
            entries=[MapEntry(**entry) for entry in data.get('entries', [])],
            reserved=set(data.get('reserved', ())),
        )
