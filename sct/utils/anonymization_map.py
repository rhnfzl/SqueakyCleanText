"""Reversible anonymization: indexed placeholders + round-trip deanonymization.

When ``replacement_mode='reversible'``, entities are replaced with indexed
placeholders (``<PERSON_0>``, ``<LOCATION_1>``) and an ``AnonymizationMap``
records the mapping so the original text can be restored.
"""

from dataclasses import dataclass, field
from typing import List
from uuid import uuid4


@dataclass
class MapEntry:
    """A single entity replacement record."""

    placeholder: str   # e.g. "<PERSON_0>"
    original: str      # e.g. "John Smith"
    entity_type: str   # e.g. "PERSON"
    start: int         # character offset in the *original* text
    end: int           # character offset in the *original* text


@dataclass
class AnonymizationMap:
    """Stores entity replacement mappings for round-trip anonymization.

    One instance per ``process()`` call — no shared state, fully thread-safe.
    """

    session_id: str = field(default_factory=lambda: str(uuid4()))
    entries: List[MapEntry] = field(default_factory=list)

    # Per-entity-type counter (tracks how many of each type have been seen).
    _counter: dict = field(default_factory=dict, repr=False)

    def next_placeholder(self, entity_type: str) -> str:
        """Generate the next indexed placeholder for *entity_type*.

        Example: first PERSON → ``<PERSON_0>``, second → ``<PERSON_1>``.
        """
        idx = self._counter.get(entity_type, 0)
        self._counter[entity_type] = idx + 1
        return f"<{entity_type}_{idx}>"

    def add(self, placeholder: str, original: str, entity_type: str,
            start: int, end: int) -> None:
        """Record a replacement entry."""
        self.entries.append(MapEntry(
            placeholder=placeholder,
            original=original,
            entity_type=entity_type,
            start=start,
            end=end,
        ))

    def deanonymize(self, text: str) -> str:
        """Restore original values by replacing placeholders.

        Sorts longest-placeholder-first to avoid partial-match collisions
        (e.g. ``<PERSON_10>`` must be replaced before ``<PERSON_1>``).
        """
        for entry in sorted(self.entries, key=lambda e: len(e.placeholder), reverse=True):
            text = text.replace(entry.placeholder, entry.original)
        return text

    def to_dict(self) -> dict:
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
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'AnonymizationMap':
        """Deserialize from a dict (inverse of ``to_dict()``)."""
        obj = cls(session_id=data['session_id'])
        for e in data.get('entries', []):
            obj.entries.append(MapEntry(**e))
        return obj
