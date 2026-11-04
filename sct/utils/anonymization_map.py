"""Reversible anonymization: indexed placeholders + round-trip deanonymization.

When ``replacement_mode='reversible'``, entities are replaced with indexed
placeholders (``<PERSON_0>``, ``<LOCATION_1>``) and an ``AnonymizationMap``
records the mapping so the original text can be restored.
"""

from dataclasses import dataclass, field
from uuid import uuid4


@dataclass
class MapEntry:
    """A single entity replacement record."""

    placeholder: str   # e.g. "<PERSON_0>"
    original: str = field(repr=False)  # e.g. "John Smith"; kept out of repr
    entity_type: str   # e.g. "PERSON"
    # Offsets index the text the replacing stage saw: for NER entries that is
    # the whole post-regex document (not the raw input, not a chunk).
    start: int
    end: int


@dataclass
class AnonymizationMap:
    """Stores entity replacement mappings for round-trip anonymization.

    One instance per ``process()`` call — no shared state, fully thread-safe.
    """

    session_id: str = field(default_factory=lambda: str(uuid4()))
    entries: list[MapEntry] = field(default_factory=list)

    # Per-entity-type counter (tracks how many of each type have been seen).
    _counter: dict = field(default_factory=dict, repr=False)
    # Placeholder spellings already in the source text; never allocated, so a
    # literal "<EMAIL_0>" cannot collide with a generated one.
    reserved: set = field(default_factory=set, repr=False)

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
            'reserved': sorted(self.reserved),
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'AnonymizationMap':
        """Deserialize from a dict (inverse of ``to_dict()``).

        New placeholders continue after the restored ones. Older payloads have no
        ``reserved``, so it starts empty.
        """
        obj = cls(session_id=data['session_id'], reserved=set(data.get('reserved', ())))
        for e in data.get('entries', []):
            entry = MapEntry(**e)
            obj.entries.append(entry)
            index = entry.placeholder.removeprefix(f"<{entry.entity_type}_").removesuffix(">")
            if index.isdecimal():
                obj._counter[entry.entity_type] = max(obj._counter.get(entry.entity_type, 0), int(index) + 1)
        return obj
