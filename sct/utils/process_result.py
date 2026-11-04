"""ProcessResult — backward-compatible result wrapper with metadata.

Unpacks as a 3-tuple for existing ``a, b, c = cleaner.process(text)`` code,
while exposing ``.metadata`` for new features (reversible anonymization maps,
document classification, etc.).
"""

from typing import TYPE_CHECKING, Any, TypedDict
from collections.abc import Sequence

if TYPE_CHECKING:
    from sct.privacy import EntityFinding
    from sct.signals import Signals
    from sct.utils.anonymization_map import AnonymizationMap


class ProcessMetadata(TypedDict, total=False):
    """Keys of ``ProcessResult.metadata``; see the ``ProcessResult`` docstring."""

    source_chars: int
    regex_counts: dict[str, dict[str, int]]
    entity_counts: dict[str, dict[str, int]]
    language_confidence: float | None
    language_source: str
    warnings: list[str]
    provenance: dict[str, Any]
    anon_map: "AnonymizationMap"
    anon_map_reference: str
    classes: list[dict]
    findings: "Sequence[EntityFinding]"
    policy: dict[str, str]


class ProcessResult:
    """Result of ``TextCleaner.process()`` / ``process_batch()``.

    Behaves as a 3-tuple ``(lm_text, stat_text, language)`` for backward
    compatibility, but carries an optional ``.metadata`` dict for extended
    features.

    Always present (from ``TextCleaner``; value-free):
        ``source_chars``  — ``len`` of the input text
        ``regex_counts``  — ``{TYPE: {"count": int, "chars": int}}`` per regex replacement
        ``entity_counts`` — same shape, per NER entity type
        ``language_confidence``, ``language_source``, ``warnings``, ``provenance``

    ``.signals`` is the value-free public view of these (see :mod:`sct.signals`).

    Metadata keys present only while their feature is active (absent otherwise):
        ``anon_map``  — :class:`AnonymizationMap` when ``replacement_mode='reversible'``
        ``anon_map_reference`` — opaque external token-store reference
        ``classes``   — ``list[dict]`` when ``check_classify_document=True``
        ``findings``  — policy decisions and detected entity spans, with
        ``include_entities=True`` or a privacy policy
        ``policy``    — active policy name and version
    """

    __slots__ = ('lm_text', 'stat_text', 'language', 'metadata')

    def __init__(
        self,
        lm_text: str,
        stat_text: str | None,
        language: str | None,
        metadata: ProcessMetadata | None = None,
    ):
        self.lm_text = lm_text
        self.stat_text = stat_text
        self.language = language
        self.metadata = {} if metadata is None else metadata

    @property
    def signals(self) -> "Signals":
        """Value-free view: counts, shares, types, language, models; never entity text."""
        from sct.signals import public_view
        return public_view(self)

    # --- 3-tuple protocol (backward compat) ---

    def __iter__(self):
        yield self.lm_text
        yield self.stat_text
        yield self.language

    def __len__(self):
        return 3

    def __getitem__(self, idx):
        return (self.lm_text, self.stat_text, self.language)[idx]

    def __repr__(self):
        # Never findings, anon_map or stat_text: they can hold the replaced values.
        return (
            f'ProcessResult(lm_text={self.lm_text!r}, '
            f'language={self.language!r}, signals={self.signals!r})'
        )

    def __eq__(self, other):
        if isinstance(other, ProcessResult):
            return (
                self.lm_text == other.lm_text
                and self.stat_text == other.stat_text
                and self.language == other.language
            )
        if isinstance(other, tuple) and len(other) == 3:
            return (self.lm_text, self.stat_text, self.language) == other
        return NotImplemented
