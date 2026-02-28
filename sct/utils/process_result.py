"""ProcessResult — backward-compatible result wrapper with metadata.

Unpacks as a 3-tuple for existing ``a, b, c = cleaner.process(text)`` code,
while exposing ``.metadata`` for new features (reversible anonymization maps,
document classification, etc.).
"""

from typing import Optional


class ProcessResult:
    """Result of ``TextCleaner.process()`` / ``process_batch()``.

    Behaves as a 3-tuple ``(lm_text, stat_text, language)`` for backward
    compatibility, but carries an optional ``.metadata`` dict for extended
    features.

    Metadata keys (all optional, ``None`` when feature is inactive):
        ``anon_map``  — :class:`AnonymizationMap` when ``replacement_mode='reversible'``
        ``classes``   — ``list[dict]`` when ``check_classify_document=True``
    """

    __slots__ = ('lm_text', 'stat_text', 'language', 'metadata')

    def __init__(
        self,
        lm_text: str,
        stat_text: Optional[str],
        language: Optional[str],
        metadata: Optional[dict] = None,
    ):
        self.lm_text = lm_text
        self.stat_text = stat_text
        self.language = language
        self.metadata = metadata

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
        meta = f', metadata={self.metadata!r}' if self.metadata else ''
        return (
            f'ProcessResult(lm_text={self.lm_text!r}, '
            f'stat_text={self.stat_text!r}, '
            f'language={self.language!r}{meta})'
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
