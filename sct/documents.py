"""Clean extracted document text while preserving safe layout provenance."""
from __future__ import annotations

from collections.abc import Mapping, Set
from dataclasses import dataclass, field
import math
from pathlib import Path
import re
from typing import TYPE_CHECKING, Literal, Protocol

from sct.findings import DetectionContext

if TYPE_CHECKING:
    from sct.sct import TextCleaner
    from sct.unified import UnifiedDocument


@dataclass(frozen=True)
class DocumentElement:
    """Text plus structural coordinates, never a source filename or SDK object."""

    id: str
    text: str = field(repr=False)
    kind: Literal['text', 'table_cell'] = 'text'
    page: int | None = None
    bbox: tuple[float, float, float, float] | None = None
    cell: tuple[int, int, int, int] | None = None
    table: int | None = None
    separator: str = '\n'

    def __post_init__(self) -> None:
        if type(self.id) is not str or not re.fullmatch(r'e[0-9]+', self.id):
            raise ValueError('Element IDs must be structural e-prefixed integers')
        if type(self.text) is not str or type(self.kind) is not str or self.kind not in ('text', 'table_cell'):
            raise ValueError('Element text or kind is invalid')
        if self.page is not None and (type(self.page) is not int or self.page < 1):
            raise ValueError('Element page must be a positive integer')
        if self.bbox is not None:
            if (type(self.bbox) is not tuple or len(self.bbox) != 4
                    or any(type(v) not in (int, float) or not math.isfinite(v) for v in self.bbox)
                    or self.bbox[0] > self.bbox[2] or self.bbox[1] > self.bbox[3]):
                raise ValueError('Element bounds must be finite ordered coordinates')
        if self.table is not None and (self.kind != 'table_cell' or type(self.table) is not int or self.table < 0):
            raise ValueError('Table identity must be a nonnegative integer')
        if self.cell is not None:
            if (self.kind != 'table_cell' or type(self.cell) is not tuple or len(self.cell) != 4
                    or any(type(v) is not int or v < 0 for v in self.cell)
                    or self.cell[0] >= self.cell[1] or self.cell[2] >= self.cell[3]):
                raise ValueError('Table cell requires nonempty row and column spans')
        if type(self.separator) is not str or any(c not in ' \t\r\n' for c in self.separator):
            raise ValueError('Element separator must contain only layout whitespace')


def _validate_elements(elements: tuple[DocumentElement, ...]) -> None:
    if type(elements) is not tuple or any(type(e) is not DocumentElement for e in elements):
        raise ValueError('Document requires an immutable tuple of typed elements')
    for element in elements:
        element.__post_init__()
    if len({e.id for e in elements}) != len(elements):
        raise ValueError('Document element IDs must be unique')


def _join(elements: tuple[DocumentElement, ...]) -> str:
    return ''.join(e.text + (e.separator if i + 1 < len(elements) else '') for i, e in enumerate(elements))


@dataclass(frozen=True)
class ExtractedDocument:
    """Sensitive extracted text and its canonical Unicode source ranges."""

    elements: tuple[DocumentElement, ...] = field(repr=False)

    def __post_init__(self) -> None:
        _validate_elements(self.elements)

    @property
    def text(self) -> str:
        return _join(self.elements)

    @property
    def source_ranges(self) -> tuple[tuple[int, int], ...]:
        start = 0
        ranges = []
        for element in self.elements:
            end = start + len(element.text)
            ranges.append((start, end))
            start = end + len(element.separator)
        return tuple(ranges)


@dataclass(frozen=True)
class ProcessedDocument:
    """Cleaned text and allowlisted structure without retained input state."""

    text: str = field(repr=False)
    elements: tuple[DocumentElement, ...] = field(repr=False)

    def __post_init__(self) -> None:
        _validate_elements(self.elements)
        if type(self.text) is not str:
            raise ValueError('Cleaned text must use a plain string')
        if _join(self.elements) != self.text:
            raise ValueError('Cleaned structure must reconstruct the rendered text')

    @property
    def element_ids(self) -> tuple[str, ...]:
        return tuple(element.id for element in self.elements)


class DocumentExtractor(Protocol):
    def extract(self, path: Path) -> ExtractedDocument: ...


def extract_document(path: Path, *, provider: DocumentExtractor) -> ExtractedDocument:
    """Extract without modifying the caller source or exposing conversion errors."""
    try:
        result = provider.extract(path)
        if type(result) is not ExtractedDocument:
            raise ValueError('Expected ExtractedDocument')
        result.__post_init__()
    except Exception:
        raise ValueError('Document extraction failed; check the provider and local artifacts') from None
    return result


DocumentOutput = Literal['text', 'ocr', 'tables', 'coordinates']


class CapableDocumentExtractor(DocumentExtractor, Protocol):
    @property
    def capabilities(self) -> Mapping[str, frozenset[DocumentOutput]]: ...


class DocumentExtractionRouter:
    """Use compatible extractors in caller order for one declared format."""

    def __init__(
        self,
        *,
        format: str,
        required: Set[DocumentOutput],
        providers: Mapping[str, CapableDocumentExtractor],
        fallback: bool,
        pin: str | None = None,
    ) -> None:
        if type(format) is not str or not format.strip() or type(fallback) is not bool:
            raise ValueError('Document routing requires a format and a boolean fallback policy')
        self._required = frozenset(required)
        if ('text' not in self._required
                or not self._required <= {'text', 'ocr', 'tables', 'coordinates'}):
            raise ValueError('Document routing requires text and supports ocr, tables and coordinates')
        compatible = tuple(
            provider for name, provider in providers.items()
            if (pin is None or name == pin)
            and self._required <= provider.capabilities.get(format, frozenset())
        )
        if not compatible:
            if pin is not None:
                raise ValueError('Pinned document extractor is unavailable or incompatible')
            raise ValueError('No document extractor matches the format and required output')
        self._providers = compatible if fallback and pin is None else compatible[:1]

    def extract(self, path: Path) -> ExtractedDocument:
        for provider in self._providers:
            try:
                document = extract_document(path, provider=provider)
                self._check_required_structure(document)
            except ValueError:
                continue
            return document
        raise ValueError('No compatible document extractor produced the required output')

    def _check_required_structure(self, document: ExtractedDocument) -> None:
        for element in document.elements:
            if 'coordinates' in self._required:
                box = element.bbox
                if element.page is None or box is None or box[0] >= box[2] or box[1] >= box[3]:
                    raise ValueError('Required document coordinates are missing')
            if ('tables' in self._required and element.kind == 'table_cell'
                    and (element.cell is None or element.table is None)):
                raise ValueError('Required table cell structure is missing')


def clean_document(document: ExtractedDocument, cleaner: TextCleaner) -> ProcessedDocument:
    """Detect once across elements; place each rendered span at its first source element."""
    return _clean_document(document, cleaner, None)[0]


def _clean_document(document: ExtractedDocument, cleaner: TextCleaner,
                    context: DetectionContext | None) -> tuple[ProcessedDocument, UnifiedDocument]:
    if type(document) is not ExtractedDocument:
        raise ValueError('Expected ExtractedDocument')
    document.__post_init__()
    if cleaner.cfg.pii_processing != 'unified':
        raise ValueError('Document cleaning requires explicit unified processing')
    if cleaner.cfg.replacement_mode in ('indexed', 'reversible'):
        raise ValueError('Document output does not support indexed or reversible modes')
    try:
        processed = (
            cleaner._process_unified_document(document.text) if context is None
            else cleaner._process_unified_document(document.text, context)
        )
    except Exception:
        raise ValueError('Document cleaning failed; check the cleaner configuration') from None
    text = processed.result.lm_text
    mapping = processed.source_map
    rendered = processed.rendered_edits
    boundaries = {position for edit in rendered for position in (edit.start, edit.end)}
    ranges = document.source_ranges
    texts = [''] * len(ranges)
    separators = [''] * len(ranges)
    index = 0
    while index < len(text):
        projection = mapping.project(index, index + 1)
        end = index + 1
        while end < len(text) and end not in boundaries and mapping.project(end, end + 1) == projection:
            end += 1
        if projection.mapping_quality == 'unavailable':
            raise ValueError('Document projection is unavailable for this transformation')
        touched = [i for i, (start, stop) in enumerate(ranges)
                   if any(r.start < stop and start < r.end for r in projection.ranges)]
        if len(touched) > 1:
            # Only actual rendered edits prove an association across element boundaries.
            if not any(edit.start <= index and end <= edit.end
                       and edit.source_isolated
                       and edit.source_projection.ranges == projection.ranges for edit in rendered):
                raise ValueError('Document projection has ambiguous cross-element coverage')
        if touched:
            texts[touched[0]] += text[index:end]
        else:
            owners = [i for i, (_, stop) in enumerate(ranges[:-1])
                      if all(stop <= r.start and r.end <= ranges[i + 1][0] for r in projection.ranges)]
            if len(owners) != 1 or any(c not in ' \t\r\n' for c in text[index:end]):
                raise ValueError('Document projection cannot assign rendered separators')
            separators[owners[0]] += text[index:end]
        index = end
    elements = tuple(DocumentElement(
        id=e.id, text=texts[i], kind=e.kind, page=e.page, bbox=e.bbox, cell=e.cell, table=e.table,
        separator=separators[i]) for i, e in enumerate(document.elements))
    return ProcessedDocument(text, elements), processed
