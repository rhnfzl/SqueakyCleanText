"""Clean an entire document and split only the cleaned result for downstream use."""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
import re
from typing import TYPE_CHECKING

from sct.chunking import ChunkRequest, ChunkingPolicy, ChunkingRule
from sct.documents import DocumentElement, ExtractedDocument, _clean_document
from sct.findings import DetectionContext, TextRange
from sct.utils.markers import PLACEHOLDER_TOKEN

if TYPE_CHECKING:
    from sct.sct import TextCleaner
    from sct.unified import RenderedEdit


@dataclass(frozen=True)
class CleanedChunk:
    """A chunk and its half-open Unicode range in the cleaned document."""

    text: str = field(repr=False)
    range: TextRange
    element_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ChunkedDocument:
    chunks: tuple[CleanedChunk, ...] = field(repr=False)
    chunker: str
    language: str | None


OUTPUT_CHUNKING = ChunkingPolicy(
    'auto', candidates=('text',), rules=(ChunkingRule('markdown', formats=('markdown', 'text/markdown')),),
)


def _element_ranges(elements: tuple[DocumentElement, ...]) -> tuple[tuple[TextRange, str], ...]:
    start = 0
    result = []
    for item in elements:
        end = start + len(item.text)
        if end > start:
            result.append((TextRange(start, end), item.id))
        start = end + len(item.separator)
    return tuple(result)


def _protected(text: str, edits: Iterable[tuple[int, int]], custom: tuple[str, ...]) -> tuple[TextRange, ...]:
    boundaries = sorted((
        *edits,
        *(match.span() for match in PLACEHOLDER_TOKEN.finditer(text)),
        *(match.span(1) for marker in custom for match in re.finditer(rf'(?=({re.escape(marker)}))', text)),
    ))
    spans: list[TextRange] = []
    for start, end in boundaries:
        if start == end:
            continue
        if spans and spans[-1].end > start:
            prior = spans[-1]
            spans[-1] = TextRange(prior.start, max(prior.end, end))
        else:
            spans.append(TextRange(start, end))
    return tuple(spans)


def chunk_document(
    cleaner: TextCleaner, document: str | ExtractedDocument, *,
    chunking: ChunkingPolicy | str | Mapping | None = None,
    max_size: int | None = None, count: Callable[[str], int] = len,
    document_format: str | None = None, context: DetectionContext | None = None,
) -> ChunkedDocument:
    """Apply the cleaner once, then return chunks in the selected size units."""
    if cleaner.cfg.replacement_mode in ('indexed', 'reversible'):
        raise ValueError('Chunked output does not support indexed or reversible modes')
    if cleaner.cfg.replacement_mode == 'synthetic' and cleaner.cfg.pii_processing != 'unified':
        raise ValueError('Chunked synthetic output requires unified processing')
    if context is not None and not isinstance(context, DetectionContext):
        raise ValueError('Expected DetectionContext')
    context = context or DetectionContext()
    if document_format is not None:
        context = replace(context, document_format=document_format)
    document_format = context.document_format
    policy = ChunkingPolicy.coerce(chunking) or cleaner.cfg.output_chunking
    edits: tuple[RenderedEdit, ...] = ()
    elements: tuple[tuple[TextRange, str], ...] = ()
    if type(document) is ExtractedDocument:
        if document_format is None:
            document_format = 'document'
        cleaned, processed = _clean_document(document, cleaner, context)
        elements = _element_ranges(cleaned.elements)
        result = processed.result
        edits = processed.rendered_edits
        policy = policy or ChunkingPolicy('elements')
    elif isinstance(document, str):
        if cleaner.cfg.pii_processing == 'unified':
            processed = cleaner._process_unified_document(document, context)
            result, edits = processed.result, processed.rendered_edits
        else:
            if replace(context, document_format=None) != DetectionContext():
                raise ValueError('Detection context requires unified processing')
            result = cleaner.process(document)
        policy = policy or OUTPUT_CHUNKING
    else:
        raise ValueError('Expected a string or ExtractedDocument')
    text = result.lm_text
    ranges = tuple(span for span, _ in elements)
    protected = _protected(text, ((edit.start, edit.end) for edit in edits), cleaner._custom_markers)
    budget = cleaner.cfg.output_chunk_max_size if max_size is None else max_size
    request = ChunkRequest(text, budget, count, 'output', result.language, document_format,
                           ranges, cleaner.cfg.allow_network_chunkers, protected)
    registry = cleaner._chunker_registry
    name = registry.resolve(policy, stage='output', language=result.language, document_format=document_format,
                            length=len(text), allow_network=request.allow_network)
    if name == 'native':
        name = 'elements' if type(document) is ExtractedDocument else 'text'
    chosen = registry.plan(name, request) if text else ()
    starts, ends = tuple(r.start for r in ranges), tuple(r.end for r in ranges)
    return ChunkedDocument(tuple(
        CleanedChunk(text[span.start:span.end], span,
                     tuple(identifier for _, identifier in elements[
                         bisect_right(ends, span.start):bisect_left(starts, span.end)]))
        for span in chosen
    ), name, result.language)
