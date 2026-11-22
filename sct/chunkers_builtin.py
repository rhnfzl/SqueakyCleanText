"""Built-in slice planners that preserve text, replacements and structure."""
from bisect import bisect_right
from collections.abc import Iterable
from importlib.util import find_spec
import re

from sct.chunking import ChunkBudgetError, ChunkerRegistry, ChunkRequest, legal_cut
from sct.findings import TextRange

_MARKDOWN_HEADING = re.compile(r'^ {0,3}#{1,6}(?=\s)', re.MULTILINE)
_PARAGRAPHS = re.compile(r'\n[ \t]*\n')
_LINE_ENDS = re.compile(r'\r?\n')
_WHITESPACE = re.compile(r'\s+')


def _limit(request: ChunkRequest, start: int) -> int:
    text, count, budget = request.text, request.count, request.max_size
    end = len(text)
    if count is len:
        return min(start + budget, end)
    # Bound the search from the current position to avoid recounting the whole tail.
    lo, hi = start, min(end, start + budget)
    while count(text[start:hi]) <= budget:
        if hi == end:
            return hi
        lo, hi = hi, min(end, start + 2 * (hi - start))
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if count(text[start:mid]) <= budget:
            lo = mid
        else:
            hi = mid
    return lo


def bounded_ranges(request: ChunkRequest, preferred: Iterable[int] = ()) -> Iterable[TextRange]:
    text = request.text
    protected_starts = tuple(span.start for span in request.protected)
    boundaries = [
        sorted({legal_cut(pos, request.protected, protected_starts) for pos in preferred if 0 < pos < len(text)}),
        *[
            sorted({legal_cut(match.end(), request.protected, protected_starts) for match in pattern.finditer(text)})
            for pattern in (_PARAGRAPHS, _LINE_ENDS, _WHITESPACE)
        ],
    ]
    start = 0
    while start < len(text):
        limit = legal_cut(_limit(request, start), request.protected, protected_starts)
        if limit <= start:
            raise ChunkBudgetError('The size budget cannot fit the next character or replacement')
        end = limit
        if limit < len(text):
            for cuts in boundaries:
                index = bisect_right(cuts, limit) - 1
                if index >= 0 and cuts[index] > start and request.count(text[start:cuts[index]]) <= request.max_size:
                    end = cuts[index]
                    break
        yield TextRange(start, end)
        start = end


def _markdown(request: ChunkRequest) -> Iterable[TextRange]:
    return bounded_ranges(request, (match.start() for match in _MARKDOWN_HEADING.finditer(request.text)))


def _elements(request: ChunkRequest) -> Iterable[TextRange]:
    return bounded_ranges(request, (element.start for element in request.elements))


def _semchunk(request: ChunkRequest) -> Iterable[TextRange]:
    from semchunk import chunk

    chunks, offsets = chunk(request.text, request.max_size, request.count, offsets=True, memoize=False)
    cuts = [
        end for text, (start, end) in zip(chunks, offsets, strict=True)
        if isinstance(start, int) and isinstance(end, int) and request.text[start:end] == text
    ]
    return bounded_ranges(request, cuts)


def _chonkie(request: ChunkRequest) -> Iterable[TextRange]:
    from chonkie import RecursiveChunker

    # Chonkie's offsets are hints. SCT copies and counts only slices of its own text.
    chunks = RecursiveChunker(tokenizer='character', chunk_size=request.max_size).chunk(request.text)
    cuts = [item.end_index for item in chunks
            if request.text[item.start_index:item.end_index] == item.text]
    return bounded_ranges(request, cuts)


def register_builtin_chunkers(registry: ChunkerRegistry) -> None:
    registry.register('text', bounded_ranges, stages=('detection', 'output'))
    registry.register('markdown', _markdown)
    registry.register('elements', _elements)
    registry.register('semchunk', _semchunk, stages=('detection', 'output'),
                      available=lambda: find_spec('semchunk') is not None)
    registry.register('chonkie', _chonkie, available=lambda: find_spec('chonkie') is not None)
