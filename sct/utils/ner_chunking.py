"""Split text into NER windows: plain functions over text and a tokenizer, no model state."""

import re
from collections.abc import Callable
from typing import Any

from sct.utils import constants


def token_count(text: str, tokenizer: Callable[..., Any]) -> int:
    """Count tokens in text."""
    return len(tokenizer(text).input_ids)


def split_pieces(pattern: re.Pattern[str], text: str) -> list[tuple]:
    """Split *text* on *pattern* into ``(separator_before, piece)`` pairs.

    Keeping the matched separator lets chunks be rejoined exactly as in
    the source (no invented spaces between CJK sentences). Empty pieces
    are dropped; their separators fold into the next piece's separator.
    """
    pieces = []
    last = 0
    separator = ''
    for match in pattern.finditer(text):
        if match.start() > last:
            pieces.append((separator, text[last:match.start()]))
            separator = ''
        separator += match.group()
        last = max(last, match.end())
    if last < len(text):
        pieces.append((separator, text[last:]))
    return pieces


def merge_pieces(pieces: list[tuple], fits: Callable[[str], bool]) -> list[str]:
    """Greedily merge adjacent ``(separator, piece)`` pairs while ``fits(chunk)``.

    Pieces inside a chunk are rejoined with their original separator.
    """
    if not pieces:
        return []
    chunks = []
    current = pieces[0][1]
    for separator, piece in pieces[1:]:
        merged = current + separator + piece
        if fits(merged):
            current = merged
        else:
            chunks.append(current)
            current = piece
    chunks.append(current)
    return chunks


def simple_chunk(text: str, max_tokens: int = 384) -> list[str]:
    """Split text into chunks of at most ``max_tokens * 2`` characters for GLiNER.

    Used for GLiNER-only mode where no ONNX/torch tokenizer is available.
    Every chunk fits the budget: oversized pieces are split again with
    finer delimiters, and runs without any delimiter are cut hard.
    """
    max_chars = max_tokens * 2  # conservative: CJK=1, Arabic=2-3, Latin=4 chars/token
    if len(text) <= max_chars:
        return [text]

    for delimiter in constants.CHUNK_DELIMITERS:
        pieces = split_pieces(delimiter, text)
        if len(pieces) <= 1:
            continue
        chunks: list[str] = []
        for chunk in merge_pieces(pieces, lambda c: len(c) <= max_chars):
            chunks.extend(simple_chunk(chunk, max_tokens))
        return chunks

    # shortcut: hard cut can split an entity; ner_chunk_overlap > 0 covers it
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]


def align_chunk(
    source: str,
    chunk: str,
    search_start: int,
) -> tuple[str, int, list[int]]:
    """Locate a whitespace-normalized chunk and map its offsets to source."""
    # Chunks are normally exact source slices; a greedy \s+ match would eat
    # whitespace that belongs to the next chunk (hard cuts, blank chunks).
    start = source.find(chunk, search_start)
    if start >= 0:
        return chunk, start, list(range(len(chunk) + 1))
    parts = re.split(r'(\s+)', chunk)
    pattern = ''.join(
        r'\s+' if part.isspace() else re.escape(part)
        for part in parts if part
    )
    match = re.compile(pattern).search(source, search_start)
    if match is None:
        raise RuntimeError("Could not align NER chunk to its source text")

    source_chunk = match.group()
    offsets = [0] * (len(chunk) + 1)
    chunk_index = 0
    source_index = 0
    while chunk_index < len(chunk):
        if chunk[chunk_index].isspace():
            chunk_end = chunk_index
            while chunk_end < len(chunk) and chunk[chunk_end].isspace():
                chunk_end += 1
            source_end = source_index
            while (
                source_end < len(source_chunk)
                and source_chunk[source_end].isspace()
            ):
                source_end += 1
            chunk_width = chunk_end - chunk_index
            source_width = source_end - source_index
            for offset in range(chunk_width + 1):
                offsets[chunk_index + offset] = (
                    source_index
                    + (offset * source_width // chunk_width)
                )
            chunk_index = chunk_end
            source_index = source_end
        else:
            offsets[chunk_index] = source_index
            chunk_index += 1
            source_index += 1
    offsets[len(chunk)] = len(source_chunk)
    return source_chunk, match.start(), offsets


def split_text(text: str, max_tokens: int, tokenizer: Callable[..., Any]) -> list[str]:
    """Split text into token-bounded chunks using a delimiter hierarchy.

    Tries sentence boundaries first (preferred: respects sentence integrity),
    then falls back to CHUNK_DELIMITERS from coarsest (paragraph) to finest
    (whitespace). Recurses on oversized chunks. Falls back to character-level
    splitting via token offsets as last resort.
    """
    def fits(chunk):
        return token_count(chunk, tokenizer) <= max_tokens

    if fits(text):
        return [text]

    # Sentence boundaries first, to avoid fragmenting entities mid-sentence.
    for pattern in (constants.SENTENCE_BOUNDARY_PATTERN, *constants.CHUNK_DELIMITERS):
        pieces = split_pieces(pattern, text)
        if len(pieces) <= 1:
            continue
        # Greedily merge small adjacent pieces; recurse on any chunk still over the limit.
        result = []
        for chunk in merge_pieces(pieces, fits):
            if fits(chunk):
                result.append(chunk)
            else:
                result.extend(split_text(chunk, max_tokens, tokenizer))
        return result

    return _split_by_token_offsets(text, max_tokens, tokenizer, fits)


def _split_by_token_offsets(text: str, max_tokens: int, tokenizer: Callable[..., Any],
                            fits: Callable[[str], bool]) -> list[str]:
    """Last resort: character-level splitting via token offsets."""
    tokenized = tokenizer(text, return_offsets_mapping=True)
    offsets = tokenized.offset_mapping
    # Reserve room for the special tokens ([CLS]/[SEP]) each chunk gets back.
    budget = max(1, max_tokens - token_count('', tokenizer))
    chunks = []
    last_end = 0
    current_tokens = 0

    for start, end in offsets:
        if end <= start:
            continue  # special token, no source text
        if current_tokens >= budget:
            chunks.append(text[last_end:start])
            last_end = start
            current_tokens = 0
        current_tokens += 1

    if last_end < len(text):
        chunks.append(text[last_end:])

    # Re-tokenizing a slice can add tokens (word-start markers), so halve any
    # slice that still does not fit until it does.
    result = []
    for chunk in chunks:
        if len(chunk) > 1 and not fits(chunk):
            mid = len(chunk) // 2
            result.extend(split_text(chunk[:mid], max_tokens, tokenizer))
            result.extend(split_text(chunk[mid:], max_tokens, tokenizer))
        else:
            result.append(chunk)
    return result
