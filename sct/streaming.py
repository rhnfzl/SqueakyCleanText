"""Streaming interfaces with explicit privacy guarantees."""

import unicodedata

from sct.sct import TextCleaner
from sct.utils import constants

_CONTACT_REGEXES = (
    constants.EMAIL_REGEX, constants.PHONE_REGEX, constants.URL_REGEX, constants.IP_REGEX,
)


class BufferedStreamingRedactor:
    """Buffer a document so entities split across chunks remain detectable."""

    def __init__(self, cleaner: TextCleaner) -> None:
        self._cleaner = cleaner
        self._chunks: list[str] = []
        self._finished = False

    def feed(self, chunk: str) -> str:
        """Buffer a chunk and emit no text before the document is complete."""
        if self._finished:
            raise RuntimeError("Cannot feed a finished stream")
        self._chunks.append(chunk)
        return ""

    def finish(self) -> str:
        """Process and return the complete buffered document."""
        if self._finished:
            raise RuntimeError("Stream is already finished")
        self._finished = True
        return self._cleaner.process("".join(self._chunks)).lm_text


class WindowedStreamingRedactor:
    """Emit bounded chunks while retaining a configurable token overlap.

    This mode is suitable for bounded contact tokens such as emails and phone
    numbers. Use ``BufferedStreamingRedactor`` when multiword NER entities must
    have document-wide guarantees.
    """

    def __init__(
        self,
        cleaner: TextCleaner,
        *,
        max_buffer_chars: int = 4096,
        overlap_chars: int = 256,
    ) -> None:
        if overlap_chars < 1:
            raise ValueError("overlap_chars must be >= 1")
        if max_buffer_chars <= overlap_chars:
            raise ValueError("max_buffer_chars must be greater than overlap_chars")
        self._cleaner = cleaner
        self._max_buffer_chars = max_buffer_chars
        self._overlap_chars = overlap_chars
        self._buffer = ""
        self._finished = False

    def feed(self, chunk: str) -> str:
        """Process a stable prefix and retain a trailing overlap window."""
        if self._finished:
            raise RuntimeError("Cannot feed a finished stream")
        self._buffer += chunk
        if len(self._buffer) <= self._max_buffer_chars:
            return ""

        cutoff = len(self._buffer) - self._overlap_chars
        # Never split inside a contact token that holds spaces ("06 12 34 56 78"). Match
        # on a per-character NFKC view (same offsets) so fullwidth "０６ １２" counts too.
        view = "".join(
            folded if len(folded := unicodedata.normalize("NFKC", character)) == 1 else character
            for character in self._buffer
        )
        inside = {
            index
            for regex in _CONTACT_REGEXES
            for match in regex.finditer(view)
            for index in range(match.start() + 1, match.end())
        }
        split_at = max(
            (
                index
                for index, character in enumerate(self._buffer[:cutoff + 1])
                if character.isspace() and index not in inside
            ),
            default=-1,
        )
        if split_at < 0:
            raise BufferError(
                "Streaming buffer exceeded max_buffer_chars without a safe whitespace split. "
                "Increase the limit or use BufferedStreamingRedactor."
            )

        prefix = self._buffer[:split_at]
        separator = self._buffer[split_at]
        self._buffer = self._buffer[split_at + 1:]
        return self._cleaner.process(prefix).lm_text + separator

    def finish(self) -> str:
        """Process the final retained window."""
        if self._finished:
            raise RuntimeError("Stream is already finished")
        self._finished = True
        return self._cleaner.process(self._buffer).lm_text
