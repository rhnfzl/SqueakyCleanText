"""Streaming interfaces with explicit privacy guarantees."""

import html
import re
import unicodedata

from sct.sct import TextCleaner
from sct.utils import constants
from sct.utils.contact import _DIGIT_VIEW
from sct.utils.normtext import _STYLED_SEP

_CONTACT_REGEXES = (
    constants.EMAIL_REGEX, constants.PHONE_REGEX, constants.URL_REGEX, constants.IP_REGEX,
)
# A digit run at the end of the buffer may be a phone the next feed completes ("06 12 34").
_PHONE_TAIL = re.compile(r"[+(]?\d[\d ()+./-]*\Z")
# Longer than any phone the regex accepts; a longer digit tail (a number table) may still split.
_PHONE_TAIL_MAX = 32
# One HTML entity (decoded later by the cleaner's ftfy step) or one character.
_VIEW_UNIT = re.compile(r"&(?:#[0-9]+|#[xX][0-9a-fA-F]+|[A-Za-z][A-Za-z0-9]*);|.", re.DOTALL)


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
        # Never split inside a contact token that holds spaces ("06 12 34 56 78"). Match on a
        # view close to what the cleaner sees: HTML entities decoded ("12&#32;34") and each letter,
        # digit, space and contact punctuation NFKC-folded ("０６ １２") with any decimal digit as
        # ASCII ("٠٦ ١٢", as the cleaner's digit view); starts/ends map view offsets back to the buffer.
        view, starts, ends = [], [], []
        for unit in _VIEW_UNIT.finditer(self._buffer):
            decoded = html.unescape(unit.group())
            if len(decoded) != 1:  # an unknown entity stays as its raw characters
                decoded, unit_ends = unit.group(), range(unit.start() + 1, unit.end() + 1)
            else:
                unit_ends = (unit.end(),)
            for character, end in zip(decoded, unit_ends):
                folded = unicodedata.normalize("NFKC", character).translate(_DIGIT_VIEW)
                # punctuation folds only where the cleaner's styled fold does (contact punctuation),
                # so a "（" glued after a phone stays wide and does not end the match early
                if not (folded.isalnum() or folded.isspace()) and character not in _STYLED_SEP:
                    folded = character
                # styled multi-character folds ("⑫" -> "12") keep every character, all on one source span
                if len(folded) != 1 and not (folded.isascii() and folded.isalnum()):
                    folded = character
                for view_character in folded:
                    view.append(view_character)
                    starts.append(end - 1 if len(unit_ends) > 1 else unit.start())
                    ends.append(end)
        view = "".join(view)
        inside = {
            index
            for regex in _CONTACT_REGEXES
            for match in regex.finditer(view)
            for index in range(starts[match.start()] + 1, ends[match.end() - 1])
        }
        tail = _PHONE_TAIL.search(view, max(0, len(view) - _PHONE_TAIL_MAX))
        if tail:
            inside.update(range(starts[tail.start()] + 1, len(self._buffer)))
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
