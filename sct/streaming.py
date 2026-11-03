"""Streaming interfaces with explicit privacy guarantees."""

import re
from bisect import bisect_right

from sct.sct import TextCleaner
from sct.utils import constants, contact

# A digit run at the end of the buffer may be a phone the next feed completes ("06 12 34").
_PHONE_TAIL = re.compile(r"[+(]?\d[\d ()+./-]*\Z")
# Longer than any phone the regex accepts; a longer digit tail (a number table) may still split.
_PHONE_TAIL_MAX = 32


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


def _user_rules_split_safe(user_rules, view, cut, right_start):
    """Whether each user-rule match of *view* is found again in the half that holds it: a custom pattern
    may read context outside its match ("(?<=account )\\d{6}"), which a split can take away."""
    for pattern, spans in user_rules:
        found = {m.span() for m in pattern.finditer(view[:cut])}
        found |= {(a + right_start, b + right_start)
                  for a, b in (m.span() for m in pattern.finditer(view[right_start:], partial=True))}
        if not spans <= found:
            return False
    return True


class WindowedStreamingRedactor:
    """Emit bounded chunks while retaining a configurable token overlap.

    This mode is suitable for bounded contact tokens such as emails, phone
    numbers and dates. Use ``BufferedStreamingRedactor`` when multiword NER entities must
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

        # Never split inside a contact token that holds spaces ("06 12 34 56 78"). The guard reads the
        # cleaner's own normalised text (cleaner.normalise, the steps process runs before its regexes)
        # and the regexes on the views the cleaner's steps use, so the two cannot differ.
        cleaner, cfg = self._cleaner, self._cleaner.cfg
        normalise = cleaner.normalise
        # The unfinished last token is left out when it changes the view before it: a cut escape ("\\U000000")
        # stops the cleaner's escape decoding for the whole buffer (the whole text decodes it once complete).
        buffer, view = self._buffer, normalise(self._buffer)
        head = buffer[:max((i for i, c in enumerate(buffer) if c.isspace()), default=-1) + 1]
        if head != buffer and not view.startswith(head_view := normalise(head)):
            buffer, view = head, head_view
        # The overlap is kept on the text the guard reads: a split needs overlap_chars of it to its right, so a
        # rule that reads past a space (a phone's next group: "㉚ ⓪⑦" folds to "30 07") sees what the whole
        # document gives it.
        cutoff = len(buffer) - self._overlap_chars
        phone_view = contact.PHONE_VIEW if cfg.fold_styled_characters else contact.PHONE_DIGIT_VIEW
        views = [cleaner._url_regex, constants.EMAIL_REGEX, contact.IP_VIEW, phone_view]
        if cfg.check_replace_html:  # a tag holding a space ('<a title="x y">')
            views.append(cleaner._html_regex)
        if cfg.check_replace_dates:
            views.append(cleaner._date_view)
        if cfg.check_fuzzy_replace_dates:  # every candidate the fuzzy step may mask ("15 Janury 1990")
            views.append(constants.FUZZY_DATE_CANDIDATE_REGEX)
        if cfg.check_replace_iban_cards:  # "NL91 ABNA 0417 1643 00"
            views += [constants.IBAN_REGEX, constants.CARD_REGEX]
        if cfg.check_replace_numbers:  # a grouped number ("5 379") is one token too
            views.append(contact.numbers_regex(cfg.replace_numbers_in_words))
        # (start, end) of view text a split must not cut: start < split < end
        protected = [
            match.span()
            for regex in views
            for match in (regex.pattern.finditer(view.translate(regex.table))
                          if isinstance(regex, contact.DigitView) else regex.finditer(view))
        ]
        # User rules (a deny term holding a space, "Blue Sky Initiative"): their matches, and a match the buffer
        # end cuts off ("... the Blue Sky"), which the next feed completes.
        user_rules = [(pattern, {m.span() for m in pattern.finditer(view, partial=True)})
                      for pattern, _ in cleaner._user_rules]
        protected += [span for _, spans in user_rules for span in spans]
        phone_digits = view.translate(phone_view.table)
        tail = _PHONE_TAIL.search(phone_digits, max(0, len(phone_digits) - _PHONE_TAIL_MAX))
        if tail:
            protected.append((tail.start(), len(view) + 1))

        # A whitespace split is safe where the view is the two halves' own views with at most one whitespace
        # character between them ("\r\n" is one "\n"), so no step reads across it, and no protected match
        # covers the cut. Candidates are tried from the cutoff back; a protected one jumps to the last candidate
        # whose own view ends at or before the match start (a bisect on view offsets, since NFKC can make the
        # view longer, "⑫" -> "12", or escapes shorter), so a feed normalises a few times, not once per space.
        candidates = [index for index, character in enumerate(buffer[:cutoff + 1]) if character.isspace()]
        split_at, index = -1, len(candidates) - 1
        while index >= 0:
            position = candidates[index]
            left = normalise(buffer[:position])
            offset = len(left)
            if not view.startswith(left):
                index -= 1
                continue
            cover = next((start for start, end in protected if start < offset < end), None)
            if cover is not None:
                index = bisect_right(candidates, cover, hi=index, key=lambda p: len(normalise(buffer[:p]))) - 1
                continue
            right = normalise(buffer[position + 1:])
            separator = view[offset:len(view) - len(right)]
            if len(view) - offset - len(right) not in (0, 1) or not view.endswith(right) \
                    or not (separator == "" or separator.isspace()):
                index -= 1
                continue
            if not _user_rules_split_safe(user_rules, view, offset, len(view) - len(right)):
                index -= 1
                continue
            split_at = position
            break
        if split_at < 0:
            raise BufferError(
                "Streaming buffer exceeded max_buffer_chars without a safe whitespace split. "
                "Increase the limit or use BufferedStreamingRedactor."
            )

        prefix = self._buffer[:split_at]
        # the cleaner's whitespace step turns any space (\u00a0, \u202f) into " "; the separator follows it
        separator = " " if cfg.check_normalize_whitespace else separator
        self._buffer = self._buffer[split_at + 1:]
        return self._cleaner.process(prefix).lm_text + separator

    def finish(self) -> str:
        """Process the final retained window."""
        if self._finished:
            raise RuntimeError("Stream is already finished")
        self._finished = True
        return self._cleaner.process(self._buffer).lm_text
