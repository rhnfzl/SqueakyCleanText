"""Prepare raw HTML fragment edits without transforming text or assigning provenance."""

import re
from html.parser import HTMLParser

_INLINE = frozenset({"b", "em", "i", "span", "strong"})
_BLOCK = frozenset({"div", "p", "table", "tr", "td", "th"})
_TABLE_PARENT = {"tr": "table", "td": "tr", "th": "tr"}
_TABLE_CONTAINERS = frozenset(_TABLE_PARENT.values())
_NO_BLOCK_CHILDREN = _INLINE | {"p"}
_START_TAG = re.compile(
    r"""<[a-z]+(?:[ \t\r\n\f]+[^\s"'<>/=`\x00]+"""
    r"""(?:[ \t\r\n\f]*=[ \t\r\n\f]*(?:"[^"\x00]*"|'[^'\x00]*'|[^\s"'=<>`\x00]+))?)*[ \t\r\n\f]*>""",
    re.IGNORECASE | re.ASCII,
)
_END_TAG = re.compile(r"</([a-z]+)[ \t\r\n\f]*>", re.IGNORECASE | re.ASCII)


class _UnsupportedFragment(ValueError):
    pass


class _FragmentParser(HTMLParser):
    def __init__(self, source: str):
        super().__init__(convert_charrefs=False)
        self.source = source
        self.line_starts = [0, *(match.end() for match in re.finditer("\n", source))]
        self.cursor = 0
        self.stack: list[str] = []
        self.edits: list[tuple[int, int, str]] = []

    def _position(self) -> int:
        line, column = self.getpos()
        return self.line_starts[line - 1] + column

    def _consume(self, raw: str) -> None:
        if self._position() != self.cursor or not self.source.startswith(raw, self.cursor):
            raise _UnsupportedFragment
        self.cursor += len(raw)

    def _tag(self, tag: str, raw: str) -> None:
        start = self.cursor
        self._consume(raw)
        separator = "\n" if tag in _BLOCK else ""
        if self.edits and self.edits[-1][1] == start:
            start, _, previous = self.edits.pop()
            separator = previous or separator
        self.edits.append((start, self.cursor, separator))

    def handle_starttag(self, tag, attrs):
        parent = self.stack[-1] if self.stack else None
        table_parent = _TABLE_PARENT.get(tag)
        if (
            tag not in _INLINE and tag not in _BLOCK
            or table_parent is not None and parent != table_parent
            or parent in _TABLE_CONTAINERS and parent != table_parent
            or tag in _BLOCK and parent in _NO_BLOCK_CHILDREN
        ):
            raise _UnsupportedFragment
        raw = self.get_starttag_text()
        if raw is None or not _START_TAG.fullmatch(raw) or len({name for name, _ in attrs}) != len(attrs):
            raise _UnsupportedFragment
        self._tag(tag, raw)
        self.stack.append(tag)

    def handle_endtag(self, tag):
        match = _END_TAG.match(self.source, self.cursor)
        if not match or match[1].lower() != tag or not self.stack or self.stack[-1] != tag:
            raise _UnsupportedFragment
        self._tag(tag, match[0])
        self.stack.pop()

    def handle_data(self, data):
        if "<" in data or "\x00" in data:
            raise _UnsupportedFragment
        if self.stack and self.stack[-1] in _TABLE_CONTAINERS and data.strip():
            raise _UnsupportedFragment
        self._consume(data)

    def handle_entityref(self, name):
        self._reference("&" + name)

    def handle_charref(self, name):
        self._reference("&#" + name)

    def _reference(self, raw: str) -> None:
        if self.source.startswith(";", self.cursor + len(raw)):
            raw += ";"
        self.handle_data(raw)

    def _reject(self, *args):
        raise _UnsupportedFragment

    handle_comment = handle_decl = handle_pi = unknown_decl = handle_startendtag = _reject


def fragment_edits(source: str) -> tuple[tuple[int, int, str], ...]:
    """Return ordered, disjoint edits, or an empty tuple for fixed-path fallback."""
    parser = _FragmentParser(source)
    try:
        parser.feed(source)
        parser.close()
    except (_UnsupportedFragment, AssertionError):
        # Some Python versions raise AssertionError for malformed declarations.
        return ()
    if parser.stack or parser.cursor != len(source):
        return ()
    return tuple((start, end, "" if start == 0 or end == len(source) else separator)
                 for start, end, separator in parser.edits)
