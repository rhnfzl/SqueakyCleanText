"""Escaped markup (&lt;b&gt;) is text, not a tag: tags are found on the markup, escapes decode between them."""

import time

import pytest

from sct import TextCleaner, TextCleanerConfig
from sct.streaming import WindowedStreamingRedactor
from sct.utils import normtext

CASES = [
    ("<p>Use &lt;b&gt; for bold, mail john@example.com</p>", "<HTML>Use <b> for bold, mail <EMAIL><HTML>"),
    ("<div>a &lt;script&gt;x&lt;/script&gt; b</div>", "<HTML>a <script>x</script> b<HTML>"),
    ("<p>Escaped &lt;i&gt;mail john@example.com&lt;/i&gt; here</p>", "<HTML>Escaped <i>mail <EMAIL></i> here<HTML>"),
    ("<p>Write &lt;john@example.com&gt; now</p>", "<HTML>Write <<EMAIL>> now<HTML>"),
    ('<a title="x &gt; y">link</a> &lt;i&gt;', "<HTML>link<HTML> <i>"),
    ("Mail john&lt;at&gt;example.com now", "Mail <EMAIL> now"),  # brackets hide from the tag step only
    ("Mail john&#60;at&#62;example.com now", "Mail <EMAIL> now"),
]


def _cleaner(mode="placeholder"):
    return TextCleaner(TextCleanerConfig(check_ner_process=False, replacement_mode=mode))


@pytest.mark.parametrize("text,expected", CASES)
def test_escaped_tags_stay_text(text, expected):
    cleaner = _cleaner()
    assert cleaner.process(text).lm_text == expected
    assert [r.lm_text for r in cleaner.process_batch([text, text])] == [expected, expected]
    doc = " ".join([text] * 6)
    for size in (64, 96):
        stream = WindowedStreamingRedactor(cleaner, max_buffer_chars=size, overlap_chars=32)
        out = "".join(stream.feed(doc[i:i + 7]) for i in range(0, len(doc), 7)) + stream.finish()
        assert out == cleaner.process(doc).lm_text


@pytest.mark.parametrize("mode", ["reversible", "synthetic"])
@pytest.mark.parametrize("text,expected", CASES[:3])
def test_escaped_tags_stay_text_in_other_modes(text, expected, mode):
    result = _cleaner(mode).process(text)
    assert "<b>" in result.lm_text or "<script>" in result.lm_text or "<i>" in result.lm_text
    assert "john@example.com" not in result.lm_text
    if mode == "reversible":  # the map gives back the tags and the email around the decoded escapes
        assert result.metadata["anon_map"].deanonymize(result.lm_text) == normtext.unescape_markup(text)


def test_escaped_bracket_in_contact_restores_as_text():
    text = "See https://example.com/?q=a&lt;b now."
    result = _cleaner("reversible").process(text)
    assert result.metadata["anon_map"].deanonymize(result.lm_text) == normtext.unescape_markup(text)
    assert all(ord(ch) < 0xF0000 for e in result.metadata["anon_map"].to_dict()["entries"] for ch in e["original"])


def test_escaped_email_in_json():
    from sct.adapters import process_json
    assert "john" not in str(process_json(_cleaner(), {"a": "Mail john&lt;at&gt;example.com now"}).value)


def test_escaped_marker_is_still_reserved():
    result = _cleaner("reversible").process("We live in &lt;EMAIL1&gt; mail a@example.com")
    assert result.lm_text.startswith("We live in <EMAIL1> mail <EMAIL") and result.lm_text != (
        "We live in <EMAIL1> mail <EMAIL1>")  # the source spelling is never allocated
    assert result.metadata["anon_map"].deanonymize(result.lm_text) == "We live in <EMAIL1> mail a@example.com"


@pytest.mark.parametrize("text", ["<" * 200_000 + "&lt;", "&lt;" * 50_000, "<a " * 50_000 + "&gt;"])
def test_unescape_markup_linear(text):
    start = time.perf_counter()
    normtext.unescape_markup(text)
    normtext.decode_html_entities(text)
    assert time.perf_counter() - start < 1.0
