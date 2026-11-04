"""The windowed stream never splits inside a phone, date, address, escape or markup token."""


import pytest

from sct import TextCleaner, TextCleanerConfig
from sct.presets import preset_config
from sct.streaming import WindowedStreamingRedactor
from tests.helpers import assert_near_linear


FAST = preset_config("training", check_ner_process=False)


@pytest.fixture(scope="module")
def fast_cleaner():
    return TextCleaner(cfg=FAST)


# --- the windowed stream split a spaced phone number across two process() calls ---
def test_windowed_stream_never_splits_inside_a_phone_number(fast_cleaner):
    text = "aaaa aaaa aaaa Call 06 12 34 56 78 tail"
    redactor = WindowedStreamingRedactor(fast_cleaner, max_buffer_chars=32, overlap_chars=16)
    out = redactor.feed(text) + redactor.finish()
    assert out == fast_cleaner.process(text).lm_text == "aaaa aaaa aaaa Call <PHONE> tail"


# --- stream boundaries see contacts that only match once normalised ---
def test_window_never_splits_a_fullwidth_phone():
    cleaner = TextCleaner(cfg=preset_config(
        "training", check_ner_process=False, check_replace_numbers=False))
    text = "aaaa aaaa aaaa Call ０６ １２ ３４ ５６ ７８ tail"
    stream = WindowedStreamingRedactor(cleaner, max_buffer_chars=32, overlap_chars=16)
    assert stream.feed(text) + stream.finish() == cleaner.process(text).lm_text
    assert "<PHONE>" in cleaner.process(text).lm_text


# --- a phone cut off by a feed boundary is held back until it completes ---
@pytest.mark.parametrize("chunks", [
    ["aaaa aaaa aaaa Call 06 12 34", " 56 78 tail"],
    ["aaaa aaaa aaaa Call ０６ １２ ３４", " ５６ ７８ tail"],
    ["aaaa aaaa aaaa Call 06", " 12", " 34", " 56", " 78 tail"],
])
def test_window_holds_an_incomplete_phone(chunks):
    cleaner = TextCleaner(cfg=preset_config(
        "training", check_ner_process=False, check_replace_numbers=False))
    stream = WindowedStreamingRedactor(cleaner, max_buffer_chars=24, overlap_chars=4)
    out = "".join(stream.feed(c) for c in chunks) + stream.finish()
    assert out == cleaner.process("".join(chunks)).lm_text
    assert "<PHONE>" in out


def test_window_still_splits_a_long_number_table():
    cleaner = TextCleaner(cfg=preset_config("training", check_ner_process=False))
    stream = WindowedStreamingRedactor(cleaner, max_buffer_chars=48, overlap_chars=4)
    assert stream.feed("1 " * 40) != ""  # a digit tail holds back at most a phone's length


# --- an HTML entity inside a spaced phone does not hide it from the split check ---
@pytest.mark.parametrize("entity", ["&#32;", "&#x20;", "&nbsp;"])
def test_streaming_never_splits_inside_html_escaped_phone(entity):
    cleaner = TextCleaner(cfg=preset_config("training", check_ner_process=False))
    redactor = WindowedStreamingRedactor(cleaner, max_buffer_chars=32, overlap_chars=16)
    text = f"aaaa aaaa aaaa Call 06 12{entity}34 56 78 tail"
    out = redactor.feed(text) + redactor.finish()
    assert out == cleaner.process(text).lm_text  # "<PHONE>" for the entities that decode to a space


def test_streaming_unknown_entity_keeps_raw_offsets():
    cleaner = TextCleaner(cfg=preset_config("training", check_ner_process=False))
    redactor = WindowedStreamingRedactor(cleaner, max_buffer_chars=24, overlap_chars=8)
    text = "word &bogus; word word word word word"
    assert redactor.feed(text) + redactor.finish() == cleaner.process(text).lm_text


# --- native-digit phones are not split between stream windows ---
@pytest.mark.parametrize("phone", ["٠٦ ١٢ ٣٤ ٥٦ ٧٨", "०६ १२ ३४ ५६ ७८"])
def test_stream_keeps_native_digit_phone_whole(phone):
    cfg = TextCleanerConfig.from_preset("training", check_ner_process=False)
    cleaner = TextCleaner(cfg)
    text = f"aaaa aaaa aaaa Call {phone} tail"
    stream = WindowedStreamingRedactor(cleaner, max_buffer_chars=32, overlap_chars=16)
    out = stream.feed(text) + stream.finish()
    assert out == cleaner.process(text).lm_text == "aaaa aaaa aaaa Call <PHONE> tail"


# --- multi-digit styled numbers (⑫) stay whole across stream windows ---
def test_stream_keeps_multi_digit_styled_phone_whole():
    cleaner = TextCleaner(TextCleanerConfig.from_preset("training", check_ner_process=False))
    text = "a a a a a Call ⓪⑥ ⑫③④ ⑤⑥⑦⑧ tail"
    stream = WindowedStreamingRedactor(cleaner, max_buffer_chars=30, overlap_chars=8)
    out = "".join(stream.feed(c) for c in text) + stream.finish()
    assert out == cleaner.process(text).lm_text == "a a a a a Call <PHONE> tail"


# --- a window split never lands inside a configured date ---
DATES = ["15 January 1990", "January 15, 1990", "15 de enero de 1990"]


@pytest.mark.parametrize("date", DATES)
def test_window_split_keeps_date_whole(date):
    cleaner = TextCleaner(TextCleanerConfig.from_preset(
        "training", check_ner_process=False, check_replace_dates=True,
        check_replace_numbers=False, check_replace_years=False))
    for pad in range(0, 30):
        stream = WindowedStreamingRedactor(cleaner, max_buffer_chars=40, overlap_chars=16)
        out = stream.feed("x " * pad + f"The patient was born on {date}. Call tomorrow.") + stream.finish()
        assert "1990" not in out and "<DATE>" in out, (pad, out)


# --- the streaming split guard composes a combining mark with its letter, as the cleaner's NFC ---
@pytest.mark.parametrize("chunk", [64, 1024, 4096, None])
def test_stream_keeps_decomposed_month_date_whole(chunk):
    cleaner = TextCleaner(cfg=preset_config("training", check_ner_process=False, check_replace_dates=True))
    text = "Introduction. " * 300 + "Born 15 février 1990. " + "Afterwards. " * 21
    stream = WindowedStreamingRedactor(cleaner)
    size = chunk or len(text)
    out = "".join(stream.feed(text[i:i + size]) for i in range(0, len(text), size)) + stream.finish()
    assert "Born <DATE>." in cleaner.process(text).lm_text
    assert "Born <DATE>." in out and "1990" not in out


B = chr(92)  # a literal backslash, as JSON-escaped text carries it


def _stream(cleaner, text, chunk):
    stream = WindowedStreamingRedactor(cleaner)
    size = chunk or len(text)
    return "".join(stream.feed(text[i:i + size]) for i in range(0, len(text), size)) + stream.finish()


# --- the streaming split guard decodes backslash escapes, as the cleaner's escape step ---
@pytest.mark.parametrize("chunk", [1024, 4096, None])
@pytest.mark.parametrize("value, n, token", [
    ("06 12" + B + "u0033" + B + "u0034 56 78", 19, "<PHONE>"),
    ("06 12" + B + "u200b 34 56 78", 20, "<PHONE>"),
    ("06 12" + B + "u00a034 56 78", 20, "<PHONE>"),
    ("06 12 " + B + "x334 56 78", 20, "<PHONE>"),
    ("15 f" + B + "u00e9vrier 1990", 20, "<DATE>"),
])
def test_stream_keeps_escaped_token_whole(chunk, value, n, token):
    cleaner = TextCleaner(cfg=preset_config(
        "training", check_ner_process=False, check_replace_dates=True, check_decode_unicode_escapes=True))
    text = "Introduction. " * 300 + "Value " + value + " here. " + "Afterwards. " * n
    assert f"Value {token} here." in cleaner.process(text).lm_text
    assert f"Value {token} here." in _stream(cleaner, text, chunk)


def test_stream_keeps_escapes_raw_when_decoding_is_off():
    cleaner = TextCleaner(cfg=preset_config("training", check_ner_process=False))
    text = "Introduction. " * 300 + "Path C:" + B + "u0033 " + B + "x41 done. " + "Afterwards. " * 20
    assert _stream(cleaner, text, None) == cleaner.process(text).lm_text


# --- with fuzzy dates on, the guard keeps a misspelled-month date in one window ---
@pytest.mark.parametrize("chunk", [1024, 4096, None])
@pytest.mark.parametrize("value", ["15 Janury 1990", "Janury 15, 1990"])
def test_stream_keeps_fuzzy_date_whole(chunk, value):
    pytest.importorskip("rapidfuzz")
    cleaner = TextCleaner(cfg=preset_config(
        "training", check_ner_process=False, check_replace_dates=True, check_fuzzy_replace_dates=True,
        language="en"))
    text = "Introduction. " * 300 + "Value " + value + " here. " + "Afterwards. " * 20
    assert "Value <DATE> here." in cleaner.process(text).lm_text
    assert "Value <DATE> here." in _stream(cleaner, text, chunk)


# --- the stream guard decodes entities to a fixpoint, keeps multi-character decodes and repairs mojibake ---
@pytest.mark.parametrize("chunk", [1024, 4096, None])
@pytest.mark.parametrize("value, token", [
    ("06 12&amp;#32;34 56 78", "<PHONE>"),
    ("15 January&ThickSpace;1990", "<DATE>"),
    ("15 fÃ©vrier 1990", "<DATE>"),
])
def test_stream_keeps_repaired_token_whole(chunk, value, token):
    cleaner = TextCleaner(cfg=preset_config("training", check_ner_process=False, check_replace_dates=True))
    text = "Introduction. " * 300 + "Value " + value + " here. " + "Afterwards. " * 20
    stream = WindowedStreamingRedactor(cleaner)
    size = chunk or len(text)
    out = "".join(stream.feed(text[i:i + size]) for i in range(0, len(text), size)) + stream.finish()
    assert f"Value {token} here." in cleaner.process(text).lm_text
    assert f"Value {token} here." in out


# --- the stream guard drops terminal escapes and controls as the cleaner's ftfy step does ---
@pytest.mark.parametrize("date", [
    "\x1b[32m15 février 1990\x1b[0m", "15 \x1b[1mfévrier 1990", "15\x00 février 1990", "15 février￼ 1990",
])
def test_stream_keeps_date_with_terminal_escape_or_control_whole(date):
    cleaner = TextCleaner(cfg=preset_config("training", check_ner_process=False, check_replace_dates=True))
    text = "Introduction. " * 300 + f"Value {date} here. " + "Afterwards. " * 20
    stream = WindowedStreamingRedactor(cleaner)
    out = stream.feed(text) + stream.finish()
    assert "Value <DATE> here." in cleaner.process(text).lm_text
    assert "Value <DATE> here." in out
    assert "1990" not in out


# --- HTML entities decode on their own, so a stream of markup splits in a few normalise calls ---
def _counting_cleaner(**overrides):
    cleaner = TextCleaner(cfg=TextCleanerConfig.from_preset("training", check_ner_process=False, **overrides))
    calls, normalise = [0], cleaner.normalise

    def counted(text):
        calls[0] += 1
        return normalise(text)

    cleaner.normalise = counted
    return cleaner, calls


@pytest.mark.parametrize("repeats", [400, 800])
def test_stream_of_markup_with_entities_splits(repeats):
    cleaner, calls = _counting_cleaner()
    text = "intro <b> " + "word &amp; " * repeats + "x" * 252
    stream = WindowedStreamingRedactor(cleaner)
    first = stream.feed(text)
    assert calls[0] < 20  # was 1605 / 3205 (one prefix and suffix normalise per rejected space)
    assert first.startswith("intro <b> word & word")
    assert first + stream.feed(" ") + stream.finish() == cleaner.process(text + " ").lm_text


def test_feed_of_markup_with_entities_stays_near_linear(fast_cleaner):
    # the smaller size must still pass max_buffer_chars, or feed() only buffers and the ratio means nothing
    assert_near_linear(lambda text: WindowedStreamingRedactor(fast_cleaner).feed(text),
                       lambda n: "intro <b> " + "word &amp; " * n + "x" * 252, 3200)


def test_stream_split_matches_whole_document_for_phone_glued_to_cjk():
    cleaner = TextCleaner(cfg=preset_config("training", check_ner_process=False))
    text = "ok 鸯搊緘薦袹 握！+31 6 2702 1527（櫿菲襋咈醩靭胰鐑聈螮頁湪土諵櫿菲襋咈醩靭胰鐑） ok ok"
    stream = WindowedStreamingRedactor(cleaner, max_buffer_chars=48, overlap_chars=12)
    out = "".join(stream.feed(text[i:i + 5]) for i in range(0, len(text), 5)) + stream.finish()
    assert out == cleaner.process(text).lm_text


def _cleaner(mode="placeholder", **kw):
    return TextCleaner(TextCleanerConfig.from_preset("training", check_ner_process=False, replacement_mode=mode, **kw))


# The stream's split guard did not see "06​ 12 ..." as one phone and cut it between windows.
@pytest.mark.parametrize("text", [
    "see you call 06​ 12 34 56 78 now ok bye",
    "see you call 06 12­ 34 56 78 now ok bye",
])
def test_stream_does_not_cut_phone_at_invisible_hint(text):
    cleaner = _cleaner()
    stream = WindowedStreamingRedactor(cleaner, max_buffer_chars=30, overlap_chars=8)
    out = "".join(stream.feed(ch) for ch in text) + stream.finish()
    assert out == cleaner.process(text).lm_text == "see you call <PHONE> now ok bye"


def _preset_cleaner(preset="training", **overrides):
    return TextCleaner(cfg=preset_config(preset, check_ner_process=False, **overrides))


# The stream kept a no-break space it split at, while the whole text's whitespace step made it " ".
def test_stream_separator_follows_whitespace_step():
    cleaner = _preset_cleaner("sct")
    text = "alpha beta\u202fgamma delta\u00a0epsilon zeta\u2009eta theta iota"
    stream = WindowedStreamingRedactor(cleaner, max_buffer_chars=20, overlap_chars=6)
    out = "".join(stream.feed(ch) for ch in text) + stream.finish()
    assert out == cleaner.process(text).lm_text
