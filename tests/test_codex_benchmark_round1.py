"""Codex review of the benchmark-fix branch, round 1 (C1, C2, C3, C5, C6, C7)."""
import re
import time

import pytest

from sct import TextCleaner, TextCleanerConfig
from sct.browser import build_browser_manifest
from sct.utils import constants
from sct.utils.datetime import ProcessDateTime
from sct.utils.special import _json_object_spans
from tests.test_final_check_fixes import _finalize_spans


@pytest.fixture(scope="module")
def cleaner():
    return TextCleaner(cfg=TextCleanerConfig(check_ner_process=False))


@pytest.mark.parametrize("text", [
    '{"a":' * 20000,                 # no closing brace: was quadratic, then RecursionError
    '{"a":' * 20000 + '}',           # deep nesting with a far '}'
    '{ x ' * 50000 + '}',            # a failed parse per brace
    '{"a":' * 900 + 'x' * 1000 + '}',
])
def test_c1_malformed_json_scan_is_fast_and_does_not_raise(text):
    t0 = time.perf_counter()
    assert _json_object_spans(text) == []
    assert time.perf_counter() - t0 < 1.0


def test_c1_default_pipeline_survives_malformed_json(cleaner):
    cleaner.process('{"a":' * 20000)


def test_c1_valid_objects_still_found():
    text = 'x {"a":"{"} y {"b": {"c": 1}} {bad} {"d":"}"}'
    assert [text[a:b] for a, b in _json_object_spans(text)] == ['{"a":"{"}', '{"b": {"c": 1}}', '{"d":"}"}']


def test_c2_json_with_replaced_number_is_kept(cleaner):
    out = cleaner.process('Order: {"n":4242,"name":"Alice"} done').lm_text
    assert out == 'Order: {"n":<NUMBER>,"name":"Alice"} done'


def test_c3_propagation_does_not_match_inside_a_word_with_combining_mark():
    text = "राज met राजा, राज."
    ner, windows, results = _finalize_spans(text, [(0, 3, 'PER', 0.99)])
    assert ner._finalize(text, windows, results, 0.85, None).text == "<PERSON> met राजा, <PERSON>."


@pytest.mark.parametrize("text, expected", [
    ("<!-- a > secret --> hello", "<HTML> hello"),
    ("<!-- <b>x</b> --> end", "<HTML> end"),
    ("<!--[if mso]><table><![endif]--> end", "<HTML> end"),
    ("<!-- note --> end", "<HTML> end"),
    ("<!-- open > tail", "<HTML> tail"),  # unclosed comment: ends at the next '>' as before
])
def test_c5_html_comment_ends_at_close_marker(cleaner, text, expected):
    assert cleaner.process(text).lm_text == expected


def test_c5_many_unclosed_comments_stay_linear():
    t0 = time.perf_counter()
    constants.HTML_REGEX.sub('', '<!--' * 50000)
    assert time.perf_counter() - t0 < 1.0


def test_c6_browser_date_rule_follows_date_options():
    cfg = TextCleanerConfig(date_month_name_slash_year=False, date_builtin_month_languages=(),
                            check_ner_process=False)
    manifest = build_browser_manifest(cfg)
    assert not re.search(manifest["rules"]["date"]["pattern"], "Born October/55 here", re.I)
    assert TextCleaner(cfg=cfg).process("Born October/55 here").lm_text == "Born October/55 here"
    assert build_browser_manifest(TextCleanerConfig())["rules"]["date"]["pattern"] == constants.DATE_REGEX.pattern


def test_c7_empty_month_selection_keeps_fuzzy_dates_off():
    pytest.importorskip("rapidfuzz")
    cfg = TextCleanerConfig(date_builtin_month_languages=(), check_fuzzy_replace_dates=True,
                            check_ner_process=False)
    assert TextCleaner(cfg=cfg).process("Born 15 January 1990 here").lm_text == "Born 15 January <YEAR> here"
    assert ProcessDateTime(fuzzy_vocabulary=(), fuzzy_vocabulary_by_lang={})._fuzzy_vocabulary == ()
    assert ProcessDateTime()._fuzzy_vocabulary == constants.FUZZY_MONTH_VOCABULARY
