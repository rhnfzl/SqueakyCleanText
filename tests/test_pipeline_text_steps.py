"""Pipeline text steps: JSON spans, HTML comments, isolated symbols, JSON keys, language confidence."""

import time
import types

import pytest

from sct import TextCleaner, TextCleanerConfig
from sct.presets import preset_config
from sct.privacy import DetectedEntity, EntityFinding
from sct.signals import public_view
from sct.utils import constants
from sct.utils.special import _json_object_spans

from tests.conftest import TEST_NER_MODELS


@pytest.fixture(scope="module")
def cleaner():
    return TextCleaner(cfg=TextCleanerConfig(check_ner_process=False))


@pytest.mark.parametrize("text", [
    '{"a":' * 20000,                 # no closing brace: was quadratic, then RecursionError
    '{"a":' * 20000 + '}',           # deep nesting with a far '}'
    '{ x ' * 50000 + '}',            # a failed parse per brace
    '{"a":' * 900 + 'x' * 1000 + '}',
])
def test_malformed_json_scan_is_fast_and_does_not_raise(text):
    t0 = time.perf_counter()
    assert _json_object_spans(text) == []
    assert time.perf_counter() - t0 < 1.0


def test_default_pipeline_survives_malformed_json(cleaner):
    cleaner.process('{"a":' * 20000)


def test_valid_objects_still_found():
    text = 'x {"a":"{"} y {"b": {"c": 1}} {bad} {"d":"}"}'
    assert [text[a:b] for a, b in _json_object_spans(text)] == ['{"a":"{"}', '{"b": {"c": 1}}', '{"d":"}"}']


def test_json_with_replaced_number_is_kept(cleaner):
    out = cleaner.process('Order: {"n":4242,"name":"Alice"} done').lm_text
    assert out == 'Order: {"n":<NUMBER>,"name":"Alice"} done'


@pytest.mark.parametrize("text, expected", [
    ("<!-- a > secret --> hello", "<HTML> hello"),
    ("<!-- <b>x</b> --> end", "<HTML> end"),
    ("<!--[if mso]><table><![endif]--> end", "<HTML> end"),
    ("<!-- note --> end", "<HTML> end"),
    ("<!-- open > tail", "<HTML> tail"),  # unclosed comment: ends at the next '>' as before
])
def test_html_comment_ends_at_close_marker(cleaner, text, expected):
    assert cleaner.process(text).lm_text == expected


def test_many_unclosed_comments_stay_linear():
    t0 = time.perf_counter()
    constants.HTML_REGEX.sub('', '<!--' * 50000)
    assert time.perf_counter() - t0 < 1.0


FAST = preset_config("training", check_ner_process=False)


@pytest.fixture(scope="module")
def fast_cleaner():
    return TextCleaner(cfg=FAST)


# --- overlapping findings from two models were counted twice in the metadata ---
def test_overlapping_findings_count_once(fast_cleaner):
    text = "John Smith works here."
    state = fast_cleaner._before_ner(text)

    def finding(end, score):
        return EntityFinding(DetectedEntity("PERSON", score, text[:end], 0, end),
                             "placeholder", "")

    result = fast_cleaner._after_ner(
        state, "<PERSON> works here.", (finding(4, 0.99), finding(10, 0.9)))
    assert result.metadata["entity_counts"] == {"PERSON": {"count": 1, "chars": 10}}
    view = public_view(result)
    assert (view["pii_count"], view["pii_chars"]) == (1, 10)


# --- Lingua confidence is rounded so workers and in-process agree ---
def test_language_confidence_is_rounded():
    cleaner = TextCleaner(TextCleanerConfig(check_ner_process=False, check_detect_language=True))
    lang = types.SimpleNamespace(name="ENGLISH")
    noisy = [types.SimpleNamespace(language=lang, value=0.5735166974819533)]
    cleaner._detector = types.SimpleNamespace(compute_language_confidence_values=lambda t: noisy)
    assert cleaner._detect_language("x")[1] == 0.573517


# --- importing sct does not pull presidio (and through it torch/GLiNER) ---
def test_import_sct_does_not_load_presidio():
    import subprocess
    import sys
    code = "import sct, sys; print('presidio_anonymizer' in sys.modules)"
    out = subprocess.run(  # noqa: S603 - fixed argv, own interpreter
        [sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


# --- isolated-symbol removal keeps brackets and punctuation glued to a word or placeholder ---
@pytest.mark.parametrize("text, expected", [
    ("He was (born 1980) here.", "He was (born <YEAR>) here."),
    ("He was (born 1980).", "He was (born <YEAR>)."),
    ("He was (born in Paris). Ok", "He was (born in Paris). Ok"),
    ("Voor vragen, bel +31 20 123 4567.", "Voor vragen, bel <PHONE>."),
    ("end , start", "end start"),
])
def test_isolated_symbols_need_space_on_both_sides(text, expected):
    cleaner = TextCleaner(TextCleanerConfig(check_ner_process=False))
    assert cleaner.process(text).lm_text == expected


# --- JSON keys get regex PII only by default; NER on keys is opt-in ---
def test_json_keys_skip_ner_by_default():
    from sct.adapters import _pii_only_cleaner
    cleaner = TextCleaner(TextCleanerConfig(ner_models=TEST_NER_MODELS))
    assert _pii_only_cleaner(cleaner, ner=False).cfg.check_ner_process is False
    assert _pii_only_cleaner(cleaner, ner=True).cfg.check_ner_process is True


def test_process_json_default_keys_do_not_reach_ner(monkeypatch):
    from sct import adapters
    seen = []
    real = adapters._pii_only_cleaner
    monkeypatch.setattr(adapters, "_pii_only_cleaner",
                        lambda c, ner=True: seen.append(ner) or real(c, ner))
    cleaner = TextCleaner(TextCleanerConfig(check_ner_process=False))
    out = adapters.process_json(cleaner, {"user": "x", "to a@b.com": 1})
    assert seen == [False]
    assert list(out.value) == ["user", "to <EMAIL>"]
    TextCleanerConfig(json_key_cleaning="pii_ner")  # opt-in mode is valid
