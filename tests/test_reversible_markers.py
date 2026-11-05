"""Reversible tags and source markers: literal placeholders, custom markers, tags cut by NER spans."""

import pickle
import re
from types import SimpleNamespace

import pytest

from sct import TextCleaner, TextCleanerConfig
from sct.presets import preset_config
from sct.utils.anonymization_map import AnonymizationMap

from tests.conftest import TEST_NER_MODELS, TEST_NER_REVISIONS
from tests.helpers import FakeRunDetector, word_ner


# --- marker-shaped source text is not protected like SCT's own placeholders ---
def test_source_marker_text_gets_no_placeholder_protection():
    detector = FakeRunDetector()
    detector.pattern = re.compile(r"[A-Z]{2,}")
    ner = word_ner(0, detector)
    text = "call <PHONE>, live in <LONDON>"
    # Old code dropped the LONDON span too, as if SCT had written "<LONDON>".
    result = ner.ner_process_detailed(
        text, positional_tags=("PER",), ner_confidence_threshold=0.5,
        source_markers=frozenset({"<LONDON>"}))
    assert result.text == "call <PHONE>, live in <<PERSON>>"


@pytest.fixture(scope="module")
def ner_cleaner():
    return TextCleaner(cfg=preset_config("training", ner_models=TEST_NER_MODELS,
                                         ner_model_revisions=TEST_NER_REVISIONS, ner_batch_size=8))


def test_source_marker_pii_is_masked_single_and_batch(ner_cleaner):
    text = "I live in <LONDON>."
    assert "LONDON" not in ner_cleaner.process(text).lm_text
    assert all("LONDON" not in r.lm_text for r in ner_cleaner.process_batch([text, text]))


@pytest.mark.parametrize("text", ["I live in ＜LONDON＞.", "I live in &lt;LONDON&gt;."])
def test_normalised_source_marker_gets_no_ner_protection(ner_cleaner, text):
    assert "LONDON" not in ner_cleaner.process(text).lm_text
    assert all("LONDON" not in r.lm_text for r in ner_cleaner.process_batch([text, text]))


# --- reversible tokens never reuse a placeholder spelling from the source ---
def test_reversible_skips_literal_source_placeholders():
    cleaner = TextCleaner(cfg=preset_config(
        "training", check_ner_process=False, check_replace_html=False, replacement_mode="reversible"))
    text = "<EMAIL_0> alice@example.com"
    result = cleaner.process(text)
    assert result.lm_text == "<EMAIL_0> <EMAIL_1>"
    assert result.metadata["anon_map"].deanonymize(result.lm_text) == text


# --- a tag inside an NER span keeps the restore of every other tag, and its value ---
def test_swap_deferred_tag_inside_ner_span():
    source = "University of <EMAIL>. Contact <EMAIL> today."
    anon_map = AnonymizationMap()
    anon_map.add("<EMAIL_0>", "ann@example.com", "EMAIL", 14, 29)
    anon_map.add("<EMAIL_1>", "bob@example.org", "EMAIL", 39, 54)
    anon_map.add("<ORGANISATION_0>", source[:21], "ORGANISATION", 0, 21)
    findings = [SimpleNamespace(entity=SimpleNamespace(start=0, end=21))]
    order = [("<EMAIL>", "<EMAIL_0>"), ("<EMAIL>", "<EMAIL_1>")]
    text = TextCleaner._swap_deferred("<ORGANISATION_0>. Contact <EMAIL> today.", order, source,
                                      findings, anon_map)
    assert text == "<ORGANISATION_0>. Contact <EMAIL_1> today."
    assert anon_map.deanonymize(text) == "University of ann@example.com. Contact bob@example.org today."


# --- a tag NER cuts in half goes back whole in reversible mode ---
def test_reversible_tag_cut_by_ner_span():
    text = "University of ann@example.com. Contact bob@example.org today."
    cleaner = TextCleaner(TextCleanerConfig(language="ENGLISH", replacement_mode="reversible",
                                            ner_models=TEST_NER_MODELS, ner_model_revisions=TEST_NER_REVISIONS))
    result = cleaner.process(text)
    assert "ann@example.com" not in result.lm_text
    assert result.metadata["anon_map"].deanonymize(result.lm_text) == text


# --- a display tag cleanup removes still carries its value ---
def test_bracket_tag_survives_bracket_removal_in_reversible():
    text = "Mail ann@example.com today."
    cleaner = TextCleaner(TextCleanerConfig(language="ENGLISH", replacement_mode="reversible",
                                            ner_models=TEST_NER_MODELS,
                                            ner_model_revisions=TEST_NER_REVISIONS, replace_with_email="[EMAIL]"))
    result = cleaner.process(text)
    assert "<EMAIL_0>" in result.lm_text
    assert "ann@example.com" in result.metadata["anon_map"].deanonymize(result.lm_text)


# --- an NER span that cuts a deferred tag leaves no fragment in any mode ---
@pytest.mark.parametrize("mode", ["synthetic", "placeholder"])
def test_cut_tag_leaves_no_fragment(mode):
    text = "University of ann@example.com. Contact bob@example.org today."
    cleaner = TextCleaner(TextCleanerConfig.from_preset(
        "training", language="ENGLISH", replacement_mode=mode, ner_models=TEST_NER_MODELS,
        ner_model_revisions=TEST_NER_REVISIONS))
    out = cleaner.process(text).lm_text
    assert out.count("<") == out.count(">") and "ann@example.com" not in out


# --- a display marker inside an NER placeholder does not break restoration ---
@pytest.mark.parametrize("field, value, text", [
    ("replace_with_email", "ann@example.com", "John Smith wrote to ann@example.com today."),
    ("replace_with_url", "https://x.com/a", "John Smith wrote. See https://x.com/a today."),
])
def test_marker_inside_ner_placeholder_round_trips(field, value, text):
    cfg = TextCleanerConfig.from_preset("training", language="ENGLISH", ner_models=TEST_NER_MODELS,
                                        ner_model_revisions=TEST_NER_REVISIONS,
                                        replacement_mode="reversible", **{field: "PERSON"})
    result = TextCleaner(cfg).process(text)
    assert value not in result.lm_text
    assert result.metadata["anon_map"].deanonymize(result.lm_text) == text


# --- custom regex markers are protected from NER like <TAG> markers ---
TEXT = "John Smith at ann@example.com and Jane Doe at bob@example.org, in London."


def _ner_cfg(mode, **kw):
    return TextCleanerConfig.from_preset("training", language="ENGLISH", ner_models=TEST_NER_MODELS,
                                         ner_model_revisions=TEST_NER_REVISIONS,
                                         replacement_mode=mode, **kw)


def test_custom_marker_reversible_round_trip():
    cleaner = TextCleaner(_ner_cfg("reversible", replace_with_email="EMAIL"))
    for result in (cleaner.process(TEXT), cleaner.process_batch([TEXT])[0]):
        assert result.metadata["anon_map"].deanonymize(result.lm_text) == TEXT
        assert "MA" not in result.lm_text.replace("EMAIL", "")


def test_custom_marker_pickled_config_round_trip():
    cfg = pickle.loads(pickle.dumps(_ner_cfg("reversible", replace_with_email="EMAIL")))  # noqa: S301 - own bytes
    result = TextCleaner(cfg).process(TEXT)
    assert result.metadata["anon_map"].deanonymize(result.lm_text) == TEXT


def test_custom_marker_placeholder_kept_whole():
    out = TextCleaner(_ner_cfg("placeholder", replace_with_email="EMAIL")).process(TEXT).lm_text
    assert out.count("EMAIL") == 2 and "example" not in out


MODES = ["placeholder", "synthetic", "reversible"]


# --- a custom marker that is also source text keeps protection for its emitted occurrences ---
SOURCE = "The EMAIL field is mandatory. John Smith at ann@example.com and Jane Doe at bob@example.org, in London."


@pytest.fixture(scope="module")
def marker_cleaners():
    return {mode: TextCleaner(TextCleanerConfig.from_preset(
        "training", language="en", ner_models=TEST_NER_MODELS,
        ner_model_revisions=TEST_NER_REVISIONS, replacement_mode=mode, replace_with_email="EMAIL"))
        for mode in MODES}


def test_literal_custom_marker_reversible_round_trip(marker_cleaners):
    cleaner = marker_cleaners["reversible"]
    for result in [cleaner.process(SOURCE), *cleaner.process_batch([SOURCE, SOURCE])]:
        assert "MA<" not in result.lm_text, result.lm_text
        assert result.metadata["anon_map"].deanonymize(result.lm_text) == SOURCE


@pytest.mark.parametrize("mode", ["placeholder", "synthetic"])
def test_literal_custom_marker_emitted_marker_not_split(marker_cleaners, mode):
    out = marker_cleaners[mode].process(SOURCE).lm_text
    assert not re.search(r"[a-z>]MA[A-Z<]", out), out
    assert "ann@example.com" not in out and "bob@example.org" not in out


def test_source_marker_spans_exempt_only_those_occurrences():
    from tests.helpers import FakeRunDetector, word_ner
    detector = FakeRunDetector()
    detector.pattern = re.compile(r"[A-Z]{2,}")
    ner = word_ner(0, detector)
    text = "<LONDON> and <LONDON>"
    result = ner.ner_process_detailed(
        text, positional_tags=("PER",), ner_confidence_threshold=0.5, source_markers=frozenset({(0, 8)}))
    assert result.text == "<<PERSON>> and <LONDON>"
