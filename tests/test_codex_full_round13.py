"""Codex 0.8.0 full review, round 13: ids 1, 2, 3, 4."""

from types import SimpleNamespace

import pytest

from sct import TextCleaner, TextCleanerConfig
from sct.utils import constants
from sct.utils.anonymization_map import AnonymizationMap


def _clean(text, **kw):
    return TextCleaner(TextCleanerConfig(check_ner_process=False, **kw)).process(text).lm_text


# --- 1: letters of the CJK punctuation block (々 〆 〇) are domain letters ---

@pytest.mark.parametrize("mode", ["placeholder", "reversible"])
def test_email_domain_with_iteration_mark(mode):
    assert "佐々木" not in _clean("Mail info@佐々木.jp now", replacement_mode=mode)


def test_email_domain_still_stops_at_cjk_punctuation():
    assert constants.EMAIL_REGEX.search("a@b.com（担当：鈴木）").group() == "a@b.com"
    assert constants.EMAIL_REGEX.search("a@b.com、次").group() == "a@b.com"


# --- 2: full-width contact punctuation folds beside letters of any script ---

@pytest.mark.parametrize("text, tag", [
    ("taro＠例え.jp", "<EMAIL>"),
    ("山田＠example.com", "<EMAIL>"),
    ("taro＠例え．jp", "<EMAIL>"),
    ("https：／／例え.jp/パス", "<URL>"),
])
def test_fullwidth_contact_beside_cjk(text, tag):
    assert _clean(text) == tag


def test_fullwidth_cjk_prose_punctuation_kept():
    assert _clean("担当：鈴木様。東京．大阪") == "担当：鈴木様。東京．大阪"


# --- 3: a tag inside an NER span keeps the restore of every other tag, and its value ---

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


# --- 4: year-last numeric dates need a valid month in one of the two parts ---

@pytest.mark.parametrize("text", ["31/31/2024", "14.25.2024", "31-13-2024", "31/12-2024"])
def test_numeric_date_impossible_month(text):
    assert constants.DATE_REGEX.search(text) is None


@pytest.mark.parametrize("text", ["31/12/2024", "12/31/2024", "1/1/2024", "15.01.2024", "15-01-2024 10:30"])
def test_numeric_date_valid(text):
    assert constants.DATE_REGEX.search(text).group() == text


def test_impossible_date_not_tagged_in_pipeline():
    out = _clean("Reference 31/31/2024", check_replace_numbers=False, check_replace_years=False)
    assert out == "Reference 31/31/2024"
