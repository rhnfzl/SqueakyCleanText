"""Codex 0.8.0 full review, round 14: ids 1, 2, 3, 4."""

import os
import subprocess
import sys

import pytest

from sct import TextCleaner, TextCleanerConfig
from sct.utils.synthetic import SyntheticReplacer
from tests.conftest import TEST_NER_MODELS


def _clean(text, **kw):
    return TextCleaner(TextCleanerConfig(check_ner_process=False, **kw)).process(text).lm_text


# --- 1: ZWNJ/ZWJ inside an IDNA domain label ---

@pytest.mark.parametrize("fold", [True, False])
@pytest.mark.parametrize("text", ["info@نامه‌ای.ir", "info@क्‍ष.com"])
def test_email_domain_with_join_control(text, fold):
    assert _clean(text, fold_styled_characters=fold) == "<EMAIL>"


# --- 2: full-width separators inside an address fold beside any script ---

@pytest.mark.parametrize("fold", [True, False])
@pytest.mark.parametrize("text", ["john@例え．テスト", "john@例え－テスト.jp", "josé＿smith@example.com",
                                  "山田＿太郎@example.com"])
def test_fullwidth_separator_in_address_beside_non_ascii(text, fold):
    assert _clean(text, fold_styled_characters=fold) == "<EMAIL>"


@pytest.mark.parametrize("fold", [True, False])
def test_fullwidth_separator_in_prose_kept(fold):
    assert _clean("東京－大阪．名古屋＿線", fold_styled_characters=fold) == "東京－大阪．名古屋＿線"


# --- 3: a tag NER cuts in half goes back whole in reversible mode ---

def test_reversible_tag_cut_by_ner_span():
    text = "University of ann@example.com. Contact bob@example.org today."
    cleaner = TextCleaner(TextCleanerConfig(language="ENGLISH", replacement_mode="reversible",
                                            ner_models=TEST_NER_MODELS))
    result = cleaner.process(text)
    assert "ann@example.com" not in result.lm_text
    assert result.metadata["anon_map"].deanonymize(result.lm_text) == text


# --- 4: locale fakes do not depend on global random or the hash seed ---

@pytest.mark.parametrize("locale, group", [("th_TH", "SOCIAL_SECURITY_NUMBER"), ("uk_UA", "TAX_ID"),
                                           ("uk_UA", "LICENSE_PLATE")])
def test_locale_fake_ignores_global_random(locale, group):
    import random
    values = set()
    for i in range(3):
        random.seed(i)
        replacer = SyntheticReplacer(seed=42, locale=locale)
        replacer.reset_cache("same doc")
        values.add(replacer.get_replacement(group, "x"))
    assert len(values) == 1


def test_italian_city_independent_of_hash_seed():
    code = ("from sct.utils.synthetic import SyntheticReplacer\n"
            "r = SyntheticReplacer(seed=42, locale='it_IT'); r.reset_cache('same doc')\n"
            "print(r.get_replacement('CITY', 'Rome'))")
    out = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,  # noqa: S603
                          env={**os.environ, "PYTHONHASHSEED": seed}).stdout
           for seed in ("1", "42")}
    assert len(out) == 1
