"""Codex 0.8.0 full review, round 18: ids 1-4."""

import datetime

import pytest

from sct import TextCleaner, TextCleanerConfig
from sct.utils.synthetic import SyntheticReplacer
from tests.conftest import TEST_NER_MODELS

MODES = ("placeholder", "synthetic", "reversible")


def _clean(text, mode):
    cfg = TextCleanerConfig.from_preset("training", check_ner_process=False, replacement_mode=mode)
    return TextCleaner(cfg).process(text).lm_text


# --- 1: full-width ？ ＃ after a non-Latin host keep the query value inside the URL ---


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("url, secret", [("https://例え.テスト？email=alice%40example.com", "alice"),
                                         ("https://例え.テスト＃token=SecretABC", "SecretABC")])
def test_fullwidth_query_after_non_latin_host_masked(url, secret, mode):
    out = _clean(f"see {url} now", mode)
    assert secret not in out and "=" not in out


def test_fullwidth_question_mark_in_prose_kept():
    assert _clean("本当？はい", "placeholder") == "本当？はい"


# --- 2: one circled local-part character before a full-width ＠ ---


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("email", ["ⓙ＠example.com", "ⓙ＠例え.jp"])
def test_single_circled_local_part_fullwidth_at(email, mode):
    out = _clean(f"mail {email} now", mode)
    assert "ⓙ" not in out and "＠" not in out and "例え" not in out


# --- 3: a display marker inside an NER placeholder does not break restoration ---


@pytest.mark.parametrize("field, value, text", [
    ("replace_with_email", "ann@example.com", "John Smith wrote to ann@example.com today."),
    ("replace_with_url", "https://x.com/a", "John Smith wrote. See https://x.com/a today."),
])
def test_marker_inside_ner_placeholder_round_trips(field, value, text):
    cfg = TextCleanerConfig.from_preset("training", language="ENGLISH", ner_models=TEST_NER_MODELS,
                                        replacement_mode="reversible", **{field: "PERSON"})
    result = TextCleaner(cfg).process(text)
    assert value not in result.lm_text
    assert result.metadata["anon_map"].deanonymize(result.lm_text) == text


# --- 4: native SSN / TAX_ID fakes follow the reference date, not the clock ---


@pytest.mark.parametrize("locale", ["zh_CN", "et_EE", "fi_FI", "lv_LV", "no_NO", "sv_SE", "pl_PL", "sr_BA"])
@pytest.mark.parametrize("entity", ["SOCIAL_SECURITY_NUMBER", "TAX_ID"])
def test_identity_fakes_follow_reference_date(locale, entity):
    def fake(reference_date):
        replacer = SyntheticReplacer(seed=42, locale=locale, reference_date=reference_date)
        replacer.reset_cache("same document")
        return replacer.get_replacement(entity, "x")

    assert fake("1990-06-01") == fake("1990-06-01") != fake("2026-01-01")


def test_reference_clock_restored():
    import faker.providers.date_time as faker_dt
    import faker.providers.ssn.zh_CN as zh

    SyntheticReplacer(seed=42, locale="zh_CN", reference_date="1990-06-01").get_replacement("TAX_ID", "x")
    assert zh.datetime is datetime and faker_dt.datetime is datetime.datetime
    assert faker_dt.dtdate is datetime.date
