"""K2: every regex replacement is counted once into metadata['regex_counts']."""

import pytest

from sct import TextCleaner, TextCleanerConfig
from sct.adapters import is_sensitive_result
from tests.conftest import TEST_NER_MODELS

EMAIL = "maria@example.com"


def _cleaner(**overrides):
    overrides.setdefault("check_ner_process", False)
    overrides.setdefault("check_detect_language", False)
    overrides.setdefault("check_statistical_model_processing", False)
    return TextCleaner(cfg=TextCleanerConfig(**overrides))


@pytest.mark.parametrize("mode", ["placeholder", "reversible", "synthetic"])
def test_one_email_counted_in_every_mode(mode):
    if mode == "synthetic":
        pytest.importorskip("faker")
    result = _cleaner(replacement_mode=mode).process(f"Mail {EMAIL} today")

    assert result.metadata["regex_counts"] == {
        "EMAIL": {"count": 1, "chars": len(EMAIL)},
    }


def test_marker_text_in_source_is_not_counted():
    result = _cleaner(check_replace_html=False).process("<EMAIL> only")

    assert result.lm_text == "<EMAIL> only"
    assert result.metadata["regex_counts"] == {}
    assert is_sensitive_result(result) is False


def test_kept_private_ip_is_not_counted():
    result = _cleaner(replace_ips_public_only=True).process(
        "lan 192.168.1.4 wan 8.8.8.8",
    )

    assert "192.168.1.4" in result.lm_text
    assert result.metadata["regex_counts"]["IP"] == {"count": 1, "chars": 7}


def test_each_step_type_counted():
    text = "<b>x</b> https://a.example/p 2024-01-15 in 1999 call +1-202-555-0147 pay $ 42 ref 4711"
    counts = _cleaner(check_replace_dates=True).process(text).metadata["regex_counts"]

    assert counts["HTML"] == {"count": 2, "chars": 7}
    assert counts["URL"]["count"] == 1
    assert counts["DATE"] == {"count": 1, "chars": 10}
    assert counts["YEAR"] == {"count": 1, "chars": 4}
    assert counts["PHONE"]["count"] == 1
    assert counts["NUMBER"] == {"count": 1, "chars": 4}  # "$ 42" is an amount, "4711" an ID
    assert counts["CURRENCY"] == {"count": 1, "chars": 1}


def test_currency_codes_counted_per_symbol():
    result = _cleaner(
        replace_with_currency_symbols=None, check_to_ascii_unicode=False,
    ).process("cost €€ or £")

    assert "EUREUR" in result.lm_text and "GBP" in result.lm_text
    assert result.metadata["regex_counts"]["CURRENCY"] == {"count": 3, "chars": 3}


def test_fuzzy_dates_counted():
    pytest.importorskip("rapidfuzz")
    text = "Septembr 12, 2020"
    plain = _cleaner(check_replace_dates=True, check_replace_years=False).process(text)
    fuzzy = _cleaner(
        check_replace_dates=True, check_replace_years=False, check_fuzzy_replace_dates=True,
    ).process(text)

    assert "DATE" not in plain.metadata["regex_counts"]
    assert fuzzy.metadata["regex_counts"]["DATE"]["count"] == 1


def test_batch_path_equals_single_path():
    cleaner = TextCleaner(cfg=TextCleanerConfig(
        ner_models=TEST_NER_MODELS,
        ner_batch_size=8,
        check_statistical_model_processing=False,
    ))
    docs = [f"John Smith mailed {EMAIL} in 2020.", "call +1-202-555-0147", "plain words"]

    batched = cleaner.process_batch(docs, batch_size=1)
    single = [cleaner._process_single(doc) for doc in docs]

    for b, s in zip(batched, single):
        assert b.metadata["regex_counts"] == s.metadata["regex_counts"]
        assert b.metadata["entity_counts"] == s.metadata["entity_counts"]
    assert batched[0].metadata["regex_counts"]["EMAIL"]["count"] == 1
    assert batched[2].metadata["regex_counts"] == {}
    assert cleaner._document_context.regex_counts is None


def test_sensitive_when_marker_is_empty_string():
    # The old marker-count guess saw no new marker in the output and said False.
    cleaner = _cleaner(replace_with_phone_numbers="")
    result = cleaner.process("call +1-202-555-0147")

    assert "555" not in result.lm_text
    assert is_sensitive_result(result) is True
