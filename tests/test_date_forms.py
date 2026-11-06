"""Dates: impossible values, native digits, typeset and glued forms, month options."""

import pytest

from sct import TextCleaner, TextCleanerConfig
from sct.streaming import WindowedStreamingRedactor
from sct.utils import constants
from sct.utils.datetime import ProcessDateTime


def test_empty_month_selection_keeps_fuzzy_dates_off():
    pytest.importorskip("rapidfuzz")
    cfg = TextCleanerConfig(date_builtin_month_languages=(), check_fuzzy_replace_dates=True,
                            check_ner_process=False)
    assert TextCleaner(cfg=cfg).process("Born 15 January 1990 here").lm_text == "Born 15 January <YEAR> here"
    assert ProcessDateTime(fuzzy_vocabulary=(), fuzzy_vocabulary_by_lang={})._fuzzy_vocabulary == ()
    assert ProcessDateTime()._fuzzy_vocabulary == constants.FUZZY_MONTH_VOCABULARY


def _clean(text, **kw):
    return TextCleaner(TextCleanerConfig(check_ner_process=False, **kw)).process(text).lm_text


# --- year-last numeric dates need a valid month in one of the two parts ---
@pytest.mark.parametrize("text", ["31/31/2024", "14.25.2024", "31-13-2024", "31/12-2024"])
def test_numeric_date_impossible_month(text):
    assert constants.DATE_REGEX.search(text) is None


@pytest.mark.parametrize("text", ["31/12/2024", "12/31/2024", "1/1/2024", "15.01.2024", "15-01-2024 10:30"])
def test_numeric_date_valid(text):
    assert constants.DATE_REGEX.search(text).group() == text


def test_impossible_date_not_tagged_in_pipeline():
    out = _clean("Reference 31/31/2024", check_replace_numbers=False, check_replace_years=False)
    assert out == "Reference 31/31/2024"


MODES = ["placeholder", "synthetic", "reversible"]


def _clean_training(text, mode="placeholder", **kw):
    cfg = TextCleanerConfig.from_preset("training", check_ner_process=False, replacement_mode=mode, **kw)
    return TextCleaner(cfg).process(text).lm_text


def _stream(text, **kw):
    stream = WindowedStreamingRedactor(TextCleaner(TextCleanerConfig(check_ner_process=False, **kw)),
                                       max_buffer_chars=48, overlap_chars=12)
    return "".join(stream.feed(text[i:i + 3]) for i in range(0, len(text), 3)) + stream.finish()


# --- dates written in native decimal digits are dates ---
DATES = ["١٥/٠١/١٩٩٠", "१५/०१/१९९०", "۱۵/۰۱/۱۹۹۰", "١٩٩٠-٠١-١٥", "১৫.০১.১৯৯০"]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("date", DATES)
def test_native_digit_date_replaced(date, mode):
    out = _clean_training(f"Born {date}.", mode, check_replace_dates=True)
    assert date not in out and "DATE" in out


def test_native_digit_date_restores_original():
    cfg = TextCleanerConfig.from_preset("training", check_ner_process=False, replacement_mode="reversible",
                                        check_replace_dates=True)
    result = TextCleaner(cfg).process("Born ١٥/٠١/١٩٩٠.")
    assert result.metadata["anon_map"].deanonymize(result.lm_text) == "Born ١٥/٠١/١٩٩٠."


def test_native_digit_date_in_stream():
    out = _stream("The patient was born on ١٥/٠١/١٩٩٠ in the city.", check_replace_dates=True)
    assert "١٩٩٠" not in out and "<DATE>" in out


# --- fuzz follow-ups (extended harness, training with dates on) ---
@pytest.mark.parametrize("text", ["วันเกิด15/01/1990ครับ", "生日1990-01-15です", "Born ꘡꘩꘧꘡–꘠꘧–꘠7 ok",
                                  "Born 1971‑07‑07 ok"])
def test_glued_or_typeset_date_replaced(text):
    out = _clean_training(text, check_replace_dates=True)
    assert "<DATE>" in out and "1990" not in out and "07" not in out


def test_typeset_date_kept_with_fold_off():
    assert "<DATE>" not in _clean_training("Born 1971‑07‑07 ok", check_replace_dates=True,
                                  fold_styled_characters=False)


def test_latin_glued_date_not_replaced():
    assert "<DATE>" not in _clean_training("code ab15/01/1990 ok", check_replace_dates=True)
