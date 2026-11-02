"""Full numeric dates and date-times become one <DATE> by default, in every mode."""

import pytest

from sct import TextCleaner, TextCleanerConfig
from sct.presets import preset_config
from sct.utils.constants import DATE_REGEX


def _cleaner(**overrides):
    overrides.setdefault("check_ner_process", False)
    overrides.setdefault("check_detect_language", False)
    overrides.setdefault("check_statistical_model_processing", False)
    return TextCleaner(cfg=TextCleanerConfig(**overrides))


DATES = [
    "2023-10-01", "2024/1/15", "2024.01.15", "15.01.2024", "01/15/2024", "1-2-2024",
    "2024-01-15T10:30:00Z", "2024-01-15T10:30:00.123+01:00", "2023-10-01 12:30",
    "15.01.2024 08:15:00",
]

NOT_DATES = [
    "3.12.0", "1.2.3", "10.1.12.2024", "1.5.2024.3", "2024.1.5", "192.168.1.1", "10.0.0.1",
    "3.14", "1,234.56", "+31 6 1234 5678", "555-123-4567", "+49 30 12.34.5678",
    "45-67-2023", "2023-13-45",
]


@pytest.mark.parametrize("date", DATES)
def test_full_date_is_one_date_token_by_default(date):
    result = _cleaner().process(f"Signed {date} here")

    assert result.lm_text == "Signed <DATE> here"
    assert result.metadata["regex_counts"] == {"DATE": {"count": 1, "chars": len(date)}}


@pytest.mark.parametrize("text", NOT_DATES)
def test_non_dates_are_not_date_matches(text):
    assert DATE_REGEX.search(f"see {text} now") is None


def test_date_inside_range_and_sentence_end():
    result = _cleaner().process("From 01.10.2023-05.10.2023.")

    assert result.metadata["regex_counts"]["DATE"]["count"] == 2
    assert "<YEAR>" not in result.lm_text and "<NUMBER>" not in result.lm_text


def test_old_behaviour_reachable_by_config():
    result = _cleaner(check_replace_dates=False).process("Signed 2023-10-01 here")

    assert result.lm_text == "Signed <YEAR><NUMBER><NUMBER> here"
    assert "DATE" not in result.metadata["regex_counts"]


def test_reversible_mode_round_trips_one_date():
    result = _cleaner(replacement_mode="reversible").process("Signed 2023-10-01T09:00:00Z here")

    assert result.lm_text == "Signed <DATE_0> here"
    assert result.metadata["anon_map"].deanonymize(result.lm_text) == "Signed 2023-10-01T09:00:00Z here"
    assert result.metadata["regex_counts"]["DATE"]["count"] == 1


def test_synthetic_mode_replaces_whole_date():
    pytest.importorskip("faker")
    result = _cleaner(replacement_mode="synthetic").process("Signed 2023-10-01 here")

    assert "2023-10-01" not in result.lm_text
    assert "<YEAR>" not in result.lm_text and "<NUMBER>" not in result.lm_text
    assert result.metadata["regex_counts"] == {"DATE": {"count": 1, "chars": 10}}


def test_training_preset_keeps_dates_as_written():
    result = TextCleaner(cfg=preset_config(
        check_ner_process=False, check_detect_language=False,
    )).process("Signed 2023-10-01 here")

    assert result.lm_text == "Signed 2023-10-01 here"


# Month/YY(YY).
MONTH_NAME_DATES = [
    ("He was born in October/55", "He was born in <DATE>"),
    ("Joined Oct/2019 here", "Joined <DATE> here"),
]


@pytest.mark.parametrize("text,expected", MONTH_NAME_DATES)
def test_month_slash_year_dates(text, expected):
    assert _cleaner().process(text).lm_text == expected


def test_thai_month_names_come_from_custom_month_names():
    # No built-in Thai month data (no word lists); the user passes month names.
    text = "วันที่ 24 ก.พ. 2562"
    assert _cleaner().process(text).lm_text == "วันที่ <NUMBER> ก.พ. <NUMBER>"
    custom = _cleaner(custom_month_names={"th": ("ก.พ.",)}).process(text)
    assert custom.lm_text == "วันที่ <DATE>"


def test_dotted_month_abbreviation_is_literal():
    assert DATE_REGEX.search("วันที่ 24 กXพX 2562") is None
    assert DATE_REGEX.search("October/555") is None


def test_month_slash_year_off_by_config():
    assert _cleaner(date_month_name_slash_year=False).process("born October/55").lm_text == "born October/<NUMBER>"


def test_builtin_month_languages_by_config():
    result = _cleaner(date_builtin_month_languages=("en",)).process("วันที่ 24 ก.พ. 2562, 3 maart 2020, 3 March 2020")

    assert result.lm_text == "วันที่ <NUMBER> ก.พ. <NUMBER>, <NUMBER> maart <YEAR>, <DATE>"


def test_builtin_month_languages_rejects_unknown_code():
    with pytest.raises(ValueError):
        TextCleanerConfig(date_builtin_month_languages=("xx",))
