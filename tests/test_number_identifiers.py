"""replace_numbers_mode / replace_numbers_min_digits / phone_reject_embedded (made-up values)."""

import warnings

import pytest

from sct import TextCleaner, TextCleanerConfig
from sct.browser import build_browser_manifest
from sct.utils import constants, contact


def _clean(text, **overrides):
    cfg = dict(
        check_ner_process=False, check_detect_language=False, check_statistical_model_processing=False,
        check_to_ascii_unicode=False, check_casefold=False, check_remove_punctuation=False,
        check_remove_stopwords=False, check_remove_isolated_letters=False,
        check_remove_isolated_special_symbols=False, check_replace_currency_symbols=False,
    )
    cfg.update(overrides)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return TextCleaner(cfg=TextCleanerConfig(**cfg)).process(text).lm_text


@pytest.mark.parametrize("text, expected", [
    ("Order #12345 shipped", "Order #<NUMBER> shipped"),
    ("customer 4567 and 3456.", "customer <NUMBER> and <NUMBER>."),
    ("id=98765 ID:98765", "id=<NUMBER> ID:<NUMBER>"),
    ("金额50000元", "金额<NUMBER>元"),  # unspaced scripts: letters do not glue, as in 0.8.1
    ("رقم ٤٥٦٧٨ هنا", "رقم <NUMBER> هنا"),  # Arabic-Indic digits
    ("संख्या ४५६७८ है", "संख्या <NUMBER> है"),  # Devanagari digits
    ("番号１２３４５です", "番号<NUMBER>です"),  # full-width digits
])
def test_identifier_shaped_numbers_masked(text, expected):
    assert _clean(text) == expected


@pytest.mark.parametrize("text", [
    "Pay 1,250.00 or 12.5 or 1.234.567",  # amounts and decimals
    "Rate 12345% and 12345 % and 12345‰",  # percent
    "Cost €12345, $ 12345, 12345€ and 12345 ₹",  # currency sign beside it
    "From 1000-3000 or 1000–3000 at 14:30",  # ranges, times
    "age 42, 7 items, 123 pages",  # under 4 digits
    "abc_1234567 x-55555 55555-y ID55555 55555kg 123/4567 ab1234567",  # part of a longer token
    "Version 3.12345 and 12345.6",
])
def test_non_identifier_numbers_kept(text):
    # phones off: "1000-2000" is also phone-shaped and the phone step (not this rule) takes it
    assert _clean(text, check_replace_phone_numbers=False) == text


def test_code_fragment_is_left_untouched_not_partially_masked():
    # Years off: the whole code stays. Years on: the year step still takes its year (unchanged), and the
    # number step does not add a second piece ("INV-<YEAR><NUMBER>" in 0.8.1).
    assert _clean("Invoice INV-2023-107683 paid", check_replace_years=False) == "Invoice INV-2023-107683 paid"
    assert _clean("Invoice INV-2023-107683 paid") == "Invoice INV-<YEAR>-107683 paid"
    assert _clean("Invoice INV-2023-107683 paid", replacement_mode="reversible") == \
        "Invoice INV-<YEAR_0>-107683 paid"
    assert "107683" in _clean("Invoice INV-2023-107683 paid", replacement_mode="synthetic")


@pytest.mark.parametrize("text, expected", [
    ("SSN: 123-45-6789", "SSN: <NUMBER>"),  # a grouped ID: every group alone is a joined part
    ("tax 1234-56-78-90 x", "tax <NUMBER> x"),
    ("on 2023-11-05 or 05-11-23 or 5-1-2023", "on 2023-11-05 or 05-11-23 or 5-1-2023"),  # date-shaped
    ("pages 100-200, v1-2-3, INV-123-45-6789", "pages 100-200, v1-2-3, INV-123-45-6789"),
])
def test_grouped_identifier_numbers(text, expected):
    assert _clean(text, check_replace_phone_numbers=False, check_replace_dates=False,
                  check_replace_years=False) == expected


def test_grouped_identifier_in_default_config():
    assert "6789" not in _clean("SSN: 123-45-6789")


def test_min_digits():
    assert _clean("code 12345 and 123456", replace_numbers_min_digits=6) == "code 12345 and <NUMBER>"
    assert _clean("code 7 and 42", replace_numbers_min_digits=1) == "code <NUMBER> and <NUMBER>"


def test_all_mode_is_081_behaviour():
    text = "Pay 1,250.00 at 14:30 on INV-2023-107683, age 42"
    assert _clean(text, replace_numbers_mode="all") == \
        "Pay <NUMBER> at <NUMBER>:<NUMBER> on INV-<YEAR><NUMBER>, age <NUMBER>"
    assert _clean("GPT4 and mp3", replace_numbers_mode="all", replace_numbers_in_words=True) == \
        "GPT<NUMBER> and mp<NUMBER>"


def test_in_words_lets_letters_touch_in_identifier_mode():
    assert _clean("ID55555 x", replace_numbers_in_words=True) == "ID<NUMBER> x"
    assert _clean("ab-55555", replace_numbers_in_words=True) == "ab-<NUMBER>"
    assert _clean("12-55555", replace_numbers_in_words=True) == "12-55555"


def test_check_replace_numbers_still_switches_off():
    assert _clean("Order 12345", check_replace_numbers=False) == "Order 12345"


def test_number_modes_in_every_replacement_mode():
    assert _clean("Order 12345 of 12.5", replacement_mode="reversible") == "Order <NUMBER_0> of 12.5"
    out = _clean("Order 12345 of 12.5", replacement_mode="synthetic")
    assert "12345" not in out and out.endswith(" of 12.5")


def test_process_batch_uses_mode():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cleaner = TextCleaner(cfg=TextCleanerConfig(
            check_ner_process=False, check_detect_language=False, check_statistical_model_processing=False))
    outs = [r.lm_text for r in cleaner.process_batch(["order 12345", "price 12.50"])]
    assert "<number>" in outs[0].lower() and "12" in outs[1]


@pytest.mark.parametrize("text", [
    "ID9876543210 x",  # glued to letters
    "ref-0612345678 x",  # joined to a token part
    "0612345678-x2 y",
    "ts 20231105123456 x",  # a yyyymmddhhmmss timestamp
    "on 20231105 x",  # a bare yyyymmdd date
])
def test_phone_reject_embedded(text):
    assert "<PHONE>" not in _clean(text, check_replace_iban_cards=False)
    assert "<PHONE>" in _clean(text, check_replace_iban_cards=False, phone_reject_embedded=False)


@pytest.mark.parametrize("text", [
    "Call 0612345678 now", "Tel:0612345678", "Ring +31 6 1234 5678", "Call 020 7946 0958",
    "Ring 06-12345678 now",  # 0.8.1 gives 06-<PHONE>; rejecting the tail would leave the whole phone
    "电话9876543210",  # unspaced script: still a phone
    "رقم ٠٦١٢٣٤٥٦٧٨ هنا",  # Arabic-Indic digits
    "फोन ९८७६५४३२१० पर",  # Devanagari digits
    "Call 2023010123 now",  # starts like a date (20230101) but is not a date or timestamp
])
def test_phone_reject_embedded_keeps_real_phones(text):
    assert "<PHONE>" in _clean(text)


@pytest.mark.parametrize("mode", ["placeholder", "reversible", "synthetic"])
def test_date_like_phone_is_masked_in_training_preset(mode):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = TextCleanerConfig.from_preset('training', check_ner_process=False, replacement_mode=mode)
        assert "2023010123" not in TextCleaner(cfg).process("Call 2023010123 now").lm_text


def test_embedded_phone_predicate():
    assert contact.embedded_phone("ab0612345678", 2, 12)
    assert not contact.embedded_phone("06 12345678", 0, 11)  # formatted: never rejected
    assert contact.embedded_phone("١٩٩٠٠١٠١٢٣٤٥", 0, 12)  # yyyymmdd in Arabic-Indic digits


def test_config_validation_and_legacy_globals(monkeypatch):
    from sct import config
    with pytest.raises(ValueError):
        TextCleanerConfig(replace_numbers_mode="some")
    for bad in (0, -1, True, 2.5):
        with pytest.raises(ValueError):
            TextCleanerConfig(replace_numbers_min_digits=bad)
    cfg = config._config_from_module_globals()
    assert (cfg.replace_numbers_mode, cfg.replace_numbers_min_digits, cfg.phone_reject_embedded) == \
        ("identifiers", 4, True)
    monkeypatch.setattr(config, "REPLACE_NUMBERS_MODE", "all")
    monkeypatch.setattr(config, "REPLACE_NUMBERS_MIN_DIGITS", 6)
    monkeypatch.setattr(config, "PHONE_REJECT_EMBEDDED", False)
    cfg = config._config_from_module_globals()
    assert (cfg.replace_numbers_mode, cfg.replace_numbers_min_digits, cfg.phone_reject_embedded) == \
        ("all", 6, False)


def test_browser_number_rule_follows_mode():
    rule = build_browser_manifest(TextCleanerConfig())["rules"]["number"]
    assert rule["pattern"] == constants.identifier_numbers_regex(
        4, letter_prefix=True, two_part=True, spaced_pair=True).pattern and rule["fold_decimal_digits"]
    old = build_browser_manifest(TextCleanerConfig(replace_numbers_mode="all"))["rules"]["number"]
    assert old["pattern"] == constants.NUMBERS_REGEX.pattern and "fold_decimal_digits" not in old


def test_streaming_redactors_follow_number_and_phone_rules():
    from sct.streaming import BufferedStreamingRedactor, WindowedStreamingRedactor
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cleaner = TextCleaner(cfg=TextCleanerConfig(
            check_ner_process=False, check_detect_language=False, check_statistical_model_processing=False,
            check_remove_punctuation=False, check_casefold=False, check_remove_isolated_letters=False))
    text = "order 12345 costs 12.50 and id9876543210 stays, zip 123 45 call 0612345678 today " * 3
    expected = cleaner.process(text).lm_text
    assert "<NUMBER>" in expected and "12.50" in expected and "id9876543210" in expected and "<PHONE>" in expected
    assert "123 45" not in expected
    for stream in (BufferedStreamingRedactor(cleaner),
                   WindowedStreamingRedactor(cleaner, max_buffer_chars=48, overlap_chars=24)):
        out = "".join(stream.feed(text[i:i + 7]) for i in range(0, len(text), 7)) + stream.finish()
        assert out.split() == expected.split()


# ID shapes the plain digit-run rule rejects (numbers_letter_prefix_ids, numbers_two_part_ids,
# numbers_spaced_pair_ids); made-up values.
_NO_STEPS = dict(check_replace_phone_numbers=False, check_replace_dates=False, check_replace_years=False)


@pytest.mark.parametrize("field, text, expected", [
    ("numbers_letter_prefix_ids", "Passport AB1234567, ref X12345678Z.", "Passport <NUMBER>, ref <NUMBER>."),
    ("numbers_letter_prefix_ids", "Ausweis Ж1234567 L٤٥٦٧٨٩٠", "Ausweis Ж1234567 <NUMBER>"),  # Latin only, any digits
    ("numbers_two_part_ids", "case 123456/7890 and 12345-678901 or 123456-789K.",
     "case <NUMBER> and <NUMBER> or <NUMBER>."),
    ("numbers_two_part_ids", "Az. 12345/67, nr 06123.45678", "Az. <NUMBER>, nr <NUMBER>"),
    ("numbers_two_part_ids", "ملف ١٢٣٤٥٦/٧٨٩٠ هنا", "ملف <NUMBER> هنا"),  # Arabic-Indic digits
    ("numbers_spaced_pair_ids", "Zip 123 45 Stockholm", "Zip <NUMBER> Stockholm"),
    ("numbers_spaced_pair_ids", "पिन ४५६ ७८ है", "पिन <NUMBER> है"),  # Devanagari digits
])
def test_id_shape_rules(field, text, expected):
    assert _clean(text, **_NO_STEPS) == expected
    assert _clean(text, **_NO_STEPS, **{field: False}) == text


@pytest.mark.parametrize("text", [
    "pay USD1000000 or EUR2500000",  # a currency code is an amount
    "ab1234567 ABCD1234567 AB12345 AB1234567abc",  # lowercase, 4+ letters, under 6 digits, glued letters
    "years 2023-2024, 1990/2010 and range 1000-3000, 10000/20000",  # year pairs, round ranges
    "at 1234-5678:30, €1234-5678 and 1234-5678€, 1234.5678%",  # time, amounts, percent
    "123/4567 12-34567 1234.567 3.14159",  # too few digits, decimals, more groups
    "cost 123 45% or €123 45, 1 123 45, 123 45.6",  # amounts, longer groups
])
def test_id_shape_rules_keep_non_ids(text):
    assert _clean(text, **_NO_STEPS) == text


def test_id_shape_rules_only_in_identifiers_mode():
    # 'all' = 0.8.1: digit parts as before, the new fields change nothing
    text = "Passport AB1234567, case 123456/7890, zip 123 45"
    assert _clean(text, replace_numbers_mode="all", **_NO_STEPS) == \
        _clean(text, replace_numbers_mode="all", numbers_letter_prefix_ids=False, numbers_two_part_ids=False,
               numbers_spaced_pair_ids=False, **_NO_STEPS) == \
        "Passport AB1234567, case <NUMBER>/<NUMBER>, zip <NUMBER> <NUMBER>"


def test_id_shape_legacy_globals(monkeypatch):
    from sct import config
    cfg = config._config_from_module_globals()
    assert (cfg.numbers_letter_prefix_ids, cfg.numbers_two_part_ids, cfg.numbers_spaced_pair_ids) == (True,) * 3
    for name in ("NUMBERS_LETTER_PREFIX_IDS", "NUMBERS_TWO_PART_IDS", "NUMBERS_SPACED_PAIR_IDS"):
        monkeypatch.setattr(config, name, False)
    cfg = config._config_from_module_globals()
    assert (cfg.numbers_letter_prefix_ids, cfg.numbers_two_part_ids, cfg.numbers_spaced_pair_ids) == (False,) * 3
