"""Codex 0.8.0 full review, round 21: ids 1-5."""

import pickle

import pytest

from sct import TextCleaner, TextCleanerConfig
from sct.browser import build_browser_manifest
from sct.streaming import WindowedStreamingRedactor
from sct.utils.normtext import fold_styled
from tests.conftest import TEST_NER_MODELS

MODES = ["placeholder", "synthetic", "reversible"]


def _clean(text, mode="placeholder", **kw):
    cfg = TextCleanerConfig.from_preset("training", check_ner_process=False, replacement_mode=mode, **kw)
    return TextCleaner(cfg).process(text).lm_text


def _stream(text, **kw):
    stream = WindowedStreamingRedactor(TextCleaner(TextCleanerConfig(check_ner_process=False, **kw)),
                                       max_buffer_chars=48, overlap_chars=12)
    return "".join(stream.feed(text[i:i + 3]) for i in range(0, len(text), 3)) + stream.finish()


# --- 1: dates written in native decimal digits are dates ---

DATES = ["١٥/٠١/١٩٩٠", "१५/०१/१९९०", "۱۵/۰۱/۱۹۹۰", "١٩٩٠-٠١-١٥", "১৫.০১.১৯৯০"]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("date", DATES)
def test_native_digit_date_replaced(date, mode):
    out = _clean(f"Born {date}.", mode, check_replace_dates=True)
    assert date not in out and "DATE" in out


def test_native_digit_date_restores_original():
    cfg = TextCleanerConfig.from_preset("training", check_ner_process=False, replacement_mode="reversible",
                                        check_replace_dates=True)
    result = TextCleaner(cfg).process("Born ١٥/٠١/١٩٩٠.")
    assert result.metadata["anon_map"].deanonymize(result.lm_text) == "Born ١٥/٠١/١٩٩٠."


def test_native_digit_date_in_stream():
    out = _stream("The patient was born on ١٥/٠١/١٩٩٠ in the city.", check_replace_dates=True)
    assert "١٩٩٠" not in out and "<DATE>" in out


def test_browser_date_rule_folds_digits():
    assert build_browser_manifest(TextCleanerConfig())["rules"]["date"]["fold_decimal_digits"]


# --- 2: IDNA dot separators (。 ｡) in a mail domain or URL host ---

CONTACTS = ["alice@acmemail。com", "alice@acmemail｡com", "https://www。acmemail。com/?token=SecretABC",
            "www。acmemail。com/?token=SecretABC"]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("contact", CONTACTS)
def test_idna_dot_contact_masked(contact, mode):
    out = _clean(f"See {contact} now", mode)
    assert "acmemail" not in out and "Secret" not in out
    assert out.startswith("See ") and out.endswith(" now")


def test_idna_dot_contact_masked_in_stream():
    out = _stream("please write to alice@example。com today")
    assert "alice" not in out and "example" not in out


@pytest.mark.parametrize("text", ["日本語の文。次の文。", "メールはinfo@example.jp。よろしく",
                                  "サイトはhttps://example.com。次"])
def test_cjk_sentence_end_kept(text):
    assert fold_styled(text) == text


# --- 3: squared Latin letters fold like circled ones ---


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("email", ["🄹🄾🄷🄽@acmemail.com", "JOHN@🄰🄲🄼🄴🄼🄰🄸🄻.com"])
def test_squared_letter_email_masked(email, mode):
    out = _clean(f"Mail {email} now", mode)
    assert "JOHN" not in out and "🄹" not in out and "acmemail" not in out and "🄰" not in out


def test_squared_units_stay():
    assert fold_styled("5㎏ and 3㎞") == "5㎏ and 3㎞"


# --- 4: layout hints inside a non-Latin mail domain ---


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("hint", ["​", "­", "⁠"])
def test_layout_hint_in_idn_domain_masked(hint, mode):
    out = _clean(f"Mail info@例{hint}え.jp now", mode)
    assert "info" not in out and "例" not in out


def test_layout_hint_in_idn_domain_masked_in_stream():
    out = _stream("please write to info@例​え.jp today")
    assert "info" not in out and "例" not in out


def test_join_controls_in_domain_kept():
    assert fold_styled("john@نامه‌ای.ایران") == "john@نامه‌ای.ایران"


# --- 5: custom regex markers are protected from NER like <TAG> markers ---

TEXT = "John Smith at ann@example.com and Jane Doe at bob@example.org, in London."


def _ner_cfg(mode, **kw):
    return TextCleanerConfig.from_preset("training", language="ENGLISH", ner_models=TEST_NER_MODELS,
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


# --- fuzz follow-ups (extended harness, training with dates on) ---


@pytest.mark.parametrize("text", ["วันเกิด15/01/1990ครับ", "生日1990-01-15です", "Born ꘡꘩꘧꘡–꘠꘧–꘠7 ok",
                                  "Born 1971‑07‑07 ok"])
def test_glued_or_typeset_date_replaced(text):
    out = _clean(text, check_replace_dates=True)
    assert "<DATE>" in out and "1990" not in out and "07" not in out


def test_typeset_date_kept_with_fold_off():
    assert "<DATE>" not in _clean("Born 1971‑07‑07 ok", check_replace_dates=True,
                                  fold_styled_characters=False)


def test_latin_glued_date_not_replaced():
    assert "<DATE>" not in _clean("code ab15/01/1990 ok", check_replace_dates=True)


@pytest.mark.parametrize("email", ["alice@acmemail｡​com", "info@例え.テ​スト"])
def test_idna_dot_or_later_label_hint_masked(email):
    out = _clean(f"Mail {email} now")
    assert "alice" not in out and "info" not in out


def test_circled_number_between_typeset_dashes_folds():
    out = _clean("Born 2026‒⑩–02 ok", check_replace_dates=True)
    assert out == "Born <DATE> ok"
