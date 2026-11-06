"""Folding for detection leaves ordinary prose punctuation, units and join controls unchanged."""

import pytest

from sct import TextCleaner, TextCleanerConfig
from sct.utils.normtext import fold_styled


def _clean(text, **kw):
    return TextCleaner(TextCleanerConfig(check_ner_process=False, **kw)).process(text).lm_text


def test_fullwidth_cjk_prose_punctuation_kept():
    assert _clean("担当：鈴木様。東京．大阪") == "担当：鈴木様。東京．大阪"


@pytest.mark.parametrize("fold", [True, False])
def test_fullwidth_separator_in_prose_kept(fold):
    assert _clean("東京－大阪．名古屋＿線", fold_styled_characters=fold) == "東京－大阪．名古屋＿線"


def _clean_training(text, mode="placeholder", **kw):
    cfg = TextCleanerConfig.from_preset("training", check_ner_process=False, replacement_mode=mode, **kw)
    return TextCleaner(cfg).process(text).lm_text


def test_fullwidth_question_mark_in_prose_kept():
    assert _clean_training("本当？はい", "placeholder") == "本当？はい"


def test_fullwidth_dot_in_prose_kept():
    assert _clean_training("例え．テスト", "placeholder") == "例え．テスト"


def test_zwnj_word_keeps_its_join_control():
    out = TextCleaner(TextCleanerConfig(check_ner_process=False)).process("نامه‌ای．ایران").lm_text
    assert out == "نامه‌ای．ایران"


@pytest.mark.parametrize("text", ["日本語の文。次の文。", "メールはinfo@example.jp。よろしく",
                                  "サイトはhttps://example.com。次"])
def test_cjk_sentence_end_kept(text):
    assert fold_styled(text) == text


def test_squared_units_stay():
    assert fold_styled("5㎏ and 3㎞") == "5㎏ and 3㎞"


def _cleaner(mode="placeholder", **kw):
    return TextCleaner(TextCleanerConfig.from_preset("training", check_ner_process=False, replacement_mode=mode, **kw))


def test_prose_punctuation_without_at_keeps_width():
    assert _cleaner().process("你好！世界．テスト＿です").lm_text == "你好！世界．テスト＿です"


@pytest.mark.parametrize("text, expected", [
    ("তানিয়া ইসলাম-এর অভিযোগ", "তানিয়া ইসলাম-এর অভিযোগ"),
    ("東京-大阪", "東京-大阪"),
    ("कर्म-योग", "कर्म-योग"),
    ("l'été", "l'été"),
    ("東京 - 大阪", "東京 大阪"),
])
def test_hyphen_between_letters_of_any_script_is_kept(text, expected):
    cleaner = TextCleaner(cfg=TextCleanerConfig(check_ner_process=False))
    assert cleaner.process(text).lm_text == expected


@pytest.mark.parametrize("text", [
    "ดู​สวัสดี",  # Thai word break
    "نامه‌ای",  # Persian ZWNJ
    "👨‍👩",  # emoji ZWJ sequence
])
def test_invisible_between_other_letters_stays(text):
    assert fold_styled(text) == text
