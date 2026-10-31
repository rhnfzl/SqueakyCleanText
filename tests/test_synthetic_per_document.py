"""Synthetic mode: fakes differ across documents, same document -> same fakes."""

import pickle
import re
import unicodedata

import pytest

from sct import config as config_module
from sct.config import TextCleanerConfig
from sct.integrations._runtime import CleanerHandle
from sct.sct import TextCleaner
from sct.utils.synthetic import SyntheticReplacer, faker_locale_for
from tests.conftest import TEST_NER_MODELS

pytest.importorskip("faker")

DOCS = [f"mail user{i}@example.com or call +1-202-555-01{i:02d}" for i in range(6)]


def _regex_cleaner(**overrides):
    return TextCleaner(TextCleanerConfig(**{
        'replacement_mode': 'synthetic', 'check_ner_process': False,
        'check_detect_language': False, 'check_statistical_model_processing': False,
        **overrides,
    }))


def _texts(results):
    return [r.lm_text for r in results]


def test_batch_documents_get_different_fakes():
    texts = _texts(_regex_cleaner().process_batch(DOCS, batch_size=4))
    assert len(set(texts)) == len(DOCS)  # 0.7: every thread's first document got the same fakes


def test_same_document_same_fakes_across_calls_workers_and_order():
    expected = _texts(_regex_cleaner().process_batch(DOCS, batch_size=1))
    cleaner = _regex_cleaner()
    assert [cleaner.process(d).lm_text for d in reversed(DOCS)] == expected[::-1]
    assert _texts(cleaner.process_batch(DOCS, batch_size=4)) == expected
    assert _regex_cleaner().process(DOCS[3]).lm_text == expected[3]


def test_consistent_mapping_inside_one_document():
    text = _regex_cleaner().process("ann@example.com wrote; reply to ann@example.com").lm_text
    first, second = text.split(" wrote; reply to ")
    assert first == second and "ann@example.com" not in text


def test_seed_is_configurable_and_legacy_global_flows():
    assert _regex_cleaner().process(DOCS[0]).lm_text != _regex_cleaner(synthetic_seed=7).process(DOCS[0]).lm_text
    old = config_module.SYNTHETIC_SEED, config_module.SYNTHETIC_PER_DOCUMENT
    try:
        config_module.SYNTHETIC_SEED, config_module.SYNTHETIC_PER_DOCUMENT = 7, False
        cfg = config_module._config_from_module_globals()
    finally:
        config_module.SYNTHETIC_SEED, config_module.SYNTHETIC_PER_DOCUMENT = old
    assert (cfg.synthetic_seed, cfg.synthetic_per_document) == (7, False)


def test_per_document_off_keeps_the_single_stream():
    # 0.7: each fresh thread/cleaner restarts one stream, so different documents get the same first fake
    emails = [_regex_cleaner(synthetic_per_document=False).process(d).lm_text.split()[1] for d in DOCS[:3]]
    assert len(set(emails)) == 1


def test_adapter_handle_in_a_new_process_reproduces_fakes():
    handle = CleanerHandle(TextCleanerConfig(replacement_mode='synthetic', check_ner_process=False))
    expected = handle.get().process(DOCS[2]).lm_text
    clone = pickle.loads(pickle.dumps(handle))  # noqa: S301 - trusted round trip
    assert clone.get().process(DOCS[2]).lm_text == expected


def test_batched_ner_matches_single_document_processing():
    cfg = TextCleanerConfig(
        replacement_mode='synthetic', ner_models=TEST_NER_MODELS, ner_batch_size=8,
        check_detect_language=False, language='ENGLISH',
    )
    docs = ["John Smith wrote to anna@example.com.", "Angela Merkel wrote to anna@example.com."]
    cleaner = TextCleaner(cfg)
    batched = _texts(cleaner.process_batch(docs, batch_size=2))
    assert batched == [cleaner.process(d).lm_text for d in docs]
    assert "John Smith" not in batched[0] and "Angela Merkel" not in batched[1]
    assert batched[0].split()[-1] != batched[1].split()[-1]  # same email, different documents


# --- synthetic_locale: fakes follow the document language ---

_EMAIL = re.compile(r"^[\w.+-]+@[\w-]+(\.[\w-]+)+$", re.ASCII)


@pytest.mark.parametrize(("language", "locale", "script"), [
    ('DUTCH', 'nl_NL', 'Latin'), ('GERMAN', 'de_DE', 'Latin'), ('CHINESE', 'zh_CN', 'Han'),
    ('JAPANESE', 'ja_JP', 'Han'), ('ARABIC', 'ar_AA', 'Arabic'), ('HINDI', 'hi_IN', 'Devanagari'),
])
def test_fakes_follow_the_document_language(language, locale, script):
    assert faker_locale_for(language) == locale
    replacer = SyntheticReplacer()
    replacer.reset_cache("some document", language)
    name = replacer.get_replacement('PERSON', 'Jan')
    letters = [c for c in name if c.isalpha()]
    # Script of the fake name, from Unicode character names (no word lists).
    assert all(unicodedata.name(c).split()[0] in {'Latin': ('LATIN',), 'Han': ('CJK', 'HIRAGANA', 'KATAKANA'),
               'Arabic': ('ARABIC',), 'Devanagari': ('DEVANAGARI',)}[script] for c in letters), name
    assert _EMAIL.match(replacer.get_replacement('EMAIL', 'a@b.com'))
    assert sum(c.isdigit() for c in replacer.get_replacement('PHONE', '+31 6 1234 5678')) >= 7


def test_dutch_fakes_differ_from_us_english_and_en_us_restores_them():
    replacer, legacy = SyntheticReplacer(), SyntheticReplacer(locale='en_US')
    names = []
    for r in (replacer, legacy):
        r.reset_cache("doc", 'DUTCH')
        names.append([r.get_replacement('PERSON', f'p{i}') for i in range(5)])
    old = SyntheticReplacer()
    old.reset_cache("doc")  # no language: Faker's default locale, as in 0.7
    assert names[1] == [old.get_replacement('PERSON', f'p{i}') for i in range(5)]
    assert names[0] != names[1]


def test_unknown_or_unsupported_language_falls_back_to_en_us():
    assert faker_locale_for(None) == 'en_US'
    assert faker_locale_for('ENGLISH') == 'en_US'
    assert faker_locale_for('BOKMAL') == 'en_US'  # Faker has no nb locale


def test_cleaner_uses_the_detected_language_and_option_flows():
    text = "Neem contact op met jan.devries@voorbeeld.nl over de vergadering van morgen in het kantoor."
    dutch = _regex_cleaner(check_detect_language=True).process(text)
    assert dutch.language == 'DUTCH'
    forced = _regex_cleaner(check_detect_language=True, synthetic_locale='en_US').process(text)
    assert dutch.lm_text != forced.lm_text
    assert _regex_cleaner(check_detect_language=True).process(text).lm_text == dutch.lm_text
    old = config_module.SYNTHETIC_LOCALE
    try:
        config_module.SYNTHETIC_LOCALE = 'de_DE'
        assert config_module._config_from_module_globals().synthetic_locale == 'de_DE'
    finally:
        config_module.SYNTHETIC_LOCALE = old


def test_localised_fakes_reproduce_in_a_new_process():
    cfg = TextCleanerConfig(replacement_mode='synthetic', check_ner_process=False, language='CHINESE')
    handle = CleanerHandle(cfg)
    text = "请联系 wang.wei@example.cn"
    expected = handle.get().process(text).lm_text
    clone = pickle.loads(pickle.dumps(handle))  # noqa: S301 - trusted round trip
    assert clone.get().process(text).lm_text == expected
