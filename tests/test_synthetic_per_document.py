"""Synthetic mode: fakes differ across documents, same document -> same fakes."""

import pickle

import pytest

from sct import config as config_module
from sct.config import TextCleanerConfig
from sct.integrations._runtime import CleanerHandle
from sct.sct import TextCleaner
from tests.conftest import TEST_NER_MODELS

pytest.importorskip("faker")

DOCS = [f"mail user{i}@example.com or call +1-202-555-01{i:02d}" for i in range(6)]


def _regex_cleaner(**overrides):
    return TextCleaner(TextCleanerConfig(
        replacement_mode='synthetic', check_ner_process=False,
        check_detect_language=False, check_statistical_model_processing=False,
        **overrides,
    ))


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
