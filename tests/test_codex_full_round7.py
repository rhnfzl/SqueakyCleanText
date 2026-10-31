"""Codex 0.8.0 full review, round 7: ids 1, 2, 3."""

import os
from types import SimpleNamespace

import pytest

from sct import TextCleaner
from sct.presets import preset_config
from sct.streaming import WindowedStreamingRedactor
from sct.utils import resources
from tests.test_chunking import _FakeRunDetector, _word_ner
from tests.test_review_round1 import _Tok


# --- 1: an HTML entity inside a spaced phone does not hide it from the split check ---

@pytest.mark.parametrize("entity", ["&#32;", "&#x20;", "&nbsp;"])
def test_streaming_never_splits_inside_html_escaped_phone(entity):
    cleaner = TextCleaner(cfg=preset_config("training", check_ner_process=False))
    redactor = WindowedStreamingRedactor(cleaner, max_buffer_chars=32, overlap_chars=16)
    text = f"aaaa aaaa aaaa Call 06 12{entity}34 56 78 tail"
    out = redactor.feed(text) + redactor.finish()
    assert out == cleaner.process(text).lm_text  # "<PHONE>" for the entities that decode to a space


def test_streaming_unknown_entity_keeps_raw_offsets():
    cleaner = TextCleaner(cfg=preset_config("training", check_ner_process=False))
    redactor = WindowedStreamingRedactor(cleaner, max_buffer_chars=24, overlap_chars=8)
    text = "word &bogus; word word word word word"
    assert redactor.feed(text) + redactor.finish() == cleaner.process(text).lm_text


# --- 2: overlap offsets come from the route's own tokenizer ---

def test_overlap_offsets_use_route_tokenizer():
    def one_span(text, **kwargs):  # primary: one token over the whole text
        spans = [(0, len(text))]
        return SimpleNamespace(input_ids=spans, offset_mapping=spans)

    one_span.max_len_single_sentence = 1000
    ner = _word_ner(2, _FakeRunDetector(min_words=2), budget=12)
    ner.tokenizer = one_span
    ner._ensemble_models = {"ENGLISH": ("ENGLISH",), "HINDI": ("HINDI",)}
    ner._tokenizers = {"ENGLISH": one_span, "HINDI": _Tok(1)}
    text = "a b c d e f g John Smith h i j k l m n"
    windows = [w[0] for w in ner._plan_windows(text, ner._get_ensemble_keys("HINDI"))]
    assert all(len(w.split()) <= 12 for w in windows)
    assert any("John Smith" in w for w in windows)
    kw = {"positional_tags": ("PER",), "ner_confidence_threshold": 0.5}
    masked = text.replace("John Smith", "<PERSON>")
    assert ner.ner_process_detailed(text, language="HINDI", **kw).text == masked
    assert [r.text for r in ner.ner_process_batch_detailed([text], ["HINDI"], **kw)] == [masked]


# --- 3: a forked child does not inherit the detector lock held ---

@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")
def test_forked_child_gets_free_detector_lock():
    with resources._detector_lock:  # another thread is building a detector at fork time
        pid = os.fork()
        if pid == 0:
            os._exit(0 if resources._detector_lock.acquire(timeout=2) else 1)
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0
