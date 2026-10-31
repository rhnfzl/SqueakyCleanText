"""Codex 0.8.0 full review, round 2: ids 1, 2, 3, 4."""

import sys
import types

import pytest

from sct import TextCleaner, TextCleanerConfig
from sct.presets import preset_config
from sct.privacy import DetectedEntity, EntityFinding
from sct.signals import public_view
from sct.streaming import WindowedStreamingRedactor

FAST = preset_config("training", check_ner_process=False)


@pytest.fixture(scope="module")
def fast_cleaner():
    return TextCleaner(cfg=FAST)


# --- 1: RFC punctuation in the local part left the address or a prefix visible ---

@pytest.mark.parametrize("text, expected", [
    ("Contact alice!@example.com", "Contact <EMAIL>"),
    ("Mail o'connor@example.com now", "Mail <EMAIL> now"),
    ("Mail jo&co@example.com now", "Mail <EMAIL> now"),
    ("'alice@example.com'", "'<EMAIL>'"),
    ("ends with dot. alice@example.com", "ends with dot. <EMAIL>"),
])
def test_email_local_part_punctuation_is_fully_masked(fast_cleaner, text, expected):
    assert fast_cleaner.process(text).lm_text == expected


# --- 2: the windowed stream split a spaced phone number across two process() calls ---

def test_windowed_stream_never_splits_inside_a_phone_number(fast_cleaner):
    text = "aaaa aaaa aaaa Call 06 12 34 56 78 tail"
    redactor = WindowedStreamingRedactor(fast_cleaner, max_buffer_chars=32, overlap_chars=16)
    out = redactor.feed(text) + redactor.finish()
    assert out == fast_cleaner.process(text).lm_text == "aaaa aaaa aaaa Call <PHONE> tail"


# --- 3: models_for grew a list ensemble route with GLINER on every call ---

def test_models_for_leaves_a_list_route_unchanged(monkeypatch):
    fake = types.ModuleType("sct.utils.gliner_adapter")

    class GLiNERAdapter:
        _onnx = False

        def __init__(self, **kwargs):
            pass

    fake.GLiNERAdapter = GLiNERAdapter
    monkeypatch.setitem(sys.modules, "sct.utils.gliner_adapter", fake)
    route = {"ENGLISH": ["ENGLISH", "MULTILINGUAL"]}
    ner = TextCleaner(cfg=TextCleanerConfig(
        ner_backend="ensemble_onnx", gliner_model="x/y", ner_ensemble=route)).GeneralNER
    ner._provenance["GLINER"] = {"key": "GLINER", "id": "x/y", "revision": None, "provider": None}
    first = ner.models_for("ENGLISH")
    assert ner.models_for("ENGLISH") == first
    assert route["ENGLISH"] == ["ENGLISH", "MULTILINGUAL"]
    assert list(ner._get_ensemble_keys("ENGLISH")) == ["ENGLISH", "MULTILINGUAL"]


# --- 4: overlapping findings from two models were counted twice in the metadata ---

def test_overlapping_findings_count_once(fast_cleaner):
    text = "John Smith works here."
    state = fast_cleaner._before_ner(text)

    def finding(end, score):
        return EntityFinding(DetectedEntity("PERSON", score, text[:end], 0, end),
                             "placeholder", "")

    result = fast_cleaner._after_ner(
        state, "<PERSON> works here.", (finding(4, 0.99), finding(10, 0.9)))
    assert result.metadata["entity_counts"] == {"PERSON": {"count": 1, "chars": 10}}
    view = public_view(result)
    assert (view["pii_count"], view["pii_chars"]) == (1, 10)
