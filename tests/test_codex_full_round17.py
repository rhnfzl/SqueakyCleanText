"""Codex 0.8.0 full review, round 17: ids 1-4."""

import dataclasses

import pytest

from tests.helpers import bare_ner, set_cfg
from sct import TextCleaner, TextCleanerConfig
from sct.privacy import EntityRule, PrivacyPolicy
from sct.streaming import WindowedStreamingRedactor
from sct.utils.ner import _presidio_type_map
from tests.conftest import TEST_NER_MODELS

# --- 1: multi-digit styled numbers (⑫) stay whole across stream windows ---


def test_stream_keeps_multi_digit_styled_phone_whole():
    cleaner = TextCleaner(TextCleanerConfig.from_preset("training", check_ner_process=False))
    text = "a a a a a Call ⓪⑥ ⑫③④ ⑤⑥⑦⑧ tail"
    stream = WindowedStreamingRedactor(cleaner, max_buffer_chars=30, overlap_chars=8)
    out = "".join(stream.feed(c) for c in text) + stream.finish()
    assert out == cleaner.process(text).lm_text == "a a a a a Call <PHONE> tail"


# --- 2: Presidio aliases (PER/LOC/ORG) reach the canonical policy rules ---


class _Result:
    def __init__(self, entity_type, start, end):
        self.entity_type, self.start, self.end, self.score = entity_type, start, end, 0.9


class _Analyzer:
    def analyze(self, *, text, language):
        return [_Result("PERSON", 0, 12), _Result("LOCATION", 16, 22), _Result("ORGANIZATION", 26, 30)]


def test_presidio_aliases_use_canonical_types():
    ner = bare_ner()
    set_cfg(ner, ner_backend="presidio_gliner")
    ner._analyzer = _Analyzer()
    ner._presidio_types = _presidio_type_map(False, {"person": "PER", "location": "LOC", "organization": "ORG"})
    policy = PrivacyPolicy(name="p", version="1",
                           rules={"PERSON": EntityRule(action="redact"),
                                  "LOCATION": EntityRule(action="placeholder"),
                                  "ORGANISATION": EntityRule(action="placeholder")},
                           default_rule=EntityRule(action="keep"))
    ner._privacy_policy = policy
    result = ner.ner_process_detailed("Anna Schmidt in London at Acme", positional_tags=("PER",))
    assert result.text == "<REDACTED> in <LOCATION> at <ORGANISATION>"
    assert [f.entity.entity_type for f in result.findings] == ["PERSON", "LOCATION", "ORGANISATION"]


# --- 3: a Haystack round trip keeps the PII label map derived ---


@pytest.mark.parametrize("legacy", [False, True])
def test_haystack_round_trip_rederives_pii_map(legacy):
    pytest.importorskip("haystack")
    from sct.integrations.haystack import SCTDocumentCleaner

    cfg = TextCleanerConfig(ner_mode="pii", legacy_type_names=legacy)
    loaded = SCTDocumentCleaner.from_dict(SCTDocumentCleaner(config=cfg).to_dict())._config
    flipped = dataclasses.replace(loaded, legacy_type_names=not legacy)
    direct = dataclasses.replace(cfg, legacy_type_names=not legacy)
    assert dict(flipped.gliner_label_map) == dict(direct.gliner_label_map)
    assert flipped.positional_tags == direct.positional_tags


# --- 4: an NER span that cuts a deferred tag leaves no fragment in any mode ---


@pytest.mark.parametrize("mode", ["synthetic", "placeholder"])
def test_cut_tag_leaves_no_fragment(mode):
    text = "University of ann@example.com. Contact bob@example.org today."
    cleaner = TextCleaner(TextCleanerConfig.from_preset(
        "training", language="ENGLISH", replacement_mode=mode, ner_models=TEST_NER_MODELS))
    out = cleaner.process(text).lm_text
    assert out.count("<") == out.count(">") and "ann@example.com" not in out
