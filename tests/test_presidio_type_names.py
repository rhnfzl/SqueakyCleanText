"""Presidio aliases and cloned PII configs keep one canonical type name and label map."""

import copy
import dataclasses
import pickle

import pytest

from sct import TextCleanerConfig
from sct.privacy import EntityRule, PrivacyPolicy
from sct.utils.ner import _presidio_type_map

from tests.helpers import bare_ner, set_cfg, set_policy


# --- dataclasses.replace re-derives the PII label map ---
def test_cloned_pii_config_follows_legacy_type_names():
    canonical = TextCleanerConfig(ner_mode="pii", check_ner_process=False)
    legacy = TextCleanerConfig(ner_mode="pii", legacy_type_names=True, check_ner_process=False)
    for source, flag, direct in ((canonical, True, legacy), (legacy, False, canonical),
                                 (pickle.loads(pickle.dumps(canonical)), True, legacy)):  # noqa: S301 - own bytes
        clone = dataclasses.replace(source, legacy_type_names=flag)
        assert dict(clone.gliner_label_map) == dict(direct.gliner_label_map)
        assert clone.positional_tags == direct.positional_tags


def test_cloned_pii_config_keeps_explicit_label_map():
    cfg = TextCleanerConfig(ner_mode="pii", gliner_label_map={"phone_number": "TEL"}, check_ner_process=False)
    assert dict(dataclasses.replace(cfg, legacy_type_names=True).gliner_label_map) == {"phone_number": "TEL"}


# --- Presidio entity types follow the configured names ---
class _Result:
    def __init__(self, entity_type, start, end):
        self.entity_type, self.start, self.end, self.score = entity_type, start, end, 0.9


@pytest.mark.parametrize("legacy, phone", [(False, "PHONE"), (True, "PHONE_NUMBER")])
def test_presidio_types_mapped(legacy, phone):
    from sct.utils.ner import _presidio_type_map

    class Analyzer:
        def analyze(self, *, text, language):
            return [_Result("PHONE_NUMBER", 5, 15), _Result("NAME", 0, 4)]

    ner = bare_ner()
    set_cfg(ner, ner_backend="presidio_gliner")
    ner._analyzer = Analyzer()
    ner._presidio_types = _presidio_type_map(legacy, {"name": "PER"})
    policy = PrivacyPolicy(name="p", version="1", rules={phone: EntityRule(action="redact")},
                           default_rule=EntityRule(action="keep"))
    set_policy(ner, policy)
    result = ner.ner_process_detailed("Anna 0612345678 now", positional_tags=("PER",))
    assert "0612345678" not in result.text
    assert {f.entity.entity_type for f in result.findings} == {phone, "PERSON"}  # PER alias -> canonical (round 17)


class _Analyzer:
    def analyze(self, *, text, language):
        return [_Result("PHONE_NUMBER", 5, 15)]


def test_empty_presidio_type_map_keeps_legacy_names():
    ner = bare_ner()
    set_cfg(ner, ner_backend="presidio_gliner")
    ner._analyzer = _Analyzer()
    ner._presidio_types = _presidio_type_map(True, None)
    assert ner._presidio_types == {}
    policy = PrivacyPolicy(name="p", version="1", rules={"PHONE_NUMBER": EntityRule(action="redact")},
                           default_rule=EntityRule(action="keep"))
    set_policy(ner, policy)
    result = ner.ner_process_detailed("Anna 0612345678 now", positional_tags=("PER",))
    assert "0612345678" not in result.text
    assert [f.entity.entity_type for f in result.findings] == ["PHONE_NUMBER"]


class _PersonAnalyzer:
    def analyze(self, *, text, language):
        return [_Result("PERSON", 0, 12), _Result("LOCATION", 16, 22), _Result("ORGANIZATION", 26, 30)]


def test_presidio_aliases_use_canonical_types():
    ner = bare_ner()
    set_cfg(ner, ner_backend="presidio_gliner")
    ner._analyzer = _PersonAnalyzer()
    ner._presidio_types = _presidio_type_map(False, {"person": "PER", "location": "LOC", "organization": "ORG"})
    policy = PrivacyPolicy(name="p", version="1",
                           rules={"PERSON": EntityRule(action="redact"),
                                  "LOCATION": EntityRule(action="placeholder"),
                                  "ORGANISATION": EntityRule(action="placeholder")},
                           default_rule=EntityRule(action="keep"))
    set_policy(ner, policy)
    result = ner.ner_process_detailed("Anna Schmidt in London at Acme", positional_tags=("PER",))
    assert result.text == "<REDACTED> in <LOCATION> at <ORGANISATION>"
    assert [f.entity.entity_type for f in result.findings] == ["PERSON", "LOCATION", "ORGANISATION"]


# --- a PII label map kept after replace() to another mode survives pickle, deepcopy and Haystack ---
def _kept_map_config():
    pii = TextCleanerConfig.from_preset("training", ner_mode="pii", language="ENGLISH")
    return dataclasses.replace(pii, ner_mode="standard")


def test_kept_pii_label_map_survives_round_trip():
    cfg = _kept_map_config()
    assert cfg.gliner_label_map["name"] == "PER"
    for loaded in (pickle.loads(pickle.dumps(cfg)), copy.deepcopy(cfg)):  # noqa: S301 - own bytes
        assert dict(loaded.gliner_label_map) == dict(cfg.gliner_label_map)
        assert loaded == cfg


def test_kept_pii_label_map_survives_haystack():
    pytest.importorskip("haystack")
    from sct.integrations.haystack import SCTDocumentCleaner

    cfg = _kept_map_config()
    loaded = SCTDocumentCleaner.from_dict(SCTDocumentCleaner(config=cfg).to_dict())._config
    assert dict(loaded.gliner_label_map) == dict(cfg.gliner_label_map)
