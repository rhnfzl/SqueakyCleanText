"""Rule-only credential scanning, with value-free canonical findings."""
import builtins
import dataclasses
import importlib.util
import os
import sys
from types import SimpleNamespace

import pytest

from sct.detectors import DetectorUnavailable
from sct.findings import DetectionContext


def detector(**kwargs):
    from sct.detectors_datafog import DataFogSecretDetector
    return DataFogSecretDetector(**kwargs)


def scan(instance, text):
    return instance.detect(text, language=None, context=DetectionContext())


def finding(start=2, end=5, **kwargs):
    return SimpleNamespace(entity_type="API_KEY", codepoint_range=SimpleNamespace(start=start, end=end),
                           byte_range=SimpleNamespace(start=start + 3, end=end + 3),
                           confidence=None, matched_text="abc", **kwargs)


def fake_sdk(monkeypatch, results):
    monkeypatch.setitem(sys.modules, "datafog_core", SimpleNamespace(scan=lambda text, options: results))


def test_canonical_conversion_drops_raw_objects(monkeypatch):
    raw = finding()
    raw.private_payload = "sensitive"
    fake_sdk(monkeypatch, [raw])
    instance = detector(entities=("API_KEY",))
    hit, = scan(instance, "😀 abc")
    assert (hit.start, hit.end, hit.entity_type, hit.score, hit.coordinate_space) == (2, 5, "API_KEY", None, "canonical")
    assert set(dataclasses.asdict(hit)) == {"start", "end", "entity_type", "score", "detector_id", "coordinate_space",
                                         "source_ranges", "mapping_quality"}
    assert "abc" not in repr(hit) + repr(instance)
    assert raw not in vars(instance).values()
    assert not instance.capabilities.supports_context
    assert not instance.capabilities.network_access
    assert instance.capabilities.model_identities == ()


@pytest.mark.parametrize("start,end", [(-1, 3), (2, 99), (3, 3), (True, 4), (2.0, 4), (3, 2)])
def test_bad_ranges_fail_value_free(monkeypatch, start, end):
    fake_sdk(monkeypatch, [finding(start, end)])
    with pytest.raises(ValueError) as exc:
        scan(detector(), "😀 abc")
    assert "abc" not in str(exc.value)
    assert exc.value.__suppress_context__


def test_inconsistent_range_and_type_fail(monkeypatch):
    raw = finding()
    raw.byte_range.end = 6
    fake_sdk(monkeypatch, [raw])
    with pytest.raises(ValueError):
        scan(detector(), "😀 abc")
    raw = finding()
    raw.entity_type = "EMAIL"
    fake_sdk(monkeypatch, [raw])
    with pytest.raises(ValueError):
        scan(detector(), "😀 abc")


@pytest.mark.parametrize("entities", [("EMAIL",), (), "API_KEY", (42,)])
def test_unknown_or_empty_selection_rejected(entities):
    with pytest.raises(ValueError):
        detector(entities=entities)


def test_missing_sdk_and_scanner_failure_hide_values(monkeypatch):
    original = builtins.__import__
    def blocked(name, *args, **kwargs):
        if name == "datafog_core":
            raise ImportError("sensitive literal")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(DetectorUnavailable) as exc:
        detector()
    assert "sensitive" not in str(exc.value)
    assert exc.value.__suppress_context__
    monkeypatch.setattr(builtins, "__import__", original)
    def broken(*args):
        raise RuntimeError("sensitive literal")
    monkeypatch.setitem(sys.modules, "datafog_core", SimpleNamespace(scan=broken))
    with pytest.raises(DetectorUnavailable) as exc:
        scan(detector(), "sensitive literal")
    assert "sensitive" not in str(exc.value)
    assert exc.value.__suppress_context__


@pytest.fixture
def native(monkeypatch):
    if importlib.util.find_spec("datafog_core") is None:
        if os.environ.get("SCT_OPTIONAL_PROFILE") == "secrets":
            pytest.fail("Required secrets SDK is missing")
        pytest.skip("secrets extra is not installed")
    import datafog_core
    def forbidden(*args, **kwargs):
        pytest.fail("Rule scanner attempted model or network work")
    import socket
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(datafog_core, "ModelSession", forbidden)
    return detector()


def samples():
    import base64
    import json
    jwt = ".".join(base64.urlsafe_b64encode(json.dumps(x).encode()).decode().rstrip("=")
                   for x in [{"alg": "HS256", "typ": "JWT"}, {"sub": "synthetic"}]) + "." + "Ab12" * 10
    return [("API_KEY", "ghp_" + "Ab12" * 9, ""),
            ("BEARER_TOKEN", "abcdefghijklmnopqrstuvwxyz123456", "Authorization: Bearer "),
            ("JWT", jwt, ""),
            ("PRIVATE_KEY", "-----BEGIN PRIVATE KEY-----\n" + "a" * 64 + "\n-----END PRIVATE KEY-----", "")]


def test_e2_acceptance(native):
    for entity, value, label in samples():
        prefix = "😀 é 漢字\n" + label
        text = prefix + value + "\n😀é\n" + label + value
        hits = scan(native, text)
        assert [(h.entity_type, h.start, h.end) for h in hits] == [
            (entity, len(prefix), len(prefix) + len(value)),
            (entity, len(prefix) + len(value) + 4 + len(label), len(text))]
        assert all(text[h.start:h.end] == value for h in hits)
        assert value not in repr(hits)


def test_native_near_miss_cohort(native):
    texts = ["api_key = 'example'", "Authorization: Bearer short", "const jwt = 'a.b.c'",
             "-----BEGIN PRIVATE KEY-----", "const value = 'ordinary_code_literal'", "const count = 42"]
    assert [len(scan(native, text)) for text in texts] == [0, 1, 0, 0, 0, 0]


@pytest.mark.parametrize("score", [float("nan"), float("inf"), True, -0.1, 1.1, "private"])
def test_bad_confidence_is_rejected(monkeypatch, score):
    raw = finding()
    raw.confidence = score
    fake_sdk(monkeypatch, [raw])
    with pytest.raises(ValueError):
        scan(detector(), "😀 abc")


def test_incomplete_or_wrong_coordinate_units_rejected(monkeypatch):
    for raw in [SimpleNamespace(), finding(3, 6), finding()]:
        if hasattr(raw, "byte_range"):
            raw.byte_range.start = 4  # A UTF-16 position cannot stand in for a UTF-8 byte boundary.
        fake_sdk(monkeypatch, [raw])
        with pytest.raises(ValueError):
            scan(detector(), "😀 abc")


def test_score_is_plain_float_and_no_raw_properties_are_read(monkeypatch):
    class Raw:
        entity_type = "API_KEY"
        codepoint_range = SimpleNamespace(start=2, end=5)
        byte_range = SimpleNamespace(start=5, end=8)
        confidence = 1
        @property
        def matched_text(self):
            pytest.fail("Adapter read a raw credential")
    fake_sdk(monkeypatch, [Raw()])
    hit, = scan(detector(), "😀 abc")
    assert type(hit.score) is float
    assert hit.score == 1.0


@pytest.mark.parametrize("mode", ["placeholder", "synthetic", "reversible"])
def test_native_registry_policy_and_marker_collision(native, mode):
    import json
    from sct import TextCleaner, TextCleanerConfig
    from sct.detectors import DetectorRegistry, DetectorSpec
    from sct.policy_v2 import HandlingRule, UnifiedPrivacyPolicy
    registry = DetectorRegistry()
    registry.register("credentials", lambda cfg, options: detector(**options))
    value = samples()[0][1]
    cfg = TextCleanerConfig(
        pii_processing="unified", replacement_mode=mode,
        detector_specs=(DetectorSpec("credentials", {"entities": ["API_KEY"]}),),
        check_detect_language=False, check_ner_process=False, check_fix_bad_unicode=False,
        check_replace_html=False, check_remove_isolated_letters=False,
        check_remove_isolated_special_symbols=False, check_normalize_whitespace=False,
    )
    cleaner = TextCleaner(cfg, detector_registry=registry)
    text = "<API_KEY_0> " + value
    result = cleaner.process(text)
    assert value not in result.lm_text
    assert value not in json.dumps(result.metadata["findings_v2"])
    if mode == "reversible":
        assert "<API_KEY_1>" in result.lm_text
        assert result.metadata["anon_map"].deanonymize(result.lm_text) == text
    registry2 = DetectorRegistry()
    registry2.register("credentials", lambda cfg, options: detector(**options))
    redact = TextCleaner(cfg, detector_registry=registry2, unified_policy=UnifiedPrivacyPolicy(
        "credential-policy", "1", {"API_KEY": HandlingRule(action="redact")}))
    assert value not in redact.process(value).lm_text


def test_unicode_inside_credential_formats_is_not_invented(native):
    for entity, value, label in samples():
        midpoint = len(value) // 2
        altered = value[:midpoint] + "é漢😀" + value[midpoint:]
        assert scan(native, label + altered) == ()


def test_native_inline_bearer_limit(native):
    text = "😀 é Authorization: Bearer abcdefghijklmnopqrstuvwxyz123456"
    assert scan(native, text) == ()


def test_native_binding_units(native):
    import datafog_core
    value = samples()[0][1]
    text = "😀é漢\n" + value
    raw, = datafog_core.scan(text, {"entities": ["API_KEY"]})
    assert (raw.codepoint_range.start, raw.codepoint_range.end) == (4, 44)
    assert (raw.byte_range.start, raw.byte_range.end) == (10, 50)
    assert not hasattr(raw, "utf16_range")
    assert text[raw.codepoint_range.start:raw.codepoint_range.end] == value
