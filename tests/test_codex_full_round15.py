"""Codex 0.8.0 full review, round 15: ids 1-9."""

import ast
import dataclasses
import os
import pickle
import re
from pathlib import Path

import pytest

from tests.ner_helpers import bare_ner
from sct import TextCleaner, TextCleanerConfig
from sct.privacy import EntityRule, PrivacyPolicy
from sct.utils import synthetic
from tests.conftest import TEST_NER_MODELS

MODES = ["placeholder", "synthetic", "reversible"]


def _training(text, mode, **kw):
    cfg = TextCleanerConfig.from_preset("training", check_ner_process=False, replacement_mode=mode, **kw)
    return TextCleaner(cfg).process(text).lm_text


# --- 1: combining marks in an email local part ---

@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("fold", [True, False])
@pytest.mark.parametrize("local", ["প্রিয়া", "பிரியா", "संपर्क"])
def test_email_local_part_with_combining_marks(local, fold, mode):
    out = _training(f"mail {local}@example.com now", mode, fold_styled_characters=fold)
    assert local[:2] not in out and "example.com" not in out


@pytest.mark.parametrize("fold", [True, False])
def test_fullwidth_at_after_combining_mark(fold):
    assert _training("प्रिया＠example.com", "placeholder", fold_styled_characters=fold) == "<EMAIL>"


# --- 2: supplementary-plane, punycode and IPv4 hosts ---

@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("text", [
    "info@𠮷野.jp", "x@example.xn--p1ai",
    "https://𠮷野.jp/?email=alice%40example.com",
    "https://example.xn--p1ai/?email=alice%40example.com",
    "http://10.0.0.0/?email=alice%40example.com",
    "http://10.0.0.255/?email=alice%40example.com",
])
def test_whole_host_is_matched(text, mode):
    out = _training(f"see {text} now", mode)
    assert "alice" not in out and "p1ai" not in out and "𠮷" not in out and "10.0" not in out
    assert out.startswith("see ") and out.endswith(" now")


def test_url_ipv4_host_followed_by_label_is_host_name():
    assert _training("http://10.0.0.1.example.com/x", "placeholder") == "<URL>"


# --- 3 / 4: native-digit phones and IPs ---

@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("text", ["٠٦ ١٢ ٣٤ ٥٦ ٧٨", "٩٨٧٦٥ ٤٣٢١٠", "٠٩٨٧٦٥-٤٣٢١٠", "٠١٠-٦٥٥٢ ١٢٣٤",
                                  "०६ १२ ३४ ५६ ७८", "١٩٢．١٦٨．١．١", "1.2.3.4١"])
def test_native_digit_phone_and_ip_fully_replaced(text, mode):
    out = _training(f"رقم {text} هنا", mode)
    assert not any(c.isdigit() and not c.isascii() for c in out) and "1.2.3.4" not in out, out


def test_native_digit_ip_public_only_check():
    out = _training("ip ١٩٢．١٦٨．١．١ and ٨．٨．٨．٨", "placeholder", replace_ips_public_only=True)
    # ． between digits of any script folds like between ASCII digits (fuzz round 1); the private IP stays
    assert out == "ip ١٩٢.١٦٨.١.١ and <IP>"


# --- 5: dataclasses.replace re-derives the PII label map ---

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


# --- 6: Presidio entity types follow the configured names ---

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
    ner._ner_backend = "presidio_gliner"
    ner._analyzer = Analyzer()
    ner._presidio_types = _presidio_type_map({"legacy_type_names": legacy, "label_map": {"name": "PER"}})
    policy = PrivacyPolicy(name="p", version="1", rules={phone: EntityRule(action="redact")},
                           default_rule=EntityRule(action="keep"))
    result = ner.ner_process_detailed("Anna 0612345678 now", positional_tags=("PER",), privacy_policy=policy)
    assert "0612345678" not in result.text
    assert {f.entity.entity_type for f in result.findings} == {phone, "PERSON"}  # PER alias -> canonical (round 17)


# --- 7: a display tag cleanup removes still carries its value ---

def test_bracket_tag_survives_bracket_removal_in_reversible():
    text = "Mail ann@example.com today."
    cleaner = TextCleaner(TextCleanerConfig(language="ENGLISH", replacement_mode="reversible",
                                            ner_models=TEST_NER_MODELS, replace_with_email="[EMAIL]"))
    result = cleaner.process(text)
    assert "<EMAIL_0>" in result.lm_text
    assert "ann@example.com" in result.metadata["anon_map"].deanonymize(result.lm_text)


# --- 8: a forked child gets a free random lock ---

@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs fork")
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_forked_child_gets_unlocked_random_lock():
    with synthetic._GLOBAL_RANDOM_LOCK:
        pid = os.fork()
        if pid == 0:  # pragma: no cover - child
            os._exit(1 if synthetic._GLOBAL_RANDOM_LOCK.locked() else 0)
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0


# --- 9: llms.txt lists the current defaults ---

def test_llms_txt_defaults_match_config():
    defaults = {f.name: f.default for f in dataclasses.fields(TextCleanerConfig)}
    listed = {}
    for line in (Path(__file__).parent.parent / "llms.txt").read_text().splitlines():
        m = re.match(r"^(\w+)=(.+?)\s+#", line)
        if m and m.group(1) in defaults:
            listed[m.group(1)] = ast.literal_eval(m.group(2))
    assert {"check_replace_dates", "language_confidence_warning", "language_margin_warning"} <= set(listed)
    assert {k: v for k, v in listed.items() if v != defaults[k]} == {}
