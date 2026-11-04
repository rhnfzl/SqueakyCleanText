"""Codex 0.8.0 full review, round 16: ids 1-3."""

import datetime

import pytest

from tests.ner_helpers import bare_ner, set_cfg
from sct import TextCleaner, TextCleanerConfig
from sct.browser import build_browser_manifest
from sct.privacy import EntityRule, PrivacyPolicy
from sct.streaming import WindowedStreamingRedactor
from sct.utils.ner import _presidio_type_map
from sct.utils.synthetic import SyntheticReplacer

# --- 1: native-digit phones are not split between stream windows ---


@pytest.mark.parametrize("phone", ["٠٦ ١٢ ٣٤ ٥٦ ٧٨", "०६ १२ ३४ ५६ ७८"])
def test_stream_keeps_native_digit_phone_whole(phone):
    cfg = TextCleanerConfig.from_preset("training", check_ner_process=False)
    cleaner = TextCleaner(cfg)
    text = f"aaaa aaaa aaaa Call {phone} tail"
    stream = WindowedStreamingRedactor(cleaner, max_buffer_chars=32, overlap_chars=16)
    out = stream.feed(text) + stream.finish()
    assert out == cleaner.process(text).lm_text == "aaaa aaaa aaaa Call <PHONE> tail"


def test_browser_phone_and_ip_rules_fold_digits():
    rules = build_browser_manifest(TextCleanerConfig())["rules"]
    assert rules["phone"]["fold_decimal_digits"] and rules["ip"]["fold_decimal_digits"]


# --- 2: an empty Presidio type map keeps analyzer names ---


class _Result:
    def __init__(self, entity_type, start, end):
        self.entity_type, self.start, self.end, self.score = entity_type, start, end, 0.9


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
    ner._privacy_policy = policy
    result = ner.ner_process_detailed("Anna 0612345678 now", positional_tags=("PER",))
    assert "0612345678" not in result.text
    assert [f.entity.entity_type for f in result.findings] == ["PHONE_NUMBER"]


# --- 3: seeded date fakes do not depend on the clock ---


def _date_fakes(day, monkeypatch, reference_date):
    import faker.providers.date_time as faker_dt

    class _Clock(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime.datetime.combine(day, datetime.time(12), tzinfo=tz)

        @classmethod
        def today(cls):
            return cls.now()

    monkeypatch.setattr(faker_dt, "datetime", _Clock)
    replacer = SyntheticReplacer(seed=42, locale="en_US", reference_date=reference_date)
    replacer.reset_cache("DOB 1990-01-01")
    return (replacer.get_replacement("DATE_OF_BIRTH", "1990-01-01"),
            replacer.get_replacement("DATE", "x"))


def test_date_fakes_same_on_any_day(monkeypatch):
    days = (datetime.date(2026, 10, 31), datetime.date(2026, 11, 1))
    ref = TextCleanerConfig().synthetic_reference_date
    assert _date_fakes(days[0], monkeypatch, ref) == _date_fakes(days[1], monkeypatch, ref)
    # None keeps the clock (earlier 0.8.0)
    assert _date_fakes(days[0], monkeypatch, None) != _date_fakes(days[1], monkeypatch, None)


def test_bad_reference_date_rejected():
    with pytest.raises(ValueError):
        TextCleanerConfig(synthetic_reference_date="2026-13-01")
