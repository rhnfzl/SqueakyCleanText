"""Synthetic fakes do not depend on global random state, hash seed, today's date or timezone."""

import ast
import datetime
import os
import random
import subprocess
import sys
import threading

import pytest

from sct import TextCleanerConfig
from sct.utils.synthetic import SyntheticReplacer


# --- locale fakes do not depend on global random or the hash seed ---
@pytest.mark.parametrize("locale, group", [("th_TH", "SOCIAL_SECURITY_NUMBER"), ("uk_UA", "TAX_ID"),
                                           ("uk_UA", "LICENSE_PLATE")])
def test_locale_fake_ignores_global_random(locale, group):
    import random
    values = set()
    for i in range(3):
        random.seed(i)
        replacer = SyntheticReplacer(seed=42, locale=locale)
        replacer.reset_cache("same doc")
        values.add(replacer.get_replacement(group, "x"))
    assert len(values) == 1


def test_italian_city_independent_of_hash_seed():
    code = ("from sct.utils.synthetic import SyntheticReplacer\n"
            "r = SyntheticReplacer(seed=42, locale='it_IT'); r.reset_cache('same doc')\n"
            "print(r.get_replacement('CITY', 'Rome'))")
    out = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,  # noqa: S603
                          env={**os.environ, "PYTHONHASHSEED": seed}).stdout
           for seed in ("1", "42")}
    assert len(out) == 1


# --- seeded date fakes do not depend on the clock ---
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


# --- native SSN / TAX_ID fakes follow the reference date, not the clock ---
@pytest.mark.parametrize("locale", ["zh_CN", "et_EE", "fi_FI", "lv_LV", "no_NO", "sv_SE", "pl_PL", "sr_BA"])
@pytest.mark.parametrize("entity", ["SOCIAL_SECURITY_NUMBER", "TAX_ID"])
def test_identity_fakes_follow_reference_date(locale, entity):
    def fake(reference_date):
        replacer = SyntheticReplacer(seed=42, locale=locale, reference_date=reference_date)
        replacer.reset_cache("same document")
        return replacer.get_replacement(entity, "x")

    assert fake("1990-06-01") == fake("1990-06-01") != fake("2026-01-01")


def test_reference_clock_restored():
    import faker.providers.date_time as faker_dt
    import faker.providers.ssn.zh_CN as zh

    SyntheticReplacer(seed=42, locale="zh_CN", reference_date="1990-06-01").get_replacement("TAX_ID", "x")
    assert zh.datetime is datetime and faker_dt.datetime is datetime.datetime
    assert faker_dt.dtdate is datetime.date


# --- a seeded locale fake does not use the shared global random, so other threads cannot shift it ---
@pytest.mark.parametrize("locale, group", [("th_TH", "SOCIAL_SECURITY_NUMBER"), ("uk_UA", "TAX_ID"),
                                           ("uk_UA", "LICENSE_PLATE")])
def test_locale_fake_ignores_concurrent_global_random(locale, group):
    replacer = SyntheticReplacer(locale=locale, reference_date="2026-01-01")

    def fake():
        replacer.reset_cache("same document")
        return replacer.get_replacement(group, "x")

    stop = threading.Event()
    noise = threading.Thread(target=lambda: [random.random() for _ in iter(stop.is_set, True)])  # noqa: S311
    interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    noise.start()
    try:
        values = {fake() for _ in range(300)}
    finally:
        stop.set()
        noise.join()
        sys.setswitchinterval(interval)
    assert values == {fake()}


# --- birth-date fakes do not depend on the worker's timezone ---
_DOB = """
from sct.utils.synthetic import SyntheticReplacer
r = SyntheticReplacer(seed=42, locale="en_US", reference_date="2026-01-01")
out = []
for i in range(8):
    r.reset_cache(f"Patient {i} was born on 1990-01-01.")
    out.append(r.get_replacement("DATE_OF_BIRTH", "1990-01-01"))
print(out)
"""


def test_date_of_birth_fakes_same_in_any_timezone():
    def run(tz):
        env = {**os.environ, "TZ": tz}
        return subprocess.run([sys.executable, "-c", _DOB], env=env, capture_output=True, text=True,  # noqa: S603
                              check=True).stdout
    outputs = {run(tz) for tz in ("UTC", "America/Los_Angeles", "Asia/Tokyo")}
    assert len(outputs) == 1
    assert all(len(value) == 10 for value in ast.literal_eval(outputs.pop()))  # dates, not datetimes
