"""A forked child gets free locks, a restored clock and the parent's pinned model commits."""

import os
import sys
import threading
from datetime import datetime
from types import SimpleNamespace

import pytest

from sct import TextCleanerConfig
from sct.utils import resources, synthetic


# --- a forked child does not inherit the detector lock held ---
@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")
def test_forked_child_gets_free_detector_lock():
    with resources._detector_lock:  # another thread is building a detector at fork time
        pid = os.fork()
        if pid == 0:
            os._exit(0 if resources._detector_lock.acquire(timeout=2) else 1)
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0


# --- a forked child gets a free random lock ---
@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs fork")
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_forked_child_gets_unlocked_random_lock():
    with synthetic._GLOBAL_RANDOM_LOCK:
        pid = os.fork()
        if pid == 0:  # pragma: no cover - child
            os._exit(1 if synthetic._GLOBAL_RANDOM_LOCK.locked() else 0)
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0


# --- a child forked while another thread holds the Faker clock gets the real clock back ---
@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs fork")
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_forked_child_restores_held_clock():
    import faker.providers.date_time
    import faker.providers.ssn.zh_CN as ssn_zh

    def tax_id():
        replacer = synthetic.SyntheticReplacer(seed=42, locale="zh_CN", reference_date="2026-01-01")
        replacer.reset_cache("same document")
        return replacer.get_replacement("TAX_ID", "x")

    expected = tax_id()
    inside, release = threading.Event(), threading.Event()

    def hold():
        with synthetic._clock_at(datetime(1990, 6, 1), {ssn_zh, sys.modules[faker.providers.date_time.__name__]}):
            inside.set()
            release.wait()

    thread = threading.Thread(target=hold)
    thread.start()
    inside.wait()
    try:
        pid = os.fork()
        if pid == 0:  # pragma: no cover - child
            os._exit(0 if tax_id() == expected else 1)
    finally:
        release.set()
        thread.join()
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0
    assert tax_id() == expected


def test_forked_handle_pins_the_parent_commits(monkeypatch):
    import pickle

    from sct.integrations import _runtime as rt

    commit = "parent-commit"
    monkeypatch.setattr(rt, "TextCleaner", lambda cfg: SimpleNamespace(
        cfg=cfg, GeneralNER=SimpleNamespace(provenance={"ENGLISH": {"revision": commit}})))
    handle = rt.CleanerHandle(TextCleanerConfig(ner_models={"ENGLISH": "org/en", "MULTILINGUAL": "org/multi"}))
    assert "ENGLISH" not in handle.get().cfg.ner_model_revisions
    copied = pickle.loads(pickle.dumps(handle))  # noqa: S301 - own object
    assert copied.config.ner_model_revisions["ENGLISH"] == commit
    monkeypatch.setattr(rt.os, "getpid", lambda: -1)  # as in a forked child
    assert handle.get().cfg.ner_model_revisions["ENGLISH"] == commit
