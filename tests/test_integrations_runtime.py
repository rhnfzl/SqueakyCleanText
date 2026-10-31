"""Framework-free tests of the adapter runtime (CleanerHandle, clean_columns, clean_records)."""

import pickle

import pytest

from sct.integrations import _runtime
from sct.integrations._runtime import CleanerHandle, clean_columns, clean_records
from sct.presets import preset_config
from sct.utils.process_result import ProcessResult

EMAIL = "mail john@example.com now"
FAST = preset_config("training", check_ner_process=False)


def _fake_view(result):
    md = result.metadata or {}
    types = {
        t: c["count"]
        for t, c in {**md.get("regex_counts", {}), **md.get("entity_counts", {})}.items()
        if t not in ("NUMBER", "YEAR", "DATE", "HTML", "CURRENCY")
    }
    return {
        "pii_count": sum(types.values()),
        "pii_types": dict(sorted(types.items())),
        "sensitive": bool(md.get("regex_counts") or md.get("entity_counts")),
    }


@pytest.fixture
def signals_view(monkeypatch):
    """Use the real ProcessResult.signals (K1) when present, else a value-free stand-in."""
    if not hasattr(ProcessResult, "signals"):
        monkeypatch.setattr(ProcessResult, "signals", property(_fake_view), raising=False)


def test_default_config_is_training_preset():
    assert CleanerHandle().config == preset_config("training")
    assert CleanerHandle("sct").config == preset_config("sct")


def test_reversible_and_bad_config_rejected():
    with pytest.raises(ValueError, match="reversible"):
        CleanerHandle(preset_config("training", replacement_mode="reversible"))
    with pytest.raises(TypeError):
        CleanerHandle(42)
    with pytest.raises(ValueError, match="unknown preset"):
        CleanerHandle("nope")


def test_get_builds_once_and_rebuilds_on_pid_change(monkeypatch):
    handle = CleanerHandle(FAST)
    first = handle.get()
    assert handle.get() is first
    monkeypatch.setattr(_runtime.os, "getpid", lambda: -1)
    assert handle.get() is not first


def test_pickle_drops_cleaner():
    handle = CleanerHandle(FAST)
    handle.get()
    blob = pickle.dumps(handle)
    assert blob == pickle.dumps(CleanerHandle(FAST))
    clone = pickle.loads(blob)  # noqa: S301 - trusted round trip
    assert clone._cleaner is None and clone.config == FAST


def test_two_handles_never_share_a_cleaner():
    a = CleanerHandle(FAST)
    b = CleanerHandle(preset_config("training", check_ner_process=False, check_replace_emails=False))
    batch = {"text": [EMAIL]}
    assert clean_columns(a, batch)["text"] == ["mail <EMAIL> now"]
    assert clean_columns(b, batch)["text"] == [EMAIL]
    assert a.get() is not b.get()


def test_clean_columns_overwrite_new_column_and_passthrough():
    handle = CleanerHandle(FAST)
    batch = {"text": [EMAIL, None, ""], "title": [EMAIL, "x", None], "id": [1, 2, 3]}
    out = clean_columns(handle, batch)
    assert out["text"] == ["mail <EMAIL> now", None, ""]
    assert out["title"] == batch["title"] and out["id"] == [1, 2, 3]
    new = clean_columns(handle, batch, ("text", "title"), "new_column")
    assert new["text"] == batch["text"]
    assert new["text_clean"] == out["text"]
    assert new["title_clean"] == ["mail <EMAIL> now", "x", None]


def test_clean_columns_validation():
    handle = CleanerHandle(FAST)
    with pytest.raises(KeyError):
        clean_columns(handle, {"text": []}, ("body",))
    with pytest.raises(ValueError):
        clean_columns(handle, {"text": []}, output="append")
    with pytest.raises(ValueError):
        clean_columns(handle, {"text": []}, ())


@pytest.mark.skipif(not hasattr(ProcessResult, "signals"),
                    reason="needs ProcessResult.signals (K1)")
def test_clean_columns_score_adds_score_a_columns():
    score = pytest.importorskip("sct.score")
    out = clean_columns(CleanerHandle(FAST), {"text": [EMAIL, None]}, score=True)
    for col in score.SCORE_A_COLUMNS:
        assert len(out[col]) == 2 and out[col][1] is None
    assert out["sct_pii_count"][0] >= 1


def test_clean_records_text_and_metadata(signals_view):
    handle = CleanerHandle(FAST)
    records = [(EMAIL, {"author": EMAIL, "n": 1}), ("plain", {})]
    out = clean_records(handle, records, "body", ("body", "author"), "overwrite")
    assert out[0][0] == "mail <EMAIL> now"
    assert out[0][1]["author"] == "mail <EMAIL> now" and out[0][1]["n"] == 1
    assert out[0][1]["sct"]["pii_count"] == 1
    assert "author" not in out[1][1] and out[1][0] == "plain"
    assert records[0][1]["author"] == EMAIL  # input metadata not modified
    new = clean_records(handle, records, "body", ("body", "author"), "new_column")
    assert new[0][0] == EMAIL
    assert new[0][1]["body_clean"] == "mail <EMAIL> now"
    assert new[0][1]["author_clean"] == "mail <EMAIL> now" and new[0][1]["author"] == EMAIL
