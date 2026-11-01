"""Hugging Face datasets adapter."""

import os
import pickle
import subprocess
import sys

import pytest

datasets = pytest.importorskip("datasets")

from sct.integrations.hf import SCTMap, clean_dataset  # noqa: E402
from sct.presets import preset_config  # noqa: E402
from tests.conftest import TEST_NER_MODELS  # noqa: E402

FAST = preset_config("training", check_ner_process=False)


def test_clean_dataset_overwrite_and_new_column():
    ds = datasets.Dataset.from_dict({"text": ["mail john@example.com", None], "id": [1, 2]})
    out = clean_dataset(ds, config=FAST)
    assert out["text"] == ["mail <EMAIL>", None] and out["id"] == [1, 2]
    new = clean_dataset(ds, config=FAST, output="new_column")
    assert new["text"] == ds["text"] and new["text_clean"] == out["text"]


def test_clean_iterable_dataset():
    ds = datasets.Dataset.from_dict({"text": ["call 192.168.1.1 now"]}).to_iterable_dataset()
    assert [r["text"] for r in clean_dataset(ds, config=FAST)] == ["call <IP> now"]


def test_map_pickles_config_only():
    fn = SCTMap(config=FAST)
    blob = pickle.dumps(fn)
    fn({"text": ["x"]})
    assert pickle.dumps(fn) == blob
    assert pickle.loads(blob)({"text": ["a@b.co"]})["text"] == ["<EMAIL>"]  # noqa: S301 - trusted round trip


_FINGERPRINT = (
    "import datasets; from sct.integrations.hf import clean_dataset; from sct.presets import preset_config; "
    "ds = datasets.Dataset.from_dict({'text': ['a']}); "
    "print(clean_dataset(ds, config=preset_config('training', check_ner_process=False))._fingerprint)"
)


def test_fingerprint_stable_across_processes():
    # Old code: datasets hashed the config's frozensets in per-process order, so the cache never hit.
    prints = {
        subprocess.run(  # noqa: S603 - fixed argv, own interpreter
            [sys.executable, "-c", _FINGERPRINT], capture_output=True, text=True, check=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
        ).stdout
        for seed in ("1", "2", "3")
    }
    assert len(prints) == 1
    ds = datasets.Dataset.from_dict({"text": ["a"]})
    assert clean_dataset(ds, config=FAST)._fingerprint != clean_dataset(ds, config="sct")._fingerprint


def test_clean_dataset_with_test_ner_model():
    cfg = preset_config("training", ner_models=TEST_NER_MODELS, language="ENGLISH")
    ds = datasets.Dataset.from_dict({"text": ["Angela Merkel visited Paris."]})
    assert "Merkel" not in clean_dataset(ds, config=cfg)["text"][0]


def test_map_overwrites_existing_clean_column():
    # A stale input `text_clean` with raw text must be replaced, not kept.
    out = SCTMap(output="new_column", config=FAST)({"text": ["mail a@b.com"], "text_clean": ["mail a@b.com"]})
    assert out == {"text_clean": ["mail <EMAIL>"]}
