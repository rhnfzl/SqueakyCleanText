"""Hugging Face datasets adapter."""

import os
import pickle
import subprocess
import sys
from types import ModuleType

import pytest

datasets = pytest.importorskip("datasets")

from sct.integrations.hf import SCTMap, clean_dataset  # noqa: E402
from sct.detectors import DetectorSpec  # noqa: E402
from sct.presets import preset_config  # noqa: E402
from tests.conftest import TEST_NER_MODELS, TEST_NER_REVISIONS  # noqa: E402

FAST = preset_config("training", check_ner_process=False)


@pytest.fixture
def custom_registry_module(monkeypatch):
    module = ModuleType("sct_hf_cache_fixture")
    exec("""
from sct.detectors import DetectorCapabilities, DetectorRegistry
from sct.findings import Detection

class Provider:
    capabilities = DetectorCapabilities('local', entity_types=('PERSON',))
    def detect(self, text, *, language, context):
        return ()

class ProtectedProvider(Provider):
    def detect(self, text, *, language, context):
        return (Detection('PERSON', 0, len(text), 'local', .9, 'canonical'),)

def registry_factory(cfg):
    registry = DetectorRegistry()
    registry.register('local', lambda cfg, options: Provider())
    return registry
""", vars(module))  # noqa: S102 - fixed application fixture, no untrusted code
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module


@pytest.mark.parametrize("change", ["factory", "provider"])
@pytest.mark.parametrize("fixed_cache_file", [False, True])
@pytest.mark.parametrize("fingerprint_options", [{}, {"new_fingerprint": None}])
def test_custom_registry_updates_bypass_disk_and_downstream_caches(
        tmp_path, custom_registry_module, change, fixed_cache_file, fingerprint_options):
    module = custom_registry_module
    source_path = tmp_path / "source"
    datasets.Dataset.from_dict({"text": ["Alice"]}).save_to_disk(source_path)
    source = datasets.load_from_disk(source_path)
    cfg = preset_config("training", pii_processing="unified", check_ner_process=False,
                        detector_specs=(DetectorSpec("local"),))
    options = {"load_from_cache_file": True, **fingerprint_options}
    if fixed_cache_file:
        options["cache_file_name"] = str(tmp_path / "clean.arrow")
    before = clean_dataset(source, config=cfg, registry_factory=module.registry_factory, **options)
    assert before["text"] == ["Alice"] and before.cache_files
    def copy_text(row):
        return {"copy": row["text"]}

    downstream_before = before.map(copy_text)
    assert downstream_before["copy"] == ["Alice"] and downstream_before.cache_files

    if change == "factory":
        exec("""
def registry_factory(cfg):
    registry = DetectorRegistry()
    registry.register('local', lambda cfg, options: ProtectedProvider())
    return registry
""", vars(module))  # noqa: S102 - fixed same-name application update
    else:
        module.Provider = module.ProtectedProvider
    after = clean_dataset(source, config=cfg, registry_factory=module.registry_factory, **options)
    assert after["text"] == ["<PERSON>"]
    assert after._fingerprint != before._fingerprint
    downstream_after = after.map(copy_text)
    assert downstream_after["copy"] == ["<PERSON>"]
    assert downstream_after._fingerprint != downstream_before._fingerprint


def test_explicit_custom_registry_fingerprint_controls_cache_reuse(tmp_path, custom_registry_module, monkeypatch):
    module = custom_registry_module
    source_path = tmp_path / "source"
    datasets.Dataset.from_dict({"text": ["Alice"]}).save_to_disk(source_path)
    source = datasets.load_from_disk(source_path)
    cfg = preset_config("training", pii_processing="unified", check_ner_process=False,
                        detector_specs=(DetectorSpec("local"),))
    before = clean_dataset(source, config=cfg, registry_factory=module.registry_factory,
                           new_fingerprint="caller_identity_v1")
    assert before["text"] == ["Alice"] and before.cache_files
    module.Provider = module.ProtectedProvider
    after = clean_dataset(source, config=cfg, registry_factory=module.registry_factory,
                          new_fingerprint="caller_identity_v2")
    assert after["text"] == ["<PERSON>"] and after._fingerprint == "caller_identity_v2"

    def unexpected_execution(*args, **kwargs):
        pytest.fail("An explicit unchanged caller identity must retain cache reuse")

    monkeypatch.setattr(SCTMap, "__call__", unexpected_execution)
    cached = clean_dataset(source, config=cfg, registry_factory=module.registry_factory,
                           new_fingerprint="caller_identity_v2")
    assert cached["text"] == ["<PERSON>"] and cached.cache_files == after.cache_files


def test_builtin_registry_keeps_disk_cache_reuse(tmp_path, monkeypatch):
    source_path = tmp_path / "source"
    datasets.Dataset.from_dict({"text": ["alice@example.com"]}).save_to_disk(source_path)
    source = datasets.load_from_disk(source_path)
    before = clean_dataset(source, config=FAST)
    assert before["text"] == ["<EMAIL>"] and before.cache_files

    def unexpected_execution(*args, **kwargs):
        pytest.fail("Unchanged built-in configuration must reuse its disk cache")

    monkeypatch.setattr(SCTMap, "__call__", unexpected_execution)
    after = clean_dataset(source, config=FAST)
    assert after["text"] == ["<EMAIL>"]
    assert after._fingerprint == before._fingerprint
    assert after.cache_files == before.cache_files


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
    cfg = preset_config("training", ner_models=TEST_NER_MODELS, ner_model_revisions=TEST_NER_REVISIONS, language="ENGLISH")
    ds = datasets.Dataset.from_dict({"text": ["Angela Merkel visited Paris."]})
    assert "Merkel" not in clean_dataset(ds, config=cfg)["text"][0]


def test_map_overwrites_existing_clean_column():
    # A stale input `text_clean` with raw text must be replaced, not kept.
    out = SCTMap(output="new_column", config=FAST)({"text": ["mail a@b.com"], "text_clean": ["mail a@b.com"]})
    assert out == {"text_clean": ["mail <EMAIL>"]}
