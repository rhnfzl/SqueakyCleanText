"""GPU device choice (ner_device, cuda_device_id, ONNX CUDA options) and model provenance."""

import pickle
import re

import onnxruntime as ort
import pytest

from sct import TextCleaner, config
from sct.config import TextCleanerConfig
from sct.utils import ner as ner_module
from sct.utils.ner import GeneralNER
from tests.conftest import TEST_NER_MODELS

CPU = 'CPUExecutionProvider'
CUDA = 'CUDAExecutionProvider'
TEXT = "John Smith visited the office in London yesterday afternoon."


def _cfg(**overrides):
    return TextCleanerConfig(
        ner_models=TEST_NER_MODELS, language="en", check_statistical_model_processing=False,
        **overrides,
    )


@pytest.fixture
def spy_sessions(monkeypatch):
    """Pretend CUDA exists; record the providers each session asks for; report *reported*."""
    real = ort.InferenceSession
    state = {"requested": [], "reported": None}

    class Spy:
        def __init__(self, path, sess_options=None, providers=None):
            state["requested"].append(providers)
            self._session = real(path, sess_options=sess_options, providers=[CPU])

        def __getattr__(self, name):
            return getattr(self._session, name)

        def get_providers(self):
            return state["reported"] or self._session.get_providers()

    monkeypatch.setattr(ort, "get_available_providers", lambda: [CUDA, CPU])
    monkeypatch.setattr(ort, "InferenceSession", Spy)
    return state


def test_cuda_device_id_and_options_reach_the_onnx_session(spy_sessions):
    spy_sessions["reported"] = [CUDA, CPU]
    result = TextCleaner(cfg=_cfg(
        ner_device="cuda", cuda_device_id=1, onnx_cuda_provider_options={"gpu_mem_limit": 123},
    )).process(TEXT)
    assert spy_sessions["requested"]
    for providers in spy_sessions["requested"]:
        assert providers == [(CUDA, {'device_id': 1, 'gpu_mem_limit': 123}), CPU]
    assert {m["provider"] for m in result.metadata["provenance"]["models"]} == {CUDA}
    assert "cuda_fallback" not in result.metadata["warnings"]


def test_cpu_device_never_asks_for_cuda(spy_sessions):
    TextCleaner(cfg=_cfg(ner_device="cpu")).process(TEXT)
    assert spy_sessions["requested"]
    assert all(providers == [CPU] for providers in spy_sessions["requested"])


def test_cuda_fallback_warning_when_session_runs_on_cpu(spy_sessions):
    spy_sessions["reported"] = [CPU]
    result = TextCleaner(cfg=_cfg(ner_device="cuda")).process(TEXT)
    assert "cuda_fallback" in result.metadata["warnings"]
    assert {m["provider"] for m in result.metadata["provenance"]["models"]} == {CPU}


def test_provenance_lists_the_english_ensemble():
    result = TextCleaner(cfg=_cfg(ner_device="cpu")).process(TEXT)
    provenance = result.metadata["provenance"]
    assert provenance["backend"] == "onnx"
    # ENGLISH and MULTILINGUAL share one loaded model; each key keeps its own entry.
    assert [m["key"] for m in provenance["models"]] == ["ENGLISH", "MULTILINGUAL"]
    for m in provenance["models"]:
        assert m["id"] == TEST_NER_MODELS["ENGLISH"]
        assert m["provider"] == CPU
        assert re.fullmatch(r"[0-9a-f]{40}", m["revision"])
    assert "detector" not in result.metadata


def test_provenance_empty_when_ner_off():
    result = TextCleaner(cfg=TextCleanerConfig(check_ner_process=False)).process(TEXT)
    assert result.metadata["provenance"] == {"backend": None, "models": []}


def test_torch_backend_gets_indexed_cuda_device(monkeypatch):
    seen = []

    class Tok:
        max_len_single_sentence = 510

    def fake_load(model_name, device='cpu', **kwargs):
        seen.append(device)
        return object(), Tok()

    monkeypatch.setattr("sct.utils.torch_pipeline.load_torch_ner_model", fake_load)
    ner = GeneralNER(device="cuda", cuda_device_id=2, ner_backend="torch",
                     torch_model_names={"ENGLISH": "a", "MULTILINGUAL": "b"})
    assert seen == ["cuda:2", "cuda:2"]
    assert [m["provider"] for m in ner.models_for("ENGLISH")] == ["cuda:2", "cuda:2"]


def test_onnx_loader_gets_device_id_and_options(monkeypatch):
    seen = []

    class Tok:
        max_len_single_sentence = 510

    def fake_load(model_name, **kwargs):
        seen.append((kwargs["device"], kwargs["cuda_device_id"], kwargs["cuda_provider_options"]))
        return object(), Tok()

    monkeypatch.setattr(ner_module, "load_onnx_ner_model", fake_load)
    GeneralNER(device="cuda", cuda_device_id=3, onnx_cuda_provider_options={"a": 1},
               model_names={"ENGLISH": "a", "MULTILINGUAL": "b"})
    assert seen == [("cuda", 3, {"a": 1})] * 2


@pytest.mark.parametrize("overrides, match", [
    ({"ner_device": "gpu"}, "ner_device"),
    ({"cuda_device_id": -1}, "cuda_device_id"),
])
def test_device_config_validation(overrides, match):
    with pytest.raises(ValueError, match=match):
        TextCleanerConfig(check_ner_process=False, **overrides)


def test_device_config_defaults_globals_and_pickle(monkeypatch):
    cfg = TextCleanerConfig(check_ner_process=False)
    assert (cfg.ner_device, cfg.cuda_device_id, cfg.onnx_cuda_provider_options) == (None, 0, None)
    monkeypatch.setattr(config, "NER_DEVICE", "cpu")
    monkeypatch.setattr(config, "CUDA_DEVICE_ID", 1)
    monkeypatch.setattr(config, "ONNX_CUDA_PROVIDER_OPTIONS", {"gpu_mem_limit": 8})
    legacy = config._config_from_module_globals()
    assert (legacy.ner_device, legacy.cuda_device_id) == ("cpu", 1)
    frozen = TextCleanerConfig(check_ner_process=False, onnx_cuda_provider_options={"x": 1})
    with pytest.raises(TypeError):
        frozen.onnx_cuda_provider_options["x"] = 2
    assert pickle.loads(pickle.dumps(frozen)) == frozen  # noqa: S301 - trusted round trip
    assert dict(legacy.onnx_cuda_provider_options) == {"gpu_mem_limit": 8}
