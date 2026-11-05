"""GPU device choice (ner_device incl. Apple mps, cuda_device_id, ONNX CUDA options) and model provenance."""

import pickle
import re

import onnxruntime as ort
import pytest

from sct import TextCleaner, config
from sct.config import TextCleanerConfig
from sct.utils import ner as ner_module
from sct.utils.ner import GeneralNER
from tests.conftest import TEST_NER_MODELS, TEST_NER_REVISIONS
import sys
import types

CPU = 'CPUExecutionProvider'
CUDA = 'CUDAExecutionProvider'
TEXT = "John Smith visited the office in London yesterday afternoon."


def _cfg(**overrides):
    return TextCleanerConfig(
        ner_models=TEST_NER_MODELS,
        ner_model_revisions=TEST_NER_REVISIONS, language="en", check_statistical_model_processing=False,
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
    ner = GeneralNER(TextCleanerConfig(ner_device="cuda", cuda_device_id=2, ner_backend="torch",
                                       torch_ner_models={"ENGLISH": "a", "MULTILINGUAL": "b"}))
    assert seen == ["cuda:2", "cuda:2"]
    assert [m["provider"] for m in ner.models_for("ENGLISH")] == ["cuda:2", "cuda:2"]


def test_onnx_loader_gets_device_id_and_options(monkeypatch):
    seen = []

    class Tok:
        max_len_single_sentence = 510

    def fake_load(model_name, **kwargs):
        seen.append((kwargs["device"], kwargs["cuda_device_id"], kwargs["cuda_provider_options"]))
        return object(), Tok()

    monkeypatch.setattr("sct.utils.onnx_pipeline.load_onnx_ner_model", fake_load)
    GeneralNER(TextCleanerConfig(ner_device="cuda", cuda_device_id=3, onnx_cuda_provider_options={"a": 1},
                                 ner_models={"ENGLISH": "a", "MULTILINGUAL": "b"}))
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


# --- Apple GPU ('mps'): torch and GLiNER only; ONNX stays on CPU ---

def _fake_torch_loader(monkeypatch):
    seen = []

    class Tok:
        max_len_single_sentence = 510

    def fake_load(model_name, device='cpu', **kwargs):
        seen.append(device)
        return object(), Tok()

    monkeypatch.setattr("sct.utils.torch_pipeline.load_torch_ner_model", fake_load)
    return seen


@pytest.mark.parametrize("available, device", [(True, "mps"), (False, "cpu")])
def test_mps_moves_torch_or_falls_back_to_cpu(monkeypatch, available, device):
    monkeypatch.setattr(ner_module, "_mps_available", lambda: available)
    seen = _fake_torch_loader(monkeypatch)
    ner = GeneralNER(TextCleanerConfig(ner_device="mps", ner_backend="torch",
                                       torch_ner_models={"ENGLISH": "a", "MULTILINGUAL": "b"}))
    assert seen == [device, device]
    assert [m["provider"] for m in ner.models_for("ENGLISH")] == [device, device]


def test_mps_keeps_onnx_on_cpu(spy_sessions, monkeypatch):
    monkeypatch.setattr(ner_module, "_mps_available", lambda: True)
    result = TextCleaner(cfg=_cfg(ner_device="mps")).process(TEXT)
    assert all(providers == [CPU] for providers in spy_sessions["requested"])
    assert {m["provider"] for m in result.metadata["provenance"]["models"]} == {CPU}
    assert "mps_fallback" not in result.metadata["warnings"]


def test_mps_fallback_warning(monkeypatch):
    monkeypatch.setattr(ner_module, "_mps_available", lambda: False)
    cleaner = TextCleaner(cfg=_cfg(ner_device="mps"))
    cleaner.GeneralNER.provenance["ENGLISH"]["provider"] = "cpu"  # a torch model that fell back
    assert "mps_fallback" in cleaner.process(TEXT).metadata["warnings"]
    assert TextCleanerConfig(check_ner_process=False, ner_device="mps").ner_device == "mps"


@pytest.mark.skipif(
    __import__("platform").system() != "Darwin" or __import__("platform").machine() != "arm64",
    reason="Apple silicon only",
)
def test_real_torch_model_on_mps_matches_cpu():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    if not torch.backends.mps.is_available():
        pytest.skip("torch has no MPS backend")
    from tests.conftest import TEST_TORCH_NER_MODELS
    outs = {}
    for device in ("cpu", "mps"):
        cleaner = TextCleaner(cfg=TextCleanerConfig(
            ner_backend="torch", torch_ner_models=TEST_TORCH_NER_MODELS, language="en",
            check_statistical_model_processing=False, ner_device=device,
        ))
        result = cleaner.process(TEXT)
        outs[device] = result.text if hasattr(result, "text") else result[0]
        assert {m["provider"] for m in result.metadata["provenance"]["models"]} == {device}
        assert "mps_fallback" not in result.metadata["warnings"]
    assert outs["mps"] == outs["cpu"]


# --- GLiNER device follows the effective backend; Presidio gets the device ---
class _Model:
    def __init__(self):
        self.moved = None

    def to(self, device):
        self.moved = device
        return self


def _fake_gliner(monkeypatch, has_onnx=False):
    class GLiNER:
        @staticmethod
        def from_pretrained(model_id, load_onnx_model=False, **kwargs):
            if load_onnx_model and not has_onnx:
                raise FileNotFoundError("no model.onnx")
            return _Model()

    class GLiNER2:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            return _Model()

    monkeypatch.setitem(sys.modules, "gliner", types.SimpleNamespace(GLiNER=GLiNER))
    monkeypatch.setitem(sys.modules, "gliner2", types.SimpleNamespace(GLiNER2=GLiNER2))


def test_gliner_onnx_fallback_and_gliner2_move_to_cuda(monkeypatch):
    _fake_gliner(monkeypatch)
    from sct.utils.gliner_adapter import GLiNERAdapter

    fallback = GLiNERAdapter("m", device="cuda:2", onnx=True)
    assert fallback._onnx is False and fallback.model.moved == "cuda:2"
    assert GLiNERAdapter("m", variant="gliner2", device="cuda:2", onnx=True).model.moved == "cuda:2"


def _fake_presidio(monkeypatch, seen):
    class Recognizer:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    class Engine:
        registry = types.SimpleNamespace(add_recognizer=lambda r: None)

    monkeypatch.setitem(sys.modules, "presidio_analyzer", types.SimpleNamespace(AnalyzerEngine=Engine))
    monkeypatch.setitem(sys.modules, "presidio_analyzer.predefined_recognizers",
                        types.SimpleNamespace(GLiNERRecognizer=Recognizer))


def test_presidio_gliner_gets_device_and_torch_provenance(monkeypatch):
    seen = {}
    _fake_presidio(monkeypatch, seen)

    ner = GeneralNER(TextCleanerConfig(ner_backend="presidio_gliner", ner_device="cuda", cuda_device_id=2,
                                       gliner_model="m"))
    assert seen["map_location"] == "cuda:2"
    assert ner.provenance["GLINER"]["provider"] == "cuda:2"


# --- ner_device=None: the ONNX provider list says CUDA, torch decides what torch components get ---
def _torch_sees(monkeypatch, gpus):
    cuda = types.SimpleNamespace(is_available=lambda: gpus > 0, device_count=lambda: gpus)
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=cuda))
    monkeypatch.setattr(ort, "get_available_providers", lambda: [CUDA, CPU])


def _fake_onnx_loader(monkeypatch):
    seen = []

    class Tok:
        max_len_single_sentence = 510

    def fake_load(model_name, **kwargs):
        seen.append(kwargs["device"])
        return types.SimpleNamespace(provider=CPU), Tok()

    monkeypatch.setattr("sct.utils.onnx_pipeline.load_onnx_ner_model", fake_load)
    return seen


@pytest.mark.parametrize("gpus, device_id, torch_device", [(0, 0, "cpu"), (1, 1, "cpu"), (2, 1, "cuda:1")])
def test_auto_gliner_follows_torch_not_the_provider_list(monkeypatch, gpus, device_id, torch_device):
    _torch_sees(monkeypatch, gpus)
    _fake_gliner(monkeypatch)
    ner = GeneralNER(TextCleanerConfig(ner_backend="gliner", gliner_model="m", cuda_device_id=device_id))
    assert ner._gliner_pipe.model.moved == (None if torch_device == "cpu" else torch_device)
    assert ner.provenance["GLINER"]["provider"] == torch_device


@pytest.mark.parametrize("gpus, map_location", [(0, "cpu"), (1, "cuda:0")])
def test_auto_presidio_follows_torch_not_the_provider_list(monkeypatch, gpus, map_location):
    _torch_sees(monkeypatch, gpus)
    seen = {}
    _fake_presidio(monkeypatch, seen)
    ner = GeneralNER(TextCleanerConfig(ner_backend="presidio_gliner", gliner_model="m"))
    assert seen["map_location"] == map_location
    assert ner.provenance["GLINER"]["provider"] == map_location


def _cpu_onnx_runtime_with_a_gpu(monkeypatch):
    _torch_sees(monkeypatch, 1)
    monkeypatch.setattr(ort, "get_available_providers", lambda: [CPU])


def test_auto_gliner_uses_the_gpu_with_a_cpu_onnx_runtime(monkeypatch):
    _cpu_onnx_runtime_with_a_gpu(monkeypatch)
    _fake_gliner(monkeypatch)
    ner = GeneralNER(TextCleanerConfig(ner_backend="gliner", gliner_model="m"))
    assert ner._gliner_pipe.model.moved == "cuda:0"
    assert ner.provenance["GLINER"]["provider"] == "cuda:0"


def test_auto_presidio_uses_the_gpu_with_a_cpu_onnx_runtime(monkeypatch):
    _cpu_onnx_runtime_with_a_gpu(monkeypatch)
    seen = {}
    _fake_presidio(monkeypatch, seen)
    GeneralNER(TextCleanerConfig(ner_backend="presidio_gliner", gliner_model="m"))
    assert seen["map_location"] == "cuda:0"


def test_auto_ensemble_onnx_splits_devices_with_a_cpu_onnx_runtime(monkeypatch):
    _cpu_onnx_runtime_with_a_gpu(monkeypatch)
    _fake_gliner(monkeypatch)
    seen = _fake_onnx_loader(monkeypatch)
    ner = GeneralNER(TextCleanerConfig(ner_backend="ensemble_onnx", gliner_model="m",
                                       ner_models={"ENGLISH": "a", "MULTILINGUAL": "b"}))
    assert seen == ["cpu", "cpu"]
    assert ner._gliner_pipe.model.moved == "cuda:0"
    assert ner.provenance["GLINER"]["provider"] == "cuda:0"


def test_auto_gliner_in_onnx_mode_stays_off_the_gpu_with_a_cpu_onnx_runtime(monkeypatch):
    _cpu_onnx_runtime_with_a_gpu(monkeypatch)
    _fake_gliner(monkeypatch, has_onnx=True)
    ner = GeneralNER(TextCleanerConfig(ner_mode="pii"))
    assert ner._torch_device == "cpu"
    assert ner.provenance["GLINER"]["provider"] == "onnx"


def test_auto_ensemble_onnx_keeps_trying_cuda_for_onnx_only(monkeypatch):
    _torch_sees(monkeypatch, 0)
    _fake_gliner(monkeypatch)
    seen = _fake_onnx_loader(monkeypatch)
    ner = GeneralNER(TextCleanerConfig(ner_backend="ensemble_onnx", gliner_model="m",
                                       ner_models={"ENGLISH": "a", "MULTILINGUAL": "b"}))
    assert seen == ["cuda", "cuda"]
    assert ner._gliner_pipe.model.moved is None
    assert ner.provenance["GLINER"]["provider"] == "cpu"


def test_auto_ensemble_torch_stays_on_cpu_without_a_gpu(monkeypatch, caplog):
    _torch_sees(monkeypatch, 0)
    _fake_gliner(monkeypatch)
    seen = _fake_torch_loader(monkeypatch)
    ner = GeneralNER(TextCleanerConfig(ner_backend="ensemble_torch", gliner_model="m",
                                       torch_ner_models={"ENGLISH": "a", "MULTILINGUAL": "b"}))
    assert seen == ["cpu", "cpu"]
    assert ner._gliner_pipe.model.moved is None
    # Torch backends just run on CPU: no warning, so the docs only promise provider "cpu" for them.
    assert ner.device == "cpu" and "Torch models run on CPU" not in caplog.text


def test_auto_torch_backend_skips_a_gpu_index_torch_cannot_see(monkeypatch):
    _torch_sees(monkeypatch, 1)
    seen = _fake_torch_loader(monkeypatch)
    GeneralNER(TextCleanerConfig(ner_backend="torch", cuda_device_id=1,
                                 torch_ner_models={"ENGLISH": "a", "MULTILINGUAL": "b"}))
    assert seen == ["cpu", "cpu"]


def test_auto_gliner_on_cpu_sets_cuda_fallback(monkeypatch, caplog):
    _torch_sees(monkeypatch, 0)
    _fake_gliner(monkeypatch)
    monkeypatch.setattr(_Model, "predict_entities", lambda self, *a, **k: [], raising=False)
    result = TextCleaner(cfg=TextCleanerConfig(
        ner_backend="gliner", gliner_model="m", language="en", check_statistical_model_processing=False,
    )).process(TEXT)
    assert "cuda_fallback" in result.metadata["warnings"]
    assert "Torch models run on CPU" in caplog.text


def test_auto_gliner_in_onnx_mode_does_not_warn_about_torch(monkeypatch, caplog):
    _torch_sees(monkeypatch, 0)
    _fake_gliner(monkeypatch, has_onnx=True)
    ner = GeneralNER(TextCleanerConfig(ner_mode="pii"))
    assert ner._gliner_pipe._onnx is True
    assert ner.provenance["GLINER"]["provider"] == "onnx"
    assert "Torch models run on CPU" not in caplog.text


def test_auto_gliner_warns_when_missing_onnx_weights_force_torch(monkeypatch, caplog):
    _torch_sees(monkeypatch, 0)
    _fake_gliner(monkeypatch)
    ner = GeneralNER(TextCleanerConfig(ner_mode="pii"))
    assert ner._gliner_pipe._onnx is False
    assert ner.provenance["GLINER"]["provider"] == "cpu"
    assert "Torch models run on CPU" in caplog.text


def test_explicit_cuda_is_not_downgraded_when_torch_sees_no_gpu(monkeypatch):
    _torch_sees(monkeypatch, 0)
    _fake_gliner(monkeypatch)
    ner = GeneralNER(TextCleanerConfig(ner_backend="gliner", gliner_model="m", ner_device="cuda"))
    assert ner._gliner_pipe.model.moved == "cuda:0"


def test_auto_plain_onnx_never_asks_torch(monkeypatch):
    monkeypatch.setattr(ort, "get_available_providers", lambda: [CUDA, CPU])
    monkeypatch.setattr(ner_module, "_torch_cuda_usable", lambda _: pytest.fail("torch was consulted"))
    seen = _fake_onnx_loader(monkeypatch)
    GeneralNER(TextCleanerConfig(ner_models={"ENGLISH": "a", "MULTILINGUAL": "b"}))
    assert seen == ["cuda", "cuda"]


def test_gliclass_stays_on_cpu_unless_cuda_is_explicit(monkeypatch):
    seen = []
    monkeypatch.setattr("sct.sct.GLiClassAdapter", lambda **kwargs: seen.append(kwargs["device"]))
    for device, expected in ((None, "cpu"), ("cuda", "cuda:1")):
        cfg = TextCleanerConfig(check_ner_process=False, ner_device=device, cuda_device_id=1)
        TextCleaner._build_gliclass(types.SimpleNamespace(cfg=cfg))
        assert seen.pop() == expected
