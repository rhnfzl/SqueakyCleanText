"""ONNX model cache: quantized directories, revisions, and loading without a Hub request."""

import json
import sys
import types
from pathlib import Path

import pytest

from tests.conftest import TEST_NER_MODELS, TEST_NER_REVISIONS


# --- the quantized cache is keyed by the resolved commit, not the branch name ---
def test_quantized_cache_follows_the_moving_branch(monkeypatch, tmp_path):
    import sct.utils.onnx_pipeline as op

    current = {"commit": "a" * 40}

    def fake_download(repo, filename, **kwargs):
        if filename == "model.onnx_data":
            raise FileNotFoundError(filename)
        path = tmp_path / "snapshots" / current["commit"] / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        if filename == "config.json":
            path.write_text(json.dumps({"id2label": {"0": "O"}}))
        elif filename == "tokenizer_config.json":
            path.write_text("{}")
        else:
            path.write_text("weights-" + current["commit"][0])
        return str(path)

    class Session:
        def __init__(self, path, **kwargs):
            self.path = path

        def get_providers(self):
            return ["CPUExecutionProvider"]

    quant = types.ModuleType("onnxruntime.quantization")
    quant.QuantType = types.SimpleNamespace(QInt8=1)
    quant.quantize_dynamic = lambda src, dst, weight_type=None: Path(dst).write_text(
        "quantized-from:" + Path(src).read_text())
    monkeypatch.setitem(sys.modules, "onnxruntime.quantization", quant)
    monkeypatch.setattr(op, "hf_hub_download", fake_download)
    monkeypatch.setattr(op, "Tokenizer", types.SimpleNamespace(
        from_file=lambda p: types.SimpleNamespace(token_to_id=lambda t: 0)))
    monkeypatch.setattr(op, "TokenizerWrapper", lambda tok, model_max_length: None)
    monkeypatch.setattr(op.ort, "InferenceSession", Session)
    monkeypatch.setattr(op, "ONNXNERPipeline", lambda s, w, i, pad_id: types.SimpleNamespace(session=s))

    for commit in ("a" * 40, "b" * 40):
        current["commit"] = commit
        pipe, _ = op.load_onnx_ner_model("org/m", revision="main", quantize=True,
                                         cache_dir=str(tmp_path / "qc"))
        assert Path(pipe.session.path).read_text() == "quantized-from:weights-" + commit[0]


def test_quantize_without_the_onnx_package_names_the_extra(monkeypatch, tmp_path):
    import sct.utils.onnx_pipeline as op

    def fake_download(repo, filename, **kwargs):
        if filename == "model.onnx_data":
            raise FileNotFoundError(filename)
        path = tmp_path / filename
        path.write_text(json.dumps({"id2label": {"0": "O"}}) if filename == "config.json" else "{}")
        return str(path)

    monkeypatch.setitem(sys.modules, "onnxruntime.quantization", None)  # import raises ImportError
    monkeypatch.setattr(op, "hf_hub_download", fake_download)
    monkeypatch.setattr(op, "Tokenizer", types.SimpleNamespace(from_file=lambda p: None))
    monkeypatch.setattr(op, "TokenizerWrapper", lambda tok, model_max_length: None)
    with pytest.raises(ImportError, match=r"squeakycleantext\[quantize\]"):
        op.load_onnx_ner_model("org/m", revision="main", quantize=True)


# --- a worker rebuilding the cleaner makes no Hub request from a warm cache ---
def test_onnx_load_from_warm_cache_makes_no_hub_request(monkeypatch):
    import huggingface_hub.file_download as fd
    from sct.utils.onnx_pipeline import load_onnx_ner_model

    model = TEST_NER_MODELS["ENGLISH"]
    # warms the cache, records the missing model.onnx_data
    pipe, _ = load_onnx_ner_model(model, revision=TEST_NER_REVISIONS["ENGLISH"])
    calls = []
    monkeypatch.setattr(fd, "get_hf_file_metadata", lambda *a, **k: calls.append(a) or 1 / 0)
    load_onnx_ner_model(model, revision=pipe.revision)
    assert calls == []


def test_quantized_cache_dirs_do_not_collide():
    from sct.utils.onnx_pipeline import _quantized_dir

    assert _quantized_dir("org/model", None) == "org__model"  # 0.7.1 layout kept
    assert _quantized_dir("org/model", "abc123").startswith("org__model@")
    assert _quantized_dir("o/m", "release/v1") != _quantized_dir("o/m", "release__v1")
    assert _quantized_dir("a/b__c", None) != _quantized_dir("a__b/c", None)
    assert _quantized_dir("a__b", None) != _quantized_dir("a/b", None)


def test_revision_cache_dirs_differ_ignoring_case():
    from sct.utils.onnx_pipeline import _quantized_dir

    assert _quantized_dir("o/m", "Release").lower() != _quantized_dir("o/m", "release").lower()
    assert _quantized_dir("o/m", None) == "o__m"  # 0.7.1 layout kept
