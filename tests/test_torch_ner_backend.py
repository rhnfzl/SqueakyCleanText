"""TorchNERPipeline behaviour, driven through fake torch and transformers modules."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from sct.utils.torch_pipeline import TorchNERPipeline, load_torch_ner_model


@pytest.fixture
def backend(monkeypatch):
    torch = MagicMock(name="torch")
    transformers = MagicMock(name="transformers")
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    return SimpleNamespace(
        torch=torch,
        tokenizer_loader=transformers.AutoTokenizer.from_pretrained,
        model_loader=transformers.AutoModelForTokenClassification.from_pretrained,
        hf_pipeline=transformers.pipeline,
        hf_run=transformers.pipeline.return_value,
    )


def test_missing_torch_names_the_install_extra(backend, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)

    with pytest.raises(ImportError, match=r"squeakycleantext\[torch\]"):
        TorchNERPipeline("m")


@pytest.mark.parametrize(
    ("cache_dir", "revision", "expected"),
    [
        ("/cache", "abc123", {"cache_dir": "/cache", "revision": "abc123"}),
        (None, None, {}),
    ],
    ids=["both_set", "both_none"],
)
def test_cache_dir_and_revision_reach_both_loaders(backend, cache_dir, revision, expected):
    TorchNERPipeline("m", cache_dir=cache_dir, revision=revision)

    backend.tokenizer_loader.assert_called_once_with("m", **expected)
    backend.model_loader.assert_called_once_with("m", **expected)


def test_model_moves_to_the_device_before_the_pipeline_uses_it(backend):
    TorchNERPipeline("m", device="cuda:1")

    moved = backend.model_loader.return_value.to
    moved.assert_called_once_with("cuda:1")
    assert backend.hf_pipeline.call_args.kwargs["model"] is moved.return_value


def test_single_string_is_sent_as_a_batch_with_batch_size(backend):
    backend.hf_run.return_value = [[]]

    TorchNERPipeline("m")("x", batch_size=4)

    backend.hf_run.assert_called_once_with(["x"], batch_size=4)


def test_flat_result_is_wrapped_per_text(backend):
    entity = {"entity_group": "PER", "word": "Ann"}
    backend.hf_run.return_value = [entity]

    assert TorchNERPipeline("m")("x") == [[entity]]


def test_batched_result_stays_as_returned(backend):
    batched = [[{"entity_group": "PER", "word": "Ann"}], []]
    backend.hf_run.return_value = batched

    assert TorchNERPipeline("m")(["x", "y"]) == batched


def test_empty_input_returns_the_empty_result(backend):
    backend.hf_run.return_value = []

    assert TorchNERPipeline("m")([]) == []


@pytest.mark.parametrize(
    ("device", "expected_calls"),
    [("cuda:0", 1), ("cpu", 0)],
)
def test_cleanup_empties_the_cache_only_on_cuda(backend, device, expected_calls):
    TorchNERPipeline("m", device=device).cleanup()

    assert backend.torch.cuda.empty_cache.call_count == expected_calls


def test_load_returns_the_pipeline_and_its_tokenizer(backend):
    pipe, tokenizer = load_torch_ner_model(
        "m", device="cuda:0", cache_dir="/cache", revision="abc123"
    )

    assert isinstance(pipe, TorchNERPipeline)
    assert tokenizer is pipe._tokenizer
    assert tokenizer is backend.tokenizer_loader.return_value
    backend.tokenizer_loader.assert_called_once_with("m", cache_dir="/cache", revision="abc123")
    backend.model_loader.return_value.to.assert_called_once_with("cuda:0")
