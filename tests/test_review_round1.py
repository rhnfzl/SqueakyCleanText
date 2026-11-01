"""Regressions for the 0.8.0 review round 1: nested spans, neighbour-only
detections, per-model window sizing, fixed-batch ONNX models, synthetic cache
across stages, quantized cache keys, POSIX-only benchmark import."""

import builtins
import importlib
import sys
from types import SimpleNamespace

import pytest

from sct.utils.anonymization_map import AnonymizationMap
from sct.utils.synthetic import SyntheticReplacer
from tests.test_chunking import _FakeRunDetector, _word_ner, _word_tokenizer, _words

TEXT = "John Paul Smith. " + "filler " * 25 + "email=a@b.com"
NESTED = [
    {"entity_group": "PER", "score": 0.99, "word": "John Paul Smith", "start": 0, "end": 15},
    {"entity_group": "PER", "score": 0.95, "word": "Paul", "start": 5, "end": 9},
    {"entity_group": "EMAIL", "score": 0.99, "word": "a@b.com",
     "start": len(TEXT) - 7, "end": len(TEXT)},
]


@pytest.mark.parametrize("mode", ["placeholder", "reversible", "synthetic"])
def test_nested_spans_leave_no_part_of_the_entity(mode):
    ner = _word_ner(0, _FakeRunDetector(), mode=mode)
    ner._synthetic_replacer = SyntheticReplacer()
    anon_map = AnonymizationMap()
    result = ner.anonymize_text(TEXT, [dict(d) for d in NESTED], mode, anon_map=anon_map)
    assert "Smith" not in result.text and "Paul" not in result.text
    assert "a@b.com" not in result.text
    if mode == "placeholder":
        assert result.text.startswith("<PERSON>. ")
    if mode == "reversible":
        assert anon_map.deanonymize(result.text) == TEXT


class _ContextDetector(_FakeRunDetector):
    """Finds "John Smith" only when "has arrived" follows it in the same window."""

    def __call__(self, texts, batch_size=1):
        return [
            [{"entity_group": "PER", "score": 0.99, "word": "John Smith",
              "start": i, "end": i + 10}]
            if (i := text.find("John Smith has arrived")) != -1 else []
            for text in texts
        ]


def test_overlap_keeps_a_detection_only_a_neighbour_window_made():
    text = _words(40, {7: "John", 8: "Smith", 9: "has", 10: "arrived"})
    no_overlap = _word_ner(0, _ContextDetector()).ner_process_detailed(
        text, positional_tags=("PER",), ner_confidence_threshold=0.5)
    assert "John Smith" not in no_overlap.text
    result = _word_ner(2, _ContextDetector()).ner_process_detailed(
        text, positional_tags=("PER",), ner_confidence_threshold=0.5)
    assert result.text == text.replace("John Smith", "<PERSON>")
    assert len(result.findings) == 1


class _Tok:
    """Word tokenizer counting *per_word* tokens per word, like a heavier vocabulary."""

    max_len_single_sentence = 1000

    def __init__(self, per_word):
        self.per_word = per_word

    def __call__(self, text, **kwargs):
        spans = _word_tokenizer(text).input_ids
        return SimpleNamespace(input_ids=spans * self.per_word, offset_mapping=spans)


@pytest.mark.parametrize("overlap", [0, 2])
def test_windows_fit_every_model_tokenizer(overlap):
    ner = _word_ner(overlap, _FakeRunDetector(), budget=12)
    heavy = _Tok(2)
    ner._tokenizers = {"ENGLISH": _Tok(1), "MULTILINGUAL": heavy}
    windows = ner._plan_windows(_words(60, {}))
    assert all(ner._token_count(w[0], heavy) <= 12 for w in windows)


def test_window_budget_honours_a_smaller_model_limit():
    ner = _word_ner(0, _FakeRunDetector(), budget=12)
    small = _Tok(1)
    small.max_len_single_sentence = 5
    ner._tokenizers = {"ENGLISH": _Tok(1), "DUTCH": small}
    assert ner._chunk_budget() == 5


def test_fixed_batch_one_model_runs_one_row_per_call():
    from sct.utils.onnx_pipeline import ONNXNERPipeline, TokenizerWrapper
    from tokenizers import Tokenizer, models, pre_tokenizers

    tok = Tokenizer(models.WordLevel({"[UNK]": 0, "John": 1, "Smith": 2}, unk_token="[UNK]"))  # noqa: S106
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    rows = []

    class Session:
        def get_inputs(self):
            return [SimpleNamespace(name=n, shape=[1, "sequence"])
                    for n in ("input_ids", "attention_mask")]

        def run(self, _, feeds):
            rows.append(feeds["input_ids"].shape[0])
            assert feeds["input_ids"].shape[0] == 1, "fixed batch dimension"
            import numpy as np
            return [np.zeros((1, feeds["input_ids"].shape[1], 2), dtype=np.float32)]

    pipe = ONNXNERPipeline(Session(), TokenizerWrapper(tok, 512), {0: "O", 1: "B-PER"})
    assert pipe(["John", "Smith", "John Smith"], batch_size=8) == [[], [], []]
    assert rows == [1, 1, 1]


def test_synthetic_value_shared_between_regex_and_ner_stages(monkeypatch):
    from tests.test_batching import _cleaner

    text = "Call 2025550147 or _2025550147_."
    outputs = []
    for size in (1, 8):
        cleaner = _cleaner(ner_batch_size=size, replacement_mode="synthetic")

        def detect(chunks, *args, **kwargs):
            return [
                [{"entity_group": "PHONE_NUMBER", "score": 0.99, "word": "2025550147",
                  "start": i, "end": i + 10, "key": f"{i}:{i + 10}"}]
                if (i := chunk.find("2025550147")) != -1 else []
                for chunk in chunks
            ]

        monkeypatch.setattr(cleaner.GeneralNER, "_detect", detect)
        outputs.append(cleaner.process(text).lm_text)
    assert "2025550147" not in outputs[1]
    assert outputs[1] == outputs[0]


def test_quantized_cache_dirs_do_not_collide():
    from sct.utils.onnx_pipeline import _quantized_dir

    assert _quantized_dir("org/model", None) == "org__model"  # 0.7.1 layout kept
    assert _quantized_dir("org/model", "abc123").startswith("org__model@")
    assert _quantized_dir("o/m", "release/v1") != _quantized_dir("o/m", "release__v1")
    assert _quantized_dir("a/b__c", None) != _quantized_dir("a__b/c", None)
    assert _quantized_dir("a__b", None) != _quantized_dir("a/b", None)


def test_benchmark_imports_without_resource_module(monkeypatch):
    real_import = builtins.__import__

    def no_resource(name, *args, **kwargs):
        if name == "resource":
            raise ImportError("No module named 'resource'")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "scripts.benchmark_release", raising=False)
    monkeypatch.setattr(builtins, "__import__", no_resource)
    module = importlib.import_module("scripts.benchmark_release")
    assert module._peak_rss_mb() is None
