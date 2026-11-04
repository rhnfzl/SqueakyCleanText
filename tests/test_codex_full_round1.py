"""Codex 0.8.0 full review, round 1: ids 1, 2, 4, 5, 7."""

import re
import sys
import types

import pytest

from sct import TextCleaner, TextCleanerConfig
from sct.presets import preset_config
from tests.conftest import TEST_NER_MODELS
from tests.helpers import FakeRunDetector, word_ner

FAST = preset_config("training", check_ner_process=False)
RAW = "mail alice@example.com"


# --- 1: marker-shaped source text is not protected like SCT's own placeholders ---

def test_source_marker_text_gets_no_placeholder_protection():
    detector = FakeRunDetector()
    detector.pattern = re.compile(r"[A-Z]{2,}")
    ner = word_ner(0, detector)
    text = "call <PHONE>, live in <LONDON>"
    # Old code dropped the LONDON span too, as if SCT had written "<LONDON>".
    result = ner.ner_process_detailed(
        text, positional_tags=("PER",), ner_confidence_threshold=0.5,
        source_markers=frozenset({"<LONDON>"}))
    assert result.text == "call <PHONE>, live in <<PERSON>>"


@pytest.fixture(scope="module")
def ner_cleaner():
    return TextCleaner(cfg=preset_config("training", ner_models=TEST_NER_MODELS, ner_batch_size=8))


def test_source_marker_pii_is_masked_single_and_batch(ner_cleaner):
    text = "I live in <LONDON>."
    assert "LONDON" not in ner_cleaner.process(text).lm_text
    assert all("LONDON" not in r.lm_text for r in ner_cleaner.process_batch([text, text]))


# --- 2: a str column name is one column, not its characters ---

@pytest.mark.parametrize("module, cls, column", [
    ("langchain_core", "langchain", "page_content"),
    ("haystack", "haystack", "content"),
    ("datatrove", "datatrove", "text"),
    ("datasets", "hf", "text"),
])
def test_adapters_take_a_single_column_name(module, cls, column):
    pytest.importorskip(module)
    mod = __import__(f"sct.integrations.{cls}", fromlist=["x"])
    adapter = {"langchain": "SCTDocumentTransformer", "haystack": "SCTDocumentCleaner",
               "datatrove": "SCTCleaner", "hf": "SCTMap"}[cls]
    assert getattr(mod, adapter)(columns=column, config=FAST).columns == (column,)


def test_langchain_str_column_cleans():
    pytest.importorskip("langchain_core")
    from langchain_core.documents import Document

    from sct.integrations.langchain import SCTDocumentTransformer

    out = SCTDocumentTransformer(columns="page_content", config=FAST).transform_documents(
        [Document(page_content=RAW)])
    assert out[0].page_content == "mail <EMAIL>"


# --- 4: grouped national phone numbers are whole matches without number replacement ---

@pytest.mark.parametrize("text", [
    "Call 699 123 456 tomorrow.", "Call 912 345 678 tomorrow.", "Call 010-6552 1234 tomorrow.",
    "Ring 020 7946 0958 now.", "Call 06 12 34 56 78 tomorrow.",
])
def test_training_masks_grouped_national_phones(text):
    out = TextCleaner(cfg=FAST).process(text).lm_text
    assert "<PHONE>" in out and not re.search(r"\d", out), out


@pytest.mark.parametrize("text", [
    "On 01-02-2023 we met.", "In 2019 we had 42 bugs.", "Order 123 456 shipped.", "Paid 1 234 567 890 won.",
])
def test_training_leaves_short_digit_groups(text):
    assert "<PHONE>" not in TextCleaner(cfg=FAST).process(text).lm_text


# --- 5: Haystack pipelines keep the cleaner config through to_dict/from_dict ---

def test_haystack_pipeline_round_trip_keeps_config():
    pytest.importorskip("haystack")
    from haystack import Document, Pipeline

    from sct.integrations.haystack import SCTDocumentCleaner

    cfg = TextCleanerConfig(check_ner_process=False, replace_with_email="<MAIL>",
                            positional_tags=("PER",))
    pipe = Pipeline()
    pipe.add_component("clean", SCTDocumentCleaner(columns="content", config=cfg))
    restored = Pipeline.loads(pipe.dumps(), allowed_modules=["sct.integrations.haystack"])
    comp = restored.get_component("clean")
    assert comp.handle.config.check_ner_process is False
    assert comp.handle.config.positional_tags == ("PER",)
    out = restored.run({"clean": {"documents": [Document(content=RAW)]}})
    assert out["clean"]["documents"][0].content == "mail <MAIL>"
    named = Pipeline()
    named.add_component("clean", SCTDocumentCleaner(config="training"))
    assert named.to_dict()["components"]["clean"]["init_parameters"]["config"] == "training"


# --- 7: GLiNER device follows the effective backend; Presidio gets the device ---

class _Model:
    def __init__(self):
        self.moved = None

    def to(self, device):
        self.moved = device
        return self


def _fake_gliner(monkeypatch):
    class GLiNER:
        @staticmethod
        def from_pretrained(model_id, load_onnx_model=False, **kwargs):
            if load_onnx_model:
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


def test_presidio_gliner_gets_device_and_torch_provenance(monkeypatch):
    seen = {}

    class Recognizer:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    class Engine:
        registry = types.SimpleNamespace(add_recognizer=lambda r: None)

    monkeypatch.setitem(sys.modules, "presidio_analyzer", types.SimpleNamespace(AnalyzerEngine=Engine))
    monkeypatch.setitem(sys.modules, "presidio_analyzer.predefined_recognizers",
                        types.SimpleNamespace(GLiNERRecognizer=Recognizer))
    from sct.utils.ner import GeneralNER

    ner = GeneralNER(TextCleanerConfig(ner_backend="presidio_gliner", ner_device="cuda", cuda_device_id=2,
                                       gliner_model="m"))
    assert seen["map_location"] == "cuda:2"
    assert ner.provenance["GLINER"]["provider"] == "cuda:2"
