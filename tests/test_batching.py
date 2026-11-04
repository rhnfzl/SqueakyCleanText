"""ner_batch_size batches model calls; batched output equals unbatched output."""

import math
import threading
from dataclasses import replace

import pytest

from tests.helpers import bare_ner, set_cfg
from sct.config import TextCleanerConfig
from sct.sct import TextCleaner
from sct.utils.gliner_adapter import GLiNERAdapter
from sct.utils.ner import GeneralNER
from tests.conftest import TEST_NER_MODELS

TAGS = ('PER', 'LOC', 'ORG', 'MISC')

DOCS = [
    "John Smith met Angela Merkel in Berlin to talk about Siemens.",
    "Jan de Vries woont in Amsterdam en werkt bij Philips in Eindhoven.",
    "Hans Müller wohnt in München und arbeitet für die Deutsche Bank.",
    "Marie Curie worked in Paris. " * 120,  # chunked into several windows
    "Barack Obama visited London.",
    "Pieter Jansen reist morgen met de trein van Utrecht naar Rotterdam. " * 40,
]


class _CountingSession:
    def __init__(self, session):
        self._session = session
        self.runs = 0

    def run(self, *args, **kwargs):
        self.runs += 1
        return self._session.run(*args, **kwargs)


@pytest.fixture(scope="module")
def onnx_ner():
    return GeneralNER(TextCleanerConfig(ner_models=TEST_NER_MODELS))


def test_onnx_batch_runs_once_per_slice(onnx_ner, monkeypatch):
    pipe = onnx_ner._pipelines['ENGLISH']
    counter = _CountingSession(pipe._session)
    monkeypatch.setattr(pipe, '_session', counter)
    monkeypatch.setattr(onnx_ner, 'cfg', replace(onnx_ner.cfg, ner_batch_size=4))
    text = DOCS[3] * 2
    chunks = onnx_ner._plan_windows(text)
    assert len(chunks) >= 5
    keys = onnx_ner._get_ensemble_keys('ENGLISH')
    onnx_ner.ner_process_detailed(text, TAGS, 0.85, 'ENGLISH')
    # 0.7.1 ran one session per chunk per key: len(keys) * len(chunks)
    assert counter.runs == len(keys) * math.ceil(len(chunks) / 4)


def test_onnx_padded_batch_matches_unbatched(onnx_ner):
    pipe = onnx_ner._pipelines['ENGLISH']
    texts = [d[:400] for d in DOCS] + ["", "Berlin", "Angela Merkel and John Smith."]
    single = pipe(texts, batch_size=1)
    batched = pipe(texts, batch_size=8)
    assert len(batched) == len(texts)
    assert pipe(texts, batch_size=2, max_workers=4) == pipe(texts, batch_size=2)
    for one, many in zip(single, batched):
        assert [(e['entity_group'], e['start'], e['end'], e['word']) for e in one] == \
               [(e['entity_group'], e['start'], e['end'], e['word']) for e in many]
        for a, b in zip(one, many):
            assert abs(a['score'] - b['score']) <= 1e-5


def _cleaner(**overrides):
    cfg = TextCleanerConfig(
        check_statistical_model_processing=True,
        ner_models=TEST_NER_MODELS,
        **overrides,
    )
    return TextCleaner(cfg, include_entities=True)


def _summary(result):
    findings = result.metadata['findings']
    return (
        result.lm_text, result.stat_text, result.language,
        [(f.entity.entity_type, f.entity.start, f.entity.end, f.entity.text) for f in findings],
    )


def test_process_batch_same_output_for_any_ner_batch_size():
    expected = [_summary(r) for r in _cleaner(ner_batch_size=1).process_batch(DOCS)]
    assert any(row[3] for row in expected)  # the test model finds entities
    for size in (8, 32):
        cleaner = _cleaner(ner_batch_size=size)
        assert [_summary(r) for r in cleaner.process_batch(DOCS)] == expected
        assert [_summary(r) for r in cleaner.process_batch(DOCS, batch_size=1)] == expected
        assert [_summary(cleaner.process(d)) for d in DOCS] == expected


def test_reversible_maps_stay_per_document_under_batching():
    docs = ["John Smith met Angela Merkel in Berlin.", "Barack Obama visited London."]
    results = _cleaner(ner_batch_size=8, replacement_mode='reversible').process_batch(docs)
    unbatched = _cleaner(ner_batch_size=1, replacement_mode='reversible').process_batch(docs)
    for result, single, doc in zip(results, unbatched, docs):
        anon_map = result.metadata['anon_map']
        assert result.lm_text == single.lm_text
        assert all(entry.original in doc for entry in anon_map.entries)
        assert anon_map.deanonymize(result.lm_text) == doc


def test_synthetic_is_consistent_within_each_document():
    docs = ["John Smith called. Later John Smith left.", "Angela Merkel met Angela Merkel."]
    results = _cleaner(ner_batch_size=8, replacement_mode='synthetic').process_batch(docs)
    first, second = (r.lm_text for r in results)
    assert "John Smith" not in first and "Angela Merkel" not in second
    left, right = first.split(" called. Later ")
    assert right == left + " left."


def test_gliner_gets_one_batched_call_under_its_lock():
    calls = []

    class FakeGLiNER:
        max_context_length = 512
        labels = ["person"]
        label_map = {"person": "PER"}

        def __call__(self, texts, batch_size=1):
            calls.append((list(texts), batch_size, ner._get_lock('gliner').locked()))
            return [[] for _ in texts]

    ner = bare_ner()
    set_cfg(ner, ner_backend='gliner')
    set_cfg(ner, ner_batch_size=16)
    ner._gliner_pipe = FakeGLiNER()
    ner._inference_locks = {}
    ner._locks_lock = threading.Lock()
    ner.tokenizer = None
    results = ner.ner_process_batch_detailed(["Ann here.", "Bob there.", "Cy."], [None] * 3, TAGS)
    assert [r.text for r in results] == ["Ann here.", "Bob there.", "Cy."]
    assert calls == [(["Ann here.", "Bob there.", "Cy."], 16, True)]


class _FakeGLiNERModel:
    def __init__(self, bi_encoder):
        self.calls = []
        if bi_encoder:
            self.encode_labels = lambda labels, batch_size=8: "emb"

    def _hits(self, text):
        return [{'label': 'person', 'score': 0.9, 'text': text[:3], 'start': 0, 'end': 3}]

    def inference(self, texts, labels, threshold=0.5, batch_size=8):
        self.calls.append(('inference', len(texts), batch_size))
        return [self._hits(t) for t in texts]

    def predict_entities(self, text, labels, threshold=0.5):
        self.calls.append(('predict_entities', 1, None))
        return self._hits(text)

    def batch_predict_with_embeds(self, texts, embeds, labels, threshold=0.5, batch_size=8):
        self.calls.append(('with_embeds', len(texts), batch_size))
        return [self._hits(t) for t in texts]


@pytest.mark.parametrize("bi_encoder", [False, True])
def test_gliner_adapter_batches_only_when_asked(bi_encoder):
    adapter = object.__new__(GLiNERAdapter)
    adapter.model = _FakeGLiNERModel(bi_encoder)
    adapter.model_id = 'fake'
    adapter.variant = 'gliner'
    adapter.labels = ['person']
    adapter.label_map = {'person': 'PER'}
    adapter.label_descriptions = None
    adapter.threshold = 0.4
    adapter._is_bi_encoder = bi_encoder
    adapter._label_embeddings = None
    adapter._cached_labels_key = None
    texts = ["Ann one", "Bob two", "Cyd three"]

    batched = adapter(texts, batch_size=8)
    batched_calls, adapter.model.calls = adapter.model.calls, []
    single = adapter(texts)

    assert batched == single
    assert [r[0]['word'] for r in batched] == ["Ann", "Bob", "Cyd"]
    name = 'with_embeds' if bi_encoder else 'inference'
    assert batched_calls == [(name, 3, 8)]
    assert len(adapter.model.calls) == 3  # batch_size=1 keeps one call per text
