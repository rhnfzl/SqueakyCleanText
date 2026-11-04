"""ONNX runs lock-free, GLiNER stays serialised, models are cached by (name, revision)."""

import multiprocessing
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from tests.ner_helpers import bare_ner, set_cfg
from sct import config
from sct.config import TextCleanerConfig
from sct.utils import ner as ner_module
from sct.utils.ner import GeneralNER
from sct.utils.onnx_pipeline import load_onnx_ner_model
from tests.conftest import TEST_NER_MODELS

TAGS = ('PER', 'LOC', 'ORG', 'MISC')
NAMES = ["John Smith", "Angela Merkel", "Barack Obama", "Marie Curie"]
PLACES = ["Berlin", "Paris", "London", "New York"]


def _texts():
    texts = []
    for i in range(16):
        base = f"{NAMES[i % 4]} met the board of Siemens in {PLACES[(i + 1) % 4]} on day {i}. "
        # every fourth text is long enough to be chunked
        texts.append(base * (60 if i % 4 == 0 else 1))
    return texts


def _key(result):
    return result.text, repr(result.findings)


@pytest.fixture(scope="module")
def onnx_ner():
    return GeneralNER(TextCleanerConfig(ner_models=TEST_NER_MODELS))


def test_onnx_threaded_matches_sequential_and_takes_no_lock(onnx_ner):
    texts = _texts()
    assert len(texts[0]) > 4 * onnx_ner.min_token_length  # well over one chunk

    def run(text):
        return _key(onnx_ner.ner_process_detailed(text, TAGS, 0.85, 'ENGLISH'))

    expected = [run(t) for t in texts]
    with ThreadPoolExecutor(max_workers=8) as pool:
        for _ in range(3):
            assert list(pool.map(run, texts)) == expected
    assert onnx_ner._inference_locks == {}


def test_gliner_calls_stay_serialised():
    state = {"inside": 0, "max": 0}
    guard = threading.Lock()

    class FakeGLiNER:
        max_context_length = 512
        labels = ["person"]
        label_map = {"person": "PER"}

        def __call__(self, texts, batch_size=1):
            with guard:
                state["inside"] += 1
                state["max"] = max(state["max"], state["inside"])
            time.sleep(0.01)
            with guard:
                state["inside"] -= 1
            return [[]]

    ner = bare_ner()
    set_cfg(ner, ner_backend="gliner")
    ner._gliner_pipe = FakeGLiNER()
    set_cfg(ner, replacement_mode="placeholder")
    ner._synthetic_replacer = None
    ner._inference_locks = {}
    ner._locks_lock = threading.Lock()
    ner.tokenizer = None

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda t: ner.ner_process_detailed(t, TAGS, 0.5, 'ENGLISH'),
                      [f"Maria {i} arrived." for i in range(12)]))
    assert state["max"] == 1


class _FakeTok:
    max_len_single_sentence = 510


@pytest.mark.parametrize("revisions, loads", [
    ({'ENGLISH': 'r1', 'MULTILINGUAL': 'r2'}, [('m', 'r1'), ('m', 'r2')]),
    ({'ENGLISH': 'r1', 'MULTILINGUAL': 'r1'}, [('m', 'r1')]),
])
def test_model_cache_is_keyed_by_revision(monkeypatch, revisions, loads):
    calls = []

    def fake_load(model_name, revision=None, **kwargs):
        calls.append((model_name, revision))
        return object(), _FakeTok()

    monkeypatch.setattr("sct.utils.onnx_pipeline.load_onnx_ner_model", fake_load)
    ner = GeneralNER(TextCleanerConfig(ner_models={'ENGLISH': 'm', 'MULTILINGUAL': 'm'},
                                       ner_model_revisions=revisions))
    assert calls == loads
    assert (ner._pipelines['ENGLISH'] is ner._pipelines['MULTILINGUAL']) == (len(loads) == 1)


def test_thread_knobs_reach_the_onnx_session():
    model = TEST_NER_MODELS['ENGLISH']
    pipe, _ = load_onnx_ner_model(model, intra_op_threads=2, inter_op_threads=3)
    opts = pipe._session.get_session_options()
    assert (opts.intra_op_num_threads, opts.inter_op_num_threads) == (2, 3)
    default_pipe, _ = load_onnx_ner_model(model)
    assert default_pipe._session.get_session_options().intra_op_num_threads == 0


def test_thread_knob_passed_from_general_ner(monkeypatch):
    seen = []

    def fake_load(model_name, **kwargs):
        seen.append((kwargs["intra_op_threads"], kwargs["inter_op_threads"]))
        return object(), _FakeTok()

    monkeypatch.setattr("sct.utils.onnx_pipeline.load_onnx_ner_model", fake_load)
    GeneralNER(TextCleanerConfig(ner_models={'ENGLISH': 'a', 'MULTILINGUAL': 'b'},
                                 onnx_intra_op_threads=2, onnx_inter_op_threads=1))
    assert seen == [(2, 1), (2, 1)]


@pytest.mark.parametrize("field", ["onnx_intra_op_threads", "onnx_inter_op_threads"])
def test_thread_config_validation(field):
    with pytest.raises(ValueError, match=field):
        TextCleanerConfig(check_ner_process=False, **{field: 0})
    assert getattr(TextCleanerConfig(check_ner_process=False), field) is None


def test_thread_config_module_globals(monkeypatch):
    monkeypatch.setattr(config, "ONNX_INTRA_OP_THREADS", 2)
    monkeypatch.setattr(config, "ONNX_INTER_OP_THREADS", 1)
    cfg = config._config_from_module_globals()
    assert (cfg.onnx_intra_op_threads, cfg.onnx_inter_op_threads) == (2, 1)


def test_fork_child_pins_live_ner(monkeypatch):
    # A forked child must never free an inherited GeneralNER (ONNX teardown crashes there);
    # the at-fork hook keeps a strong reference to every live one. Called directly: no real fork on macOS.
    monkeypatch.setattr(ner_module, "_FORK_PINNED", [])
    ner = GeneralNER(TextCleanerConfig(ner_models=TEST_NER_MODELS))
    assert ner in ner_module._LIVE_NER
    ner_module._pin_live_ner_in_child()
    assert any(p is ner for p in ner_module._FORK_PINNED)


def test_import_sct_leaves_onnxruntime_unloaded():
    code = "import sys, sct; sys.exit('onnxruntime' in sys.modules)"
    child = multiprocessing.get_context("spawn").Process(target=exec, args=(code,))  # a fresh interpreter
    child.start()
    child.join()
    assert child.exitcode == 0
