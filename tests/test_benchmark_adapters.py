"""Comparison adapters in process: the contract, and each load() against a fake of its tool. No model, no download."""

import contextlib
import io
import json
import math
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from tests.test_benchmarks_compare import _adapter_module


def fake_modules(monkeypatch, **modules):
    """Install fakes in ``sys.modules``: ``a__b="x"`` stands for ``a.b``, and its parent exposes it."""
    for name, module in modules.items():
        dotted = name.replace("__", ".")
        monkeypatch.setitem(sys.modules, dotted, module)
        if "." in dotted:
            parent, child = dotted.rsplit(".", 1)
            setattr(sys.modules[parent], child, module)


def fake_torch():
    return SimpleNamespace(set_num_threads=lambda n: None, inference_mode=contextlib.nullcontext,
                           cuda=SimpleNamespace(is_available=lambda: False))


def doc(doc_id="d", text="hello world", lang="en"):
    return {"id": doc_id, "text": text, "lang": lang}


# --- _protocol ---

@pytest.fixture
def protocol():
    return _adapter_module("_protocol.py")


def test_check_normalises_a_valid_prediction(protocol):
    result = {"spans": [{"start": 0, "end": 5, "label": "X", "score": 1}], "masked_text": 7, "unplaced": 0}
    assert protocol.check(doc(), result) == {"spans": [{"start": 0, "end": 5, "label": "X", "score": 1.0}],
                                             "masked_text": "7"}
    assert protocol.check(doc(), {"unplaced": 2}) == {"spans": [], "unplaced": 2}
    assert protocol.check(doc(), {"spans": [{"start": 0, "end": 1, "label": "X"}]})["spans"][0]["score"] is None


@pytest.mark.parametrize("span", [{"start": -1, "end": 2, "label": "X"}, {"start": 3, "end": 3, "label": "X"},
                                  {"start": 0, "end": 99, "label": "X"}, {"start": 0.0, "end": 2, "label": "X"},
                                  {"start": 0, "end": 2, "label": ""}, {"start": 0, "end": 2, "label": None}])
def test_check_rejects_a_span_outside_the_contract(protocol, span):
    with pytest.raises(ValueError):
        protocol.check(doc(), {"spans": [span]})


def run_quality(protocol, predict, docs, batch_size):
    out = io.StringIO()
    protocol.run_quality(predict, docs, out, batch_size)
    return [json.loads(line) for line in out.getvalue().splitlines()]


def test_a_failing_batch_is_retried_so_one_doc_does_not_fail_its_neighbours(protocol):
    calls = []

    def predict(docs):
        calls.append([d["id"] for d in docs])
        if any(d["id"] == "bad" for d in docs):
            raise RuntimeError("boom")
        return [{"spans": []} for _ in docs]

    lines = run_quality(protocol, predict, [doc("a"), doc("bad"), doc("c")], 3)
    assert calls == [["a", "bad", "c"], ["a"], ["bad"], ["c"]]
    assert [line["id"] for line in lines] == ["a", "bad", "c"]
    assert lines[1]["error"] == "RuntimeError: boom" and lines[1]["spans"] == []
    assert "error" not in lines[0] and "error" not in lines[2]


def test_a_wrong_result_count_or_a_bad_span_is_a_doc_error(protocol):
    assert "predict returned 0 results" in run_quality(protocol, lambda docs: [], [doc()], 1)[0]["error"]

    def bad(docs):
        return [{"spans": [{"start": 0, "end": 999, "label": "X"}]} for _ in docs]

    assert "out of range" in run_quality(protocol, bad, [doc()], 1)[0]["error"]


def test_an_error_message_is_capped(protocol):
    def predict(docs):
        raise ValueError("x" * 1000)

    assert len(run_quality(protocol, predict, [doc()], 1)[0]["error"]) == 500


def test_speed_lines_come_in_order_and_cover_both_batch_sizes(protocol):
    out = io.StringIO()
    protocol.run_speed(lambda docs: [{"spans": []} for _ in docs], [doc(str(i)) for i in range(5)], out, 2, 1.5)
    lines = [json.loads(line) for line in out.getvalue().splitlines()]
    assert [line["event"] for line in lines] == ["first", "latency", "throughput", "throughput", "done"]
    assert lines[0]["load_seconds"] == 1.5 and lines[1]["docs"] == 5
    assert [(line["batch_size"], line["common"]) for line in lines[2:4]] == [(2, False), (8, True)]
    assert lines[4]["peak_rss_mb"] > 0


def test_serve_writes_ready_then_predictions_and_passes_the_config(protocol, tmp_path):
    source, target = tmp_path / "in.jsonl", tmp_path / "out.jsonl"
    source.write_text(json.dumps(doc("a")) + "\n\n" + json.dumps(doc("b")) + "\n", encoding="utf-8")
    seen = {}

    def load(config):
        seen.update(config)
        assert not target.exists()
        return lambda docs: [{"spans": []} for _ in docs]

    protocol.serve(load, ["--input", str(source), "--output", str(target), "--config", '{"k": 1}', "--batch-size", "0"])
    assert seen == {"k": 1}
    assert float((tmp_path / "out.jsonl.ready").read_text()) >= 0
    assert [json.loads(line)["id"] for line in target.read_text().splitlines()] == ["a", "b"]


# --- dummy ---

def test_dummy_marks_emails_and_digit_runs_outside_them():
    predict = _adapter_module("dummy_adapter.py").load({})
    spans = predict([doc(text="ann1@x.io call 42")])[0]["spans"]
    assert [(s["start"], s["end"], s["label"], s["score"]) for s in spans] == [(0, 9, "EMAIL", 0.9), (15, 17, "NUMBER", 0.6)]


def test_dummy_config_is_keyed_by_doc_id():
    fixed = [{"start": 0, "end": 1, "label": "X"}]
    predict = _adapter_module("dummy_adapter.py").load({"raise": ["r"], "bad": ["b"], "spans": {"s": fixed}})
    assert predict([doc("s")])[0]["spans"] == fixed
    assert predict([doc("b", "abc")])[0]["spans"][0]["end"] == 8
    with pytest.raises(RuntimeError):
        predict([doc("r")])


# --- opf ---

LABELS = ["O", "B-x", "I-x", "E-x", "S-x", "B-y", "E-y"]
BIASES = dict.fromkeys(("transition_bias_background_stay", "transition_bias_background_to_start",
                        "transition_bias_inside_to_continue", "transition_bias_inside_to_end",
                        "transition_bias_end_to_background", "transition_bias_end_to_start"), 0.0)


def test_opf_tag_and_span_helpers():
    opf = _adapter_module("opf_adapter.py")
    assert opf.tag_of("O") == ("O", None) and opf.tag_of("B-private_person") == ("B", "private_person")
    assert opf.token_spans([0, 1, 2, 3, 0, 4], LABELS) == [("x", 1, 4), ("x", 5, 6)]
    assert opf.token_spans([0, 3], LABELS) == []  # an end with no begin is not a span
    spans = [{"start": 0, "end": 3, "label": "x"}, {"start": 8, "end": 11, "label": "y"}]
    assert opf.mask("Ann met Bob", spans) == "<X> met <Y>"


def test_opf_transitions_allow_only_valid_bioes():
    torch = pytest.importorskip("torch")
    opf = _adapter_module("opf_adapter.py")
    start, end, trans = opf.transition_scores(LABELS, {**BIASES, "transition_bias_background_stay": 0.5})
    index = LABELS.index
    assert [LABELS[i] for i in torch.nonzero(start > opf.NEG_INF).flatten().tolist()] == ["O", "B-x", "S-x", "B-y"]
    assert [LABELS[i] for i in torch.nonzero(end > opf.NEG_INF).flatten().tolist()] == ["O", "E-x", "S-x", "E-y"]
    assert trans[index("O"), index("O")] == 0.5
    assert trans[index("B-x"), index("I-x")] == 0.0 and trans[index("B-x"), index("E-x")] == 0.0
    assert trans[index("B-x"), index("E-y")] == opf.NEG_INF  # a span closes with its own label
    assert trans[index("O"), index("I-x")] == opf.NEG_INF
    assert trans[index("E-x"), index("B-y")] == 0.0


def test_opf_viterbi_finds_the_best_valid_path_not_the_best_token():
    torch = pytest.importorskip("torch")
    opf = _adapter_module("opf_adapter.py")
    start, end, trans = opf.transition_scores(LABELS, BIASES)
    logp = torch.full((3, len(LABELS)), -5.0)
    logp[0, LABELS.index("I-x")] = 0.0  # token-wise best is I, E, O: not a valid start
    logp[0, LABELS.index("B-x")] = -1.0
    logp[1, LABELS.index("E-x")] = 0.0
    logp[2, LABELS.index("O")] = 0.0
    assert [LABELS[i] for i in opf.viterbi(logp, start, end, trans)] == ["B-x", "E-x", "O"]


def test_opf_viterbi_falls_back_to_argmax_when_no_path_is_finite():
    torch = pytest.importorskip("torch")
    opf = _adapter_module("opf_adapter.py")
    start, end, trans = opf.transition_scores(LABELS, BIASES)
    logp = torch.full((2, len(LABELS)), -math.inf)
    logp[0, LABELS.index("I-x")] = 0.0
    logp[1, LABELS.index("I-x")] = 0.0
    assert opf.viterbi(logp, start, end, trans) == [LABELS.index("I-x")] * 2


def _opf_predict(monkeypatch, tmp_path, config):
    torch = pytest.importorskip("torch")

    class Tokenizer:
        def __call__(self, texts, add_special_tokens, return_offsets_mapping):
            assert texts == ["Ann  lives"]
            return {"input_ids": [[1, 2]], "offset_mapping": [[(0, 5), (5, 10)]]}

    class Model:
        config = SimpleNamespace(id2label=dict(enumerate(["O", "S-person"])), pad_token_id=0)

        def to(self, device):
            return self

        def eval(self):
            return self

        def __call__(self, input_ids, attention_mask):
            # token 1 is the person, token 2 is background
            logits = torch.tensor([[[0.0, 3.0], [3.0, 0.0]][i - 1] for i in input_ids[0].tolist()])
            return SimpleNamespace(logits=logits.unsqueeze(0))

    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps({"operating_points": {"default": {"biases": BIASES}}}))
    fake_modules(
        monkeypatch,
        huggingface_hub=SimpleNamespace(hf_hub_download=lambda *a, **k: str(calibration)),
        transformers=SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: Tokenizer()),
            AutoModelForTokenClassification=SimpleNamespace(from_pretrained=lambda *a, **k: Model())),
    )
    return _adapter_module("opf_adapter.py").load(config)


@pytest.mark.parametrize("n_ctx", [None, 1])  # one window, or a window per token
def test_opf_predict_trims_whitespace_and_masks(monkeypatch, tmp_path, n_ctx):
    predict = _opf_predict(monkeypatch, tmp_path, {"device": "cpu", "n_ctx": n_ctx})
    result = predict([doc(text="Ann  lives")])[0]
    span = result["spans"][0]
    assert (span["start"], span["end"], span["label"]) == (0, 3, "person") and 0.5 < span["score"] < 1
    assert result["masked_text"] == "<PERSON>  lives"


def test_opf_rejects_an_unknown_bias(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="unknown OPF biases"):
        _opf_predict(monkeypatch, tmp_path, {"device": "cpu", "biases": {"nope": 1.0}})


# --- gliner-nvidia ---

def _nvidia(monkeypatch, config):
    calls = {}

    class Model:
        def to(self, device):
            calls["to"] = device

        def inference(self, texts, labels, threshold, batch_size):
            calls.update(labels=labels, threshold=threshold, batch_size=batch_size)
            return [[{"start": 0, "end": 3, "label": labels[0], "score": 0.7, "text": "ignored"}] for _ in texts]

    def from_pretrained(name, revision, map_location, dtype):
        calls.update(revision=revision, device=map_location, dtype=dtype)
        return Model()

    fake_modules(monkeypatch, torch=fake_torch(),
                 gliner=SimpleNamespace(GLiNER=SimpleNamespace(from_pretrained=from_pretrained)))
    adapter = _adapter_module("gliner_nvidia_adapter.py")
    return adapter, adapter.load(config), calls


def test_nvidia_default_track_is_the_quick_start_on_the_common_prompts(monkeypatch):
    adapter, predict, calls = _nvidia(monkeypatch, {})
    results = predict([doc("a"), doc("b")])
    assert results[0] == {"spans": [{"start": 0, "end": 3, "label": adapter.COMMON[0], "score": 0.7}]}
    assert calls["labels"] == adapter.COMMON and calls["threshold"] == 0.5 and calls["batch_size"] == 2
    assert calls["revision"] == adapter.REVISION and calls["dtype"] == "float32" and calls["device"] == "cpu"


def test_nvidia_knobs_change_labels_threshold_and_gpu_dtype(monkeypatch):
    adapter, predict, calls = _nvidia(monkeypatch, {"labels": "card", "threshold": 0.3, "device": "cuda"})
    predict([doc()])
    assert calls["labels"] == adapter.CARD and calls["threshold"] == 0.3
    assert calls["dtype"] == "float16" and calls["to"] == "cuda"


def test_nvidia_rejects_an_unknown_label_set(monkeypatch):
    with pytest.raises(KeyError):
        _nvidia(monkeypatch, {"labels": "other"})


# --- gliner2-fastino ---

def test_fastino_prompts_and_unknown_choice():
    fastino = _adapter_module("gliner2_fastino_adapter.py")
    assert fastino.prompts("card") == list(fastino.CARD_LABELS)
    assert "person" in fastino.prompts("gliner")
    with pytest.raises(ValueError):
        fastino.prompts("other")


def test_fastino_spans_are_sorted_and_missing_offsets_count_as_unplaced():
    fastino = _adapter_module("gliner2_fastino_adapter.py")
    result = fastino.to_spans({"entities": {"email": [{"start": 9, "end": 12, "confidence": 0.8}, {"text": "lost"}],
                                            "person": [{"start": 0, "end": 3, "confidence": 0.9}]}})
    assert result == {"spans": [{"start": 0, "end": 3, "label": "person", "score": 0.9},
                                {"start": 9, "end": 12, "label": "email", "score": 0.8}], "unplaced": 1}
    assert fastino.to_spans({}) == {"spans": [], "unplaced": 0}


def test_fastino_passes_the_chunking_and_threshold_settings(monkeypatch):
    calls = {}

    class Model:
        def to(self, device):
            calls["device"] = device

        def batch_extract_entities_long(self, texts, labels, **kwargs):
            calls.update(kwargs, labels=labels)
            return [{"entities": {}} for _ in texts]

    fake_modules(monkeypatch, torch=fake_torch(),
                 gliner2=SimpleNamespace(GLiNER2=SimpleNamespace(from_pretrained=lambda *a, **k: Model())))
    predict = _adapter_module("gliner2_fastino_adapter.py").load({"threshold": 0.2, "encoder_batch": 2, "chunk_size": 10})
    assert predict([doc()]) == [{"spans": [], "unplaced": 0}]
    assert (calls["threshold"], calls["batch_size"], calls["chunk_size"], calls["chunk_overlap"]) == (0.2, 2, 10, 64)
    assert calls["include_spans"] and calls["include_confidence"] and calls["device"] == "cpu"


# --- presidio ---

def _presidio(monkeypatch, tmp_path, config):
    calls = {"analyze": [], "engine_args": None}

    class Analyzer:
        def __init__(self, **kwargs):
            calls["engine_args"] = kwargs
            self.nlp_engine = SimpleNamespace(nlp={"en": SimpleNamespace(max_length=5),
                                                   "de": SimpleNamespace(max_length=5)})
            calls["nlp"] = self.nlp_engine.nlp

        def analyze(self, text, language, score_threshold):
            calls["analyze"].append((language, score_threshold))
            return [SimpleNamespace(start=4, end=6, entity_type="B", score=0.5),
                    SimpleNamespace(start=0, end=3, entity_type="A", score=None)]

    class Registry:
        def __init__(self, supported_languages):
            calls["registry_languages"] = supported_languages

        def load_predefined_recognizers(self, languages, nlp_engine):
            pass

        def remove_recognizer(self, name):
            calls["removed"] = name

        def add_recognizer(self, recognizer):
            calls.setdefault("added", []).append(recognizer)

    conf = tmp_path / "conf.yaml"
    conf.write_text("models: []\n")

    class Provider:
        _get_full_conf_path = staticmethod(lambda: str(conf))

        def __init__(self, nlp_configuration):
            calls["nlp_conf"] = nlp_configuration

        def create_engine(self):
            return "engine"

    def recognizer(**kwargs):
        return kwargs

    analyzer_module = SimpleNamespace(
        AnalyzerEngine=Analyzer, RecognizerRegistry=Registry,
        predefined_recognizers=SimpleNamespace(PhoneRecognizer=recognizer, UkNinoRecognizer=recognizer))
    anonymizer = SimpleNamespace(anonymize=lambda text, analyzer_results: SimpleNamespace(text=f"masked:{text}"))
    fake_modules(monkeypatch, presidio_analyzer=analyzer_module,
                 presidio_analyzer__nlp_engine=SimpleNamespace(NlpEngineProvider=Provider),
                 presidio_anonymizer=SimpleNamespace(AnonymizerEngine=lambda: anonymizer))
    return _adapter_module("presidio_adapter.py").load(config), calls


def test_presidio_default_is_the_untouched_quick_start_on_english(monkeypatch, tmp_path):
    predict, calls = _presidio(monkeypatch, tmp_path, {})
    result = predict([doc(text="x" * 20, lang="de")])[0]
    assert calls["engine_args"] == {} and calls["analyze"] == [("en", None)]
    assert [s["label"] for s in result["spans"]] == ["A", "B"]  # sorted by offset
    assert result["masked_text"] == "masked:" + "x" * 20
    assert calls["nlp"]["en"].max_length == 21  # a long doc is analysed whole


def test_presidio_gold_uses_its_models_and_warns_once_for_a_missing_one(monkeypatch, tmp_path, capsys):
    predict, calls = _presidio(monkeypatch, tmp_path, {"language": "gold", "models": {"de": "de_core"},
                                                       "score_threshold": 0.4})
    predict([doc(lang="de_AT"), doc(lang="fr"), doc(lang="fr")])
    assert calls["analyze"] == [("de", 0.4), ("en", 0.4), ("en", 0.4)]
    assert capsys.readouterr().err.count("no spaCy model") == 1
    assert calls["nlp_conf"]["models"] == [{"lang_code": "en", "model_name": "en_core_web_lg"},
                                           {"lang_code": "de", "model_name": "de_core"}]
    assert calls["engine_args"]["supported_languages"] == ["en", "de"]


def test_presidio_tuned_recognizers_replace_phone_and_enable_extras(monkeypatch, tmp_path):
    _, calls = _presidio(monkeypatch, tmp_path, {"phone_regions": ["NL"], "enable": ["UkNinoRecognizer"]})
    assert calls["removed"] == "PhoneRecognizer"
    assert calls["added"] == [{"supported_language": "en", "supported_regions": ("NL",)}, {"supported_language": "en"}]


# --- bardsai ---

DETECTOR = """
import json
import re
from pathlib import Path
import numpy as np

_EMAIL = re.compile(r"\\S+@\\S+")
TYPES = json.loads((Path(__file__).parent / "types.json").read_text())

class PIIDetector:
    types = TYPES

    def __init__(self, path, onnx, max_len, stride):
        assert (max_len, stride) == (512, 128)

    def _char_probs(self, text):
        acc = np.zeros((len(text), len(self.types)))
        cnt = np.zeros(len(text))
        acc[0, self.types.index("PERSON_NAME")] = 0.8  # the only token start inside "Ann"
        cnt[0] = 1
        return acc, cnt, None

    def predict(self, text):
        self._char_probs(text)
        return [{"start": 0, "end": 3, "label": "PERSON_NAME"}, {"start": 7, "end": 15, "label": "EMAIL_ADDRESS"}]
"""


def _bardsai(monkeypatch, tmp_path, config, drop=()):
    adapter = _adapter_module("bardsai_v3_adapter.py")
    types = [t for t in [*adapter.LABELS, *adapter.OUT_OF_SCOPE] if t not in drop]
    (tmp_path / "pii_detector.py").write_text(DETECTOR)
    (tmp_path / "types.json").write_text(json.dumps(types))
    fake_modules(monkeypatch, huggingface_hub=SimpleNamespace(snapshot_download=lambda *a, **k: str(tmp_path)))
    return adapter.load(config)


def test_bardsai_scores_from_the_decoder_and_leaves_regex_emails_unscored(monkeypatch, tmp_path):
    result = _bardsai(monkeypatch, tmp_path, {})([doc(text="Ann at ann@x.io")])[0]
    assert np.isclose(result["spans"][0]["score"], 0.8) and result["spans"][1]["score"] is None
    assert result["masked_text"] == "[PERSON_NAME] at [EMAIL_ADDRESS]"


def test_bardsai_threshold_drops_a_span_from_the_mask_too(monkeypatch, tmp_path):
    result = _bardsai(monkeypatch, tmp_path, {"threshold": 0.9})([doc(text="Ann at ann@x.io")])[0]
    assert [s["label"] for s in result["spans"]] == ["EMAIL_ADDRESS"]
    assert result["masked_text"] == "Ann at [EMAIL_ADDRESS]"


def test_bardsai_stops_when_the_model_labels_change(monkeypatch, tmp_path):
    with pytest.raises(RuntimeError, match="labels changed"):
        _bardsai(monkeypatch, tmp_path, {}, drop=("HEALTH_DATA",))


# --- datafog ---

def test_datafog_locale_gold_means_german_only():
    datafog = _adapter_module("datafog_adapter.py")
    assert datafog.doc_locales("gold", "de_AT") == ["de"]
    assert datafog.doc_locales("gold", "fr") is None and datafog.doc_locales("gold", None) is None
    assert datafog.doc_locales(["nl"], "de") == ["nl"] and datafog.doc_locales(None, "de") is None


def test_datafog_counts_values_it_could_not_place(monkeypatch):
    calls = []

    def scan_and_redact(text, **kwargs):
        calls.append(kwargs)
        entities = [SimpleNamespace(start=0, end=3, type="EMAIL"), SimpleNamespace(start=-1, end=-1, type="PHONE")]
        return SimpleNamespace(entities=entities, redacted_text="[x]")

    fake_modules(monkeypatch, datafog=SimpleNamespace(), datafog__engine=SimpleNamespace(scan_and_redact=scan_and_redact))
    predict = _adapter_module("datafog_adapter.py").load({"locales": "gold", "threads": 4})
    assert predict([doc(lang="de")]) == [{"spans": [{"start": 0, "end": 3, "label": "EMAIL", "score": None}],
                                          "masked_text": "[x]", "unplaced": 1}]
    assert calls[0] == {"engine": "regex", "entity_types": None, "strategy": "token", "locales": ["de"], "backend": "python"}


# --- sct ---

def test_sct_language_falls_back_to_the_base_code_or_to_detection():
    sct = _adapter_module("sct_adapter.py")
    assert sct.sct_language("ar-mixed_en") == sct.sct_language("ar")
    assert sct.sct_language(None) is None and sct.sct_language("zz-nope") is None


def test_sct_predict_places_masked_values_in_the_source(capsys):
    predict = _adapter_module("sct_adapter.py").load({"overrides": {"check_ner_process": False}})
    text = "Mail ana@example.com now."
    english, unknown = predict([doc("a", text, "en"), doc("b", text, "zz-nope")])
    assert [(s["start"], s["end"], s["score"]) for s in english["spans"]] == [(5, 20, None)]
    assert text[english["spans"][0]["start"]:english["spans"][0]["end"]] == "ana@example.com"
    assert "ana@example.com" not in english["masked_text"] and english["unplaced"] == 0
    assert unknown["spans"] == english["spans"]
    assert capsys.readouterr().err.count("no SCT language") == 1
