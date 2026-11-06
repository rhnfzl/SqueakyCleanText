"""Round-2 comparison entries: registry, CUDA 12 envs, label maps and span placement with fake models.

No tool package or model is loaded: each adapter's ``load`` runs against hand-written stand-ins for its
library, on hand-written text, so offsets, label maps and wiring are checked offline.
"""

import importlib.util
import json
import sys
import types

import pytest

from benchmarks.compare import entries, report, runner, speed
from benchmarks.datasets import ROUND2

NEW = ("gliner-knowledgator", "piiranha", "llm-guard", "scrubadub", "ai4privacy", "openmed", "pii-tracer",
       "presidio-full", "sct-training", "sct-gliner-nvidia")
TEXT = "Héllo Ana Lee 👋, mail ana.lee@example.org or call +31 6 1234 5678 today."
EMAIL = (TEXT.index("ana.lee"), TEXT.index("ana.lee") + len("ana.lee@example.org"))


def adapter(name):
    sys.path.insert(0, str(runner.ADAPTERS))
    try:
        spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), runner.ADAPTERS / name)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(runner.ADAPTERS))


def fake_module(monkeypatch, name, **attrs):
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def fake_torch(monkeypatch):
    cuda = types.SimpleNamespace(is_available=lambda: False)
    return fake_module(monkeypatch, "torch", set_num_threads=lambda n: None, cuda=cuda)


# --- registry ---

def test_round2_entries_are_registered_with_runs_envs_and_rules():
    assert set(NEW) == set(entries.ROUND2_ENTRIES) and set(NEW) <= set(entries.ENTRIES)
    for name in NEW:
        t = entries.tool(name)
        assert (runner.ADAPTERS / t["adapter"]).exists() and t["requirements"].exists(), name
        assert name in entries.CONTAMINATION and name in entries.CLAIMED_LANGUAGES and name in report.LICENCES, name
        assert set(t["labels"].values()) <= set(entries.SCT_TYPES) and t["gpu_gb"] <= 5, name
    assert entries.tool("presidio-full", "native")["adapter_config"]["enable"] == "all"
    assert entries.tool("sct-training", "gold")["adapter_config"] == {"mode": "default", "language": "gold",
                                                                      "preset": "training"}
    nvidia = entries.tool("sct-gliner-nvidia")["adapter_config"]
    assert nvidia["mode"] == "gliner" and nvidia["overrides"]["gliner_model"] == "nvidia/gliner-PII"


def test_round2_gpu_envs_are_cuda12():
    for name in entries.ENTRIES:  # round-1 entries too: round 2 runs them on CUDA 12 as well
        spec = entries.tool(name)["requirements"].read_text()
        pins = [line for line in spec.splitlines() if line and not line.startswith("#")]
        torch_pins = [p for p in pins if p.startswith("torch")]
        assert all("/whl/cu126/torch-2.14.1%2Bcu126-cp312" in p for p in torch_pins), name  # never PyPI's CUDA 13 torch
        if "cuda" in entries.ENTRIES[name]["device"] and name != "sct-training":
            assert torch_pins, name
    ort = entries.tool("sct-training")["requirements"].read_text()
    assert "onnxruntime-gpu[cuda,cudnn] @ " in ort and "/onnxruntime-gpu/1.29/" in ort  # PyPI's 1.27+ is CUDA 13
    assert entries.ENTRIES["presidio-full"]["device"] == "cpu"


def test_env_builder_swaps_the_cpu_twin_for_the_gpu_build(tmp_path, monkeypatch):
    spec = tmp_path / "x.txt"
    spec.write_text("# c\nSqueakyCleanText==0.8.1\nonnxruntime-gpu[cuda,cudnn]==1.31.1\n")
    assert runner.gpu_twin_fixes(spec.read_text()) == [("onnxruntime", "onnxruntime-gpu[cuda,cudnn]==1.31.1")]
    assert runner.gpu_twin_fixes("onnxruntime==1.31.1\n") == []
    url = "onnxruntime-gpu[cuda,cudnn] @ https://example.org/onnxruntime_gpu-1.29.0-cp312-cp312-manylinux_2_28_x86_64.whl"
    assert runner.gpu_twin_fixes(url) == [("onnxruntime", url)]
    assert runner.gpu_twin_fixes(url.replace("[cuda,cudnn]", "")) == [("onnxruntime", url.replace("[cuda,cudnn]", ""))]
    calls = []
    monkeypatch.setattr(runner.shutil, "which", lambda name: "uv")
    monkeypatch.setattr(runner.subprocess, "run", lambda cmd, **kw: calls.append(cmd) or types.SimpleNamespace(stdout=""))
    (tmp_path / "envs" / "x" / "bin").mkdir(parents=True)
    runner.ensure_env("x", spec, tmp_path / "envs")
    assert ["uv", "pip", "uninstall", "--python", str(tmp_path / "envs" / "x" / "bin" / "python"), "onnxruntime"] in calls
    assert calls.index(next(c for c in calls if "uninstall" in c)) > calls.index(next(c for c in calls if "-r" in c))
    assert calls[-2][-1] == "onnxruntime-gpu[cuda,cudnn]==1.31.1" and "--no-deps" in calls[-2]


def test_round2_contamination_and_shared_view():
    status = entries.status
    assert status("piiranha", "openpii") == status("piiranha", "argilla_pii") == "excluded"  # ai4privacy-400k family
    assert entries.contamination("llm-guard", "argilla_pii") == "T" and status("llm-guard", "openpii") == "excluded"
    assert entries.contamination("openmed", "nemotron") == "T" and status("openmed", "tab") == "unknown"
    assert status("pii-tracer", "tab") == "excluded" and status("pii-tracer", "gretel") == "excluded"
    assert status("ai4privacy", "openpii") == "excluded" and status("ai4privacy", "gretel_en") == "clean"
    assert status("sct-gliner-nvidia", "nemotron") == "excluded" and status("sct-training", "redact") == "excluded"
    # SHORTLIST section 2 after the U-cell pass: one residual U, opf x KDPII, carried into openmed (OPF inside it)
    for name in entries.ENTRIES:
        unknown = {"kdpii"} if name in ("opf", "openmed") else set()
        assert {d for d in ROUND2 if status(name, d) != "clean"} == unknown, name
        assert all(status(name, d) == "unknown" for d in unknown), name
    assert runner.shared_view(["gretel_en"], {"sct-default": {}}) == ("EMAIL", "PHONE", "DATE", "NUMBER")  # round 1 kept
    assert runner.shared_view(["kdpii"], {"sct-default": {}}) == ("EMAIL", "PHONE", "NUMBER")


def test_round2_label_tables_match_the_frozen_entries():
    tables = {"piiranha": ("piiranha_adapter.py", 17), "ai4privacy": ("ai4privacy_adapter.py", 20),
              "openmed": ("openmed_adapter.py", 54), "pii-tracer": ("pii_tracer_adapter.py", 9),
              "llm-guard": ("llm_guard_adapter.py", 14), "scrubadub": ("scrubadub_adapter.py", 20)}
    for name, (file, n) in tables.items():
        module = adapter(file)
        assert entries.ENTRIES[name]["labels"] == {k: v for k, v in module.LABELS.items()}, name
        assert not set(module.OUT_OF_SCOPE) & set(module.LABELS) and len(module.LABELS) + len(module.OUT_OF_SCOPE) == n
    assert entries.ENTRIES["presidio-full"]["labels"] == entries.ENTRIES["presidio"]["labels"]


# --- adapters with fake libraries ---

def test_hf_token_pipeline_places_trimmed_spans_and_windows_long_docs(monkeypatch):
    fake_torch(monkeypatch)
    made = {}

    def pipeline(task, **kw):
        made.update(kw, task=task)

        def run(texts, batch_size):
            # the subword token carries the leading space; a second hit falls below the threshold
            return [[{"entity_group": "EMAIL", "start": t.index("ana.lee") - 1, "end": EMAIL[1], "score": 0.9},
                     {"entity_group": "GIVENNAME", "start": 6, "end": 9, "score": 0.2}] for t in texts]
        return run

    tokenizer = types.SimpleNamespace(model_max_length=10**9)
    net = types.SimpleNamespace(config=types.SimpleNamespace(id2label={0: "O", 1: "I-EMAIL", 2: "I-GIVENNAME"}))
    fake_module(monkeypatch, "transformers", pipeline=pipeline,
                AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda m, revision: tokenizer),
                AutoModelForTokenClassification=types.SimpleNamespace(from_pretrained=lambda m, revision: net))
    hf = adapter("_hf_token.py")
    predict = hf.load("m", "r", {"EMAIL", "GIVENNAME"}, {"threshold": 0.5}, max_len=256, stride=64)
    assert predict([{"text": TEXT}]) == [{"spans": [{"start": EMAIL[0], "end": EMAIL[1], "label": "EMAIL", "score": 0.9}]}]
    assert made["aggregation_strategy"] == "simple" and made["stride"] == 64 and tokenizer.model_max_length == 256
    with pytest.raises(RuntimeError, match="labels changed"):
        hf.load("m", "r", {"EMAIL"}, {}, max_len=256, stride=64)
    assert hf.label_set({0: "B-CITY", 1: "I-CITY", 2: "O"}) == {"CITY"}


def test_piiranha_and_ai4privacy_wire_their_checkpoints(monkeypatch):
    for file, window in (("piiranha_adapter.py", (256, 64)), ("ai4privacy_adapter.py", (512, 128))):
        module = adapter(file)
        seen = {}
        monkeypatch.setattr(module._hf_token, "load", lambda *a, seen=seen: seen.setdefault("args", a))
        module.load({})
        model, revision, labels, _, max_len, stride = seen["args"]
        assert (model, revision) == (module.MODEL, module.REVISION) and len(revision) == 40
        assert labels == set(module.LABELS) | set(module.OUT_OF_SCOPE) and (max_len, stride) == window


def test_pii_tracer_uses_the_vendor_predict_and_mask(monkeypatch):
    fake_torch(monkeypatch)
    span = types.SimpleNamespace(start=EMAIL[0], end=EMAIL[1], label="private_email", score=0.97)

    class Model:
        viterbi = types.SimpleNamespace(pii_types=["private_person", "private_email", "private_phone", "private_address",
                                                   "private_url", "private_date", "account_number", "secret", "other_pii"])

        def to(self, device):
            return self

        def eval(self):
            return self

        def predict(self, text):
            return [span], 0.8

    loaded = {}
    fake_module(monkeypatch, "transformers", AutoModel=types.SimpleNamespace(
        from_pretrained=lambda m, **kw: loaded.update(kw, model=m) or Model()))
    tracer = adapter("pii_tracer_adapter.py")
    out = tracer.load({"device": "cpu"})([{"text": TEXT}])[0]
    assert loaded["trust_remote_code"] is True and loaded["revision"] == tracer.REVISION
    assert out["spans"] == [{"start": EMAIL[0], "end": EMAIL[1], "label": "private_email", "score": 0.97}]
    assert out["masked_text"] == TEXT[:EMAIL[0]] + "[PRIVATE_EMAIL]" + TEXT[EMAIL[1]:]
    Model.viterbi = types.SimpleNamespace(pii_types=["private_email"])
    with pytest.raises(RuntimeError, match="labels changed"):
        tracer.load({})


def test_openmed_reuses_the_opf_decoder_with_neutral_biases(monkeypatch):
    openmed = adapter("openmed_adapter.py")
    bioes = ["O"] + [f"{t}-{x}" for x in [*openmed.LABELS, *openmed.OUT_OF_SCOPE] for t in "BIES"]
    assert len(bioes) == 217  # config.json num_labels at the pinned revision
    seen = {}
    monkeypatch.setattr(openmed, "make_predict", lambda name, rev, biases, config: (seen.update(
        name=name, rev=rev, biases=biases) or (lambda docs: []), bioes))
    openmed.load({})
    assert seen["name"] == openmed.MODEL and set(seen["biases"].values()) == {0.0}
    assert len(seen["biases"]) == 6
    openmed.load({"biases": {"transition_bias_background_stay": -1}})
    assert seen["biases"]["transition_bias_background_stay"] == -1.0
    with pytest.raises(RuntimeError, match="labels changed"):
        openmed.check_labels(bioes[:-4])
    opf = adapter("opf_adapter.py")  # the shared decoder still places a B..E run on its tokens' char offsets
    labels = ["O", "B-EMAIL", "I-EMAIL", "E-EMAIL", "S-EMAIL"]
    assert opf.token_spans([0, 1, 2, 3, 0, 4], labels) == [("EMAIL", 1, 4), ("EMAIL", 5, 6)]


def test_llm_guard_spans_follow_the_scan_steps(monkeypatch):
    def result(start, end, entity_type, score):
        return types.SimpleNamespace(start=start, end=end, entity_type=entity_type, score=score)

    class Vault:
        pass

    PIPE = types.SimpleNamespace(pipeline=types.SimpleNamespace(  # DeBERTa's tokenizer ships the 1e30 "no limit"
        tokenizer=types.SimpleNamespace(model_max_length=int(1e30)),
        model=types.SimpleNamespace(config=types.SimpleNamespace(max_position_embeddings=512))))

    class Anonymize:
        def __init__(self, vault, threshold, language):
            self._entity_types = [*entries.ENTRIES["llm-guard"]["labels"], "CRYPTO", "UUID", "CUSTOM"]
            self._threshold, self._language, self._allowed_names = threshold, language, None
            self._analyzer = types.SimpleNamespace(analyze=lambda **kw: [
                result(EMAIL[0], EMAIL[1], "EMAIL_ADDRESS", 1.0), result(6, 6, "PERSON", 0.9),
                result(6, 13, "PERSON", 0.85)], registry=types.SimpleNamespace(recognizers=[types.SimpleNamespace(), PIPE]))

        @staticmethod
        def remove_single_quotes(text):
            return text.replace("'", " ")

        def _remove_conflicts_and_get_text_manipulation_data(self, results):
            return results

        @staticmethod
        def _merge_entities_with_whitespace_between(text, results):
            return results

        @staticmethod
        def _anonymize(text, results, vault, use_faker):
            for r in sorted(results, key=lambda r: -r.start):
                text = text[:r.start] + f"[REDACTED_{r.entity_type}_1]" + text[r.end:]
            return text, []

    fake_module(monkeypatch, "llm_guard")
    fake_module(monkeypatch, "llm_guard.input_scanners", Anonymize=Anonymize)
    fake_module(monkeypatch, "llm_guard.vault", Vault=Vault)
    guard = adapter("llm_guard_adapter.py")
    predict = guard.load({})
    assert PIPE.pipeline.tokenizer.model_max_length == 512  # so llm-guard's own chunking runs on long texts
    out = predict([{"text": TEXT}, {"text": "   "}])
    assert [(s["start"], s["end"], s["label"]) for s in out[0]["spans"]] == [(6, 13, "PERSON"), (*EMAIL, "EMAIL_ADDRESS")]
    assert "[REDACTED_EMAIL_ADDRESS_1]" in out[0]["masked_text"] and out[1] == {"spans": [], "masked_text": "   "}


def test_scrubadub_expands_merged_filth_and_masks_like_clean(monkeypatch):
    def filth(beg, end, kind):
        return types.SimpleNamespace(beg=beg, end=end, type=kind)

    phone = (TEXT.index("+31"), TEXT.index(" today"))
    merged = types.SimpleNamespace(beg=6, end=13, type="unknown", filths=[filth(6, 13, "name"), filth(6, 9, "organization")])

    class Scrubber:
        def __init__(self, locale):
            self.detectors = []

        def add_detector(self, detector):
            self.detectors.append(detector)

        def iter_filth(self, text):
            return iter([filth(*phone, "phone"), merged, filth(*EMAIL, "email")])

    fake_module(monkeypatch, "scrubadub", Scrubber=Scrubber)
    fake_module(monkeypatch, "scrubadub_spacy")
    fake_module(monkeypatch, "scrubadub_spacy.detectors")
    fake_module(monkeypatch, "scrubadub_spacy.detectors.spacy", SpacyEntityDetector=lambda **kw: kw)
    out = adapter("scrubadub_adapter.py").load({})([{"text": TEXT}])[0]
    assert [(s["start"], s["end"], s["label"]) for s in out["spans"]] == [
        (6, 9, "organization"), (6, 13, "name"), (*EMAIL, "email"), (*phone, "phone")]
    assert out["masked_text"] == ("Héllo {{NAME+ORGANIZATION}} 👋, mail {{EMAIL}} or call {{PHONE}} today.")


def test_gliner_knowledgator_runs_the_card_quick_start(monkeypatch):
    fake_torch(monkeypatch)
    loaded = {}

    class Model:
        def to(self, device):
            pass

        def inference(self, texts, labels, threshold, batch_size):
            loaded.update(threshold=threshold, labels=labels)
            return [[{"start": EMAIL[0], "end": EMAIL[1], "label": "email", "score": 0.8}] for _ in texts]

    fake_module(monkeypatch, "gliner", GLiNER=types.SimpleNamespace(
        from_pretrained=lambda model, **kw: loaded.update(kw, model=model) or Model()))
    know = adapter("gliner_knowledgator_adapter.py")
    out = know.load({"device": "cpu"})([{"text": TEXT}])
    assert out == [{"spans": [{"start": EMAIL[0], "end": EMAIL[1], "label": "email", "score": 0.8}]}]
    assert loaded["model"] == know.MODEL and loaded["revision"] == know.REVISION and loaded["threshold"] == 0.3
    assert loaded["labels"] == list(entries.GLINER_PROMPTS) and loaded["dtype"] == "float32"
    with pytest.raises(ValueError):
        know.load({"labels": "card"})


def test_presidio_full_enables_every_pattern_recognizer_class():
    presidio = adapter("presidio_adapter.py")

    class Base:
        pass

    module = types.SimpleNamespace(Base=Base, EmailRecognizer=type("EmailRecognizer", (Base,), {}),
                                   UkNinoRecognizer=type("UkNinoRecognizer", (Base,), {}), helper=len, other=int)
    assert presidio.pattern_recognizers(module, Base) == ["EmailRecognizer", "UkNinoRecognizer"]
    assert callable(adapter("presidio_full_adapter.py").load)


def test_sct_round2_configs_build():
    from sct.config import TextCleanerConfig

    sct = adapter("sct_adapter.py")
    cfg = TextCleanerConfig(**sct.cleaner_fields({"preset": "training"}))
    assert cfg.replacement_mode == "reversible" and not cfg.check_replace_dates
    run = entries.tool("sct-gliner-nvidia")["adapter_config"]
    cfg = TextCleanerConfig(**sct.cleaner_fields(run))
    assert cfg.gliner_model == "nvidia/gliner-PII" and cfg.gliner_revision == entries.ENTRIES["gliner-nvidia"]["revision"]


# --- GPU speed column ---

def test_speed_and_report_show_device_docs_per_second_and_vram(tmp_path):
    assert speed.entry_device("cuda|cpu", "cuda") == "cuda" and speed.entry_device("cpu", "cuda") == "cpu"
    assert speed.entry_device("cuda|cpu", "cpu") == "cpu"
    entry = {"cold_start_seconds": {"median": 3.0}, "latency_median_ms": {"median": 9.0}, "latency_p95_ms": {"median": 12.0},
             "common_batch_docs_per_second": {"median": 41.5}, "common_batch_chars_per_second": {"median": 9000.0},
             "peak_rss_mb": {"median": 900.0}, "peak_vram_mb": {"median": 2048.0}, "torch_peak_vram_mb": None,
             "install_mb": 10.0, "failed": [], "device": "cuda"}
    cpu_entry = {**entry, "peak_vram_mb": None, "device": "cpu"}
    run = {"device": "cuda", "gpu": "0", "cpus": None, "threads": 16, "docs": 100, "repeats": 3,
           "entries": {"openmed": entry, "presidio-full": cpu_entry}}
    table = report.speed_table(run)
    assert "| device |" in table[2] and "| peak VRAM MB |" in table[2]
    assert any(row.startswith("| openmed | cuda |") and "| 2,048 |" in row for row in table)
    assert any(row.startswith("| presidio-full | cpu |") and "| - |" in row for row in table)
    (tmp_path / "speed-gpu").mkdir()
    (tmp_path / "speed-gpu" / "speed.json").write_text(json.dumps(run))
    assert report.gpu_speed(tmp_path, "openmed") == "cuda, 41.5 docs/s, 2,048 MB"
    assert report.gpu_speed(tmp_path, "nope") == "n/a" and report.gpu_speed(tmp_path / "x", "openmed") == "n/a"
