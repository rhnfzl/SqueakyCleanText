"""Codex 0.8.0 full review, round 5: ids 1, 2, 3."""

import json
import sys
import threading
import time
import types
from pathlib import Path

import pytest

from sct import TextCleaner
from sct.presets import preset_config
from sct.streaming import WindowedStreamingRedactor


# --- 1: a phone cut off by a feed boundary is held back until it completes ---

@pytest.mark.parametrize("chunks", [
    ["aaaa aaaa aaaa Call 06 12 34", " 56 78 tail"],
    ["aaaa aaaa aaaa Call ０６ １２ ３４", " ５６ ７８ tail"],
    ["aaaa aaaa aaaa Call 06", " 12", " 34", " 56", " 78 tail"],
])
def test_window_holds_an_incomplete_phone(chunks):
    cleaner = TextCleaner(cfg=preset_config(
        "training", check_ner_process=False, check_replace_numbers=False))
    stream = WindowedStreamingRedactor(cleaner, max_buffer_chars=24, overlap_chars=4)
    out = "".join(stream.feed(c) for c in chunks) + stream.finish()
    assert out == cleaner.process("".join(chunks)).lm_text
    assert "<PHONE>" in out


def test_window_still_splits_a_long_number_table():
    cleaner = TextCleaner(cfg=preset_config("training", check_ner_process=False))
    stream = WindowedStreamingRedactor(cleaner, max_buffer_chars=48, overlap_chars=4)
    assert stream.feed("1 " * 40) != ""  # a digit tail holds back at most a phone's length


# --- 2: threads sharing a handle build one cleaner ---

def test_handle_builds_one_cleaner_across_threads(monkeypatch):
    from sct.integrations import _runtime as rt

    built = []

    def slow_cleaner(cfg):
        time.sleep(0.2)
        built.append(cfg)
        return object()

    monkeypatch.setattr(rt, "TextCleaner", slow_cleaner)
    handle = rt.CleanerHandle(preset_config("training", check_ner_process=False))
    got = []
    threads = [threading.Thread(target=lambda: got.append(handle.get())) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(built) == 1
    assert len({id(c) for c in got}) == 1


# --- 3: the quantized cache is keyed by the resolved commit, not the branch name ---

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
