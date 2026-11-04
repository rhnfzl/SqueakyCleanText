"""0.8.0 full test round 2 (Linux run A, Mac round 2) bug fixes."""

import types

import pytest

from sct import TextCleaner, TextCleanerConfig
from tests.conftest import TEST_NER_MODELS


# --- Linux 2: Lingua confidence is rounded so workers and in-process agree ---

def test_language_confidence_is_rounded():
    cleaner = TextCleaner(TextCleanerConfig(check_ner_process=False, check_detect_language=True))
    lang = types.SimpleNamespace(name="ENGLISH")
    noisy = [types.SimpleNamespace(language=lang, value=0.5735166974819533)]
    cleaner._detector = types.SimpleNamespace(compute_language_confidence_values=lambda t: noisy)
    assert cleaner._detect_language("x")[1] == 0.573517


# --- Linux 5: importing sct does not pull presidio (and through it torch/GLiNER) ---

def test_import_sct_does_not_load_presidio():
    import subprocess
    import sys
    code = "import sct, sys; print('presidio_anonymizer' in sys.modules)"
    out = subprocess.run(  # noqa: S603 - fixed argv, own interpreter
        [sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


# --- Mac C: isolated-symbol removal keeps brackets and punctuation glued to a word or placeholder ---

@pytest.mark.parametrize("text, expected", [
    ("He was (born 1980) here.", "He was (born <YEAR>) here."),
    ("He was (born 1980).", "He was (born <YEAR>)."),
    ("He was (born in Paris). Ok", "He was (born in Paris). Ok"),
    ("Voor vragen, bel +31 20 123 4567.", "Voor vragen, bel <PHONE>."),
    ("end , start", "end start"),
])
def test_isolated_symbols_need_space_on_both_sides(text, expected):
    cleaner = TextCleaner(TextCleanerConfig(check_ner_process=False))
    assert cleaner.process(text).lm_text == expected


# --- Mac A: JSON keys get regex PII only by default; NER on keys is opt-in ---

def test_json_keys_skip_ner_by_default():
    from sct.adapters import _pii_only_cleaner
    cleaner = TextCleaner(TextCleanerConfig(ner_models=TEST_NER_MODELS))
    assert _pii_only_cleaner(cleaner, ner=False).cfg.check_ner_process is False
    assert _pii_only_cleaner(cleaner, ner=True).cfg.check_ner_process is True


def test_process_json_default_keys_do_not_reach_ner(monkeypatch):
    from sct import adapters
    seen = []
    real = adapters._pii_only_cleaner
    monkeypatch.setattr(adapters, "_pii_only_cleaner",
                        lambda c, ner=True: seen.append(ner) or real(c, ner))
    cleaner = TextCleaner(TextCleanerConfig(check_ner_process=False))
    out = adapters.process_json(cleaner, {"user": "x", "to a@b.com": 1})
    assert seen == [False]
    assert list(out.value) == ["user", "to <EMAIL>"]
    TextCleanerConfig(json_key_cleaning="pii_ner")  # opt-in mode is valid


def test_cli_reports_duplicate_json_key_as_usage_error(monkeypatch):
    import io
    from sct import cli
    monkeypatch.setattr(cli, "process_json", lambda *a: (_ for _ in ()).throw(
        ValueError("Cleaning JSON keys produced a duplicate key")))
    with pytest.raises(SystemExit) as exc:
        cli.main(["--json-input", "--no-ner"], stdin=io.StringIO('{"a": 1}'), stdout=io.StringIO())
    assert exc.value.code == 2


# --- Mac B: NER does not rewrite the synthetic fake a regex step inserted ---

def test_ner_leaves_regex_synthetic_fakes_alone():
    from sct.utils.synthetic import SyntheticReplacer
    from tests.test_chunking import _FakeRunDetector, _word_ner

    ner = _word_ner(0, _FakeRunDetector(), mode="synthetic")
    ner._synthetic_replacer = SyntheticReplacer()
    ner._synthetic_replacer.reset_cache()
    fake = ner._synthetic_replacer.get_replacement("EMAIL", "anna@beispiel.de")
    # The fake detector tags capitalised words, like the real models tag pieces of the fake email.
    fake = fake[0].upper() + fake[1:]
    ner._synthetic_replacer._get_cache()[("EMAIL", "anna@beispiel.de")] = fake
    result = ner.ner_process_detailed(
        f"Write to {fake} today, Angela.", positional_tags=("PER",), ner_confidence_threshold=0.5)
    assert fake in result.text
    assert "Angela" not in result.text


# --- Linux 1: a worker rebuilding the cleaner makes no Hub request from a warm cache ---

def test_onnx_load_from_warm_cache_makes_no_hub_request(monkeypatch):
    import huggingface_hub.file_download as fd
    from sct.utils.onnx_pipeline import load_onnx_ner_model

    model = TEST_NER_MODELS["ENGLISH"]
    pipe, _ = load_onnx_ner_model(model)  # warms the cache, records the missing model.onnx_data
    calls = []
    monkeypatch.setattr(fd, "get_hf_file_metadata", lambda *a, **k: calls.append(a) or 1 / 0)
    load_onnx_ner_model(model, revision=pipe.revision)
    assert calls == []


def test_forked_handle_pins_the_parent_commits(monkeypatch):
    import pickle

    from sct.integrations import _runtime as rt

    handle = rt.CleanerHandle(TextCleanerConfig(ner_models=TEST_NER_MODELS))
    handle.get().GeneralNER.load_language("ENGLISH")
    commit = handle.get().GeneralNER.provenance["ENGLISH"]["revision"]
    copied = pickle.loads(pickle.dumps(handle))  # noqa: S301 - own object
    assert copied.config.ner_model_revisions["ENGLISH"] == commit
    monkeypatch.setattr(rt.os, "getpid", lambda: -1)  # as in a forked child
    assert handle.get().cfg.ner_model_revisions["ENGLISH"] == commit


# --- Mac D: WordPiece entities cover the whole word, not "John Do" + a visible "e" ---

def test_wordpiece_entity_ends_on_a_whole_word():
    from sct.utils.onnx_pipeline import load_onnx_ner_model

    pipe, _ = load_onnx_ner_model(TEST_NER_MODELS["ENGLISH"])
    text = "Contact John Doe today."
    spans = [(e["start"], e["end"]) for e in pipe(text)[0] if e["entity_group"] == "PER"]
    assert spans and all(text[end] in " ." for _, end in spans)
    assert any(text[s:e].endswith("Doe") for s, e in spans)


# --- Mac F: an MCP root that does not exist fails at start-up, not on every call ---

def test_mcp_missing_root_fails_at_start(tmp_path):
    pytest.importorskip("fastmcp")
    from sct.mcp_server import create_mcp_server

    with pytest.raises(ValueError, match="SCT_MCP_ROOT"):
        create_mcp_server(TextCleaner(TextCleanerConfig(check_ner_process=False)),
                          root=str(tmp_path / "missing"))
