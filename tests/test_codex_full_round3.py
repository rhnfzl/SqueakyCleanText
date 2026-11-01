"""Codex 0.8.0 full review, round 3: ids 1, 3, 4, 5, 6, 7, 8 (2 is in test_adapters)."""

import json
from io import StringIO

import pytest

from sct import TextCleaner, TextCleanerConfig
from sct.cli import main
from sct.presets import preset_config
from sct.streaming import WindowedStreamingRedactor
from tests.conftest import TEST_NER_MODELS

FAST = preset_config("training", check_ner_process=False)


# --- 1: a marker that only appears after Unicode/HTML repair is still source text ---

@pytest.fixture(scope="module")
def ner_cleaner():
    return TextCleaner(cfg=preset_config("training", ner_models=TEST_NER_MODELS, ner_batch_size=8))


@pytest.mark.parametrize("text", ["I live in ＜LONDON＞.", "I live in &lt;LONDON&gt;."])
def test_normalised_source_marker_gets_no_ner_protection(ner_cleaner, text):
    assert "LONDON" not in ner_cleaner.process(text).lm_text
    assert all("LONDON" not in r.lm_text for r in ner_cleaner.process_batch([text, text]))


# --- 3: stream boundaries see contacts that only match once normalised ---

def test_window_never_splits_a_fullwidth_phone():
    cleaner = TextCleaner(cfg=preset_config(
        "training", check_ner_process=False, check_replace_numbers=False))
    text = "aaaa aaaa aaaa Call ０６ １２ ３４ ５６ ７８ tail"
    stream = WindowedStreamingRedactor(cleaner, max_buffer_chars=32, overlap_chars=16)
    assert stream.feed(text) + stream.finish() == cleaner.process(text).lm_text
    assert "<PHONE>" in cleaner.process(text).lm_text


# --- 4: a query or fragment straight after the host is part of the URL ---

@pytest.mark.parametrize("url", [
    "https://example.com?patient=AliceSmith", "https://example.com#token=SecretABC",
])
def test_bare_host_query_and_fragment_are_replaced(url):
    assert TextCleaner(cfg=FAST).process(f"see {url} now").lm_text == "see <URL> now"


def test_bare_trailing_question_mark_stays_punctuation():
    assert TextCleaner(cfg=FAST).process("Seen https://example.com? Yes").lm_text == (
        "Seen <URL>? Yes")


# --- 5: a saved tokenizer truncation caps the window budget ---

def test_window_budget_respects_saved_truncation():
    from tokenizers import Regex, Tokenizer, models, pre_tokenizers

    from sct.utils.onnx_pipeline import TokenizerWrapper

    tok = Tokenizer(models.WordLevel({"[UNK]": 0, "x": 1}, unk_token="[UNK]"))  # noqa: S106
    tok.pre_tokenizer = pre_tokenizers.Split(Regex("."), "isolated")
    tok.enable_truncation(max_length=20)
    assert TokenizerWrapper(tok, model_max_length=512).max_len_single_sentence == 18
    assert TokenizerWrapper(tok, model_max_length=10).max_len_single_sentence == 8


# --- 6: Haystack serializes nested config sets (custom stopwords) ---

def test_haystack_serializes_custom_stopwords():
    pytest.importorskip("haystack")
    from haystack import Pipeline

    from sct.integrations.haystack import SCTDocumentCleaner

    cfg = TextCleanerConfig(check_ner_process=False, custom_stopwords={"ENGLISH": frozenset({"foo"})})
    pipe = Pipeline()
    pipe.add_component("clean", SCTDocumentCleaner(config=cfg))
    restored = Pipeline.loads(pipe.dumps(), allowed_modules=["sct.integrations.haystack"])
    assert restored.get_component("clean").handle.config.custom_stopwords["ENGLISH"] == frozenset({"foo"})


# --- 7: reversible tokens never reuse a placeholder spelling from the source ---

def test_reversible_skips_literal_source_placeholders():
    cleaner = TextCleaner(cfg=preset_config(
        "training", check_ner_process=False, check_replace_html=False, replacement_mode="reversible"))
    text = "<EMAIL_0> alice@example.com"
    result = cleaner.process(text)
    assert result.lm_text == "<EMAIL_0> <EMAIL_1>"
    assert result.metadata["anon_map"].deanonymize(result.lm_text) == text


# --- 8: empty reversible input still carries a map, and the CLI writes it ---

def test_cli_reversible_empty_input_writes_empty_map(tmp_path):
    token_map = tmp_path / "tokens.json"
    exit_code = main(
        ["--no-ner", "--replacement-mode", "reversible", "--token-map", str(token_map)],
        stdin=StringIO(""), stdout=StringIO(),
    )
    assert exit_code == 0
    assert json.loads(token_map.read_text())["entries"] == []
