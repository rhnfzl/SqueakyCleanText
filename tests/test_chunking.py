"""Chunker regressions: hard size cap, CJK/Indic sentence ends, exact rejoin,
document-relative reversible map offsets. No model download needed."""

import threading
from types import SimpleNamespace

import pytest

from sct.utils.anonymization_map import AnonymizationMap
from sct.utils.ner import GeneralNER

BUDGET = 384  # _simple_chunk max_tokens -> 768 chars


def _ner():
    return object.__new__(GeneralNER)


@pytest.mark.parametrize("text", [
    "张" * 2841,                                   # Chinese run, no punctuation
    "東京都の山田太郎は会社員です。" * 250,           # Japanese, 。 without spaces
    "राम ने कहा कि वह कल आएगा। " * 60,             # Hindi, danda
    "intro\n" + "ক" * 900 + "\nend",              # Bengali run between newlines
    "தமிழ்" * 200 + "॥" + "தமிழ்" * 200,           # Tamil runs, double danda
])
def test_simple_chunk_caps_every_chunk(text):
    chunks = _ner()._simple_chunk(text, BUDGET)
    assert max(len(c) for c in chunks) <= BUDGET * 2
    assert "".join("".join(chunks).split()) == "".join(text.split())  # no content lost
    for chunk in chunks:
        assert chunk in text  # source slice, no invented separators


def test_simple_chunk_splits_after_cjk_and_danda_terminators():
    ner = _ner()
    assert ner._simple_chunk("甲。乙！丙？", 1) == ["甲。", "乙！", "丙？"]
    assert ner._simple_chunk("क। ख॥ ग", 1) == ["क।", "ख॥", "ग"]


def test_split_text_rejoins_cjk_without_spaces():
    def char_tokenizer(text, **kwargs):
        return SimpleNamespace(
            input_ids=list(text),
            offset_mapping=[(i, i + 1) for i in range(len(text))],
        )

    text = "甲。乙。" * 20
    chunks = _ner().split_text(text, 10, char_tokenizer)
    assert "".join(chunks) == text
    assert all(len(c) <= 10 and c.endswith("。") for c in chunks)


def test_split_text_offset_fallback_reserves_special_tokens():
    def bert_like(text, **kwargs):
        # [CLS] + one token per char + [SEP]
        return SimpleNamespace(
            input_ids=[0] * (len(text) + 2),
            offset_mapping=[(0, 0)] + [(i, i + 1) for i in range(len(text))] + [(0, 0)],
        )

    ner = _ner()
    chunks = ner.split_text("x" * 100, 20, bert_like)
    assert "".join(chunks) == "x" * 100
    assert all(ner._token_count(c, bert_like) <= 20 for c in chunks)


def test_split_text_offset_fallback_rechecks_retokenized_slices():
    def marker_tokenizer(text, **kwargs):
        # a word-start marker token on every non-empty text, like XLM-R's "_"
        return SimpleNamespace(
            input_ids=[0] * (len(text) + 1 if text else 0),
            offset_mapping=[(0, 0)] + [(i, i + 1) for i in range(len(text))],
        )

    ner = _ner()
    chunks = ner.split_text("x" * 100, 20, marker_tokenizer)
    assert "".join(chunks) == "x" * 100
    assert all(ner._token_count(c, marker_tokenizer) <= 20 for c in chunks)


def test_split_text_ignores_tokenizer_file_truncation():
    from tokenizers import Regex, Tokenizer, models, pre_tokenizers

    from sct.utils.onnx_pipeline import TokenizerWrapper

    tok = Tokenizer(models.WordLevel({"[UNK]": 0, "x": 1}, unk_token="[UNK]"))  # noqa: S106
    tok.pre_tokenizer = pre_tokenizers.Split(Regex("."), "isolated")
    tok.enable_truncation(max_length=20)  # as shipped in some tokenizer.json files
    wrapper = TokenizerWrapper(tok, model_max_length=20)

    text = "x" * 100 + "Alice"
    chunks = _ner().split_text(text, 20, wrapper)
    assert "".join(chunks) == text
    assert chunks[-1].endswith("Alice") and len(chunks[-1]) <= 20
    assert tok.truncation is not None  # inference tokenizer keeps its own setting


class _FakeGLiNER:
    """Finds every occurrence of NAMES in a chunk, like a perfect detector."""

    max_context_length = 100  # -> 90 tokens -> 180-char chunks
    labels = ["person"]
    label_map = {"person": "PER"}
    names = ("王小明", "राहुल", "ரவி")

    def __call__(self, texts):
        out = []
        for text in texts:
            found = []
            for name in self.names:
                start = text.find(name)
                while start != -1:
                    found.append({"entity_group": "PER", "score": 0.99, "word": name,
                                  "start": start, "end": start + len(name)})
                    start = text.find(name, start + 1)
            out.append(found)
        return out


def _gliner_ner(mode):
    ner = _ner()
    ner._ner_backend = "gliner"
    ner._gliner_pipe = _FakeGLiNER()
    ner._replacement_mode = mode
    ner._synthetic_replacer = None
    ner._inference_locks = {}
    ner._locks_lock = threading.Lock()
    ner.tokenizer = None
    return ner


@pytest.mark.parametrize("text, name", [
    ("这是一段很长的中文文本。" * 60 + "联系人王小明。", "王小明"),
    ("東京都の会社です。" * 80 + "王小明に連絡。", "王小明"),
    ("यह एक लंबा वाक्य है। " * 40 + "संपर्क राहुल है।", "राहुल"),
    ("இது ஒரு நீண்ட வாக்கியம். " * 40 + "தொடர்பு ரவி.", "ரவி"),
])
def test_reversible_map_offsets_are_document_relative(text, name):
    anon_map = AnonymizationMap()
    result = _gliner_ner("reversible").ner_process_detailed(
        text, positional_tags=("PER",), ner_confidence_threshold=0.5, anon_map=anon_map,
    )
    assert name not in result.text
    entry = anon_map.entries[-1]
    assert entry.start > 180  # past the first window
    assert text[entry.start:entry.end] == name
    finding = result.findings[-1].entity
    assert (finding.start, finding.end) == (entry.start, entry.end)
    assert anon_map.deanonymize(result.text) == text


@pytest.mark.parametrize("text", [
    " " * 400 + "王小明",                  # hard cuts leave whitespace-only chunks
    "王小明" + " " * 400 + "王小明",       # chunk ends and next starts in one gap
    "\n" + "x" * 179 + " " * 300 + "王小明",
])
def test_whitespace_at_hard_cuts_keeps_alignment(text):
    anon_map = AnonymizationMap()
    result = _gliner_ner("reversible").ner_process_detailed(
        text, positional_tags=("PER",), ner_confidence_threshold=0.5, anon_map=anon_map,
    )
    assert "王小明" not in result.text
    assert all(text[f.entity.start:f.entity.end] == "王小明" for f in result.findings)
    assert anon_map.deanonymize(result.text) == text


def test_align_chunk_maps_normalized_whitespace():
    source_chunk, start, offsets = GeneralNER._align_chunk("x  a\t\tb", "a b", 0)
    assert (source_chunk, start) == ("a\t\tb", 3)
    assert offsets[2] == 3 and offsets[3] == 4
