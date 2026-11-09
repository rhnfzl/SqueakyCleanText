"""Chunker regressions: hard size cap, CJK/Indic sentence ends, exact rejoin,
document-relative reversible map offsets. No model download needed."""

import re
from types import SimpleNamespace

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.helpers import bare_ner, FakeRunDetector, gliner_ner, numbered_words, set_cfg, windows_for, word_ner
from sct.utils.ner_chunking import align_chunk, simple_chunk, split_text, token_count
from sct.utils.anonymization_map import AnonymizationMap
from sct.config import TextCleanerConfig
from sct.utils.ner import GeneralNER
from tests.conftest import TEST_NER_MODELS, TEST_NER_REVISIONS

BUDGET = 384  # simple_chunk max_tokens -> 768 chars


def _ner():
    return bare_ner()


@pytest.mark.parametrize("text", [
    "张" * 2841,                                   # Chinese run, no punctuation
    "東京都の山田太郎は会社員です。" * 250,           # Japanese, 。 without spaces
    "राम ने कहा कि वह कल आएगा। " * 60,             # Hindi, danda
    "intro\n" + "ক" * 900 + "\nend",              # Bengali run between newlines
    "தமிழ்" * 200 + "॥" + "தமிழ்" * 200,           # Tamil runs, double danda
])
def test_simple_chunk_caps_every_chunk(text):
    chunks = simple_chunk(text, BUDGET)
    assert max(len(c) for c in chunks) <= BUDGET * 2
    assert "".join("".join(chunks).split()) == "".join(text.split())  # no content lost
    for chunk in chunks:
        assert chunk in text  # source slice, no invented separators


def test_simple_chunk_splits_after_cjk_and_danda_terminators():
    assert simple_chunk("甲。乙！丙？", 1) == ["甲。", "乙！", "丙？"]
    assert simple_chunk("क। ख॥ ग", 1) == ["क।", "ख॥", "ग"]


def test_split_text_rejoins_cjk_without_spaces():
    def char_tokenizer(text, **kwargs):
        return SimpleNamespace(
            input_ids=list(text),
            offset_mapping=[(i, i + 1) for i in range(len(text))],
        )

    text = "甲。乙。" * 20
    chunks = split_text(text, 10, char_tokenizer)
    assert "".join(chunks) == text
    assert all(len(c) <= 10 and c.endswith("。") for c in chunks)


def test_split_text_offset_fallback_reserves_special_tokens():
    def bert_like(text, **kwargs):
        # [CLS] + one token per char + [SEP]
        return SimpleNamespace(
            input_ids=[0] * (len(text) + 2),
            offset_mapping=[(0, 0)] + [(i, i + 1) for i in range(len(text))] + [(0, 0)],
        )

    chunks = split_text("x" * 100, 20, bert_like)
    assert "".join(chunks) == "x" * 100
    assert all(token_count(c, bert_like) <= 20 for c in chunks)


def test_split_text_offset_fallback_rechecks_retokenized_slices():
    def marker_tokenizer(text, **kwargs):
        # a word-start marker token on every non-empty text, like XLM-R's "_"
        return SimpleNamespace(
            input_ids=[0] * (len(text) + 1 if text else 0),
            offset_mapping=[(0, 0)] + [(i, i + 1) for i in range(len(text))],
        )

    chunks = split_text("x" * 100, 20, marker_tokenizer)
    assert "".join(chunks) == "x" * 100
    assert all(token_count(c, marker_tokenizer) <= 20 for c in chunks)


def test_split_text_ignores_tokenizer_file_truncation():
    from tokenizers import Regex, Tokenizer, models, pre_tokenizers

    from sct.utils.onnx_pipeline import TokenizerWrapper

    tok = Tokenizer(models.WordLevel({"[UNK]": 0, "x": 1}, unk_token="[UNK]"))  # noqa: S106
    tok.pre_tokenizer = pre_tokenizers.Split(Regex("."), "isolated")
    tok.enable_truncation(max_length=20)  # as shipped in some tokenizer.json files
    wrapper = TokenizerWrapper(tok, model_max_length=20)

    text = "x" * 100 + "Alice"
    chunks = split_text(text, 20, wrapper)
    assert "".join(chunks) == text
    assert chunks[-1].endswith("Alice") and len(chunks[-1]) <= 20
    assert tok.truncation is not None  # inference tokenizer keeps its own setting


def test_split_text_ignores_tokenizer_file_padding():
    from tokenizers import Regex, Tokenizer, models, pre_tokenizers

    from sct.utils.onnx_pipeline import TokenizerWrapper

    tok = Tokenizer(models.WordLevel({"[UNK]": 0, "x": 1}, unk_token="[UNK]"))  # noqa: S106
    tok.pre_tokenizer = pre_tokenizers.Split(Regex("."), "isolated")
    tok.enable_padding(length=64)  # fixed padding, as in some tokenizer.json files
    wrapper = TokenizerWrapper(tok, model_max_length=20)

    assert len(wrapper("xx").input_ids) == 2
    assert split_text("x" * 30, 20, wrapper) == ["x" * 20, "x" * 10]


@pytest.mark.parametrize("text, name", [
    ("这是一段很长的中文文本。" * 60 + "联系人王小明。", "王小明"),
    ("東京都の会社です。" * 80 + "王小明に連絡。", "王小明"),
    ("यह एक लंबा वाक्य है। " * 40 + "संपर्क राहुल है।", "राहुल"),
    ("இது ஒரு நீண்ட வாக்கியம். " * 40 + "தொடர்பு ரவி.", "ரவி"),
])
def test_reversible_map_offsets_are_document_relative(text, name):
    anon_map = AnonymizationMap()
    result = gliner_ner("reversible").ner_process_detailed(
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
    result = gliner_ner("reversible").ner_process_detailed(
        text, positional_tags=("PER",), ner_confidence_threshold=0.5, anon_map=anon_map,
    )
    assert "王小明" not in result.text
    assert all(text[f.entity.start:f.entity.end] == "王小明" for f in result.findings)
    assert anon_map.deanonymize(result.text) == text


def test_align_chunk_maps_normalized_whitespace():
    source_chunk, start, offsets = align_chunk("x  a\t\tb", "a b", 0)
    assert (source_chunk, start) == ("a\t\tb", 3)
    assert offsets[2] == 3 and offsets[3] == 4


# --- ner_chunk_overlap: token windows, document-level merge, replace once ---

def test_overlap_finds_name_split_between_chunks():
    # Cores of 12 (overlap 0) and 8 (overlap 2) words both end right after "John".
    text = numbered_words(40, {23: "John", 24: "Smith"})
    old = word_ner(0, FakeRunDetector(min_words=2)).ner_process_detailed(
        text, positional_tags=("PER",), ner_confidence_threshold=0.5)
    assert "John Smith" in old.text and not old.findings  # the 0.7.1 limit

    result = word_ner(2, FakeRunDetector(min_words=2)).ner_process_detailed(
        text, positional_tags=("PER",), ner_confidence_threshold=0.5)
    assert result.text == text.replace("John Smith", "<PERSON>")
    [finding] = result.findings
    assert text[finding.entity.start:finding.entity.end] == "John Smith"


def test_overlap_keeps_one_copy_of_a_span_seen_by_two_windows():
    # Window 3 (core w16..w23) also sees "Paul Smith" in its left overlap.
    text = numbered_words(40, {13: "John", 14: "Paul", 15: "Smith"})
    anon_map = AnonymizationMap()
    ner = word_ner(2, FakeRunDetector(min_words=2), mode="reversible")
    result = ner.ner_process_detailed(
        text, positional_tags=("PER",), ner_confidence_threshold=0.5, anon_map=anon_map)
    [finding] = result.findings
    assert finding.entity.text == "John Paul Smith"
    [entry] = anon_map.entries
    assert text[entry.start:entry.end] == "John Paul Smith"
    assert anon_map.deanonymize(result.text) == text


def test_overlap_windows_fit_the_budget_at_hard_cuts():
    def bert_like(text, **kwargs):
        return SimpleNamespace(
            input_ids=[0] * (len(text) + 2),
            offset_mapping=[(0, 0)] + [(i, i + 1) for i in range(len(text))] + [(0, 0)],
        )

    ner = word_ner(3, FakeRunDetector(), budget=20)
    ner.tokenizer = bert_like
    windows = windows_for(ner, "x" * 100)
    assert all(token_count(w[0], bert_like) <= 20 for w in windows)
    assert any(w[1] > 0 and len(w[0]) > 14 for w in windows)  # overlap was added
    owned = [(w[3], w[4]) for w in windows]
    assert owned[0][0] == 0 and all(a[1] == b[0] for a, b in zip(owned, owned[1:]))

    gliner = gliner_ner("placeholder")
    set_cfg(gliner, ner_chunk_overlap=10)
    budget = gliner._chunk_budget()
    windows = windows_for(gliner, "张" * 2841)
    assert len(windows) > 1 and all(len(w[0]) <= budget * 2 for w in windows)
    assert "".join(w[0][w[3] - w[1]:w[4] - w[1]] for w in windows) == "张" * 2841


def _finalize_071(ner, text, threshold):
    """Frozen 0.7.1 path: vote and replace chunk by chunk (placeholder mode)."""
    chunks = split_text(text, ner.min_token_length, ner.tokenizer)
    out, findings, search_start = [], [], 0
    for chunk, results in zip(chunks, ner._detect(chunks, [None] * len(chunks), ("PER",))):
        source_chunk, chunk_start, offsets = align_chunk(text, chunk, search_start)
        out.append(text[search_start:chunk_start])
        search_start = chunk_start + len(source_chunk)
        voted = [
            {**item, "start": offsets[item["start"]], "end": offsets[item["end"]]}
            for item in ner.vote_entities(results, threshold)
        ]
        out.append(ner.anonymize_text(source_chunk, voted) if voted else source_chunk)
        findings += [(i["start"] + chunk_start, i["end"] + chunk_start) for i in voted]
    out.append(text[search_start:])
    return "".join(out), findings


@settings(max_examples=60, deadline=None)
@given(
    words=st.lists(st.sampled_from(["lorem", "ipsum", "dolor", "Ann,", "Bob.", "Eve,"]),
                   min_size=1, max_size=60),
    seps=st.lists(st.sampled_from([" ", "  ", "\n", "\n\n", ". ", "! "]), min_size=60, max_size=60),
)
def test_overlap_zero_matches_071_per_chunk_replacement(words, seps):
    text = "".join(word + sep for word, sep in zip(words, seps))
    ner = word_ner(0, FakeRunDetector(), budget=6)
    result = ner.ner_process_detailed(text, positional_tags=("PER",), ner_confidence_threshold=0.5)
    expected_text, expected_spans = _finalize_071(ner, text, 0.5)
    assert result.text == expected_text
    assert [(f.entity.start, f.entity.end) for f in result.findings] == expected_spans


def test_whole_document_replacement_merges_names_across_chunk_boundary():
    # Deliberate 0.8.0 change: Presidio merges same-type spans split only by
    # spaces, now also when a chunk boundary falls between them.
    text = numbered_words(30, {11: "Ann", 12: "Bob"})
    ner = word_ner(0, FakeRunDetector())
    result = ner.ner_process_detailed(text, positional_tags=("PER",), ner_confidence_threshold=0.5)
    assert result.text == text.replace("Ann Bob", "<PERSON>")
    assert _finalize_071(ner, text, 0.5)[0] == text.replace("Ann Bob", "<PERSON> <PERSON>")
    assert len(result.findings) == 2


def test_overlap_config_validation():
    with pytest.raises(ValueError, match="ner_chunk_overlap"):
        TextCleanerConfig(check_ner_process=False, ner_chunk_overlap=-1)
    assert TextCleanerConfig(check_ner_process=False).ner_chunk_overlap == 0
    cfg = TextCleanerConfig(ner_chunk_overlap=16, ner_batch_size=2, check_ner_process=False)
    assert (cfg.ner_chunk_overlap, cfg.ner_batch_size) == (16, 2)


def test_overlap_with_real_tokenizer():
    with pytest.raises(ValueError, match="ner_chunk_overlap"):
        GeneralNER(TextCleanerConfig(ner_models=TEST_NER_MODELS,
                                     ner_model_revisions=TEST_NER_REVISIONS, ner_chunk_overlap=200))
    ner = GeneralNER(TextCleanerConfig(ner_models=TEST_NER_MODELS,
                                       ner_model_revisions=TEST_NER_REVISIONS, ner_chunk_overlap=32,
                                       replacement_mode="reversible"))
    text = "Marie Curie worked in Paris with Pierre Curie. " * 80
    windows = windows_for(ner, text)
    assert len(windows) > 1
    assert all(token_count(w[0], ner.tokenizer) <= ner.min_token_length for w in windows)
    anon_map = AnonymizationMap()
    result = ner.ner_process_detailed(text, ("PER", "LOC"), 0.85, "ENGLISH", anon_map)
    assert "Paris" not in result.text
    assert len({(f.entity.start, f.entity.end) for f in result.findings}) == len(result.findings)
    assert anon_map.deanonymize(result.text) == text


def test_ner_span_inside_regex_placeholder_is_dropped():
    # Old code: a model tagging "PHONE" in "<PHONE>" gave "call <<PERSON>>" (0.7.1 too).
    detector = FakeRunDetector()
    detector.pattern = re.compile(r"[A-Z]{2,}|Ann")
    ner = word_ner(0, detector)
    result = ner.ner_process_detailed(
        "call <PHONE> or <EMAIL_0>, ask Ann", positional_tags=("PER",), ner_confidence_threshold=0.5)
    assert result.text == "call <PHONE> or <EMAIL_0>, ask <PERSON>"
    assert [f.entity.text for f in result.findings] == ["Ann"]
