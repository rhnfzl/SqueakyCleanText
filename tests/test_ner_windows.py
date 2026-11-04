"""NER windows: size per model tokenizer, overlap voting across neighbours, and model routes."""

import sys
import types
from types import SimpleNamespace

import pytest

from sct import TextCleaner, TextCleanerConfig
from sct.presets import preset_config
from sct.privacy import EntityRule, PrivacyPolicy
from sct.utils.ner import GeneralNER
from sct.utils.ner_chunking import token_count

from tests.helpers import (
    both_paths, FakeRunDetector, gliner_ner, numbered_words, route_models, set_cfg, windows_for, word_ner, WordCountTokenizer,
)


# --- models_for grew a list ensemble route with GLINER on every call ---
def test_models_for_leaves_a_list_route_unchanged(monkeypatch):
    fake = types.ModuleType("sct.utils.gliner_adapter")

    class GLiNERAdapter:
        _onnx = False

        def __init__(self, **kwargs):
            pass

    fake.GLiNERAdapter = GLiNERAdapter
    monkeypatch.setitem(sys.modules, "sct.utils.gliner_adapter", fake)
    route = {"ENGLISH": ["ENGLISH", "MULTILINGUAL"]}
    ner = TextCleaner(cfg=TextCleanerConfig(
        ner_backend="ensemble_onnx", gliner_model="x/y", ner_ensemble=route)).GeneralNER
    ner.provenance["GLINER"] = {"key": "GLINER", "id": "x/y", "revision": None, "provider": None}
    first = ner.models_for("ENGLISH")
    assert ner.models_for("ENGLISH") == first
    assert route["ENGLISH"] == ["ENGLISH", "MULTILINGUAL"]
    assert list(ner._get_ensemble_keys("ENGLISH")) == ["ENGLISH", "MULTILINGUAL"]


# --- a saved tokenizer truncation caps the window budget ---
def test_window_budget_respects_saved_truncation():
    from tokenizers import Regex, Tokenizer, models, pre_tokenizers

    from sct.utils.onnx_pipeline import TokenizerWrapper

    tok = Tokenizer(models.WordLevel({"[UNK]": 0, "x": 1}, unk_token="[UNK]"))  # noqa: S106
    tok.pre_tokenizer = pre_tokenizers.Split(Regex("."), "isolated")
    tok.enable_truncation(max_length=20)
    assert TokenizerWrapper(tok, model_max_length=512).max_len_single_sentence == 18
    assert TokenizerWrapper(tok, model_max_length=10).max_len_single_sentence == 8


# --- a route without the primary model is not split by the primary's count ---
def test_route_without_primary_counts_with_its_own_tokenizer():
    ner = word_ner(0, FakeRunDetector(min_words=2), budget=12)
    heavy = WordCountTokenizer(2)
    ner.tokenizer = heavy  # primary (ENGLISH), 2 tokens per word
    route_models(ner, {"ENGLISH": heavy, "HINDI": WordCountTokenizer(1)}, {"ENGLISH": ("ENGLISH",), "HINDI": ("HINDI",)})
    text = "a b c d e John Smith f g h"  # 10 HINDI tokens, inside the budget of 12
    assert [w[0] for w in windows_for(ner, text, "HINDI")] == [text]
    kw = {"positional_tags": ("PER",), "ner_confidence_threshold": 0.5}
    masked = text.replace("John Smith", "<PERSON>")
    assert ner.ner_process_detailed(text, language="HINDI", **kw).text == masked
    assert [r.text for r in ner.ner_process_batch_detailed([text], ["HINDI"], **kw)] == [masked]
    # A route that does use the primary still counts with it.
    assert len(windows_for(ner, text, "ENGLISH")) == 2


# --- overlap offsets come from the route's own tokenizer ---
def test_overlap_offsets_use_route_tokenizer():
    def one_span(text, **kwargs):  # primary: one token over the whole text
        spans = [(0, len(text))]
        return SimpleNamespace(input_ids=spans, offset_mapping=spans)

    one_span.max_len_single_sentence = 1000
    ner = word_ner(2, FakeRunDetector(min_words=2), budget=12)
    ner.tokenizer = one_span
    route_models(ner, {"ENGLISH": one_span, "HINDI": WordCountTokenizer(1)}, {"ENGLISH": ("ENGLISH",), "HINDI": ("HINDI",)})
    text = "a b c d e f g John Smith h i j k l m n"
    windows = [w[0] for w in windows_for(ner, text, "HINDI")]
    assert all(len(w.split()) <= 12 for w in windows)
    assert any("John Smith" in w for w in windows)
    kw = {"positional_tags": ("PER",), "ner_confidence_threshold": 0.5}
    masked = text.replace("John Smith", "<PERSON>")
    assert ner.ner_process_detailed(text, language="HINDI", **kw).text == masked
    assert [r.text for r in ner.ner_process_batch_detailed([text], ["HINDI"], **kw)] == [masked]


# --- merged language_source follows the chosen language's leaves ---
def test_merged_language_source_follows_chosen_language():
    from sct.adapters import process_json
    cleaner = TextCleaner(cfg=preset_config("training", check_ner_process=False,
                                            language="ENGLISH"))
    signals = process_json(cleaner, [" " * 15, "Hello world!", "Hello world!"]).signals
    assert (signals["language"], signals["language_source"]) == ("ENGLISH", "pinned")


class _ContextDetector(FakeRunDetector):
    """Finds "John Smith" only when "has arrived" follows it in the same window."""

    def __call__(self, texts, batch_size=1):
        return [
            [{"entity_group": "PER", "score": 0.99, "word": "John Smith",
              "start": i, "end": i + 10}]
            if (i := text.find("John Smith has arrived")) != -1 else []
            for text in texts
        ]


def test_overlap_keeps_a_detection_only_a_neighbour_window_made():
    text = numbered_words(40, {7: "John", 8: "Smith", 9: "has", 10: "arrived"})
    no_overlap = word_ner(0, _ContextDetector()).ner_process_detailed(
        text, positional_tags=("PER",), ner_confidence_threshold=0.5)
    assert "John Smith" not in no_overlap.text
    result = word_ner(2, _ContextDetector()).ner_process_detailed(
        text, positional_tags=("PER",), ner_confidence_threshold=0.5)
    assert result.text == text.replace("John Smith", "<PERSON>")
    assert len(result.findings) == 1


@pytest.mark.parametrize("overlap", [0, 2])
def test_windows_fit_every_model_tokenizer(overlap):
    ner = word_ner(overlap, FakeRunDetector(), budget=12)
    heavy = WordCountTokenizer(2)
    route_models(ner, {"ENGLISH": WordCountTokenizer(1), "MULTILINGUAL": heavy})
    windows = windows_for(ner, numbered_words(60, {}))
    assert all(token_count(w[0], heavy) <= 12 for w in windows)


def test_window_budget_honours_a_smaller_model_limit():
    ner = word_ner(0, FakeRunDetector(), budget=12)
    small = WordCountTokenizer(1)
    small.max_len_single_sentence = 5
    route_models(ner, {"ENGLISH": WordCountTokenizer(1), "DUTCH": small})
    assert ner._chunk_budget() == 5


class _ScoredDetector(FakeRunDetector):
    """Finds "John Smith"; 0.99 when "has arrived" follows in the window, else 0.60."""

    def __call__(self, texts, batch_size=1):
        return [
            [{"entity_group": "PER", "score": 0.99 if "John Smith has arrived" in text else 0.60,
              "word": "John Smith", "start": i, "end": i + 10}]
            if (i := text.find("John Smith")) != -1 else []
            for text in texts
        ]


def test_rejected_owner_does_not_hide_an_eligible_neighbour():
    # The owning window sees "John Smith has" (0.60), its neighbour "... has arrived" (0.99).
    text = numbered_words(40, {7: "John", 8: "Smith", 9: "has", 10: "arrived"})
    expected = text.replace("John Smith", "<PERSON>")
    ner = word_ner(2, _ScoredDetector())
    single = ner.ner_process_detailed(text, positional_tags=("PER",), ner_confidence_threshold=0.85)
    assert single.text == expected and len(single.findings) == 1
    batch = ner.ner_process_batch_detailed(
        [text, text], [None, None], positional_tags=("PER",), ner_confidence_threshold=0.85)
    assert [r.text for r in batch] == [expected, expected]


def _two_language_ner(overlap=0):
    ner = word_ner(overlap, FakeRunDetector(min_words=2), budget=12)
    route_models(ner, {"ENGLISH": WordCountTokenizer(1)}, {"ENGLISH": ("ENGLISH",), "DUTCH": ("DUTCH",)})
    return ner


def test_unrelated_loaded_model_does_not_change_a_document():
    # A heavier Dutch tokenizer halves the window and would split "John | Smith".
    text = numbered_words(30, {5: "John", 6: "Smith"})
    expected = text.replace("John Smith", "<PERSON>")
    ner = _two_language_ner()
    kwargs = {"positional_tags": ("PER",), "ner_confidence_threshold": 0.5}
    assert ner.ner_process_detailed(text, language="ENGLISH", **kwargs).text == expected
    route_models(ner, {"ENGLISH": WordCountTokenizer(1), "DUTCH": WordCountTokenizer(2)})
    assert ner.ner_process_detailed(text, language="ENGLISH", **kwargs).text == expected
    batch = ner.ner_process_batch_detailed([text, text], ["ENGLISH", "DUTCH"], **kwargs)
    assert batch[0].text == expected
    # The Dutch document is still sized for the Dutch tokenizer.
    windows = windows_for(ner, text, "DUTCH")
    assert all(token_count(w[0], WordCountTokenizer(2)) <= 12 for w in windows)


def test_overlap_rechecked_after_a_smaller_model_loads():
    ner = _two_language_ner(overlap=2)  # 2 * 2 < 12 // 2 passes at construction
    small = WordCountTokenizer(1)
    small.max_len_single_sentence = 5
    route_models(ner, {"ENGLISH": WordCountTokenizer(1), "DUTCH": small})
    text = numbered_words(30, {15: "John", 16: "Smith"})
    kwargs = {"positional_tags": ("PER",), "ner_confidence_threshold": 0.5}
    with pytest.raises(ValueError, match="ner_chunk_overlap"):
        ner.ner_process_detailed(text, language="DUTCH", **kwargs)
    with pytest.raises(ValueError, match="ner_chunk_overlap"):
        ner.ner_process_batch_detailed([text, text], ["ENGLISH", "DUTCH"], **kwargs)
    assert ner.ner_process_detailed(text, language="ENGLISH", **kwargs).findings


class _AliceDetector(FakeRunDetector):
    """Finds "Alice"; 0.99 when "alpha" is in the window, else 0.60."""

    def __call__(self, texts, batch_size=1):
        return [
            [{"entity_group": "PER", "score": 0.99 if "alpha" in text else 0.60,
              "word": "Alice", "start": i, "end": i + 5}]
            if (i := text.find("Alice")) != -1 else []
            for text in texts
        ]


def test_repeated_neighbour_copies_are_one_model_vote():
    # Short "Alice." core: both neighbours see Alice (0.99 and 0.60), the owner 0.60.
    text = "alpha w1 w2 w3 w4 w5 w6 w7. Alice. Beta b1 b2 b3 b4 b5 b6 b7."
    for result in both_paths(word_ner(2, _AliceDetector()), text):
        assert result.text == text.replace("Alice", "<PERSON>")
        assert len(result.findings) == 1


class _FallbackContextDetector(FakeRunDetector):
    """"John Smith" at 0.99 PER when "has arrived" follows; otherwise *fallback*."""

    def __init__(self, fallback):
        self.fallback = fallback

    def __call__(self, texts, batch_size=1):
        out = []
        for text in texts:
            i = text.find("John Smith")
            if i == -1:
                out.append([])
            elif "John Smith has arrived" in text:
                out.append([{"entity_group": "PER", "score": 0.99, "word": "John Smith",
                             "start": i, "end": i + 10}])
            else:
                group, score, word = self.fallback
                out.append([{"entity_group": group, "score": score, "word": word,
                             "start": i, "end": i + len(word)}])
        return out


TEXT = numbered_words(40, {7: "John", 8: "Smith", 9: "has", 10: "arrived"})


def test_policy_threshold_sees_the_neighbours_score():
    # Owner: 0.86 passes the detector but not the policy; neighbour: 0.99.
    policy = PrivacyPolicy("p", "1", {"PERSON": EntityRule("redact", 0.95)})
    ner = word_ner(2, _FallbackContextDetector(("PER", 0.86, "John Smith")))
    for result in both_paths(ner, TEXT, privacy_policy=policy):
        assert result.text == TEXT.replace("John Smith", "<REDACTED>")


def test_kept_owner_span_does_not_hide_a_redacted_neighbour():
    # Owner: "John Smith has" as LOC (keep); neighbour: "John Smith" as PER (redact).
    policy = PrivacyPolicy("p", "1", {"PERSON": EntityRule("redact"),
                                      "LOCATION": EntityRule("keep")})
    ner = word_ner(2, _FallbackContextDetector(("LOC", 0.99, "John Smith has")))
    for result in both_paths(ner, TEXT, privacy_policy=policy):
        assert result.text == TEXT.replace("John Smith", "<REDACTED>")
    # Without a policy the covering owner span is replaced and the copy dropped.
    for result in both_paths(ner, TEXT):
        assert result.text == TEXT.replace("John Smith has", "<LOCATION>")
        assert len(result.findings) == 1


class _FakeTok:
    max_len_single_sentence = 510


def test_tokenizer_is_published_before_its_pipeline(monkeypatch):
    monkeypatch.setattr("sct.utils.onnx_pipeline.load_onnx_ner_model", lambda *a, **k: (object(), _FakeTok()))
    ner = GeneralNER(TextCleanerConfig(ner_models={"ENGLISH": "a", "MULTILINGUAL": "b", "DUTCH": "d", "GERMAN": "d"}))

    class Pipelines(dict):
        def __setitem__(self, key, value):
            assert key in ner._tokenizers, "pipeline visible before its tokenizer"
            super().__setitem__(key, value)

    ner._pipelines = Pipelines(ner._pipelines)
    ner._ensure_loaded("DUTCH")   # fresh load
    ner._ensure_loaded("GERMAN")  # shares the Dutch model
    assert ner._pipelines["GERMAN"] is ner._pipelines["DUTCH"]


class _LabelSwitchDetector(FakeRunDetector):
    """"Alice" as PER 0.99 beside "alpha", as LOC 0.60 beside "b7", else nothing."""

    def __call__(self, texts, batch_size=1):
        out = []
        for text in texts:
            i = text.find("Alice")
            if i == -1 or ("alpha" in text) == ("b7" in text):
                out.append([])
            else:
                group, score = ("PER", 0.99) if "alpha" in text else ("LOC", 0.60)
                out.append([{"entity_group": group, "score": score, "word": "Alice",
                             "start": i, "end": i + 5}])
        return out


def test_label_change_across_windows_is_one_model_vote():
    # The owner window sees nothing; the neighbours see PER 0.99 and LOC 0.60.
    text = "alpha w1 w2 w3 w4 w5 w6 w7. Alice. Beta b1 b2 b3 b4 b5 b6 b7."
    policy = PrivacyPolicy("p", "1", {"PERSON": EntityRule("redact")})
    for result in both_paths(word_ner(2, _LabelSwitchDetector()), text):
        assert result.text == text.replace("Alice", "<PERSON>")
        assert len(result.findings) == 1
    for result in both_paths(word_ner(2, _LabelSwitchDetector()), text, privacy_policy=policy):
        assert result.text == text.replace("Alice", "<REDACTED>")


class _BothMarkersDetector(FakeRunDetector):
    """"Alice" at 0.99 only when the window holds both "alpha" and "omega"."""

    def __call__(self, texts, batch_size=1):
        return [
            [{"entity_group": "PER", "score": 0.99, "word": "Alice", "start": i, "end": i + 5}]
            if (i := text.find("Alice")) != -1 and "alpha" in text and "omega" in text else []
            for text in texts
        ]


def test_overlap_keeps_a_fitting_document_in_one_window():
    text = "alpha w1 w2 w3 w4 Alice w6 w7 w8 w9 omega"  # 11 tokens, budget 12
    ner = word_ner(2, _BothMarkersDetector())
    assert [w[0] for w in windows_for(ner, text)] == [text]
    for result in both_paths(ner, text):
        assert result.text == text.replace("Alice", "<PERSON>")


def test_overlap_keeps_a_fitting_document_in_one_window_gliner_only():
    ner = gliner_ner("placeholder")
    set_cfg(ner, ner_chunk_overlap=10)
    text = "x" * (ner._chunk_budget() * 2)
    assert [w[0] for w in windows_for(ner, text)] == [text]
    assert len(windows_for(ner, text + "x")) > 1
