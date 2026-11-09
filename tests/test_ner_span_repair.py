"""NER span repair: word-piece widening, ensemble votes, value propagation, gap merge, edge snapping."""

import types

import numpy as np
import pytest

from sct import TextCleanerConfig
from sct.utils.anonymization_map import AnonymizationMap
from sct.utils.ner import GeneralNER
from sct.utils.onnx_pipeline import ONNXNERPipeline
from sct.utils.synthetic import SyntheticReplacer

from tests.conftest import TEST_NER_MODELS, TEST_NER_REVISIONS
from tests.helpers import bare_ner, FakeRunDetector, finalize_spans, finalized_text, set_cfg, word_ner, word_tokenizer


def test_propagation_does_not_match_inside_a_word_with_combining_mark():
    text = "राज met राजा, राज."
    ner, windows, results = finalize_spans(text, [(0, 3, 'PER', 0.99)])
    assert finalized_text(ner, text, windows, results) == "<PERSON> met राजा, <PERSON>."


# --- one model's two labels for a widened span are one vote, its best ---
class _TwoLabelDetector(FakeRunDetector):
    """Like a widened "Do" + "##e": PER 0.99 and MISC 0.60 over the same word."""

    def __call__(self, texts, batch_size=1):
        return [[{**span, "entity_group": group, "score": score}
                 for span in found for group, score in (("PER", 0.99), ("MISC", 0.60))]
                for found in super().__call__(texts, batch_size)]


def test_same_model_two_labels_do_not_average_below_threshold():
    ner = word_ner(0, _TwoLabelDetector())
    result = ner.ner_process_detailed("w0 w1 Doe w3", positional_tags=("PER", "MISC"),
                                      ner_confidence_threshold=0.85)
    assert result.text == "w0 w1 <PERSON> w3"


def _aggregate(text, tokens, offsets, tagged):
    """Run the WordPiece aggregation on fake tokens; ``tagged`` maps token index -> label id."""
    pipe = object.__new__(ONNXNERPipeline)
    pipe._id2label, pipe._subword_prefix = {0: "O", 1: "B-PER", 2: "I-PER"}, "##"
    scores = np.zeros((len(tokens), 3))
    for idx in range(len(tokens)):
        scores[idx, tagged.get(idx, 0)] = 1.0
    mask = [1] + [0] * (len(tokens) - 2) + [1]
    encoding = types.SimpleNamespace(tokens=tokens, offsets=offsets, special_tokens_mask=mask)
    return [(e["start"], e["end"]) for e in pipe._aggregate_simple(text, encoding, scores)]


def test_wordpiece_widening_adds_a_missed_tail():
    tokens = ["[CLS]", "John", "Do", "##e", "[SEP]"]
    offsets = [(0, 0), (0, 4), (5, 7), (7, 8), (0, 0)]
    assert _aggregate("John Doe", tokens, offsets, {1: 1, 2: 2}) == [(0, 8)]


def test_wordpiece_widening_does_not_swallow_an_unspaced_sentence():
    # Thai / Japanese runs have no spaces: the whole run is one "word" of ## pieces.
    text = "ย้ายไปนครราชสีมาเมื่อปีที่แล้ว"
    tokens = ["[CLS]", "ย้ายไป", "##นครราชสีมา", "##เมื่อปีที่แล้ว", "[SEP]"]
    offsets = [(0, 0), (0, 6), (6, 16), (16, len(text)), (0, 0)]
    assert _aggregate(text, tokens, offsets, {2: 1}) == [(6, 16)]


def test_subword_widen_option_reaches_the_old_behaviours():
    text = "ย้ายไปนครราชสีมาเมื่อปีที่แล้ว"
    tokens = ["[CLS]", "ย้ายไป", "##นครราชสีมา", "##เมื่อปีที่แล้ว", "[SEP]"]
    offsets = [(0, 0), (0, 6), (6, 16), (16, len(text)), (0, 0)]
    pipe = object.__new__(ONNXNERPipeline)
    pipe._id2label, pipe._subword_prefix = {0: "O", 1: "B-PER", 2: "I-PER"}, "##"
    scores = np.zeros((5, 3))
    scores[:, 0], scores[2] = 1.0, [0, 1, 0]
    encoding = types.SimpleNamespace(tokens=tokens, offsets=offsets, special_tokens_mask=[1, 0, 0, 0, 1])
    spans = {w: [(e["start"], e["end"]) for e in pipe._aggregate_simple(text, encoding, scores, w)]
             for w in (None, 0, 1.0)}
    assert spans == {None: [(0, len(text))], 0: [(6, 16)], 1.0: [(6, 16)]}


def test_subword_widen_config():
    assert TextCleanerConfig().ner_subword_widen == 1.0
    with pytest.raises(ValueError):
        TextCleanerConfig(ner_subword_widen=-1)


@pytest.mark.parametrize("same_script, expected", [(True, (3, 15)), (False, (2, 17))])
def test_wordpiece_widening_stops_at_a_script_change(same_script, expected):
    # mBERT reads "はDavid Millerで" as one word: the Latin name must not take the kana around it.
    text = "名前はDavid Millerです"
    tokens = ["[CLS]", "名", "前", "は", "##Dav", "##id", "Miller", "##で", "##す", "[SEP]"]
    offsets = [(0, 0), (0, 1), (1, 2), (2, 3), (3, 6), (6, 8), (9, 15), (15, 16), (16, 17), (0, 0)]
    pipe = object.__new__(ONNXNERPipeline)
    pipe._id2label, pipe._subword_prefix = {0: "O", 1: "B-PER", 2: "I-PER"}, "##"
    scores = np.zeros((len(tokens), 3))
    scores[:, 0] = 1.0
    scores[4], scores[5], scores[6] = [0, 1, 0], [0, 0, 1], [0, 0, 1]
    mask = [1] + [0] * (len(tokens) - 2) + [1]
    encoding = types.SimpleNamespace(tokens=tokens, offsets=offsets, special_tokens_mask=mask)
    found = pipe._aggregate_simple(text, encoding, scores, None, same_script)
    assert [(e["start"], e["end"]) for e in found] == [expected]


def test_wordpiece_widening_still_completes_a_same_script_word():
    tokens = ["[CLS]", "ถึง", "##คุณ", "##Sar", "##ah", "[SEP]"]
    text = "ถึงคุณSarah"
    offsets = [(0, 0), (0, 3), (3, 6), (6, 9), (9, 11), (0, 0)]
    assert _aggregate(text, tokens, offsets, {3: 1}) == [(6, 11)]


def test_ensemble_agreement_never_drops_a_strong_single_model_hit():
    # F1: XLM-R 1.0 + wikineural 0.40 on the same widened span averaged to 0.70 < 0.85

    ner = bare_ner()
    strong = {'entity_group': 'LOC', 'score': 0.95, 'key': '0:5', 'start': 0, 'end': 5, 'model': 'a'}
    weak = {**strong, 'score': 0.60, 'model': 'b'}
    assert ner.vote_entities([strong, weak], 0.85) == [strong]
    assert ner.vote_entities([{**weak, 'score': 0.5}], 0.85) == []
    set_cfg(ner, ner_ensemble_vote='mean')
    assert ner.vote_entities([strong, weak], 0.85) == []


def test_ensemble_vote_config():
    assert TextCleanerConfig().ner_ensemble_vote == 'max'
    with pytest.raises(ValueError):
        TextCleanerConfig(ner_ensemble_vote='median')


def test_masked_value_is_masked_at_its_other_whole_word_occurrences():
    # F2: a masked surname stayed visible at its other occurrences in the same document
    text = "Dear Okonkwo, thanks. Okonkwo\nOkonkwos and okonkwo by Okonkwo."
    ner, windows, results = finalize_spans(text, [(5, 12, 'PER', 0.99)])
    out = finalized_text(ner, text, windows, results)
    # whole-word, case-sensitive: a longer word and a lowercase variant stay
    assert out == "Dear <PERSON>, thanks. <PERSON>\nOkonkwos and okonkwo by <PERSON>."
    set_cfg(ner, ner_propagate_masked=False)
    assert finalized_text(ner, text, windows, results).count("<PERSON>") == 1


def test_propagation_skips_short_letterless_and_low_score_values():
    text = "A. went to Rome, Rome. -- and -- again A."
    spans = [(0, 2, 'PER', 0.99), (11, 15, 'LOC', 0.90), (23, 25, 'ORG', 0.99)]
    ner, windows, results = finalize_spans(text, spans)
    assert finalized_text(ner, text, windows, results) == \
        "<PERSON> went to <LOCATION>, <LOCATION>. <ORGANISATION> and -- again <PERSON>"
    set_cfg(ner, ner_propagate_min_score=0.95)
    assert finalized_text(ner, text, windows, results) == \
        "<PERSON> went to <LOCATION>, Rome. <ORGANISATION> and -- again <PERSON>"


def test_propagation_reversible_restores_exactly():
    from sct.utils.anonymization_map import AnonymizationMap

    text = "Lars Petersen met Acme. Acme paid Lars Petersen."
    ner, windows, results = finalize_spans(
        text, [(0, 13, 'PER', 0.99), (18, 22, 'ORG', 0.99)], mode='reversible')
    anon_map = AnonymizationMap()
    out = finalized_text(ner, text, windows, results, anon_map)
    assert "Petersen" not in out and "Acme" not in out
    assert anon_map.deanonymize(out) == text


def test_propagate_config():
    assert TextCleanerConfig().ner_propagate_masked is True
    assert TextCleanerConfig().ner_propagate_min_score is None
    with pytest.raises(ValueError):
        TextCleanerConfig(ner_propagate_min_score=1.5)


def _loc_spans(text, *words):
    return [(text.index(w), text.index(w) + len(w), 'LOC', 0.99) for w in words]


def test_location_gap_merge_joins_address_parts():
    # F8: one address came out as many LOCATION/NUMBER spans with "Apt 5B" and "IL" visible
    text = "Ship to <NUMBER> Elm Street, Apt 5B, Springfield, IL <NUMBER>, USA. Then Paris.\nRome"
    ner, windows, results = finalize_spans(
        text, _loc_spans(text, "Elm Street", "Springfield", "USA", "Paris", "Rome"))
    out = finalized_text(ner, text, windows, results)
    assert out == ("Ship to <NUMBER> <LOCATION>, Apt 5B, <LOCATION>, IL <NUMBER>, <LOCATION>. "
                   "Then <LOCATION>.\n<LOCATION>")  # 0 = off, 0.8.0 output
    set_cfg(ner, ner_merge_location_gap=12)
    out = finalized_text(ner, text, windows, results)
    # placeholders in the gap do not count; a sentence end or a line break stops the merge
    assert out == "Ship to <NUMBER> <LOCATION>. Then <LOCATION>.\n<LOCATION>"
    set_cfg(ner, ner_merge_location_gap=7)
    out = finalized_text(ner, text, windows, results)
    assert out == "Ship to <NUMBER> <LOCATION>, Apt 5B, <LOCATION>. Then <LOCATION>.\n<LOCATION>"


def test_location_gap_merge_reversible_restores_exactly():
    from sct.utils.anonymization_map import AnonymizationMap

    anon_map = AnonymizationMap()
    anon_map.add('<NUMBER_0>', '62704', 'NUMBER', 0, 0)
    text = "Springfield, IL <NUMBER_0>, USA"
    ner, windows, results = finalize_spans(
        text, _loc_spans(text, "Springfield", "USA"), mode='reversible', ner_merge_location_gap=12)
    out = finalized_text(ner, text, windows, results, anon_map)
    assert out == "<LOCATION_0>"
    assert anon_map.deanonymize(out) == "Springfield, IL 62704, USA"


def test_location_gap_config():
    assert TextCleanerConfig().ner_merge_location_gap == 0
    with pytest.raises(ValueError):
        TextCleanerConfig(ner_merge_location_gap=-1)


@pytest.mark.parametrize("text, word, expected", [
    # F9: a SentencePiece / GLiNER span ending or starting inside a word left a letter visible
    ("Via della Salute 12", "Via della Salut", "<PERSON> 12"),
    ("Sig. Mario Rossi", "ario Rossi", "Sig. <PERSON>"),
    ("রহিমের বাড়ি", "রহিম", "<PERSON> বাড়ি"),  # vowel sign + letter suffix, same script
    ("Millerは東京", "Mille", "<PERSON>は東京"),  # wide / other-script letters are not taken
    ("张三说", "张三", "<PERSON>说"),  # unspaced CJK: wide characters never snap
    ("ย้ายไปนครราชสีมาเมื่อปีที่แล้ว", "นครราชสีมา", "ย้ายไป<PERSON>เมื่อปีที่แล้ว"),  # ratio cap
])
def test_span_edges_snap_to_the_whole_word(text, word, expected):
    start = text.index(word)
    ner, windows, results = finalize_spans(
        text, [(start, start + len(word), 'PER', 0.99)], ner_propagate_masked=False)
    assert finalized_text(ner, text, windows, results) == expected
    set_cfg(ner, ner_snap_word_edges=False)  # 0.8.0: the span is replaced as tagged
    assert finalized_text(ner, text, windows, results) == \
        text[:start] + "<PERSON>" + text[start + len(word):]


# --- NER does not rewrite the synthetic fake a regex step inserted ---
def test_ner_leaves_regex_synthetic_fakes_alone():
    from sct.utils.synthetic import SyntheticReplacer
    from tests.helpers import FakeRunDetector, word_ner

    ner = word_ner(0, FakeRunDetector(), mode="synthetic")
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


# --- WordPiece entities cover the whole word, not "John Do" + a visible "e" ---
def test_wordpiece_entity_ends_on_a_whole_word():
    from sct.utils.onnx_pipeline import load_onnx_ner_model

    pipe, _ = load_onnx_ner_model(TEST_NER_MODELS["ENGLISH"], revision=TEST_NER_REVISIONS["ENGLISH"])
    text = "Contact John Doe today."
    spans = [(e["start"], e["end"]) for e in pipe(text)[0] if e["entity_group"] == "PER"]
    assert spans and all(text[end] in " ." for _, end in spans)
    assert any(text[s:e].endswith("Doe") for s, e in spans)


TEXT = "John Paul Smith. " + "filler " * 25 + "email=a@b.com"


NESTED = [
    {"entity_group": "PER", "score": 0.99, "word": "John Paul Smith", "start": 0, "end": 15},
    {"entity_group": "PER", "score": 0.95, "word": "Paul", "start": 5, "end": 9},
    {"entity_group": "EMAIL", "score": 0.99, "word": "a@b.com",
     "start": len(TEXT) - 7, "end": len(TEXT)},
]


@pytest.mark.parametrize("mode", ["placeholder", "reversible", "synthetic"])
def test_nested_spans_leave_no_part_of_the_entity(mode):
    ner = word_ner(0, FakeRunDetector(), mode=mode)
    ner._synthetic_replacer = SyntheticReplacer()
    anon_map = AnonymizationMap()
    text = ner.anonymize_text(TEXT, [dict(d) for d in NESTED], mode, anon_map=anon_map)
    assert "Smith" not in text and "Paul" not in text
    assert "a@b.com" not in text
    if mode == "placeholder":
        assert text.startswith("<PERSON>. ")
    if mode == "reversible":
        assert anon_map.deanonymize(text) == TEXT


# --- GLiNER labels reach the ensemble vote, and the span rules see the long labels ---
class _FakeGLiNERModel:
    """A GLiNER model that finds one fixed word for each label it is asked about."""

    words = {"person": "Ann", "organization": "Acme", "location": "Paris"}

    def predict_entities(self, text, labels, threshold=0.5):
        return [{"label": label, "text": self.words[label], "score": 0.95,
                 "start": text.index(self.words[label]), "end": text.index(self.words[label]) + len(self.words[label])}
                for label in labels]


class _SilentTokenModel:
    """A token model that finds nothing, so every span in the output came from GLiNER."""

    max_len_single_sentence = 510

    def __call__(self, text_or_texts, **kwargs):
        if kwargs.get("batch_size"):
            return [[] for _ in text_or_texts]
        return word_tokenizer(text_or_texts)


def _gliner_ner(monkeypatch, **cfg):
    silent = _SilentTokenModel()
    monkeypatch.setattr("sct.utils.onnx_pipeline.load_onnx_ner_model", lambda *a, **k: (silent, silent))
    monkeypatch.setattr("sct.utils.torch_pipeline.load_torch_ner_model", lambda *a, **k: (silent, silent))
    monkeypatch.setattr("sct.utils.gliner_adapter.GLiNERAdapter._load_gliner", lambda *a: _FakeGLiNERModel())
    return GeneralNER(TextCleanerConfig(gliner_model="fake/gliner", ner_device="cpu", **cfg))


def _gliner_output(monkeypatch, **cfg):
    return _gliner_ner(monkeypatch, **cfg).ner_process_detailed("Ann works at Acme in Paris", language="ENGLISH").text


@pytest.mark.parametrize("backend", ["ensemble_onnx", "ensemble_torch"])
def test_ensemble_keeps_gliner_spans_with_default_labels(monkeypatch, backend):
    assert _gliner_output(monkeypatch, ner_backend=backend) == "<PERSON> works at <ORGANISATION> in <LOCATION>"


@pytest.mark.parametrize("label_map, expected", [
    ({"person": "ORG", "organization": "PER", "location": "MISC"}, "<ORGANISATION> works at <PERSON> in <MISC>"),
    ({"person": "PER"}, "<PERSON> works at Acme in Paris"),
    ({}, "Ann works at Acme in Paris"),
])
def test_ensemble_user_label_map_wins(monkeypatch, label_map, expected):
    assert _gliner_output(monkeypatch, ner_backend="ensemble_onnx", gliner_label_map=label_map) == expected


def test_ensemble_keeps_a_label_requested_under_its_long_tag(monkeypatch):
    out = _gliner_output(monkeypatch, ner_backend="ensemble_onnx", positional_tags=("ORGANIZATION", "PER", "LOC"))
    assert out == "<PERSON> works at <ORGANIZATION> in <LOCATION>"


@pytest.mark.parametrize("cfg_tags, call_tags, expected", [
    (("PER", "LOC", "ORG", "MISC"), ("PERSON", "ORGANIZATION", "LOCATION"),
     "<PERSON> works at <ORGANIZATION> in <LOCATION>. <PERSON> left <LOCATION>, <ORGANIZATION> stayed."),
    (("PERSON", "ORGANIZATION", "LOCATION"), ("PER", "ORG", "LOC"),
     "<PERSON> works at <ORGANISATION> in <LOCATION>. <PERSON> left <LOCATION>, <ORGANISATION> stayed."),
])
def test_ensemble_picks_short_or_long_tags_per_call(monkeypatch, cfg_tags, call_tags, expected):
    ner = _gliner_ner(monkeypatch, ner_backend="ensemble_onnx", positional_tags=cfg_tags)
    text = "Ann works at Acme in Paris. Ann left Paris, Acme stayed."
    assert ner.ner_process_detailed(text, language="ENGLISH", positional_tags=call_tags).text == expected
    assert ner.ner_process_batch_detailed([text], ["ENGLISH"], positional_tags=call_tags)[0].text == expected


def test_ensemble_leaves_an_uppercase_gliner_label_as_written(monkeypatch):
    monkeypatch.setitem(_FakeGLiNERModel.words, "PERSON", "Ann")
    out = _gliner_output(monkeypatch, ner_backend="ensemble_onnx", gliner_labels=("PERSON", "location"))
    assert out == "Ann works at Acme in <LOCATION>"


def test_plain_gliner_backend_keeps_its_default_tokens(monkeypatch):
    assert _gliner_output(monkeypatch, ner_backend="gliner") == "<PERSON> works at <ORGANIZATION> in <LOCATION>"


def test_long_labels_propagate_and_keep_their_tokens():
    text = "Ann of Acme in Paris. Ann, Acme, Paris."
    spans = [(0, 3, "PERSON", 0.9), (7, 11, "ORGANIZATION", 0.9), (15, 20, "LOCATION", 0.9)]
    ner, windows, results = finalize_spans(text, spans)
    assert finalized_text(ner, text, windows, results) == \
        "<PERSON> of <ORGANIZATION> in <LOCATION>. <PERSON>, <ORGANIZATION>, <LOCATION>."


def test_long_location_label_merges_across_a_gap():
    text = "Elm Street, Springfield. Rome"
    spans = [(0, 10, "LOCATION", 0.9), (12, 23, "LOCATION", 0.9), (25, 29, "LOC", 0.9)]
    ner, windows, results = finalize_spans(text, spans, ner_merge_location_gap=12)
    assert finalized_text(ner, text, windows, results) == "<LOCATION>. <LOCATION>"
