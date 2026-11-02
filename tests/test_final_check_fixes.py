"""Regressions found by the final 0.8.0 multilingual eval re-run."""
import types

import numpy as np
import pytest

from sct import TextCleaner, TextCleanerConfig
from sct.utils import constants
from sct.utils.onnx_pipeline import ONNXNERPipeline


@pytest.mark.parametrize("text, email", [
    ("お問い合わせはseikyu@example.com（担当：鈴木花子）まで。", "seikyu@example.com"),
    ("联系 a@b.cn。谢谢", "a@b.cn"),
    ("write a@b.com“quoted”", "a@b.com"),
    ("mail info@例え.jp now", "info@例え.jp"),
    ("user@उदाहरण.भारत ok", "user@उदाहरण.भारत"),
])
def test_email_domain_stops_at_cjk_and_fullwidth_punctuation(text, email):
    assert [m.group() for m in constants.EMAIL_REGEX.finditer(text)][0].endswith(email)


def test_email_keeps_the_name_after_fullwidth_bracket_visible_to_ner():
    cleaner = TextCleaner(cfg=TextCleanerConfig(check_ner_process=False))
    assert cleaner.process("seikyu@example.com（担当：鈴木花子）").lm_text == "<EMAIL>（担当：鈴木花子）"


@pytest.mark.parametrize("text, expected", [
    ("তানিয়া ইসলাম-এর অভিযোগ", "তানিয়া ইসলাম-এর অভিযোগ"),
    ("東京-大阪", "東京-大阪"),
    ("कर्म-योग", "कर्म-योग"),
    ("l'été", "l'été"),
    ("東京 - 大阪", "東京 大阪"),
])
def test_hyphen_between_letters_of_any_script_is_kept(text, expected):
    cleaner = TextCleaner(cfg=TextCleanerConfig(check_ner_process=False))
    assert cleaner.process(text).lm_text == expected


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


def test_subword_widen_config_and_legacy_global(monkeypatch):
    from sct import config as cfg_module

    assert TextCleanerConfig().ner_subword_widen == 1.0
    with pytest.raises(ValueError):
        TextCleanerConfig(ner_subword_widen=-1)
    monkeypatch.setattr(cfg_module, "NER_SUBWORD_WIDEN", None)
    assert cfg_module._config_from_module_globals().ner_subword_widen is None


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


def test_widen_same_script_config_and_legacy_global(monkeypatch):
    from sct import config as cfg_module

    assert TextCleanerConfig().ner_subword_widen_same_script is True
    monkeypatch.setattr(cfg_module, "NER_SUBWORD_WIDEN_SAME_SCRIPT", False)
    assert cfg_module._config_from_module_globals().ner_subword_widen_same_script is False


def test_ensemble_agreement_never_drops_a_strong_single_model_hit():
    # F1: XLM-R 1.0 + wikineural 0.40 on the same widened span averaged to 0.70 < 0.85
    from sct.utils.ner import GeneralNER

    ner = object.__new__(GeneralNER)
    strong = {'entity_group': 'LOC', 'score': 0.95, 'key': '0:5', 'start': 0, 'end': 5, 'model': 'a'}
    weak = {**strong, 'score': 0.60, 'model': 'b'}
    assert ner.ner_ensemble([strong, weak], 0.85) == [strong]
    assert ner.ner_ensemble([{**weak, 'score': 0.5}], 0.85) == []
    ner._ner_ensemble_vote = 'mean'
    assert ner.ner_ensemble([strong, weak], 0.85) == []


def test_ensemble_vote_config_and_legacy_global(monkeypatch):
    from sct import config as cfg_module

    assert TextCleanerConfig().ner_ensemble_vote == 'max'
    with pytest.raises(ValueError):
        TextCleanerConfig(ner_ensemble_vote='median')
    monkeypatch.setattr(cfg_module, "NER_ENSEMBLE_VOTE", 'mean')
    assert cfg_module._config_from_module_globals().ner_ensemble_vote == 'mean'


def _finalize_spans(text, spans, **attrs):
    # One window over the whole text; spans are (start, end, group, score) from one model.
    from sct.utils.ner import GeneralNER

    ner = object.__new__(GeneralNER)
    ner._replacement_mode = attrs.pop('mode', 'placeholder')
    for name, value in attrs.items():
        setattr(ner, name, value)
    items = [{'entity_group': g, 'score': s, 'start': a, 'end': b, 'model': 'm'} for a, b, g, s in spans]
    windows = [(text, 0, list(range(len(text) + 1)), 0, len(text))]
    return ner, windows, [items]


def test_masked_value_is_masked_at_its_other_whole_word_occurrences():
    # F2: a masked surname stayed visible at its other occurrences in the same document
    text = "Dear Okonkwo, thanks. Okonkwo\nOkonkwos and okonkwo by Okonkwo."
    ner, windows, results = _finalize_spans(text, [(5, 12, 'PER', 0.99)])
    out = ner._finalize(text, windows, results, 0.85, None, None).text
    # whole-word, case-sensitive: a longer word and a lowercase variant stay
    assert out == "Dear <PERSON>, thanks. <PERSON>\nOkonkwos and okonkwo by <PERSON>."
    ner._ner_propagate_masked = False
    assert ner._finalize(text, windows, results, 0.85, None, None).text.count("<PERSON>") == 1


def test_propagation_skips_short_letterless_and_low_score_values():
    text = "A. went to Rome, Rome. -- and -- again A."
    spans = [(0, 2, 'PER', 0.99), (11, 15, 'LOC', 0.90), (23, 25, 'ORG', 0.99)]
    ner, windows, results = _finalize_spans(text, spans)
    assert ner._finalize(text, windows, results, 0.85, None, None).text == \
        "<PERSON> went to <LOCATION>, <LOCATION>. <ORGANISATION> and -- again <PERSON>"
    ner._ner_propagate_min_score = 0.95
    assert ner._finalize(text, windows, results, 0.85, None, None).text == \
        "<PERSON> went to <LOCATION>, Rome. <ORGANISATION> and -- again <PERSON>"


def test_propagation_reversible_restores_exactly():
    from sct.utils.anonymization_map import AnonymizationMap

    text = "Lars Petersen met Acme. Acme paid Lars Petersen."
    ner, windows, results = _finalize_spans(
        text, [(0, 13, 'PER', 0.99), (18, 22, 'ORG', 0.99)], mode='reversible')
    anon_map = AnonymizationMap()
    out = ner._finalize(text, windows, results, 0.85, anon_map, None).text
    assert "Petersen" not in out and "Acme" not in out
    assert anon_map.deanonymize(out) == text


def test_propagate_config_and_legacy_global(monkeypatch):
    from sct import config as cfg_module

    assert TextCleanerConfig().ner_propagate_masked is True
    assert TextCleanerConfig().ner_propagate_min_score is None
    with pytest.raises(ValueError):
        TextCleanerConfig(ner_propagate_min_score=1.5)
    monkeypatch.setattr(cfg_module, "NER_PROPAGATE_MASKED", False)
    monkeypatch.setattr(cfg_module, "NER_PROPAGATE_MIN_SCORE", 0.9)
    cfg = cfg_module._config_from_module_globals()
    assert (cfg.ner_propagate_masked, cfg.ner_propagate_min_score) == (False, 0.9)
