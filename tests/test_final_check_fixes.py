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
