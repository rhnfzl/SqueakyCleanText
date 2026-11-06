import json

import pytest

from sct.score import (
    SCORE_A_COLUMNS,
    SCORE_B_COLUMNS,
    ONNXQualityScorer,
    ScoreConfig,
    build_data_card,
    minhash_tokens,
    render_data_card,
    score_record,
    score_records,
)


class FakeResult:
    def __init__(self, lm_text="", **signals):
        self.lm_text = lm_text
        self.signals = {
            "language": "HINDI", "language_confidence": 0.93, "pii_count": 2,
            "pii_share": 0.04, "pii_types": {"PERSON": 1, "EMAIL": 1},
            "warnings": ["ner_fallback_models"], "models": [],
            **signals,
        }


def test_score_a_columns_from_signals():
    row = score_record(FakeResult("x"))

    assert tuple(row) == SCORE_A_COLUMNS
    assert row == {
        "sct_language": "HINDI", "sct_language_score": 0.93,
        "sct_pii_count": 2, "sct_pii_share": 0.04,
        "sct_pii_types": ["EMAIL", "PERSON"],
        "sct_warnings": ["ner_fallback_models"],
    }


def test_quality_scorer_sees_original_text_not_anonymised():
    seen = []

    def scorer(texts):
        seen.extend(texts)
        return [0.75] * len(texts)

    row = score_record(FakeResult("Hi <PERSON>"), original="Hi Maria", quality=scorer)

    assert seen == ["Hi Maria"]
    assert row["sct_quality"] == 0.75
    assert row["sct_minhash"] is None
    assert tuple(row) == SCORE_A_COLUMNS + SCORE_B_COLUMNS
    with pytest.raises(ValueError, match="original"):
        score_record(FakeResult("x"), quality=scorer)


def test_score_records_batches_quality_once():
    calls = []

    def scorer(texts):
        calls.append(list(texts))
        return [0.1, 0.9]

    rows = score_records(
        [FakeResult("a"), FakeResult("b")], ["A", "B"],
        ScoreConfig(quality=True), scorer=scorer,
    )

    assert calls == [["A", "B"]]
    assert [r["sct_quality"] for r in rows] == [0.1, 0.9]
    assert all(r["sct_minhash"] is None for r in rows)


def test_score_b_off_by_default():
    rows = score_records([FakeResult("a")], ["a"])

    assert tuple(rows[0]) == SCORE_A_COLUMNS
    with pytest.raises(ValueError, match="quality_model"):
        score_records([FakeResult("a")], ["a"], ScoreConfig(quality=True))
    with pytest.raises(ValueError, match="same length"):
        score_records([FakeResult("a")], [])


@pytest.mark.parametrize("field", ["minhash_num_perm", "minhash_shingle"])
def test_score_config_rejects_zero(field):
    with pytest.raises(ValueError):
        ScoreConfig(**{field: 0})


def test_minhash_tokens_use_chars_for_cjk_and_words_otherwise():
    assert minhash_tokens("会议在东京举行 <PERSON_2>") == [
        "会", "议", "在", "东", "京", "举", "行", "<PERSON>",
    ]
    assert minhash_tokens("Das Treffen mit <PERSON_1> ist") == [
        "das", "treffen", "mit", "<PERSON>", "ist",
    ]
    assert minhash_tokens("ประชุมที่กรุงเทพ")[:3] == ["ป", "ร", "ะ"]


def test_minhash_identical_after_anonymisation_only_name_differs():
    pytest.importorskip("datasketch")
    cfg = ScoreConfig(minhash=True)
    doc = "Yesterday {} presented the quarterly results to the board in Utrecht"
    a = score_record(FakeResult(doc.format("<PERSON_1>")), minhash=cfg)["sct_minhash"]
    b = score_record(FakeResult(doc.format("<PERSON_7>")), minhash=cfg)["sct_minhash"]
    raw = score_record(FakeResult(doc.format("Maria")), minhash=cfg)["sct_minhash"]

    assert a == b
    assert len(a) == 128 and all(isinstance(v, int) for v in a)
    assert raw != a
    json.dumps(a)


def test_minhash_none_for_short_text_and_cjk_gets_signature():
    pytest.importorskip("datasketch")
    cfg = ScoreConfig(minhash=True)

    assert score_record(FakeResult("<PERSON> <EMAIL> hi"), minhash=cfg)["sct_minhash"] is None
    assert score_record(FakeResult("会议在东京举行"), minhash=cfg)["sct_minhash"] is not None


def test_data_card_counts_and_markdown():
    rows = score_records(
        [FakeResult("a"), FakeResult("b", language=None, pii_count=0, pii_share=0.0,
                                     pii_types={}, warnings=["language_unknown"])],
        ["a", "b"],
    )
    card = build_data_card(rows, models=[
        {"key": "MULTILINGUAL", "id": "m", "revision": "r", "provider": "CPU"},
        {"key": "MULTILINGUAL", "id": "m", "revision": "r", "provider": "CPU"},
    ])

    assert card["records"] == 2
    assert card["records_with_pii"] == 1
    assert card["languages"] == {"HINDI": 1, "unknown": 1}
    assert card["pii_types"] == {"EMAIL": 1, "PERSON": 1}
    assert card["pii_share_histogram"] == {"0": 1, "(0.01, 0.05]": 1}
    assert card["warnings"] == {"language_unknown": 1, "ner_fallback_models": 1}
    assert len(card["models"]) == 1
    md = render_data_card(card)
    assert "| HINDI | 1 |" in md and "MULTILINGUAL: m @ r" in md
    assert "None (NER off)." in render_data_card(build_data_card([]))


class _FakeSession:
    def __init__(self, path, providers):
        self.path = path

    def get_inputs(self):
        return [type("I", (), {"name": n})() for n in ("input_ids", "attention_mask")]

    def run(self, _, feeds):
        # Score = number of real tokens, to prove padding and batching line up.
        return [feeds["attention_mask"].sum(axis=1).astype(float)[:, None]]


def _tiny_model_dir(tmp_path, revision="abc"):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    tok = Tokenizer(WordLevel({"<pad>": 0, "[UNK]": 1, "a": 2}, unk_token="[UNK]"))  # noqa: S106
    tok.pre_tokenizer = Whitespace()
    tok.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "model.onnx").write_bytes(b"")
    (tmp_path / "sct_quality.json").write_text(json.dumps({"head_revision": revision}))
    return tmp_path


def test_onnx_quality_scorer_batches_and_pads(tmp_path, monkeypatch):
    import onnxruntime

    monkeypatch.setattr(onnxruntime, "InferenceSession", _FakeSession)
    scorer = ONNXQualityScorer(_tiny_model_dir(tmp_path), revision="abc", batch_size=2)

    assert scorer(["a", "a a a", "a a"]) == [1.0, 3.0, 2.0]
    with pytest.raises(ValueError, match="expected 'other'"):
        ONNXQualityScorer(tmp_path, revision="other")


@pytest.mark.parametrize("scores", [[0.5], [0.5, 0.5, 0.5]])
def test_score_records_rejects_scorer_of_wrong_length(scores):
    with pytest.raises(ValueError):
        score_records([FakeResult("a"), FakeResult("b")], ["a", "b"], scorer=lambda texts: scores)
