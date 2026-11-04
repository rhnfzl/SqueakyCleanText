"""Regressions for the 0.8.0 review round 2: a rejected owner window hiding an
eligible neighbour copy, window sizing scoped to a document's own models, and
overlap checked against models loaded after construction."""

import pytest

from sct.utils.ner_chunking import token_count
from tests.test_chunking import _FakeRunDetector, _word_ner, _words
from tests.test_review_round1 import _Tok


class _ScoredDetector(_FakeRunDetector):
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
    text = _words(40, {7: "John", 8: "Smith", 9: "has", 10: "arrived"})
    expected = text.replace("John Smith", "<PERSON>")
    ner = _word_ner(2, _ScoredDetector())
    single = ner.ner_process_detailed(text, positional_tags=("PER",), ner_confidence_threshold=0.85)
    assert single.text == expected and len(single.findings) == 1
    batch = ner.ner_process_batch_detailed(
        [text, text], [None, None], positional_tags=("PER",), ner_confidence_threshold=0.85)
    assert [r.text for r in batch] == [expected, expected]


def _two_language_ner(overlap=0):
    ner = _word_ner(overlap, _FakeRunDetector(min_words=2), budget=12)
    ner._ensemble_models = {"ENGLISH": ("ENGLISH",), "DUTCH": ("DUTCH",)}
    ner._tokenizers = {"ENGLISH": _Tok(1)}
    return ner


def test_unrelated_loaded_model_does_not_change_a_document():
    # A heavier Dutch tokenizer halves the window and would split "John | Smith".
    text = _words(30, {5: "John", 6: "Smith"})
    expected = text.replace("John Smith", "<PERSON>")
    ner = _two_language_ner()
    kwargs = {"positional_tags": ("PER",), "ner_confidence_threshold": 0.5}
    assert ner.ner_process_detailed(text, language="ENGLISH", **kwargs).text == expected
    ner._tokenizers["DUTCH"] = _Tok(2)
    assert ner.ner_process_detailed(text, language="ENGLISH", **kwargs).text == expected
    batch = ner.ner_process_batch_detailed([text, text], ["ENGLISH", "DUTCH"], **kwargs)
    assert batch[0].text == expected
    # The Dutch document is still sized for the Dutch tokenizer.
    windows = ner._plan_windows(text, ner._get_ensemble_keys("DUTCH"))
    assert all(token_count(w[0], _Tok(2)) <= 12 for w in windows)


def test_overlap_rechecked_after_a_smaller_model_loads():
    ner = _two_language_ner(overlap=2)  # 2 * 2 < 12 // 2 passes at construction
    small = _Tok(1)
    small.max_len_single_sentence = 5
    ner._tokenizers["DUTCH"] = small
    text = _words(30, {15: "John", 16: "Smith"})
    kwargs = {"positional_tags": ("PER",), "ner_confidence_threshold": 0.5}
    with pytest.raises(ValueError, match="ner_chunk_overlap"):
        ner.ner_process_detailed(text, language="DUTCH", **kwargs)
    with pytest.raises(ValueError, match="ner_chunk_overlap"):
        ner.ner_process_batch_detailed([text, text], ["ENGLISH", "DUTCH"], **kwargs)
    assert ner.ner_process_detailed(text, language="ENGLISH", **kwargs).findings
