"""Regressions for the 0.8.0 review round 4: one model's label change across
windows voting twice, overlap splitting a document that fits one window, and
revision cache dirs that alias on case-insensitive filesystems."""

from tests.ner_helpers import set_cfg
from sct.privacy import EntityRule, PrivacyPolicy
from tests.test_chunking import _FakeRunDetector, _gliner_ner, _word_ner
from tests.test_review_round3 import _both_paths


class _LabelSwitchDetector(_FakeRunDetector):
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
    for result in _both_paths(_word_ner(2, _LabelSwitchDetector()), text):
        assert result.text == text.replace("Alice", "<PERSON>")
        assert len(result.findings) == 1
    for result in _both_paths(_word_ner(2, _LabelSwitchDetector()), text, privacy_policy=policy):
        assert result.text == text.replace("Alice", "<REDACTED>")


class _BothMarkersDetector(_FakeRunDetector):
    """"Alice" at 0.99 only when the window holds both "alpha" and "omega"."""

    def __call__(self, texts, batch_size=1):
        return [
            [{"entity_group": "PER", "score": 0.99, "word": "Alice", "start": i, "end": i + 5}]
            if (i := text.find("Alice")) != -1 and "alpha" in text and "omega" in text else []
            for text in texts
        ]


def test_overlap_keeps_a_fitting_document_in_one_window():
    text = "alpha w1 w2 w3 w4 Alice w6 w7 w8 w9 omega"  # 11 tokens, budget 12
    ner = _word_ner(2, _BothMarkersDetector())
    assert [w[0] for w in ner._plan_windows(text)] == [text]
    for result in _both_paths(ner, text):
        assert result.text == text.replace("Alice", "<PERSON>")


def test_overlap_keeps_a_fitting_document_in_one_window_gliner_only():
    ner = _gliner_ner("placeholder")
    set_cfg(ner, ner_chunk_overlap=10)
    text = "x" * (ner._chunk_budget() * 2)
    assert [w[0] for w in ner._plan_windows(text)] == [text]
    assert len(ner._plan_windows(text + "x")) > 1


def test_revision_cache_dirs_differ_ignoring_case():
    from sct.utils.onnx_pipeline import _quantized_dir

    assert _quantized_dir("o/m", "Release").lower() != _quantized_dir("o/m", "release").lower()
    assert _quantized_dir("o/m", None) == "o__m"  # 0.7.1 layout kept
