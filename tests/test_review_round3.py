"""Regressions for the 0.8.0 review round 3: repeated window observations voting
as independent models, neighbour suppression ahead of the privacy policy, and a
pipeline published before its tokenizer."""

from sct.privacy import EntityRule, PrivacyPolicy
from sct.utils.ner import GeneralNER
from tests.test_chunking import _FakeRunDetector, _word_ner, _words

KWARGS = {"positional_tags": ("PER", "LOC"), "ner_confidence_threshold": 0.85}


def _both_paths(ner, text, **kwargs):
    single = ner.ner_process_detailed(text, **KWARGS, **kwargs)
    batch = ner.ner_process_batch_detailed([text, text], [None, None], **KWARGS, **kwargs)
    return [single] + batch


class _AliceDetector(_FakeRunDetector):
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
    for result in _both_paths(_word_ner(2, _AliceDetector()), text):
        assert result.text == text.replace("Alice", "<PERSON>")
        assert len(result.findings) == 1


class _ContextDetector(_FakeRunDetector):
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


TEXT = _words(40, {7: "John", 8: "Smith", 9: "has", 10: "arrived"})


def test_policy_threshold_sees_the_neighbours_score():
    # Owner: 0.86 passes the detector but not the policy; neighbour: 0.99.
    policy = PrivacyPolicy("p", "1", {"PERSON": EntityRule("redact", 0.95)})
    ner = _word_ner(2, _ContextDetector(("PER", 0.86, "John Smith")))
    for result in _both_paths(ner, TEXT, privacy_policy=policy):
        assert result.text == TEXT.replace("John Smith", "<REDACTED>")


def test_kept_owner_span_does_not_hide_a_redacted_neighbour():
    # Owner: "John Smith has" as LOC (keep); neighbour: "John Smith" as PER (redact).
    policy = PrivacyPolicy("p", "1", {"PERSON": EntityRule("redact"),
                                      "LOCATION": EntityRule("keep")})
    ner = _word_ner(2, _ContextDetector(("LOC", 0.99, "John Smith has")))
    for result in _both_paths(ner, TEXT, privacy_policy=policy):
        assert result.text == TEXT.replace("John Smith", "<REDACTED>")
    # Without a policy the covering owner span is replaced and the copy dropped.
    for result in _both_paths(ner, TEXT):
        assert result.text == TEXT.replace("John Smith has", "<LOCATION>")
        assert len(result.findings) == 1


class _FakeTok:
    max_len_single_sentence = 510


def test_tokenizer_is_published_before_its_pipeline(monkeypatch):
    monkeypatch.setattr("sct.utils.onnx_pipeline.load_onnx_ner_model", lambda *a, **k: (object(), _FakeTok()))
    ner = GeneralNER(model_names={"ENGLISH": "a", "MULTILINGUAL": "b", "DUTCH": "d", "GERMAN": "d"})

    class Pipelines(dict):
        def __setitem__(self, key, value):
            assert key in ner._tokenizers, "pipeline visible before its tokenizer"
            super().__setitem__(key, value)

    ner._pipelines = Pipelines(ner._pipelines)
    ner._ensure_loaded("DUTCH")   # fresh load
    ner._ensure_loaded("GERMAN")  # shares the Dutch model
    assert ner._pipelines["GERMAN"] is ner._pipelines["DUTCH"]
