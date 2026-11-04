"""Per-document state cleanup, the empty-input metadata, the JSON key cleaner and the streaming guard's offsets."""
import pytest

from sct.adapters import process_json
from sct.config import TextCleanerConfig
from sct.privacy import PrivacyPolicy
from sct.sct import CustomPatternTimeout, TextCleaner
from sct.streaming import WindowedStreamingRedactor, _step_matches


def _cleaner(**overrides):
    return TextCleaner(cfg=TextCleanerConfig(check_ner_process=False, **overrides))


def test_context_holds_no_document_text_after_process():
    cleaner = _cleaner(custom_allow_terms=("Acme",))
    cleaner.process("secret Acme a@b.com")
    assert vars(cleaner._document_context) == {}


def test_context_holds_no_document_text_after_a_step_raises(monkeypatch):
    cleaner = _cleaner(custom_patterns={"X": "x"})

    def timeout(*_args):
        raise CustomPatternTimeout("X", 1)
    monkeypatch.setattr(cleaner, "_user_matches", timeout)
    with pytest.raises(CustomPatternTimeout):
        cleaner.process("secret x")
    assert vars(cleaner._document_context) == {}


def test_empty_input_has_the_policy_and_findings_keys():
    policy = PrivacyPolicy(name="p", version="1", rules=())
    cleaner = TextCleaner(cfg=TextCleanerConfig(check_ner_process=False), privacy_policy=policy)
    for text in ("", "  \n"):
        metadata = cleaner.process(text).metadata
        assert metadata["findings"] == ()
        assert metadata["policy"] == {"name": "p", "version": "1"}
    assert TextCleaner(cfg=TextCleanerConfig(check_ner_process=False), include_entities=True).process(
        "").metadata["findings"] == ()


@pytest.mark.parametrize("key", ["José", "Jose"])
def test_deny_term_is_masked_in_a_json_key_whichever_way_it_was_normalised(key):
    cleaner = _cleaner(check_to_ascii_unicode=True, custom_deny_terms={"CO": ["José"]})
    assert list(process_json(cleaner, {key: "x"}).value) == ["<CO>"]


def test_guard_blanks_keep_offsets_for_one_character_matches():
    def hash_step(repl, text):
        return "".join(repl(_Match(i)) if c == "#" else c for i, c in enumerate(text))

    class _Match:
        def __init__(self, at):
            self.at = at

        def span(self):
            return self.at, self.at + 1

        def group(self):
            return "#"
    spans, blanked = _step_matches([hash_step], "# a # b")
    assert spans == [(0, 1), (4, 5)]
    assert len(blanked) == len("# a # b")


def test_one_character_custom_pattern_streams_without_a_buffer_error():
    cleaner = _cleaner(custom_patterns={"HASH": "#"})
    text = "# word user@example.com word " * 60
    redactor = WindowedStreamingRedactor(cleaner, max_buffer_chars=200, overlap_chars=20)
    out = redactor.feed(text) + redactor.finish()
    assert "user@example.com" not in out and "#" not in out
