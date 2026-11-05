"""ProcessResult.signals: same keys for every input, and never a raw value."""

import json

import pytest

from sct import TextCleaner, TextCleanerConfig, process_json
from sct.signals import PUBLIC_VIEW_KEYS, WARNING_CODES
from tests.conftest import TEST_NER_MODELS, TEST_NER_REVISIONS

NO_NER = TextCleanerConfig(check_ner_process=False)
TEXT = "John Smith wrote to john.smith@example.com from 203.0.113.9, call 415-555-0134."
RAW = ("John", "Smith", "john.smith@example.com", "203.0.113.9", "415-555-0134")


@pytest.fixture(scope="module")
def plain():
    return TextCleaner(cfg=NO_NER)


@pytest.mark.parametrize("text", ["mail a@b.co now", "", "   ", "Hallo Welt"])
def test_same_keys_for_every_text(plain, text):
    signals = plain.process(text).signals
    assert tuple(signals) == PUBLIC_VIEW_KEYS
    assert set(signals["warnings"]) <= WARNING_CODES
    assert signals["chars"] == len(text)


@pytest.mark.parametrize("value", [{}, {"a": ["x", {"b": "mail me a@b.co"}]}])
def test_process_json_keeps_signals(plain, value):
    result = process_json(plain, value)
    assert tuple(result.signals) == PUBLIC_VIEW_KEYS
    assert result.sensitive == result.signals["sensitive"]


def test_process_json_counts_every_leaf(plain):
    signals = process_json(plain, {"a": ["mail me a@b.co", {"b": "or c@d.org"}]}).signals
    assert signals["pii_types"] == {"EMAIL": 2}
    assert signals["chars"] == len("a") + len("mail me a@b.co") + len("b") + len("or c@d.org")


def test_pii_share(plain):
    assert plain.process("a@b.co").signals["pii_share"] == 1.0
    empty = plain.process("").signals
    assert empty["pii_share"] == 0.0
    assert "empty_input" in empty["warnings"] and empty["language_source"] == "none"


def test_batch_empty_input_has_view(plain):
    results = plain.process_batch(["", "a@b.co"])
    assert [r.signals["chars"] for r in results] == [0, 6]


@pytest.mark.parametrize("mode", ["placeholder", "reversible", "synthetic"])
def test_nothing_raw_in_repr_or_signals(mode):
    if mode == "synthetic":
        pytest.importorskip("faker")
    cleaner = TextCleaner(
        cfg=TextCleanerConfig(ner_models=TEST_NER_MODELS, ner_model_revisions=TEST_NER_REVISIONS, replacement_mode=mode,
                              check_to_ascii_unicode=False),
        include_entities=True,
    )
    result = cleaner.process(TEXT)
    assert result.signals["pii_count"] >= 4
    seen = [repr(result), str(result), json.dumps(result.signals)]
    seen += [repr(f) for f in result.metadata.get("findings", ())]
    if mode == "reversible":
        seen.append(repr(result.metadata["anon_map"]))
        seen += [repr(e) for e in result.metadata["anon_map"].entries]
    else:
        seen.append(json.dumps(process_json(cleaner, {"k": TEXT}).signals))
    for blob in seen:
        for raw in RAW:
            assert raw not in blob, (raw, blob)
