"""NER reads the same text in placeholder, synthetic and reversible mode (ner_sees_tags)."""

import re

import pytest

from sct import config as config_module
from sct.config import TextCleanerConfig
from sct.sct import TextCleaner
from tests.conftest import TEST_NER_MODELS

pytest.importorskip("faker")

DOCS = [
    "Jan de Vries werkt bij Philips in Eindhoven, mail jan@vries.nl.",
    "Mail john.smith@acme.com or call +1-202-555-0143, then John Smith meets Mary Jones in London.",
    "Anna Schmidt (anna@firma.de, +49 30 1234567) zahlte 25 € an Siemens in Berlin, siehe https://firma.de/a.",
    "Escribe a maria@correo.es; María García vive en Madrid y trabaja para Telefónica desde 2019.",
    "Bob <EMAIL> wrote to bob@example.org and bob@example.org again about Paris.",
]
MODES = ("placeholder", "synthetic", "reversible")


def _cleaner(mode, **overrides):
    return TextCleaner(
        TextCleanerConfig(ner_models=TEST_NER_MODELS, replacement_mode=mode, **overrides),
        include_entities=True,
    )


def _spans(result):
    return sorted((f.entity.entity_type, f.entity.start, f.entity.end, f.entity.text)
                  for f in result.metadata["findings"])


@pytest.fixture(scope="module")
def results():
    out = {}
    for mode in MODES:
        cleaner = _cleaner(mode)
        out[mode] = ([cleaner.process(d) for d in DOCS], cleaner.process_batch(DOCS, batch_size=3))
    return out


def test_every_mode_finds_the_same_entity_spans(results):
    expected = [_spans(r) for r in results["placeholder"][0]]
    assert any(expected)
    for mode in MODES:
        single, batch = results[mode]
        assert [_spans(r) for r in single] == expected, mode
        assert [_spans(r) for r in batch] == expected, mode
        assert [r.metadata["regex_counts"] for r in single] == \
            [r.metadata["regex_counts"] for r in results["placeholder"][0]]


def test_synthetic_values_land_after_ner(results):
    single, batch = results["synthetic"]
    assert [r.lm_text for r in single] == [r.lm_text for r in batch]
    for result in single:
        assert "<EMAIL>" not in result.lm_text and "<PHONE>" not in result.lm_text
        for email in ("jan@vries.nl", "john.smith@acme.com", "anna@firma.de", "bob@example.org"):
            assert email not in result.lm_text
    fakes = [w.strip(".,") for w in single[-1].lm_text.split() if "@" in w]
    assert len(fakes) == 2 and fakes[0] == fakes[1]  # one email -> one fake, twice


def test_tag_already_in_source_stays_in_place():
    text = "Bob <EMAIL> wrote to bob@example.org, then <EMAIL> and ann@example.org."
    for mode in ("synthetic", "reversible"):
        result = _cleaner(mode, check_replace_html=False).process(text)
        match = re.search(r"<EMAIL> wrote to (\S+), then <EMAIL> and (\S+)\.$", result.lm_text)
        assert match and "example.org" not in result.lm_text, result.lm_text
        first, second = match.groups()
        assert first != second and "<EMAIL>" not in first + second
        if mode == "reversible":
            assert result.metadata["anon_map"].deanonymize(first + " " + second) == \
                "bob@example.org ann@example.org"


def test_reversible_map_round_trips(results):
    for doc, result in zip(DOCS, results["reversible"][0]):
        anon_map = result.metadata["anon_map"]
        for entry in anon_map.entries:
            assert entry.placeholder in result.lm_text
            if entry.entity_type == "EMAIL":
                assert entry.original in doc


def test_old_order_is_an_option():
    text = DOCS[0]
    old = _cleaner("synthetic", ner_sees_tags=False).process(text)
    new = _cleaner("synthetic").process(text)
    assert _spans(old) != _spans(new)  # NER read the fake email in 0.7 and found Philips
    assert _spans(new) == _spans(_cleaner("placeholder").process(text))


def test_legacy_global_flows():
    old = config_module.NER_SEES_TAGS
    try:
        config_module.NER_SEES_TAGS = False
        assert config_module._config_from_module_globals().ner_sees_tags is False
    finally:
        config_module.NER_SEES_TAGS = old
    assert TextCleanerConfig().ner_sees_tags is True
