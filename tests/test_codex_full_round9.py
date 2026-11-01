"""Codex 0.8.0 full review, round 9: ids 1, 2, 4, 6."""

import pytest

from sct import TextCleaner
from sct.presets import preset_config
from sct.risk import assess_reidentification_risk

FAST = preset_config("training", check_ner_process=False)


# --- 1: a ) ] } inside a URL does not end it; one around the URL stays outside ---

@pytest.mark.parametrize("text, expected", [
    ("https://example.com/?filter=(active)&email=alice%40example.com", "<URL>"),
    ("https://example.com/?q=[a]&token=abc%40def x", "<URL> x"),
    ("https://en.wikipedia.org/wiki/Foo_(bar)_baz x", "<URL> x"),
    ("see (https://x.com/a). ok", "see (<URL>). ok"),
    ("[l](https://x.com/a), b", "[l](<URL>), b"),
])
def test_url_interior_brackets_stay_in_url(text, expected):
    assert TextCleaner(cfg=FAST).process(text).lm_text == expected


# --- 2: datasets output columns are declared, so a first null batch cannot fix a null type ---

@pytest.mark.parametrize("data, kwargs", [
    ({"text": ["plain", "alice@example.com"]}, {"score": True}),
    ({"text": [None, "hello world"]}, {"score": True}),
    ({"text": [None, "alice@example.com"]}, {"output": "new_column"}),
])
def test_clean_dataset_first_null_batch(data, kwargs):
    datasets = pytest.importorskip("datasets")
    from sct.integrations.hf import clean_dataset
    out = clean_dataset(datasets.Dataset.from_dict(data), config=FAST, batch_size=1, **kwargs)
    assert len(out) == 2
    if kwargs.get("score"):
        assert out.features["sct_pii_types"].feature.dtype == "string"


# --- 4: type tags reach inside tuple and frozenset quasi-identifiers ---

@pytest.mark.parametrize("a, b", [((True,), (1,)), ((1,), (1.0,)),
                                  (frozenset({True}), frozenset({1})), (((1,),), ((True,),))])
def test_nested_quasi_identifiers_are_type_tagged(a, b):
    report = assess_reidentification_risk([{"q": a}, {"q": b}], quasi_identifiers=("q",),
                                           k_threshold=2)
    assert (report.minimum_k, report.at_risk_records) == (1, 2)


# --- 6: merged language_source follows the chosen language's leaves ---

def test_merged_language_source_follows_chosen_language():
    from sct.adapters import process_json
    cleaner = TextCleaner(cfg=preset_config("training", check_ner_process=False,
                                            language="ENGLISH"))
    signals = process_json(cleaner, [" " * 15, "Hello world!", "Hello world!"]).signals
    assert (signals["language"], signals["language_source"]) == ("ENGLISH", "pinned")
