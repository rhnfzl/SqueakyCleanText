"""Historical fixtures and independently checked v0.9.0 outputs stay compatible."""

import json
from pathlib import Path

import pytest

from sct import TextCleaner, TextCleanerConfig

FIXTURES = Path(__file__).parent / "fixtures" / "processing_compatibility.json"


def cleaner(case):
    options = dict(
        check_ner_process=False,
        check_detect_language=False,
        check_statistical_model_processing=False,
        replacement_mode=case.get("mode", "placeholder"),
    )
    if case.get("custom_step"):
        options["custom_pipeline_steps"] = (lambda text: f"[CUSTOM] {text}",)
    cfg = (
        TextCleanerConfig.from_preset("training", **options)
        if case.get("preset") == "training"
        else TextCleanerConfig(**options)
    )
    return TextCleaner(cfg=cfg)


@pytest.mark.parametrize("fixture", ["processing_compatibility.json", "processing_compatibility_v090.json"])
def test_p0_acceptance(fixture):
    captured = json.loads(FIXTURES.with_name(fixture).read_text())
    if isinstance(captured, dict):
        assert captured["baseline_commit"] == "882342fd08c03718d21d22278814cfbfe4b883f3"
        assert captured["baseline_version"] == "0.9.0"
        cases = captured["cases"]
    else:
        cases = captured
    assert {case["id"] for case in cases} >= {
        "default_contacts",
        "training_contacts",
        "synthetic_contacts",
        "reversible_source_marker",
        "blank",
        "custom_step",
    }
    for case in cases:
        result = cleaner(case).process(case["input"])
        assert list(tuple(result)[:2]) == case["output"], case["id"]
        assert len(tuple(result)) == 3
        if case.get("restored") is not None:
            assert result.metadata["anon_map"].deanonymize(result.lm_text) == case["restored"]


@pytest.mark.parametrize("mode", ["placeholder", "synthetic", "reversible"])
def test_batch_preserves_input_order(mode):
    processor = cleaner({"mode": mode})
    texts = ["third@example.com", "", "first@example.com", "third@example.com"]
    assert [r.lm_text for r in processor.process_batch(texts)] == [processor.process(t).lm_text for t in texts]
