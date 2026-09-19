import pytest

from sct import TextCleanerConfig
from sct.browser import build_browser_manifest
from sct.config import DEFAULT_NER_MODELS


def test_browser_manifest_pins_onnx_model_and_records_limitations():
    manifest = build_browser_manifest(TextCleanerConfig())

    assert manifest["format_version"] == 1
    assert manifest["models"]["ENGLISH"]["revision"] == (
        "2c10955af9538d6ff8a68316c5dbfcd0b968761e"
    )
    assert "language detection" in " ".join(manifest["limitations"]).lower()


def test_browser_manifest_rejects_unpinned_custom_models():
    models = dict(DEFAULT_NER_MODELS)
    models["ENGLISH"] = "example/custom-onnx-model"
    config = TextCleanerConfig(
        ner_models=models,
    )

    with pytest.raises(ValueError, match="ENGLISH"):
        build_browser_manifest(config)
