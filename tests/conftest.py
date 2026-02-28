"""Shared test constants and fixtures."""

from sct.config import LANG_KEYS

# Lightweight ONNX test model shared by all NER tests (never production 7GB models)
TEST_NER_MODELS = {k: "protectai/bert-base-NER-onnx" for k in LANG_KEYS}
