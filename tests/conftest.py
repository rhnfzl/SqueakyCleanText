"""Shared test constants and fixtures."""

from sct.config import DEFAULT_TORCH_NER_MODELS, LANG_KEYS

# Lightweight ONNX test model shared by all NER tests (never production 7GB models)
TEST_NER_MODELS = {k: "protectai/bert-base-NER-onnx" for k in LANG_KEYS}
TEST_NER_REVISIONS = {k: "84d28ab2bc28401b4353b26033eb043c8c8a0739" for k in LANG_KEYS}
TEST_TORCH_NER_MODELS = {k: "dslim/bert-base-NER" for k in DEFAULT_TORCH_NER_MODELS}  # small torch twin


def pytest_collection_modifyitems(items):
    # shortcut: marks whole modules, so -m "not needs_model" skips some weight-free tests; mark per test if that subset matters
    for item in items:
        module = getattr(item, "module", None)
        if hasattr(module, "TEST_NER_MODELS") or hasattr(module, "TEST_TORCH_NER_MODELS"):
            item.add_marker("needs_model")
