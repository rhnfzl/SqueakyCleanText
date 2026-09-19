import threading
import time
import warnings

import pytest

from sct import EntityRule, PrivacyPolicy, TextCleaner, TextCleanerConfig
from sct.utils.contact import ProcessContacts
from sct.utils.ner import GeneralNER


def test_process_batch_uses_batch_size_as_concurrency_limit():
    lock = threading.Lock()
    active = 0
    max_active = 0

    def observe_concurrency(text):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.01)
        with lock:
            active -= 1
        return text

    cleaner = TextCleaner(cfg=TextCleanerConfig(
        check_ner_process=False,
        check_detect_language=False,
        check_statistical_model_processing=False,
        custom_pipeline_steps=(observe_concurrency,),
    ))

    results = cleaner.process_batch(["one", "two", "three"], batch_size=1)

    assert [result.lm_text for result in results] == ["one", "two", "three"]
    assert max_active == 1


def test_html_replacement_honors_configured_token():
    result = ProcessContacts().replace_html(
        "<p>Hello <strong>world</strong></p>",
        replace_with="[TAG]",
    )

    assert result == "[TAG]Hello [TAG]world[TAG][TAG]"


def test_presidio_backend_uses_detected_language():
    calls = []

    class FakeAnalyzer:
        def analyze(self, *, text, language):
            calls.append((text, language))
            return []

    ner = object.__new__(GeneralNER)
    ner._ner_backend = "presidio_gliner"
    ner._analyzer = FakeAnalyzer()
    policy = PrivacyPolicy(
        name="strict",
        version="1",
        rules={"PERSON": EntityRule()},
    )

    ner.ner_process_detailed(
        "Piet woont hier.",
        positional_tags=("PER",),
        language="DUTCH",
        privacy_policy=policy,
    )

    assert calls == [("Piet woont hier.", "nl")]


def test_gliclass_onnx_rejects_unsupported_configuration():
    with pytest.raises(ValueError, match="GLiClass ONNX is not supported"):
        TextCleanerConfig(gliclass_onnx=True)


def test_default_config_does_not_emit_deprecation_warning():
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        TextCleanerConfig()

    assert not [
        warning for warning in recorded
        if issubclass(warning.category, DeprecationWarning)
    ]


def test_general_ner_passes_pinned_revision_to_model_loader(monkeypatch):
    calls = []

    class FakeTokenizer:
        max_len_single_sentence = 510

    def fake_loader(model_name, *, device, cache_dir, revision):
        calls.append((model_name, revision))
        return object(), FakeTokenizer()

    monkeypatch.setattr("sct.utils.ner.load_onnx_ner_model", fake_loader)

    GeneralNER(
        model_names={"ENGLISH": "org/en", "MULTILINGUAL": "org/multi"},
        model_revisions={"ENGLISH": "sha-en", "MULTILINGUAL": "sha-multi"},
    )

    assert calls == [("org/en", "sha-en"), ("org/multi", "sha-multi")]


def test_default_onnx_models_have_pinned_revisions():
    revisions = TextCleanerConfig().ner_model_revisions

    assert set(revisions) >= {
        "ENGLISH", "DUTCH", "GERMAN", "SPANISH", "MULTILINGUAL"
    }
    assert all(len(revision) == 40 for revision in revisions.values())


def test_explicit_default_gliclass_model_keeps_pinned_revision():
    config = TextCleanerConfig(
        gliclass_model="knowledgator/gliclass-edge-v3.0",
    )

    assert config.gliclass_revision == (
        "df03993a2ed98e5e4a0d2dd7efbbd105abe874cf"
    )


def test_reversible_mode_rejects_unmapped_fuzzy_date_replacement():
    with pytest.raises(ValueError, match="fuzzy date replacement"):
        TextCleanerConfig(
            replacement_mode="reversible",
            check_fuzzy_replace_dates=True,
        )
