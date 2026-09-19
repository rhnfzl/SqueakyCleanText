import threading

from sct.privacy import (
    DetectedEntity,
    EntityRule,
    EntityFinding,
    InMemoryTokenStore,
    NERProcessingResult,
    ProcessingMetrics,
    PrivacyPolicy,
)
from sct import TextCleaner, TextCleanerConfig
from sct.utils.ner import GeneralNER
from sct.utils.anonymization_map import AnonymizationMap


def test_privacy_policy_applies_threshold_allowlist_and_actions():
    policy = PrivacyPolicy(
        name="support",
        version="1",
        rules={
            "PERSON": EntityRule(action="placeholder", min_score=0.8),
            "ORGANISATION": EntityRule(action="keep"),
        },
        allowlist=frozenset({"Jane Public"}),
    )
    entities = (
        DetectedEntity("PERSON", 0.95, "Jane Public", 0, 11),
        DetectedEntity("PERSON", 0.72, "John Smith", 16, 26),
        DetectedEntity("PERSON", 0.91, "Maria Chen", 31, 41),
        DetectedEntity("ORGANISATION", 0.99, "Example Corp", 45, 57),
    )

    findings = policy.evaluate(entities)

    assert [finding.action for finding in findings] == [
        "allow",
        "ignore",
        "placeholder",
        "keep",
    ]
    assert findings[0].reason == "allowlist"
    assert findings[1].reason == "below-threshold"


def test_token_store_keeps_reversible_map_behind_reference():
    anon_map = AnonymizationMap()
    anon_map.add("<PERSON_0>", "Maria Chen", "PERSON", 0, 10)
    store = InMemoryTokenStore()

    reference = store.save(anon_map)

    assert reference == anon_map.session_id
    assert store.load(reference) is anon_map
    store.delete(reference)
    assert store.load(reference) is None


def test_privacy_policy_transforms_only_selected_entities():
    policy = PrivacyPolicy(
        name="support",
        version="1",
        rules={
            "PERSON": EntityRule(action="placeholder"),
            "ORGANISATION": EntityRule(action="keep"),
        },
    )
    text = "Maria at Acme"
    findings = policy.evaluate((
        DetectedEntity("PERSON", 0.9, "Maria", 0, 5),
        DetectedEntity("ORGANISATION", 0.9, "Acme", 9, 13),
    ))

    transformed = policy.transform(text, findings)

    assert transformed == "<PERSON> at Acme"


def test_privacy_policy_redacts_union_of_overlapping_entities():
    policy = PrivacyPolicy(
        name="strict",
        version="1",
        rules={
            "PERSON": EntityRule(action="placeholder"),
            "ACCOUNT": EntityRule(action="redact"),
        },
    )
    text = "Maria123"
    findings = policy.evaluate((
        DetectedEntity("PERSON", 0.9, "Maria", 0, 5),
        DetectedEntity("ACCOUNT", 0.8, "ia123", 3, 8),
    ))

    assert policy.transform(text, findings) == "<REDACTED>"


def test_text_cleaner_returns_policy_findings(monkeypatch):
    class FakeNER:
        def __init__(self, **kwargs):
            pass

        def ner_process_detailed(self, text, *, privacy_policy, **kwargs):
            entity = DetectedEntity("PERSON", 0.95, "Maria", 0, 5)
            findings = privacy_policy.evaluate((entity,))
            return NERProcessingResult(
                text=privacy_policy.transform(text, findings),
                findings=findings,
            )

    monkeypatch.setattr("sct.sct.ner.GeneralNER", FakeNER)
    policy = PrivacyPolicy(
        name="strict",
        version="2026-09",
        rules={"PERSON": EntityRule(action="redact")},
    )
    cfg = TextCleanerConfig(
        check_ner_process=True,
        check_detect_language=False,
        check_fix_bad_unicode=False,
        check_to_ascii_unicode=False,
        check_replace_html=False,
        check_replace_urls=False,
        check_replace_emails=False,
        check_replace_years=False,
        check_replace_phone_numbers=False,
        check_replace_numbers=False,
        check_replace_currency_symbols=False,
        check_remove_isolated_letters=False,
        check_remove_isolated_special_symbols=False,
        check_normalize_whitespace=False,
        check_statistical_model_processing=False,
        language="en",
    )

    result = TextCleaner(
        cfg=cfg,
        privacy_policy=policy,
        include_entities=True,
    ).process("Maria arrived.")

    assert result.lm_text == "<REDACTED> arrived."
    assert result.metadata["policy"] == {"name": "strict", "version": "2026-09"}
    assert result.metadata["detector"]["backend"] == "onnx"
    assert (
        result.metadata["detector"]["models"]["ENGLISH"]["revision"]
        == "2c10955af9538d6ff8a68316c5dbfcd0b968761e"
    )
    assert result.metadata["findings"] == (
        EntityFinding(
            DetectedEntity("PERSON", 0.95, "Maria", 0, 5),
            "redact",
            "policy-rule",
        ),
    )


def test_general_ner_detailed_processing_returns_real_offsets():
    class FakeGLiNER:
        max_context_length = 512
        labels = ["person"]
        label_map = {"person": "PER"}

        def __call__(self, texts):
            return [[{
                "entity_group": "PER",
                "score": 0.95,
                "word": "Maria",
                "start": 0,
                "end": 5,
            }]]

    ner = object.__new__(GeneralNER)
    ner._ner_backend = "gliner"
    ner._gliner_pipe = FakeGLiNER()
    ner._replacement_mode = "placeholder"
    ner._synthetic_replacer = None
    ner._inference_locks = {}
    ner._locks_lock = threading.Lock()
    ner.tokenizer = None
    policy = PrivacyPolicy(
        name="strict",
        version="1",
        rules={"PERSON": EntityRule(action="redact")},
    )

    result = ner.ner_process_detailed(
        "Maria arrived.",
        positional_tags=("PER",),
        ner_confidence_threshold=0.8,
        language="ENGLISH",
        privacy_policy=policy,
    )

    assert result.text == "<REDACTED> arrived."
    assert result.findings[0].entity == DetectedEntity(
        "PERSON", 0.95, "Maria", 0, 5
    )


def test_detailed_processing_maps_normalized_chunk_offsets_to_source():
    class FakeGLiNER:
        max_context_length = 512
        labels = ["person"]
        label_map = {"person": "PER"}

        def __call__(self, texts):
            assert texts == ["Intro. Maria"]
            return [[{
                "entity_group": "PER",
                "score": 0.95,
                "word": "Maria",
                "start": 7,
                "end": 12,
            }]]

    ner = object.__new__(GeneralNER)
    ner._ner_backend = "gliner"
    ner._gliner_pipe = FakeGLiNER()
    ner._replacement_mode = "placeholder"
    ner._synthetic_replacer = None
    ner._inference_locks = {}
    ner._locks_lock = threading.Lock()
    ner.tokenizer = None
    ner._simple_chunk = lambda text, max_tokens: ["Intro. Maria"]
    policy = PrivacyPolicy(
        name="strict",
        version="1",
        rules={"PERSON": EntityRule(action="redact")},
    )

    result = ner.ner_process_detailed(
        "Intro.\n\nMaria",
        positional_tags=("PER",),
        ner_confidence_threshold=0.8,
        language="ENGLISH",
        privacy_policy=policy,
    )

    assert result.text == "Intro.\n\n<REDACTED>"
    assert result.findings[0].entity.start == 8


def test_text_cleaner_stores_reversible_map_externally(monkeypatch):
    class FakeNER:
        def __init__(self, **kwargs):
            pass

        def ner_process(self, text, *, anon_map, **kwargs):
            anon_map.add("<PERSON_0>", "Maria", "PERSON", 0, 5)
            return "<PERSON_0> arrived."

    monkeypatch.setattr("sct.sct.ner.GeneralNER", FakeNER)
    store = InMemoryTokenStore()
    cfg = TextCleanerConfig(
        replacement_mode="reversible",
        check_ner_process=True,
        check_detect_language=False,
        check_fix_bad_unicode=False,
        check_to_ascii_unicode=False,
        check_replace_html=False,
        check_replace_urls=False,
        check_replace_emails=False,
        check_replace_years=False,
        check_replace_phone_numbers=False,
        check_replace_numbers=False,
        check_replace_currency_symbols=False,
        check_remove_isolated_letters=False,
        check_remove_isolated_special_symbols=False,
        check_normalize_whitespace=False,
        check_statistical_model_processing=False,
        language="en",
    )

    result = TextCleaner(cfg=cfg, token_store=store).process("Maria arrived.")

    reference = result.metadata["anon_map_reference"]
    assert "anon_map" not in result.metadata
    assert store.load(reference).deanonymize(result.lm_text) == "Maria arrived."


def test_reversible_mode_restores_contact_values():
    cleaner = TextCleaner(cfg=TextCleanerConfig(
        replacement_mode="reversible",
        check_ner_process=False,
        check_detect_language=False,
        check_statistical_model_processing=False,
        check_remove_isolated_special_symbols=False,
    ))
    source = "Email maria@example.com or call +1-202-555-0147."

    result = cleaner.process(source)
    anon_map = result.metadata["anon_map"]

    assert "<EMAIL_0>" in result.lm_text
    assert "<PHONE_NUMBER_0>" in result.lm_text
    assert anon_map.deanonymize(result.lm_text) == source


def test_text_cleaner_emits_processing_metrics():
    received = []
    cfg = TextCleanerConfig(
        check_ner_process=False,
        check_detect_language=False,
        check_statistical_model_processing=False,
    )

    TextCleaner(cfg=cfg, metrics_callback=received.append).process("Plain text.")

    assert len(received) == 1
    assert isinstance(received[0], ProcessingMetrics)
    assert received[0].duration_ms >= 0
    assert received[0].backend is None
    assert received[0].entity_counts == {}


def test_text_cleaner_clears_reversible_context_after_failure():
    def fail(_text):
        raise RuntimeError("pipeline failed")

    cleaner = TextCleaner(cfg=TextCleanerConfig(
        replacement_mode="reversible",
        check_ner_process=False,
        check_detect_language=False,
        custom_pipeline_steps=(fail,),
    ))

    try:
        cleaner.process("Email maria@example.com")
    except RuntimeError:
        pass
    else:
        raise AssertionError("processing should propagate pipeline failures")

    assert cleaner._document_context.anon_map is None
