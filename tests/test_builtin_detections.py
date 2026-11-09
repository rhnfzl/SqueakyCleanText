"""Native analysis returns canonical spans without rendering replacements."""

from sct import TextCleanerConfig
from sct.findings import DetectionContext
from tests.helpers import FakeRunDetector, word_ner


def test_regex_keeps_overlapping_hits_on_original_stage():
    from sct.detectors_builtin import BuiltinRegexDetector

    text = 'a@example.com 4111 1111 1111 1111 999.999.999.999 4111111111111112 BE68 5390 0754 7034 EUR1'
    detector = BuiltinRegexDetector(TextCleanerConfig(check_ner_process=False))
    hits = detector.detect(text, language=None, context=DetectionContext())
    assert ('EMAIL', 0, 13) in [(h.entity_type, h.start, h.end) for h in hits]
    assert any(h.entity_type == 'CARD' and text[h.start:h.end] == '4111 1111 1111 1111' for h in hits)
    assert not any(h.entity_type == 'IP' for h in hits)
    assert not any(h.entity_type == 'CARD' and text[h.start:h.end] == '4111111111111112' for h in hits)
    assert any(h.entity_type == 'IBAN' and text[h.start:h.end] == 'BE68 5390 0754 7034' for h in hits)
    assert all(h.score is None and h.coordinate_space == 'canonical' for h in hits)


def test_ner_analysis_keeps_window_offsets_without_rendering():
    ner = word_ner(1, FakeRunDetector(min_words=2), budget=8)
    text = 'a b c d John Smith e f g h i j'
    ner.anonymize_text = lambda *a, **k: (_ for _ in ()).throw(AssertionError('rendered'))
    hits = ner.analyze(text, language=None, context=DetectionContext())
    assert [(h.entity_type, h.start, h.end) for h in hits] == [('PERSON', 8, 18)]


def test_regex_custom_rules_dates_allow_terms_and_validations():
    from sct.detectors_builtin import BuiltinRegexDetector

    cfg = TextCleanerConfig(
        check_ner_process=False, custom_patterns={'TOKEN': r'key-\d+'},
        custom_deny_terms={'PERSON': ('Alice',)}, custom_allow_terms=('key-99',),
        custom_month_names={'ENGLISH': ('testmonth',)}, date_builtin_month_languages=(),
        replace_ips_public_only=True, phone_reject_embedded=True,
    )
    text = 'Alice key-12 key-99 15 testmonth 1990 8.8.8.8 192.168.1.1 x1234567890y'
    hits = BuiltinRegexDetector(cfg).detect(text, language=None, context=DetectionContext())
    values = [(h.entity_type, text[h.start:h.end]) for h in hits]
    assert ('PERSON', 'Alice') in values
    assert ('TOKEN', 'key-12') in values
    assert ('TOKEN', 'key-99') not in values
    assert ('DATE', '15 testmonth 1990') in values
    assert ('IP', '8.8.8.8') in values
    assert ('IP', '192.168.1.1') not in values
    assert ('PHONE', '1234567890') not in values


def test_presidio_analysis_never_renders_or_applies_policy():
    from types import SimpleNamespace
    from tests.helpers import bare_ner, set_cfg

    ner = bare_ner()
    set_cfg(ner, ner_backend='presidio_gliner')
    ner._presidio_types = {'PER': 'PERSON'}

    class Analyzer:
        supported_languages = ['en']

        def analyze(self, *, text, language):
            assert text == '<ALICE>' and language == 'en'
            return [SimpleNamespace(entity_type='PER', start=1, end=6, score=0.9)]

    ner._analyzer = Analyzer()
    ner._privacy_policy = object()
    hits = ner.analyze('<ALICE>', language='DUTCH', context=DetectionContext())
    assert [(h.entity_type, h.start, h.end, h.score) for h in hits] == [('PERSON', 1, 6, 0.9)]


def test_canonical_ner_offsets_preserve_angle_brackets():
    ner = word_ner(0, FakeRunDetector(min_words=2), budget=20)
    hits = ner.analyze('<John Smith>', language=None, context=DetectionContext())
    assert [(h.start, h.end) for h in hits] == [(1, 11)]


def test_disabled_regex_steps_do_not_claim_capabilities():
    from sct.detectors_builtin import BuiltinRegexDetector

    detector = BuiltinRegexDetector(TextCleanerConfig(check_replace_emails=False, check_ner_process=False))
    assert 'EMAIL' not in detector.capabilities.entity_types
    hits = detector.detect('a@example.com', language=None, context=DetectionContext())
    assert not any(h.entity_type == 'EMAIL' for h in hits)


def test_token_and_ensemble_analysis_matches_legacy_without_extra_inference():
    from tests.helpers import route_models, set_cfg, WordCountTokenizer

    for backend in ('onnx', 'torch', 'ensemble_onnx', 'ensemble_torch', 'gliner'):
        detector = FakeRunDetector(min_words=2)
        ner = word_ner(1, detector, budget=8)
        set_cfg(ner, ner_backend=backend)
        ner._model_names = {'MULTILINGUAL': 'fake/model'}
        ner._model_revisions = {'MULTILINGUAL': None}
        calls = []

        def token_pipe(texts, **kwargs):
            calls.append(tuple(texts))
            return detector(texts)

        ner._pipelines = {'MULTILINGUAL': token_pipe}
        route_models(ner, {'MULTILINGUAL': WordCountTokenizer(1)}, {'MULTILINGUAL': ('MULTILINGUAL',)})
        text = 'a b c d John Smith e f g h i j'
        hits = ner.analyze(text, language=None, context=DetectionContext())
        assert [(h.entity_type, h.start, h.end) for h in hits] == [('PERSON', 8, 18)], backend
        count = len(calls)
        legacy = ner.ner_process_detailed(text)
        assert legacy.text == 'a b c d <PERSON> e f g h i j', backend
        assert len(calls) == 2 * count, backend


def test_fuzzy_date_offsets_follow_candidate_boundaries():
    from sct.detectors_builtin import BuiltinRegexDetector

    detector = BuiltinRegexDetector(TextCleanerConfig(check_ner_process=False, check_fuzzy_replace_dates=True))
    text = 'x15 Januarry 1990 15 Januarry 1990'
    hits = detector.detect(text, language='ENGLISH', context=DetectionContext())
    assert [(h.start, h.end) for h in hits if h.entity_type == 'DATE'] == [(18, 34)]


def test_ner_analysis_normalizes_provider_numeric_scores():
    import numpy as np
    from tests.helpers import gliner_ner

    class NumericScores:
        max_context_length = 100
        labels = ['person']
        label_map = {'person': 'PER'}

        def __call__(self, texts, batch_size=1):
            return [[{'entity_group': 'PER', 'score': np.float32(0.99), 'word': 'Alice',
                      'start': 0, 'end': 5}] for text in texts]

    ner = gliner_ner(detector=NumericScores())
    hits = ner.analyze('Alice', language=None, context=DetectionContext())
    assert len(hits) == 1 and isinstance(hits[0].score, float)
