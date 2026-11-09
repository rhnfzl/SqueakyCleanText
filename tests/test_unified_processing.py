"""Unified findings protect one canonical text stage and retain honest provenance."""
import asyncio
from dataclasses import replace
import json
import threading

import pytest

from sct import TextCleaner, TextCleanerConfig
from sct.detectors import DetectorCapabilities, DetectorRegistry, DetectorSpec
from sct.findings import Detection, DetectionContext


def config(**kwargs):
    defaults = dict(pii_processing='unified', check_detect_language=False, check_ner_process=False,
                    check_fix_bad_unicode=False, check_replace_html=False, check_remove_isolated_letters=False,
                    check_remove_isolated_special_symbols=False, check_normalize_whitespace=False)
    return TextCleanerConfig(**(defaults | kwargs))


def provider(hits, *, inspect=None):
    registry = DetectorRegistry()
    class Provider:
        capabilities = DetectorCapabilities('test', entity_types=('PERSON', 'EMAIL'), supports_context=True)
        def detect(self, text, *, language, context):
            if inspect:
                inspect(text, context, id(self))
            return tuple(Detection(kind, start, end, 'test', score, 'canonical')
                         for kind, start, end, score in hits)
    registry.register('test', lambda cfg, options: Provider())
    return registry


def policy(rules):
    from sct.policy_v2 import UnifiedPrivacyPolicy, HandlingRule
    return UnifiedPrivacyPolicy('test', '1', {key: HandlingRule(**value) for key, value in rules.items()})


@pytest.mark.parametrize('mode', ['placeholder', 'synthetic', 'reversible'])
def test_mixed_detectors_share_canonical_text_and_policy(mode):
    seen = []
    text = 'Alice a@example.com'
    cleaner = TextCleaner(config(replacement_mode=mode,
        detector_specs=(DetectorSpec('builtin_regex'), DetectorSpec('test'))),
        detector_registry=provider([('PERSON', 0, 5, .9)], inspect=lambda text, *_: seen.append(text)),
        unified_policy=policy({'EMAIL': {'action': 'redact'}}))
    result = cleaner.process(text)
    assert seen == [text]
    assert 'Alice' not in result.lm_text and 'a@example.com' not in result.lm_text
    assert result.metadata['counts_v2'] == {'version': 2, 'detector_hits': 2, 'resolved_replacements': 2}
    assert 'a@example.com' not in json.dumps(result.metadata['findings_v2'])
    if mode == 'reversible':
        amap = result.metadata['anon_map']
        assert len(amap.entries) == 1
        assert 'Alice' in amap.deanonymize(result.lm_text)
        assert 'a@example.com' not in amap.deanonymize(result.lm_text)


def test_protected_overlap_union_is_transitive_and_redact_dominates():
    cleaner = TextCleaner(config(detector_specs=(DetectorSpec('test'),), replacement_mode='reversible'),
        detector_registry=provider([('PERSON', 0, 4, .99), ('EMAIL', 3, 7, .1), ('PERSON', 6, 10, .9)]),
        unified_policy=policy({'EMAIL': {'action': 'redact'}}))
    result = cleaner.process('abcdefghij')
    assert result.lm_text == '<REDACTED>'
    assert not result.metadata['anon_map'].entries
    assert result.metadata['counts_v2']['resolved_replacements'] == 1


def test_keep_cannot_exempt_overlapping_protection():
    cleaner = TextCleaner(config(detector_specs=(DetectorSpec('test'),)),
        detector_registry=provider([('PERSON', 0, 5, .9), ('EMAIL', 3, 8, None)]),
        unified_policy=policy({'PERSON': {'action': 'keep'}}))
    assert cleaner.process('abcdefgh').lm_text == 'abc<EMAIL>'


def test_unknown_score_protects_and_threshold_equality_passes():
    cleaner = TextCleaner(config(detector_specs=(DetectorSpec('test'),)),
        detector_registry=provider([('PERSON', 0, 5, None)]),
        unified_policy=policy({'PERSON': {'min_score': .9}}))
    assert cleaner.process('Alice').lm_text == '<PERSON>'
    stricter = TextCleaner(cleaner.cfg, detector_registry=provider([('PERSON', 0, 5, None)]),
        unified_policy=policy({'PERSON': {'min_score': .9, 'unknown_score_action': 'error'}}))
    with pytest.raises(ValueError, match='score'):
        stricter.process('Alice')


def test_source_markers_are_reserved():
    result = TextCleaner(config(replacement_mode='reversible')).process('<EMAIL_0> a@example.com')
    assert result.lm_text == '<EMAIL_0> <EMAIL_1>'
    assert result.metadata['anon_map'].deanonymize(result.lm_text) == '<EMAIL_0> a@example.com'


@pytest.mark.parametrize('opaque', [lambda text: text, str.lower])
def test_opaque_callbacks_invalidate_mapping_even_when_text_is_equal(opaque):
    cleaner = TextCleaner(config(custom_pipeline_steps=(opaque,)))
    result = cleaner.process('a@example.com')
    assert result.metadata['findings_v2'][0]['mapping_quality'] == 'unavailable'
    assert result.metadata['findings_v2'][0]['source_ranges'] == []
    with pytest.raises(ValueError, match='source'):
        TextCleaner(replace(cleaner.cfg, source_mapping='require_source')).process('a@example.com')


def test_unicode_ascii_expansion_and_html_preserve_source_cover():
    text = '<b>é</b> a@example.com'
    cleaner = TextCleaner(config(check_replace_html=True, check_to_ascii_unicode=True))
    result = cleaner.process(text)
    finding = result.metadata['findings_v2'][0]
    assert finding['source_ranges'] == [{'start': 9, 'end': 22}]
    assert finding['mapping_quality'] == 'exact'
    assert result.lm_text == '<HTML>e<HTML> <EMAIL>'


def test_identical_repeated_text_cannot_move_source_ranges():
    cleaner = TextCleaner(config(check_to_ascii_unicode=True))
    text = 'é a@example.com é a@example.com'
    findings = cleaner.process(text).metadata['findings_v2']
    assert [item['source_ranges'] for item in findings] == [
        [{'start': 2, 'end': 15}], [{'start': 18, 'end': 31}]]


def test_contexts_batch_async_order_and_thread_ownership():
    seen = []
    lock = threading.Lock()
    def inspect(text, context, identity):
        with lock:
            seen.append((text, context.document_id, threading.get_ident(), identity))
    cleaner = TextCleaner(config(detector_specs=(DetectorSpec('test'),)), detector_registry=provider([], inspect=inspect))
    contexts = [DetectionContext(document_id=str(i)) for i in range(8)]
    texts = [f'row {i}' for i in range(8)]
    results = asyncio.run(cleaner.aprocess_batch(texts, max_workers=3, contexts=contexts))
    assert [result.lm_text for result in results] == texts
    assert {(text, doc) for text, doc, *_ in seen} == set(zip(texts, map(str, range(8))))
    owners = {}
    for _, _, thread, identity in seen:
        owners.setdefault(identity, set()).add(thread)
    assert all(len(threads) == 1 for threads in owners.values())
    with pytest.raises(ValueError, match='contexts'):
        cleaner.process_batch(texts, contexts=contexts[:1])


def test_external_selection_does_not_initialize_native_models(monkeypatch):
    def forbidden(*args):
        pytest.fail('Native model initialization must not occur')
    monkeypatch.setattr(TextCleaner, '_build_ner', forbidden)
    monkeypatch.setattr(TextCleaner, '_build_gliclass', forbidden)
    cleaner = TextCleaner(config(check_ner_process=True,
        detector_specs=(DetectorSpec('test'),)), detector_registry=provider([]))
    assert cleaner.process('hello').lm_text == 'hello'


def test_legacy_rejects_unified_policy_and_context():
    cfg = TextCleanerConfig(check_ner_process=False)
    with pytest.raises(ValueError, match='unified'):
        TextCleaner(cfg, unified_policy=policy({}))
    with pytest.raises(ValueError, match='context'):
        TextCleaner(cfg).process('hello', context=DetectionContext(document_id='private'))


def test_invalid_provider_result_fails_without_sensitive_output():
    cleaner = TextCleaner(config(detector_specs=(DetectorSpec('test'),)),
                          detector_registry=provider([('PERSON', 0, 100, .9)]))
    with pytest.raises(ValueError) as error:
        cleaner.process('private')
    assert 'private' not in str(error.value)


def test_empty_and_buffered_results_use_versioned_metadata():
    from sct import BufferedStreamingRedactor
    cleaner = TextCleaner(config())
    assert cleaner.process('').metadata['counts_v2']['detector_hits'] == 0
    stream = BufferedStreamingRedactor(cleaner)
    stream.feed('a@')
    stream.feed('example.com')
    assert stream.finish() == cleaner.process('a@example.com').lm_text


def test_unsupported_surfaces_reject_unified():
    from sct import build_browser_manifest, WindowedStreamingRedactor
    cleaner = TextCleaner(config())
    with pytest.raises(ValueError, match='unified'):
        build_browser_manifest(cleaner.cfg)
    with pytest.raises(ValueError, match='unified'):
        WindowedStreamingRedactor(cleaner)


@pytest.mark.parametrize('include', [False, True])
def test_sensitive_values_require_explicit_opt_in_in_single_and_batch(include):
    cleaner = TextCleaner(config(), include_entities=include, unified_policy=policy({'EMAIL': {'action': 'redact'}}))
    for result in [cleaner.process('a@example.com'), *cleaner.process_batch(['a@example.com'])]:
        assert 'a@example.com' not in repr(result)
        assert 'a@example.com' not in json.dumps(result.metadata['findings_v2'])
        if include:
            assert result.metadata['findings_v2_values'] == [
                {'finding_index': 0, 'coordinate_space': 'canonical', 'value': 'a@example.com'}]
        else:
            assert 'findings_v2_values' not in result.metadata
            assert 'a@example.com' not in json.dumps(result.metadata)


def test_reversible_map_does_not_expand_source_markers_inside_restored_values():
    cfg = config(replacement_mode='reversible', detector_specs=(DetectorSpec('test'),))
    text = '<EMAIL_0> Alice'
    cleaner = TextCleaner(cfg, detector_registry=provider([('PERSON', 0, 9, .9), ('EMAIL', 10, 15, .9)]))
    result = cleaner.process(text)
    assert result.metadata['anon_map'].deanonymize(result.lm_text) == text


def test_complex_unicode_repair_reports_covering_for_repeated_text():
    text = 'Ａlice Ａlice a@example.com'
    cleaner = TextCleaner(config(check_fix_bad_unicode=True))
    finding = cleaner.process(text).metadata['findings_v2'][0]
    assert finding['mapping_quality'] == 'covering'
    assert finding['source_ranges'] == [{'start': 0, 'end': len(text)}]


def test_post_render_map_projects_replacement_and_unmodified_tail():
    cleaner = TextCleaner(config())
    document = cleaner._process_unified_document('a@example.com tail')
    assert document.result.lm_text == '<EMAIL> tail'
    replaced = document.source_map.project(0, 7)
    tail = document.source_map.project(8, 12)
    assert [(r.start, r.end) for r in replaced.ranges] == [(0, 13)]
    assert replaced.mapping_quality == 'covering'
    assert [(r.start, r.end) for r in tail.ranges] == [(14, 18)]
    assert tail.mapping_quality == 'exact'
    assert 'a@example.com' not in repr(document)


def test_invalid_source_ranges_from_provider_are_rejected():
    from sct.findings import TextRange
    registry = DetectorRegistry()
    class BadProvider:
        capabilities = DetectorCapabilities('bad', entity_types=('PERSON',))
        def detect(self, text, **kwargs):
            return (Detection('PERSON', 0, 5, 'bad', .9, 'canonical', (TextRange(0, 900),), 'exact'),)
    registry.register('bad', lambda cfg, options: BadProvider())
    cleaner = TextCleaner(config(detector_specs=(DetectorSpec('bad'),)), detector_registry=registry)
    with pytest.raises(ValueError):
        cleaner.process('Alice')


def test_unknown_scores_below_threshold_and_tie_resolution_are_deterministic():
    cleaner = TextCleaner(config(detector_specs=(DetectorSpec('test'),)),
        detector_registry=provider([('PERSON', 0, 5, .2), ('EMAIL', 0, 5, .9)]))
    assert cleaner.process('Alice').lm_text == '<EMAIL>'
    threshold = TextCleaner(cleaner.cfg, detector_registry=provider([('PERSON', 0, 5, .9)]),
                            unified_policy=policy({'PERSON': {'min_score': .9}}))
    assert threshold.process('Alice').lm_text == '<PERSON>'
    below = TextCleaner(cleaner.cfg, detector_registry=provider([('PERSON', 0, 5, .8)]),
                        unified_policy=policy({'PERSON': {'min_score': .9}}))
    assert below.process('Alice').lm_text == 'Alice'


def test_builtin_network_permission_is_required_before_detection():
    cleaner = TextCleaner(config(check_ner_process=True))
    from sct.detectors import DetectorUnavailable
    with pytest.raises(DetectorUnavailable, match='network permission'):
        cleaner.process('Alice')


def test_token_store_metrics_and_statistical_text_remain_separate():
    from sct import InMemoryTokenStore
    store, metrics = InMemoryTokenStore(), []
    cleaner = TextCleaner(config(replacement_mode='reversible'), token_store=store, metrics_callback=metrics.append)
    result = cleaner.process('Contact a@example.com')
    assert result.stat_text == 'contact email0'
    assert 'anon_map' not in result.metadata
    amap = store.load(result.metadata['anon_map_reference'])
    assert amap.deanonymize(result.lm_text) == 'Contact a@example.com'
    assert len(metrics) == 1 and dict(metrics[0].entity_counts) == {}
    assert 'a@example.com' not in repr(metrics)


def test_provider_and_custom_errors_do_not_expose_values():
    registry = DetectorRegistry()
    class FailedProvider:
        capabilities = DetectorCapabilities('failure', entity_types=('PERSON',))
        def detect(self, text, **kwargs):
            raise RuntimeError(text)
    registry.register('failure', lambda cfg, options: FailedProvider())
    cleaner = TextCleaner(config(detector_specs=(DetectorSpec('failure'),)), detector_registry=registry)
    sensitive = 'private-value'
    with pytest.raises(ValueError) as error:
        cleaner.process(sensitive)
    import traceback
    assert 'private-value' not in ''.join(traceback.format_exception(error.value))


def test_configuration_validation_and_explicit_config():
    with pytest.raises(ValueError):
        config(pii_processing='typo')
    with pytest.raises(ValueError):
        config(source_mapping='typo')
    cleaner = TextCleaner(config(pii_processing='unified', source_mapping='require_source'))
    assert cleaner.cfg.pii_processing == 'unified'
    assert cleaner.cfg.source_mapping == 'require_source'


def test_unavailable_detector_retains_typed_safe_failure():
    from sct.detectors import DetectorUnavailable
    cleaner = TextCleaner(config(check_ner_process=True))
    with pytest.raises(DetectorUnavailable):
        cleaner.process('Alice')


def test_classification_cannot_load_before_unified_network_permission(monkeypatch):
    def forbidden(*args):
        pytest.fail('Classification initialized before permission validation')
    monkeypatch.setattr(TextCleaner, '_build_gliclass', forbidden)
    with pytest.raises(ValueError, match='network'):
        TextCleaner(config(check_ner_process=True, check_classify_document=True))


def test_cleaner_config_view_uses_its_own_unified_detector_flags():
    cleaner = TextCleaner(config())
    view = cleaner._with_config(replace(cleaner.cfg, check_replace_emails=False))
    assert view.process('a@example.com').lm_text == 'a@example.com'
    assert cleaner.process('a@example.com').lm_text == '<EMAIL>'


def test_builtin_regex_timeout_warning_reaches_document_metadata():
    cleaner = TextCleaner(config(custom_patterns={'TOKEN': '(a+)+$'}, custom_patterns_timeout=.000001,
                                  custom_patterns_on_timeout='warn'))
    result = cleaner.process('a' * 10000 + '!')
    assert 'custom_pattern_timeout' in result.metadata['warnings']


def test_legacy_rejects_strict_source_mapping_instead_of_ignoring_it():
    with pytest.raises(ValueError, match='unified'):
        TextCleaner(TextCleanerConfig(check_ner_process=False, source_mapping='require_source'))


def test_native_capability_declares_possible_network_artifact_access():
    from sct.detectors_builtin import BuiltinNERDetector
    from tests.helpers import FakeRunDetector, word_ner
    detector = BuiltinNERDetector(word_ner(0, FakeRunDetector(min_words=2)))
    assert detector.capabilities.network_access is True


def test_provider_unavailability_during_detection_is_typed_and_safe():
    from sct.detectors import DetectorUnavailable
    registry = DetectorRegistry()
    class Unavailable:
        capabilities = DetectorCapabilities('unavailable')
        def detect(self, text, **kwargs):
            raise DetectorUnavailable(text)
    registry.register('unavailable', lambda cfg, options: Unavailable())
    cleaner = TextCleaner(config(detector_specs=(DetectorSpec('unavailable'),)), detector_registry=registry)
    with pytest.raises(DetectorUnavailable) as error:
        cleaner.process('private-value')
    assert 'private-value' not in str(error.value)


def test_custom_markers_and_keep_policy_survive_rendering():
    cleaner = TextCleaner(config(replace_with_email='EMAIL', custom_patterns={'PERSON': 'Alice'}),
                          unified_policy=policy({'PERSON': {'action': 'keep'}}))
    result = cleaner.process('Alice a@example.com')
    assert result.lm_text == 'Alice EMAIL'
    assert result.metadata['counts_v2'] == {'version': 2, 'detector_hits': 2, 'resolved_replacements': 1}


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -.1, 1.1, True])
def test_invalid_policy_score_thresholds_fail_closed(value):
    from sct import HandlingRule
    with pytest.raises(ValueError):
        HandlingRule(min_score=value)


def test_current_subclass_batch_signature_keeps_single_and_async_callers():
    class ExistingCleaner(TextCleaner):
        def process_batch(self, texts, max_workers=None):
            return super().process_batch(texts, max_workers)
    cleaner = ExistingCleaner(TextCleanerConfig(check_ner_process=False, check_detect_language=False))
    assert cleaner.process('hello').lm_text == 'hello'
    assert asyncio.run(cleaner.aprocess_batch(['hello']))[0].lm_text == 'hello'


def test_native_allow_terms_preserve_normalized_and_repeated_names(monkeypatch):
    from tests.helpers import FakeRunDetector, word_ner
    import sct.utils.ner as native
    def model(cfg, **kwargs):
        instance = word_ner(0, FakeRunDetector())
        instance._allow_regex = kwargs.get('allow_regex')
        return instance
    monkeypatch.setattr(native, 'GeneralNER', model)
    monkeypatch.setattr(TextCleaner, '_detect_language', lambda self, text: (None, None, 'none', None))
    cleaner = TextCleaner(config(check_ner_process=True, allow_network_detectors=True,
                                  check_to_ascii_unicode=True, custom_allow_terms=('Álice',)))
    result = cleaner.process('Álice and Alice; Bob')
    assert result.lm_text == 'Alice and Alice; <PERSON>'
    assert [finding['entity_type'] for finding in result.metadata['findings_v2']] == ['PERSON']


@pytest.mark.parametrize('kind', ['foo', 'vendor.Person', 'PERSON-type', 'person name'])
def test_external_label_source_markers_survive_roundtrip_and_serialization(kind):
    from sct import AnonymizationMap
    text = f'<{kind}_0> Alice'
    registry = DetectorRegistry()
    class Provider:
        capabilities = DetectorCapabilities('external', entity_types=(kind,))
        def detect(self, text, **kwargs):
            return (Detection(kind, len(text) - 5, len(text), 'external', None, 'canonical'),)
    registry.register('external', lambda cfg, options: Provider())
    cleaner = TextCleaner(config(replacement_mode='reversible', detector_specs=(DetectorSpec('external'),)),
                          detector_registry=registry)
    result = cleaner.process(text)
    assert result.lm_text == f'<{kind}_0> <{kind}_1>'
    restored = AnonymizationMap.from_dict(result.metadata['anon_map'].to_dict())
    assert restored.deanonymize(result.lm_text) == text
    assert restored.next_placeholder(kind) == f'<{kind}_2>'
    assert f'<{kind}_0>' in restored.reserved


def test_allow_terms_apply_to_external_spans_without_exempting_larger_names():
    cleaner = TextCleaner(config(detector_specs=(DetectorSpec('test'),), custom_allow_terms=('Alice',)),
        detector_registry=provider([('PERSON', 0, 5, .9), ('PERSON', 10, 21, .9)]))
    result = cleaner.process('Alice and Alice Smith')
    assert result.lm_text == 'Alice and <PERSON>'
    assert result.metadata['counts_v2'] == {'version': 2, 'detector_hits': 1, 'resolved_replacements': 1}


def test_external_marker_reservation_uses_normalized_source():
    from sct import AnonymizationMap
    registry = DetectorRegistry()
    class Provider:
        capabilities = DetectorCapabilities('external', entity_types=('foo',))
        def detect(self, text, **kwargs):
            return (Detection('foo', len(text) - 5, len(text), 'external', None, 'canonical'),)
    registry.register('external', lambda cfg, options: Provider())
    cleaner = TextCleaner(config(replacement_mode='reversible', detector_specs=(DetectorSpec('external'),),
                                  check_fix_bad_unicode=True), detector_registry=registry)
    result = cleaner.process('&lt;foo_0&gt; Alice')
    assert result.lm_text == '<foo_0> <foo_1>'
    restored = AnonymizationMap.from_dict(result.metadata['anon_map'].to_dict())
    assert restored.deanonymize(result.lm_text) == '<foo_0> Alice'


def test_explicit_classification_runs_before_external_detection(monkeypatch):
    calls = []
    class Classifier:
        def classify(self, text):
            calls.append(('classify', text))
            return [{'label': 'letter', 'score': .9}]
    monkeypatch.setattr(TextCleaner, '_build_gliclass', lambda self: Classifier())
    cleaner = TextCleaner(config(check_ner_process=True, check_classify_document=True, allow_network_detectors=True,
        detector_specs=(DetectorSpec('test'),)),
        detector_registry=provider([], inspect=lambda text, *_: calls.append(('detect', text))))
    result = cleaner.process('hello')
    assert result.metadata['classes'] == [{'label': 'letter', 'score': .9}]
    assert calls == [('classify', 'hello'), ('detect', 'hello')]


def test_explicit_classification_requires_network_permission_even_with_external_only_selection(monkeypatch):
    def forbidden(*args):
        pytest.fail('Classification loaded before network permission')
    monkeypatch.setattr(TextCleaner, '_build_gliclass', forbidden)
    with pytest.raises(ValueError, match='network'):
        TextCleaner(config(check_classify_document=True, detector_specs=(DetectorSpec('test'),)),
                    detector_registry=provider([]))


def test_unified_classification_failure_hides_input_values(monkeypatch):
    class Classifier:
        def classify(self, text):
            raise RuntimeError(text)
    monkeypatch.setattr(TextCleaner, '_build_gliclass', lambda self: Classifier())
    cleaner = TextCleaner(config(check_classify_document=True, allow_network_detectors=True))
    sensitive = 'private-value'
    with pytest.raises(ValueError) as error:
        cleaner.process(sensitive)
    assert sensitive not in str(error.value)


def test_browser_manifest_rejects_strict_mapping_in_legacy_configuration():
    from sct import build_browser_manifest
    cfg = TextCleanerConfig(check_ner_process=False, source_mapping='require_source')
    with pytest.raises(ValueError, match='source mapping'):
        build_browser_manifest(cfg)


def test_windowed_streaming_rejects_strict_mapping_on_config_views():
    from sct import WindowedStreamingRedactor
    cleaner = TextCleaner(TextCleanerConfig(check_ner_process=False))
    view = cleaner._with_config(replace(cleaner.cfg, source_mapping='require_source'))
    with pytest.raises(ValueError, match='source mapping'):
        WindowedStreamingRedactor(view)


def test_buffered_unified_streaming_keeps_supported_strict_mapping():
    from sct import BufferedStreamingRedactor
    cleaner = TextCleaner(config(source_mapping='require_source'))
    stream = BufferedStreamingRedactor(cleaner)
    stream.feed('a@')
    stream.feed('example.com')
    assert stream.finish() == '<EMAIL>'
    assert cleaner.process('a@example.com').metadata['findings_v2'][0]['mapping_quality'] == 'exact'
