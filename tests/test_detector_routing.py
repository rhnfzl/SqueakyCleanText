"""Routing skips a stage only with matching calibration and declared coverage."""
from dataclasses import replace
import importlib.util
import json
import pickle

import pytest

from sct import TextCleaner, TextCleanerConfig
from sct.detectors import DetectorCapabilities, DetectorRegistry, DetectorSpec
from sct.findings import Detection, DetectionContext
from sct.language_policy import LanguageDecision


def api():
    assert importlib.util.find_spec('sct.routing') is not None, 'The routing API is missing'
    from sct import routing
    return routing


def config(**kwargs):
    values = dict(pii_processing='unified', check_ner_process=False, language='en',
                  check_fix_bad_unicode=False, check_replace_html=False, check_remove_isolated_letters=False,
                  check_remove_isolated_special_symbols=False, check_normalize_whitespace=False)
    return TextCleanerConfig(**(values | kwargs))


def cascade(**kwargs):
    return config(**(dict(routing_policy='calibrated_cascade',
        routing_fast_specs=(DetectorSpec('builtin_regex'),), routing_full_specs=(DetectorSpec('names'),),
        routing_entity_types=('EMAIL',)) | kwargs))


def artifact(cfg, capabilities=None):
    from sct.detectors_builtin import BuiltinRegexDetector
    routing = api()
    return dict(schema_version=1, config_digest=routing.routing_config_digest(cfg),
        fast_capabilities_digest=routing.capabilities_digest(
            (BuiltinRegexDetector(cfg).capabilities,) if capabilities is None else capabilities),
        language_min_confidence=.90, language_min_margin=.10, max_chars=256,
        languages=['ENGLISH'], entity_types=['EMAIL'])


def calibrated(**kwargs):
    cfg = cascade(**kwargs)
    return replace(cfg, routing_calibration=artifact(cfg))


def registry(events):
    result = DetectorRegistry()
    class Names:
        capabilities = DetectorCapabilities('names', entity_types=('PERSON',), supports_context=True)
        def detect(self, text, *, language, context):
            events.append(('detect', language))
            start = text.find('Alice')
            return () if start < 0 else (Detection('PERSON', start, start + 5, 'names', .95, 'canonical'),)
    def create(cfg, options):
        events.append(('create', None))
        return Names()
    result.register('names', create)
    return result


def test_l2_acceptance():
    routing = api()
    cfg = calibrated()
    policy = routing.RoutingPolicy(cfg)
    context = routing.RoutingContext(LanguageDecision('ENGLISH', .9, .1, 'detected', 'detected'), ('EMAIL',), 30)
    assert policy.plan(context=context, completed=()).selected_ids == ('fast',)
    from sct.detectors_builtin import BuiltinRegexDetector
    completed = (routing.DetectorRun('fast', (BuiltinRegexDetector(cfg).capabilities,), calls=1),)
    decision = policy.plan(context=context, completed=completed)
    assert decision.selected_ids == () and decision.skipped_ids == ('full',)
    assert decision.reasons == ('calibrated_fast_coverage',)
    assert decision.policy_identity.startswith('experimental:calibrated_cascade:v1:')
    mixed = replace(context, requested_entity_types=('EMAIL', 'PERSON'))
    assert policy.plan(context=mixed, completed=completed).selected_ids == ('full',)


def test_real_regex_early_exit_never_constructs_full_and_reuses_cache():
    events = []
    cleaner = TextCleaner(calibrated(), detector_registry=registry(events))
    for _ in range(2):
        result = cleaner.process('alice@example.com')
        assert result.lm_text == '<EMAIL>'
        trace = result.metadata['routing_v1']
        assert trace['executed_stages'] == ['fast'] and trace['detector_calls'] == 1
        assert trace['skipped_ids'] == ['full']
        assert 'alice@example.com' not in json.dumps(trace)
    assert not events
    assert len(cleaner._unified._local.detectors) == 1


@pytest.mark.parametrize('requests', [(), ('PERSON',), ('EMAIL', 'PERSON'), ('UNKNOWN',)])
def test_mixed_name_and_email_needs_full_for_uncovered_or_unknown_requests(requests):
    events = []
    result = TextCleaner(calibrated(routing_entity_types=requests), detector_registry=registry(events)).process(
        'Alice alice@example.com')
    assert result.lm_text == '<PERSON> <EMAIL>'
    assert result.metadata['routing_v1']['executed_stages'] == ['fast', 'full']
    assert events == [('create', None), ('detect', 'ENGLISH')]


def test_missing_calibration_runs_full():
    result = TextCleaner(cascade(), detector_registry=registry([])).process('Alice alice@example.com')
    assert result.lm_text == '<PERSON> <EMAIL>'
    assert result.metadata['routing_v1']['reasons'] == ['missing_calibration']


@pytest.mark.parametrize('changes,reason', [
    ({'language': LanguageDecision(None, None, None, 'none', 'none')}, 'unknown_language'),
    ({'language': LanguageDecision('GERMAN', .99, .5, 'detected', 'detected')}, 'uncalibrated_language'),
    ({'language': LanguageDecision('ENGLISH', .89, .5, 'detected', 'detected')}, 'language_evidence'),
    ({'language': LanguageDecision('ENGLISH', .99, .09, 'detected', 'detected')}, 'language_evidence'),
    ({'language': LanguageDecision('ENGLISH', .99, None, 'detected', 'detected')}, 'language_evidence'),
    ({'text_size': 257}, 'text_size'), ({'requires_context': True}, 'unsupported_context'),
    ({'quality_score': .8}, 'unsupported_context'), ({'has_classification': True}, 'unsupported_context'),
])
def test_uncalibrated_context_routes_full(changes, reason):
    from sct.detectors_builtin import BuiltinRegexDetector
    routing = api()
    cfg = calibrated()
    ctx = routing.RoutingContext(LanguageDecision('ENGLISH', .9, .1, 'detected', 'detected'), ('EMAIL',), 30)
    runs = (routing.DetectorRun('fast', (BuiltinRegexDetector(cfg).capabilities,), calls=1),)
    decision = routing.RoutingPolicy(cfg).plan(context=replace(ctx, **changes), completed=runs)
    assert decision.selected_ids == ('full',) and reason in decision.reasons


def test_missing_capabilities_and_failed_stage_never_authorize_exit():
    routing = api()
    ctx = routing.RoutingContext(LanguageDecision('ENGLISH', 1., None, 'pinned', 'pinned'), ('EMAIL',), 20)
    policy = routing.RoutingPolicy(calibrated())
    for run in (routing.DetectorRun('fast', (), calls=0), routing.DetectorRun('fast', (), status='failed', calls=1)):
        assert policy.plan(context=ctx, completed=(run,)).selected_ids == ('full',)


@pytest.mark.parametrize('field,value', [('ner_confidence_threshold', .5), ('ner_model_revisions', {'en': 'a' * 40}),
    ('routing_full_specs', (DetectorSpec('names', {'revision': 'changed'}),))])
def test_calibration_rejects_changed_config_model_and_revision(field, value):
    cfg = calibrated()
    with pytest.raises(ValueError, match='identity'):
        TextCleaner(replace(cfg, **{field: value}), detector_registry=registry([]))


def test_actual_fast_capability_identity_mismatch_rejected():
    cfg = calibrated()
    tampered = dict(cfg.routing_calibration) | {'fast_capabilities_digest': '0' * 64}
    with pytest.raises(ValueError, match='identity'):
        TextCleaner(replace(cfg, routing_calibration=tampered), detector_registry=registry([])).process('alice@example.com')


def test_failing_fast_returns_no_partial_success_or_raw_error():
    class Broken:
        capabilities = DetectorCapabilities('broken', entity_types=('EMAIL',))
        def detect(self, text, **kwargs):
            raise RuntimeError(text)
    reg = registry([])
    reg.register('broken', lambda cfg, opts: Broken())
    cfg = cascade(routing_fast_specs=(DetectorSpec('broken'),))
    cfg = replace(cfg, routing_calibration=artifact(cfg, (Broken.capabilities,)))
    with pytest.raises(ValueError, match='Detector failed') as caught:
        TextCleaner(cfg, detector_registry=reg).process('alice@example.com')
    assert 'alice@example.com' not in str(caught.value)


def test_always_run_preserves_default_output_and_metadata():
    cfg = config(detector_specs=(DetectorSpec('builtin_regex'), DetectorSpec('names')))
    old = TextCleaner(cfg, detector_registry=registry([])).process('Alice alice@example.com')
    new = TextCleaner(replace(cfg, routing_policy='always_run'),
                      detector_registry=registry([])).process('Alice alice@example.com')
    assert tuple(new) == tuple(old) == ('<PERSON> <EMAIL>', 'person email', 'ENGLISH')
    assert new.metadata == old.metadata and 'routing_v1' not in new.metadata


def test_stage_bounds_and_duplicate_specs():
    routing = api()
    policy = routing.RoutingPolicy(calibrated())
    ctx = routing.RoutingContext(LanguageDecision(None, None, None, 'none', 'none'), (), 20)
    runs = (routing.DetectorRun('fast', ()), routing.DetectorRun('full', ()))
    assert policy.plan(context=ctx, completed=runs).selected_ids == ()
    for invalid in ((runs[0], runs[0]), (runs[1],), (*runs, runs[0])):
        with pytest.raises(ValueError, match='stage'):
            policy.plan(context=ctx, completed=invalid)
    with pytest.raises(ValueError, match='duplicate'):
        cascade(routing_full_specs=(DetectorSpec('builtin_regex'),))


def test_config_json_pickle_and_legacy_rejection():
    cfg = calibrated()
    assert TextCleaner(cfg, detector_registry=registry([])).cfg.routing_fast_specs == cfg.routing_fast_specs
    data = {name: value for name, value in cfg._plain_state().items() if name.startswith('routing_')}
    assert config(**json.loads(json.dumps(data))).routing_calibration == cfg.routing_calibration
    assert pickle.loads(pickle.dumps(cfg)) == cfg  # noqa: S301 - trusted round trip
    with pytest.raises(ValueError, match='unified'):
        replace(cfg, pii_processing='legacy')


def test_context_hints_are_not_lost_by_fast_exit():
    cfg = calibrated()
    result = TextCleaner(cfg, detector_registry=registry([])).process(
        'Alice alice@example.com', context=DetectionContext(declared_entity_types=('PERSON',)))
    assert result.lm_text == '<PERSON> <EMAIL>'
    assert result.metadata['routing_v1']['executed_stages'] == ['fast', 'full']


@pytest.mark.parametrize('mode', ['placeholder', 'synthetic', 'reversible'])
def test_routing_preserves_replacement_modes(mode):
    cfg = calibrated(replacement_mode=mode)
    result = TextCleaner(cfg, detector_registry=registry([])).process('alice@example.com')
    assert 'alice@example.com' not in result.lm_text
    assert len(tuple(result)) == 3
    if mode == 'reversible':
        assert result.metadata['anon_map'].deanonymize(result.lm_text) == 'alice@example.com'


def test_derived_json_key_views_run_configured_groups_without_stale_calibration():
    from sct import process_json
    for key_mode, expected in [('pii', 'Alice <EMAIL>'), ('pii_ner', '<PERSON> <EMAIL>')]:
        cfg = calibrated(json_key_cleaning=key_mode, routing_entity_types=('EMAIL', 'PERSON'))
        result = process_json(TextCleaner(cfg, detector_registry=registry([])), {'Alice alice@example.com': 'Alice'})
        assert result.value == {expected: '<PERSON>'}


def test_corpus_identity_sees_routed_builtin_model_revision():
    pytest.importorskip('datatrove')
    from sct.corpus.datatrove import _model_identities
    cfg = cascade(routing_full_specs=(DetectorSpec('builtin_ner'),),
                  ner_models={'ENGLISH': 'example/model', 'MULTILINGUAL': 'example/model'},
                  ner_model_revisions={'ENGLISH': 'a' * 40, 'MULTILINGUAL': 'a' * 40})
    assert ('ner:ENGLISH', 'example/model@' + 'a' * 40) in _model_identities(cfg)


def test_callbacks_cannot_get_unfingerprinted_calibration():
    cfg = cascade(custom_pipeline_steps=(str.lower,))
    with pytest.raises(ValueError, match='finite JSON'):
        TextCleaner(cfg, detector_registry=registry([]))


@pytest.mark.parametrize('key,value', [('schema_version', 2), ('max_chars', True), ('language_min_margin', float('nan')),
    ('languages', []), ('entity_types', ['']), ('config_digest', 'arbitrary'), ('unknown', 'secret')])
def test_invalid_calibration_is_rejected(key, value):
    cfg = cascade()
    with pytest.raises(ValueError):
        replace(cfg, routing_calibration=artifact(cfg) | {key: value})


def test_aliases_never_execute_duplicate_detector_identity():
    cfg = cascade(routing_fast_specs=(DetectorSpec('alias'),), routing_entity_types=())
    reg = registry([])
    from sct.detectors_builtin import BuiltinRegexDetector
    reg.register('alias', lambda cfg, options: BuiltinRegexDetector(cfg))
    cfg = replace(cfg, routing_full_specs=(DetectorSpec('builtin_regex'),))
    with pytest.raises(ValueError, match='duplicate'):
        TextCleaner(cfg, detector_registry=reg).process('alice@example.com')


def test_unknown_language_keeps_external_full_and_skips_reserved_builtin():
    events = []
    cfg = cascade(language=None, check_detect_language=False, check_remove_stopwords=False,
        unknown_language_action='skip_ner',
        routing_full_specs=(DetectorSpec('builtin_ner'), DetectorSpec('names')))
    result = TextCleaner(cfg, detector_registry=registry(events)).process('Alice alice@example.com')
    assert result.lm_text == '<PERSON> <EMAIL>' and result.metadata['ner_skipped'] is True
    assert events == [('create', None), ('detect', None)]
    assert result.metadata['routing_v1']['detector_calls'] == 2


def test_real_provider_native_model_route():
    """The configured full stage executes the cached native ONNX provider."""
    import os
    if not os.environ.get('SCT_L2_NATIVE'):
        pytest.skip('Native model qualification runs in the isolated CPU server profile')
    from tests.conftest import TEST_NER_MODELS, TEST_NER_REVISIONS
    cfg = cascade(check_ner_process=True, allow_network_detectors=True,
        routing_full_specs=(DetectorSpec('builtin_ner'),), routing_entity_types=('EMAIL', 'PERSON'),
        ner_models=TEST_NER_MODELS, ner_model_revisions=TEST_NER_REVISIONS, ner_device='cpu', ner_batch_size=1)
    cfg = replace(cfg, routing_calibration=artifact(cfg))
    result = TextCleaner(cfg).process('John Smith lives in London. Email john@example.com')
    assert 'john@example.com' not in result.lm_text and 'John Smith' not in result.lm_text
    assert result.metadata['routing_v1']['executed_stages'] == ['fast', 'full']
    assert result.metadata['routing_v1']['detector_calls'] == 2
    assert any(hit['detector_id'] == 'builtin_ner' and hit['entity_type'] == 'PERSON'
               for hit in result.metadata['findings_v2'])


def test_blank_input_records_no_stage_execution():
    events = []
    result = TextCleaner(calibrated(), detector_registry=registry(events)).process('   ')
    trace = result.metadata['routing_v1']
    assert trace['executed_stages'] == [] and trace['detector_calls'] == 0
    assert trace['reasons'] == ['empty_input'] and trace['status'] == 'no_stages'
    assert not events


def test_full_cache_is_reused_across_fast_full_transitions():
    events = []
    cleaner = TextCleaner(calibrated(), detector_registry=registry(events))
    for text in ('alice@example.com', 'Alice ' + 'x' * 257, 'alice@example.com', 'Alice ' + 'x' * 257):
        result = cleaner.process(text)
        assert 'Alice' not in result.lm_text and 'alice@example.com' not in result.lm_text
    assert events == [('create', None), ('detect', 'ENGLISH'), ('detect', 'ENGLISH')]
    assert len(cleaner._unified._local.detectors) == 2


def test_requested_types_reject_scalar_strings():
    with pytest.raises(ValueError, match='entity types'):
        cascade(routing_entity_types='EMAIL')


def test_always_run_explicit_groups_match_original_union_metadata():
    cfg = calibrated(routing_entity_types=('EMAIL', 'PERSON'))
    baseline = TextCleaner(config(detector_specs=cfg.routing_fast_specs + cfg.routing_full_specs),
                           detector_registry=registry([])).process('Alice alice@example.com')
    actual = TextCleaner(replace(cfg, routing_policy='always_run'),
                         detector_registry=registry([])).process('Alice alice@example.com')
    assert tuple(actual) == tuple(baseline) and actual.metadata == baseline.metadata


@pytest.mark.parametrize('preexisting_warning', [False, True])
def test_fast_timeout_requires_full_even_with_preexisting_warning(monkeypatch, preexisting_warning):
    cfg = cascade(routing_entity_types=('PERSON',), language_policy='abstain',
        custom_patterns={'PERSON': '(a+)+$'}, custom_patterns_timeout=.000000000001,
        custom_patterns_on_timeout='warn')
    cfg = replace(cfg, routing_calibration=artifact(cfg) | {'entity_types': ['PERSON']})
    cleaner = TextCleaner(cfg, detector_registry=registry([]))
    if preexisting_warning:
        monkeypatch.setattr(cleaner, '_language_warnings', lambda *args, **kwargs: {'custom_pattern_timeout'})
    result = cleaner.process('Alice alice@example.com ' + 'a' * 150 + '!')
    assert 'custom_pattern_timeout' in result.metadata['warnings']
    assert result.metadata['routing_v1']['executed_stages'] == ['fast', 'full']
    assert result.metadata['routing_v1']['reasons'] == ['fast_stage_incomplete']
    assert result.metadata['routing_v1']['status'] == 'stages_incomplete'
    assert 'Alice' not in result.lm_text


def test_full_timeout_preserves_warn_and_reports_incomplete_stages():
    cfg = cascade(routing_fast_specs=(DetectorSpec('names'),), routing_full_specs=(DetectorSpec('builtin_regex'),),
        custom_patterns={'TOKEN': '(a+)+$'}, custom_patterns_timeout=.000000000001, custom_patterns_on_timeout='warn')
    result = TextCleaner(cfg, detector_registry=registry([])).process('Alice alice@example.com ' + 'a' * 150 + '!')
    assert 'custom_pattern_timeout' in result.metadata['warnings']
    assert result.metadata['routing_v1']['executed_stages'] == ['fast', 'full']
    assert result.metadata['routing_v1']['status'] == 'stages_incomplete'
    assert 'Alice' not in result.lm_text and 'alice@example.com' not in result.lm_text


@pytest.mark.parametrize('field', ['custom_stopwords', 'custom_month_names'])
@pytest.mark.parametrize('text', ['alice@example.com', ''])
def test_nested_config_mutation_rejects_calibration_before_use(field, text):
    mutable = {}
    cfg = calibrated(**{field: mutable})
    events = []
    cleaner = TextCleaner(cfg, detector_registry=registry(events))
    mutable['ENGLISH'] = ('CallerMonth',)
    with pytest.raises(ValueError, match='identity'):
        cleaner.process(text)
    assert not events
    assert not getattr(cleaner._unified._local, 'detectors', {})


def scored_registry(events):
    result = DetectorRegistry()
    def factory(name, score):
        def create(cfg, options):
            events.append(('create', name))
            class Provider:
                capabilities = DetectorCapabilities(name, entity_types=('PERSON',))
                def detect(self, text, *, language, context):
                    events.append(('detect', name))
                    return (Detection('PERSON', 0, 5, name, score, 'canonical'),)
            return Provider()
        return create
    result.register('fast', factory('fast', .6))
    result.register('full', factory('full', .9))
    return result


def scored_config(unified_policy=None):
    cfg = cascade(routing_fast_specs=(DetectorSpec('fast'),), routing_full_specs=(DetectorSpec('full'),),
                  routing_entity_types=('PERSON',))
    values = artifact(cfg, (DetectorCapabilities('fast', entity_types=('PERSON',)),))
    if unified_policy is not None:
        values['config_digest'] = api().routing_config_digest(cfg, unified_policy=unified_policy)
    return replace(cfg, routing_calibration=values | {'entity_types': ['PERSON']})


@pytest.mark.parametrize('change', ['name', 'version', 'rules'])
def test_actual_handling_policy_mismatch_rejects_before_provider_creation(change):
    from sct import UnifiedPrivacyPolicy, HandlingRule
    values = {'name': 'default', 'version': '1', 'rules': {}}
    values[change] = {'name': 'changed', 'version': '2', 'rules': {'PERSON': HandlingRule(min_score=.8)}}[change]
    policy = UnifiedPrivacyPolicy(**values)
    events = []
    with pytest.raises(ValueError, match='identity'):
        TextCleaner(scored_config(), detector_registry=scored_registry(events), unified_policy=policy).process('Alice')
    assert not events


def test_matching_custom_handling_policy_calibration_protects_requested_entity():
    from sct import UnifiedPrivacyPolicy, HandlingRule
    policy = UnifiedPrivacyPolicy('custom', '3', {'PERSON': HandlingRule(action='redact', min_score=.5)})
    events = []
    result = TextCleaner(scored_config(policy), detector_registry=scored_registry(events),
                         unified_policy=policy).process('Alice')
    assert result.lm_text == '<REDACTED>'
    assert result.metadata['routing_v1']['skipped_ids'] == ['full']
    assert events == [('create', 'fast'), ('detect', 'fast')]


def test_default_handling_policy_digest_is_implicit_or_explicit():
    from sct import UnifiedPrivacyPolicy
    cfg = cascade()
    assert api().routing_config_digest(cfg) == api().routing_config_digest(
        cfg, unified_policy=UnifiedPrivacyPolicy('default', '1', {}))


@pytest.mark.parametrize('kind', ['policy', 'rule'])
def test_cascade_rejects_unfingerprinted_policy_subclasses_but_always_run_preserves_them(kind):
    from sct import UnifiedPrivacyPolicy, HandlingRule, FindingDecision
    class CustomPolicy(UnifiedPrivacyPolicy):
        def evaluate(self, detections):
            return tuple(FindingDecision(hit, 'keep', 'custom') for hit in detections)
    class CustomRule(HandlingRule):
        pass
    policy = (CustomPolicy('default', '1', {}) if kind == 'policy'
              else UnifiedPrivacyPolicy('default', '1', {'PERSON': CustomRule(min_score=.8)}))
    cfg = scored_config()
    events = []
    with pytest.raises(ValueError, match='declarative'):
        TextCleaner(cfg, detector_registry=scored_registry(events), unified_policy=policy)
    assert not events
    result = TextCleaner(replace(cfg, routing_policy='always_run'),
                         detector_registry=scored_registry(events), unified_policy=policy).process('Alice')
    assert result.lm_text == ('Alice' if kind == 'policy' else '<PERSON>')
    assert events == [('create', 'fast'), ('create', 'full'), ('detect', 'fast'), ('detect', 'full')]
    assert 'routing_v1' not in result.metadata
