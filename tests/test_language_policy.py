"""Language abstention preserves evidence and makes reduced protection explicit."""
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from sct import TextCleaner, TextCleanerConfig
from sct.language_policy import LanguageDecision, apply_language_policy
from sct.detectors import DetectorCapabilities, DetectorRegistry, DetectorSpec, DetectorUnavailable
from sct.findings import Detection


def cfg(**kwargs):
    return TextCleanerConfig(**({'check_ner_process': False} | kwargs))


def test_l1_acceptance():
    policy = cfg(language_policy='abstain', language_min_confidence=.8, language_min_margin=.1)
    decision = LanguageDecision('ENGLISH', .79, .1, 'detected', 'detected')
    result = apply_language_policy(decision, policy)
    assert result.language is None and result.source == 'abstained'
    assert result.confidence == .79 and result.margin == .1
    assert decision.language == 'ENGLISH'
    assert apply_language_policy(replace(decision, confidence=.8), policy).language == 'ENGLISH'
    pinned = LanguageDecision('DUTCH', 1., None, 'pinned', 'pinned')
    assert apply_language_policy(pinned, policy) is pinned


@pytest.mark.parametrize('source,confidence,margin', [('none', None, None), ('fallback', 0., None),
    ('detected', .9, .09), ('detected', None, .5), ('detected', .9, None)])
def test_abstention_requires_configured_evidence(source, confidence, margin):
    decision = LanguageDecision('ENGLISH' if source != 'none' else None, confidence, margin, source, source)
    assert apply_language_policy(decision, cfg()).source == source
    assert apply_language_policy(decision, cfg(language_policy='abstain',
        language_min_confidence=.8, language_min_margin=.1)).source == 'abstained'


@pytest.mark.parametrize('name,value', [('language_policy', 'other'), ('unknown_language_action', 'other'),
    ('language_min_confidence', -.1), ('language_min_confidence', float('nan')),
    ('language_min_margin', float('inf')), ('language_min_margin', True), ('language_min_confidence', '0.8')])
def test_policy_config_validation(name, value):
    with pytest.raises(ValueError, match=name):
        cfg(**{name: value})


def test_policy_config_and_serialization():
    config = cfg(language_policy='abstain', language_min_confidence=.8,
                 language_min_margin=.1, unknown_language_action='skip_ner')
    assert TextCleaner(config).cfg is config
    assert config.language_policy == 'abstain' and config.language_min_confidence == .8
    assert config.language_min_margin == .1 and config.unknown_language_action == 'skip_ner'
    values = {name: getattr(config, name) for name in
              ('language_policy', 'language_min_confidence', 'language_min_margin', 'unknown_language_action')}
    restored = cfg(**json.loads(json.dumps(values)))
    assert all(getattr(restored, name) == value for name, value in values.items())
    restored_state = object.__new__(TextCleanerConfig)
    restored_state.__setstate__(config.__getstate__())
    assert restored_state == config


@pytest.mark.parametrize('mode', ['legacy', 'unified'])
def test_real_lingua_abstains_without_inventing_restricted_language(mode):
    cleaner = TextCleaner(cfg(language=('en', 'th'), pii_processing=mode, language_policy='abstain',
                              language_min_confidence=.8, language_min_margin=.1))
    result = cleaner.process('नमस्ते, आप कैसे हैं? मैं दिल्ली में रहता हूँ और एक बैंक में काम करता हूँ।')
    assert result.language is None
    assert result.metadata['language_source'] == 'abstained'
    assert 'language_abstained' in result.metadata['warnings']
    assert 'language_abstained' in result.signals['warnings']
    assert result.metadata['language_reason'] == 'no_language_evidence'
    assert len(tuple(result)) == 3
    assert 'language_margin' in result.signals


@pytest.mark.parametrize('mode', ['legacy', 'unified'])
@pytest.mark.parametrize('action', ['multilingual', 'skip_ner', 'error'])
def test_unknown_actions_and_no_partial_success(mode, action):
    cleaner = TextCleaner(cfg(pii_processing=mode, unknown_language_action=action, ner_batch_size=1))
    calls = []
    class NER:
        device = 'cpu'
        cfg = cleaner.cfg
        def analyze(self, text, **kwargs):
            calls.append(kwargs['language'])
            return ()
        def ner_process_detailed(self, text, **kwargs):
            calls.append(kwargs['language'])
            return SimpleNamespace(text=text, findings=())
        def models_for(self, language):
            return []
        def uses_fallback_models(self, language):
            return True
    cleaner.GeneralNER = NER()
    if action == 'error':
        with pytest.raises(ValueError, match='Language is unknown') as caught:
            cleaner.process('12345 !!!')
        assert str(caught.value) == 'Language is unknown; choose multilingual or skip_ner'
        assert not calls
    else:
        # Unified built-in NER holds a worker-local cleaner; use its real detector boundary.
        if mode == 'unified':
            from sct.detectors_builtin import BuiltinNERDetector
            detector = BuiltinNERDetector(cleaner.GeneralNER)
            cleaner._unified._local.detectors = {0: detector}
            import os
            cleaner._unified._local.pid = os.getpid()
        result = cleaner.process('12345 !!!')
        assert result.language is None
        if action == 'skip_ner':
            assert not calls
            assert result.metadata['ner_skipped'] is True
            assert 'ner_skipped_unknown_language' in result.signals['warnings']
        else:
            assert calls == [None]


def test_unified_skip_keeps_explicit_external_detectors():
    calls = []
    class External:
        capabilities = DetectorCapabilities('external', entity_types=('EMAIL',))
        def detect(self, text, *, language, context):
            calls.append(language)
            return (Detection('EMAIL', 0, 5, 'external', None, 'canonical'),)
    registry = DetectorRegistry()
    registry.register('external', lambda cfg, options: External())
    cleaner = TextCleaner(cfg(pii_processing='unified', unknown_language_action='skip_ner',
        detector_specs=(DetectorSpec('external'),)), detector_registry=registry)
    result = cleaner.process('12345 !!!')
    assert calls == [None]
    assert '12345' not in result.lm_text
    assert result.metadata['counts_v2']['detector_hits'] == 1
    assert '12345' not in json.dumps(result.signals)


@pytest.mark.parametrize('mode', ['placeholder', 'synthetic', 'reversible'])
def test_abstention_replacement_modes_preserve_contact_protection(mode):
    cleaner = TextCleaner(cfg(language_policy='abstain', language_min_confidence=1., replacement_mode=mode))
    result = cleaner.process('Hello world alice@example.com')
    assert result.language is None
    assert 'alice@example.com' not in result.lm_text
    if mode == 'reversible':
        assert 'alice@example.com' in result.metadata['anon_map'].deanonymize(result.lm_text)


def test_opted_out_default_shapes():
    result = TextCleaner(cfg()).process('Hello world')
    assert 'language_margin' not in result.metadata and 'ner_skipped' not in result.metadata
    assert 'language_reason' not in result.signals


def test_merged_signals_keep_selected_language_evidence_and_any_skip():
    from sct.signals import merge_views
    from sct.utils.process_result import ProcessResult
    def view(language, chars, margin, reason, skipped=False):
        return ProcessResult('', None, language, dict(source_chars=chars, language_confidence=.8,
            language_source='detected' if language else 'abstained', language_margin=margin,
            language_reason=reason, **({'ner_skipped': True} if skipped else {}))).signals
    leaves = [view(None, 10, .01, 'margin_below_minimum', True),
              view('ENGLISH', 7, .3, 'detected'), view('ENGLISH', 6, .2, 'detected')]
    merged = merge_views(leaves)
    assert merged['language'] == 'ENGLISH'
    assert merged['language_reason'] == 'detected' and merged['language_margin'] == .3
    assert merged['ner_skipped'] is True


@pytest.mark.parametrize('workers', [1, 2])
def test_batched_skip_preserves_order_and_active_cache_indices(monkeypatch, workers):
    cleaner = TextCleaner(cfg(unknown_language_action='skip_ner', replacement_mode='synthetic', ner_batch_size=8))
    languages = []
    finalized = []
    class NER:
        device = 'cpu'
        def ner_process_batch_detailed(self, texts, langs, *, synthetic_caches, **kwargs):
            languages.extend(langs)
            results = []
            for i, text in enumerate(texts):
                cleaner._synthetic_replacer.swap_cache(synthetic_caches[i])
                finalized.append(cleaner._synthetic_replacer._get_cache().language)
                results.append(SimpleNamespace(text=text, findings=()))
            return results
        def models_for(self, language):
            return []
        def uses_fallback_models(self, language):
            return False
    cleaner.GeneralNER = NER()
    monkeypatch.setattr(cleaner, '_detect_language', lambda text:
        (None, None, 'none', None) if text.startswith('12345') else ('ENGLISH', .9, 'detected', .4))
    texts = ['12345 a@example.com', 'Hello b@example.com', '12345 c@example.com', 'Hello d@example.com']
    result = cleaner.process_batch(texts, max_workers=workers)
    assert languages == ['ENGLISH', 'ENGLISH']
    assert finalized == ['ENGLISH', 'ENGLISH']
    assert [r.language for r in result] == [None, 'ENGLISH', None, 'ENGLISH']
    assert [r.metadata.get('ner_skipped', False) for r in result] == [True, False, True, False]
    assert all(original.split()[-1] not in cleaned.lm_text for original, cleaned in zip(texts, result, strict=True))
    assert len(cleaner._document_context.__dict__) == 0


def test_no_language_requested_does_not_raise_or_abstain():
    result = TextCleaner(cfg(language_policy='abstain', unknown_language_action='error',
        check_detect_language=False, check_remove_stopwords=False)).process('Hello world')
    assert result.language is None and result.metadata['language_source'] == 'none'
    assert 'language_abstained' not in result.metadata['warnings']


@pytest.mark.parametrize('evidence', [{'language_margin': .2}, {'language_reason': 'detected'}])
def test_public_view_accepts_partial_optional_language_metadata(evidence):
    from sct.utils.process_result import ProcessResult
    view = ProcessResult('', None, 'ENGLISH', evidence).signals
    assert all(view[key] == value for key, value in evidence.items())


def test_unified_skip_does_not_construct_builtin_ner(monkeypatch):
    cleaner = TextCleaner(cfg(pii_processing='unified', check_ner_process=True,
        language_policy='abstain', language_min_confidence=1., unknown_language_action='skip_ner'))
    calls = []
    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError('Skipped NER factory must not run')
    # The default registry holds its factory, so observe the actual registered boundary.
    original = cleaner._detector_registry.create
    def create(spec, config):
        if spec.name == 'builtin_ner':
            return forbidden()
        return original(spec, config)
    monkeypatch.setattr(cleaner._detector_registry, 'create', create)
    result = cleaner.process('Hello world alice@example.com')
    assert not calls
    assert result.language is None and result.metadata['ner_skipped'] is True
    assert 'alice@example.com' not in result.lm_text
    assert [d['id'] for d in result.metadata['provenance_v2']['detectors']] == ['builtin_regex']


def test_unified_skipped_factory_cache_does_not_disable_later_ner():
    cleaner = TextCleaner(cfg(pii_processing='unified', check_ner_process=True,
        unknown_language_action='skip_ner'))
    for text in ['12345 !!!', 'This is clearly English text about the weather', '12345 !!!']:
        if text.startswith('12345'):
            result = cleaner.process(text)
            assert result.metadata['ner_skipped'] is True
        else:
            # Known-language processing still enforces the default network permission.
            with pytest.raises(DetectorUnavailable, match='network permission'):
                cleaner.process(text)


@pytest.mark.parametrize('order', [('unknown', 'known', 'unknown'), ('known', 'unknown', 'known')])
def test_unified_skip_reuses_external_instances_and_initializes_ner_later(monkeypatch, order):
    created = []
    analyzed = []
    class External:
        capabilities = DetectorCapabilities('external')
        def detect(self, text, *, language, context):
            analyzed.append(('external', language))
            return ()
    registry = DetectorRegistry()
    registry.register('external', lambda cfg, options: External())
    config = cfg(pii_processing='unified', check_ner_process=True, unknown_language_action='skip_ner',
        detector_specs=(DetectorSpec('builtin_regex'), DetectorSpec('builtin_ner'), DetectorSpec('external')))
    class NER:
        cfg = config
        def analyze(self, text, *, language, context):
            analyzed.append(('builtin_ner', language))
            return ()
    from sct.detectors_builtin import BuiltinNERDetector
    original = registry.create
    def create(spec, config):
        created.append(spec.name)
        return BuiltinNERDetector(NER()) if spec.name == 'builtin_ner' else original(spec, config)
    monkeypatch.setattr(registry, 'create', create)
    cleaner = TextCleaner(config, detector_registry=registry)
    for index, kind in enumerate(order):
        result = cleaner.process('12345 !!!' if kind == 'unknown' else 'This is clearly English text about the weather')
        assert result.metadata.get('ner_skipped', False) == (kind == 'unknown')
        if index == 0 and kind == 'unknown':
            assert 'builtin_ner' not in created
    assert sorted(created) == ['builtin_ner', 'builtin_regex', 'external']
    assert sum(name == 'external' for name, _ in analyzed) == len(order)
    assert sum(name == 'builtin_ner' for name, _ in analyzed) == order.count('known')
    assert all(language == 'ENGLISH' for name, language in analyzed if name == 'builtin_ner')
