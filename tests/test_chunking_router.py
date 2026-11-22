import pytest

from sct.findings import TextRange


def test_auto_uses_document_rule_before_compatible_default():
    from sct.chunking import ChunkerRegistry, ChunkingPolicy, ChunkingRule

    registry = ChunkerRegistry()
    registry.register('plain', lambda request: (TextRange(0, len(request.text)),), stages=('detection',))
    registry.register('long', lambda request: (), stages=('detection',))
    policy = ChunkingPolicy(
        'auto', candidates=('markdown', 'plain'), rules=(
            ChunkingRule('markdown', formats=('markdown',)),
            ChunkingRule('long', languages=('dutch',), min_chars=20),
        ),
    )
    assert registry.resolve(policy, stage='detection', language='DUTCH', document_format='markdown', length=20) == 'long'
    assert registry.resolve(policy, stage='detection', language='DUTCH', document_format='markdown', length=19) == 'plain'
    assert registry.resolve(policy, stage='output', language='DUTCH', document_format='markdown', length=20) == 'markdown'


def test_detection_partition_rejects_gaps_and_over_budget_windows():
    from sct.chunking import ChunkerRegistry, ChunkRequest

    registry = ChunkerRegistry()
    registry.register('gap', lambda request: (TextRange(0, 1), TextRange(2, 3)), stages=('detection',))
    registry.register('oversize', lambda request: (TextRange(0, 3),), stages=('detection',))
    request = ChunkRequest('界x🙂', 2, len, stage='detection')
    with pytest.raises(ValueError, match='cover'):
        registry.plan('gap', request)
    with pytest.raises(ValueError, match='budget'):
        registry.plan('oversize', request)


def test_explicit_selection_rejects_incompatible_stage_and_auto_can_use_native():
    from sct.chunking import ChunkerRegistry, ChunkingPolicy

    registry = ChunkerRegistry()
    registry.register('only-output', lambda request: (), stages=('output',))
    details = dict(stage='detection', language=None, document_format=None, length=7)
    with pytest.raises(ValueError, match='stage'):
        registry.resolve(ChunkingPolicy('only-output'), **details)
    assert registry.resolve(ChunkingPolicy('auto', candidates=('only-output',)), **details) == 'native'


def test_router_windows_only_opted_in_detectors_and_masks_a_cross_boundary_name():
    from sct import TextCleaner, TextCleanerConfig
    from sct.chunking import ChunkerRegistry
    from sct.detectors import DetectorCapabilities, DetectorRegistry, DetectorSpec, DetectorWindowLimits
    from sct.findings import Detection, DetectionContext

    calls = []

    class Names:
        def __init__(self, identifier):
            self.capabilities = DetectorCapabilities(identifier, entity_types=('PERSON',), supports_windowed=True)

        def window_limits(self, language):
            return DetectorWindowLimits(16, len, overlap_chars=4)

        def detect(self, text, *, language, context):
            calls.append((self.capabilities.id, text))
            start = text.find('John Smith')
            if start < 0:
                return ()
            return (Detection('PERSON', start, start + 10, self.capabilities.id, .9, 'canonical'),)

    chunkers = ChunkerRegistry()
    chunkers.register('fixed', lambda request: tuple(
        TextRange(start, min(start + request.max_size, len(request.text)))
        for start in range(0, len(request.text), request.max_size)
    ), stages=('detection',))
    detectors = DetectorRegistry()
    detectors.register('bounded', lambda cfg, options: Names('bounded'))
    detectors.register('whole', lambda cfg, options: Names('whole'))
    cfg = TextCleanerConfig(
        pii_processing='unified', check_ner_process=False, language='en',
        check_remove_isolated_letters=False, check_normalize_whitespace=False,
        detector_specs=(DetectorSpec('bounded'), DetectorSpec('whole', chunking='native')),
        detector_chunking={'strategy': 'auto', 'rules': [{'strategy': 'fixed', 'formats': ['markdown']}]},
    )
    text = 'hello John Smith is here with us'
    cleaner = TextCleaner(cfg, detector_registry=detectors, chunker_registry=chunkers)
    result = cleaner.process(text, context=DetectionContext(document_format='markdown'))
    assert result.lm_text == 'hello <PERSON> is here with us'
    assert [s for name, s in calls if name == 'whole'] == [text]
    windows = [s for name, s in calls if name == 'bounded']
    assert len(windows) > 1 and all(len(window) <= 16 for window in windows)
    assert any('John Smith' in window for window in windows)
    assert [(item['id'], item.get('chunker')) for item in result.metadata['provenance_v2']['detectors']] == [
        ('bounded', 'fixed'), ('whole', None),
    ]
    calls.clear()
    short = cleaner.process('Short text.', context=DetectionContext(document_format='markdown'))
    assert calls == [('bounded', 'Short text.'), ('whole', 'Short text.')]
    assert all('chunker' not in item for item in short.metadata['provenance_v2']['detectors'])


def test_output_chunker_receives_only_cleaned_text_and_reports_cleaned_offsets():
    from sct import TextCleaner, TextCleanerConfig, chunk_document
    from sct.chunking import ChunkerRegistry

    seen = []

    def split(request):
        seen.append(request.text)
        cut = request.text.index('<EMAIL>')
        return (TextRange(0, cut), TextRange(cut, len(request.text)))

    registry = ChunkerRegistry()
    registry.register('example', split)
    cleaner = TextCleaner(TextCleanerConfig(check_ner_process=False), chunker_registry=registry)
    result = chunk_document(cleaner, 'Contact alice@example.com please', chunking='example')
    assert seen == ['Contact <EMAIL> please']
    assert result.chunker == 'example'
    assert [(c.text, c.range.start, c.range.end) for c in result.chunks] == [
        ('Contact ', 0, 8), ('<EMAIL> please', 8, len(seen[0])),
    ]


def test_output_preset_keeps_replacements_intact_and_uses_cleaned_elements():
    from sct import DocumentElement, ExtractedDocument, TextCleaner, TextCleanerConfig, chunk_document

    cleaner = TextCleaner(TextCleanerConfig(
        pii_processing='unified', check_ner_process=False, check_normalize_whitespace=False,
    ))
    document = ExtractedDocument((
        DocumentElement('e0', 'Contact alice@example.com', page=1),
        DocumentElement('e1', 'Second section', page=2),
    ))
    result = chunk_document(cleaner, document, max_size=17)
    assert result.chunker == 'elements'
    assert [(c.text, c.element_ids) for c in result.chunks] == [
        ('Contact <EMAIL>\n', ('e0',)), ('Second section', ('e1',)),
    ]
    with pytest.raises(ValueError, match='budget'):
        chunk_document(cleaner, document, max_size=6)


@pytest.mark.parametrize('plugin', ['semchunk', 'chonkie'])
def test_optional_chunker_slices_preserve_unicode_and_replacements(plugin):
    pytest.importorskip(plugin)
    from sct import TextCleaner, TextCleanerConfig, chunk_document

    cleaner = TextCleaner(TextCleanerConfig(
        check_ner_process=False, check_normalize_whitespace=False,
    ))
    text = '界 first. 🙂 Second sentence. Contact user@example.org now. 終わり。'
    result = chunk_document(cleaner, text, chunking=plugin, max_size=16)
    assert result.chunker == plugin
    assert ''.join(chunk.text for chunk in result.chunks) == cleaner.process(text).lm_text
    assert sum('<EMAIL>' in chunk.text for chunk in result.chunks) == 1
    assert all(len(chunk.text) <= 16 for chunk in result.chunks)


def test_backend_window_planner_preserves_model_token_limits_and_overlap():
    from types import SimpleNamespace

    from sct.chunking import ChunkRequest, ChunkerRegistry
    from sct.utils.ner_chunking import plan_windows

    text = '界' * 48
    seen = []
    registry = ChunkerRegistry()

    def tokenizer(source, **kwargs):
        return SimpleNamespace(input_ids=[None] * (len(source) + 2),
                               offset_mapping=[(i, i + 1) for i in range(len(source))])

    def split(budget, count):
        seen.append((budget, count('界')))
        return registry.plan('text', ChunkRequest(text, budget, count, 'detection'))

    windows = plan_windows(text, 16, tokenizer, 3, split)
    assert seen == [(10, 3)]
    assert all(len(tokenizer(window.text).input_ids) <= 16 for window in windows)
    assert ''.join(text[window.own_start:window.own_end] for window in windows) == text
    assert windows[1].start < windows[1].own_start


def test_output_config_round_trips_and_can_be_overridden():
    import json
    import pickle

    from sct import TextCleaner, TextCleanerConfig, chunk_document

    cfg = TextCleanerConfig(check_ner_process=False, output_chunking='text', output_chunk_max_size=10)
    state = cfg._plain_state()
    chunking_fields = {key: state[key] for key in ('output_chunking', 'output_chunk_max_size')}
    assert TextCleanerConfig(check_ner_process=False, **json.loads(json.dumps(chunking_fields))) == cfg
    assert pickle.loads(pickle.dumps(cfg)) == cfg  # noqa: S301
    cleaner = TextCleaner(cfg)
    text = 'A paragraph contains several words.'
    assert all(len(chunk.text) <= 10 for chunk in chunk_document(cleaner, text).chunks)
    assert len(chunk_document(cleaner, text, max_size=len(text)).chunks) == 1


def test_synthetic_output_requires_rendered_replacement_ranges():
    from sct import TextCleaner, TextCleanerConfig, chunk_document

    pytest.importorskip('faker')
    cleaner = TextCleaner(TextCleanerConfig(check_ner_process=False, replacement_mode='synthetic'))
    with pytest.raises(ValueError, match='unified'):
        chunk_document(cleaner, 'Contact alice@example.com', max_size=10)


def test_output_keeps_ambiguous_overlapping_custom_markers_together():
    from sct import ChunkBudgetError, TextCleaner, TextCleanerConfig, chunk_document

    cleaner = TextCleaner(TextCleanerConfig(check_ner_process=False, replace_with_currency_symbols='aaa'))
    with pytest.raises(ChunkBudgetError):
        chunk_document(cleaner, 'aa€', max_size=3)


def test_window_overlap_fits_noncharacter_counters_without_losing_all_context():
    from sct.chunking import ChunkerRegistry, ChunkingPolicy
    from sct.detector_windows import detect_with_chunking
    from sct.detectors import DetectorCapabilities, DetectorWindowLimits
    from sct.findings import Detection, DetectionContext

    class Names:
        capabilities = DetectorCapabilities('bounded', entity_types=('PERSON',), supports_windowed=True)

        def window_limits(self, language):
            return DetectorWindowLimits(12, lambda s: len(s.encode('utf-8')), overlap_chars=2)

        def detect(self, text, *, language, context):
            assert len(text.encode('utf-8')) <= 12
            index = text.find('丁戊己')
            return (Detection('PERSON', index, index + 3, 'bounded', .9, 'canonical'),) if index >= 0 else ()

    hits, name = detect_with_chunking(
        Names(), '甲乙丙丁戊己庚辛壬癸', language=None, context=DetectionContext(), document_format=None,
        policy=ChunkingPolicy('text'), inherited=False, registry=ChunkerRegistry(), allow_network=False,
    )
    assert name == 'text'
    assert [(hit.start, hit.end) for hit in hits] == [(3, 6)]
