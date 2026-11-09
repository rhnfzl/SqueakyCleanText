"""Combined contracts use real storage and corpus stages with deterministic detectors."""
from dataclasses import asdict, replace
import importlib.util
import json
from pathlib import Path
import traceback

import pytest

from sct import (
    Detection, DetectionContext, DetectorCapabilities, DetectorRegistry, DetectorSpec,
    DocumentElement, ExtractedDocument, FieldRule, FieldSchema, TextCleaner,
    TextCleanerConfig, clean_document, extract_document, process_json,
)


def config(**overrides):
    return TextCleanerConfig.from_preset('training', pii_processing='unified',
        check_ner_process=False, check_detect_language=False, **overrides)


@pytest.fixture
def extracted():
    class Provider:
        def extract(self, path):
            return ExtractedDocument((DocumentElement('e0', 'Ada Example', page=1),
                DocumentElement('e1', 'Write review.private@example.test for scientific research today.', page=2)))
    return extract_document(Path('private-input.pdf'), provider=Provider())


@pytest.fixture
def registry():
    contexts = []
    class Names:
        capabilities = DetectorCapabilities('acceptance_names', entity_types=('PERSON',), supports_context=True)
        def detect(self, text, *, language, context):
            contexts.append(context)
            return (Detection('PERSON', 0, 11, 'acceptance_names', .9, 'canonical'),)
    registry = DetectorRegistry()
    registry.register('acceptance_names', lambda *_: Names())
    return registry, contexts


@pytest.mark.skipif(any(importlib.util.find_spec(name) is None for name in ('cryptography', 'datasketch')),
                    reason='install token-store and corpus extras')
def test_r1_acceptance(tmp_path, extracted, registry, caplog):
    from cryptography.fernet import Fernet
    from sct.corpus import CorpusRecord, SelectionConfig, analyse_records, select_records
    from sct.corpus.manifest import RunIdentity, RunManifest, config_digest, file_digest
    from sct.corpus.runner import resume_run
    from sct.score import ScoreConfig
    from sct.token_stores import EncryptedSQLiteTokenStore

    class Keys:
        key = Fernet.generate_key()
        def current_key_id(self):
            return 'test-key'
        def fernet_key(self, key_id):
            assert key_id == 'test-key'
            return self.key

    events, metrics = [], []
    store = EncryptedSQLiteTokenStore(tmp_path / 'private' / 'tokens.sqlite', scope='test-scope',
        key_provider=Keys(), ttl_seconds=60, clock=lambda: 100, audit=events.append)
    detectors, contexts = registry
    cfg = config(detector_specs=(DetectorSpec('builtin_regex'), DetectorSpec('acceptance_names')))
    cleaner = TextCleaner(replace(cfg, replacement_mode='reversible'), detector_registry=detectors,
        token_store=store, metrics_callback=metrics.append)
    context = DetectionContext(document_id='opaque-document', path=('private-field',),
                               declared_entity_types=('PERSON',))
    direct = cleaner.process(extracted.text, context=context)
    assert contexts == [context]
    assert direct.metadata['counts_v2'] == {'version': 2, 'detector_hits': 2, 'resolved_replacements': 2}
    assert 'anon_map' not in direct.metadata
    reference = direct.metadata['anon_map_reference']
    assert store.load(reference).deanonymize(direct.lm_text) == extracted.text
    assert {row['entity_type']: row['source_ranges'] for row in direct.metadata['findings_v2']} == {
        'PERSON': [{'start': 0, 'end': 11}], 'EMAIL': [{'start': 18, 'end': 45}]}

    nonreversible = TextCleaner(cfg, detector_registry=detectors, metrics_callback=metrics.append)
    document = clean_document(extracted, nonreversible)
    assert document.text == '<PERSON>\nWrite <EMAIL> for scientific research today.'
    assert document.elements[0].page == 1 and document.elements[1].page == 2
    schema = FieldSchema('caller', '1', (FieldRule(('author',), ('PERSON',), 'protect_value'),))
    structured = process_json(TextCleaner(config()), {'author': 'Ada Example', 'body': document.text}, schema=schema)
    assert structured.value == {'author': '<PERSON>', 'body': document.text}
    with pytest.raises(ValueError, match='reversible'):
        process_json(cleaner, {'body': extracted.text})
    with pytest.raises(ValueError, match='mode'):
        clean_document(extracted, cleaner)

    selection = SelectionConfig(dedup_mode='apply', dedup_basis='pii_normalized')
    records = [CorpusRecord(record_id, extracted.text) for record_id in ('b', 'a')]
    analyses = analyse_records(records, cleaner=nonreversible, score_config=ScoreConfig(), selection=selection)
    with select_records(analyses, selection=selection) as report:
        assert report.kept_ids == ['a']
        decisions = [asdict(row) for row in report.iter_decisions()]
        output = tmp_path / 'kept.jsonl'
        output.write_text('\n'.join(json.dumps({'id': row.id, 'text': row.text}) for row in report.iter_records()))
        private_root = report.root
    assert not private_root.exists()
    assert json.loads(output.read_text()) == {'id': 'a', 'text': document.text}
    identity = RunIdentity('test-run', 'baseline', 'test-package', (), config_digest({'selection': asdict(selection)}),
                           (('policy', 'default'), ('detector', 'fixture-v1')), 1)
    manifest = RunManifest(identity, (('kept.jsonl', file_digest(output)),), {'input': 2, 'kept': 1, 'rejected': 1})
    manifest_path = tmp_path / 'manifest.json'
    manifest.publish(manifest_path)
    assert resume_run(manifest_path, expected=identity).complete
    artifacts = (json.dumps(direct.metadata), json.dumps(asdict(document)), json.dumps(structured.value),
                 json.dumps(decisions), output.read_text(), manifest_path.read_text(), repr(events), repr(metrics),
                 caplog.text, repr(extracted))
    for value in ('Ada Example', 'review.private@example.test', 'private-field'):
        assert all(value not in artifact for artifact in artifacts)
        assert value.encode() not in (tmp_path / 'private' / 'tokens.sqlite').read_bytes()
    assert all(record.text == extracted.text for record in records)


@pytest.mark.skipif(any(importlib.util.find_spec(name) is None for name in ('datatrove', 'spacy', 'orjson')),
                    reason='install corpus extra for native replay')
def test_extracted_unified_output_survives_interrupted_native_publication(tmp_path, extracted, registry, monkeypatch):
    from sct.corpus import curate_resumable
    from sct.corpus.runner import _Lifecycle
    detectors, _ = registry
    cleaner = TextCleaner(config(detector_specs=(DetectorSpec('builtin_regex'), DetectorSpec('acceptance_names'))),
                          detector_registry=detectors)
    document = clean_document(extracted, cleaner)
    source = tmp_path / 'input.jsonl'
    source.write_text(json.dumps({'id': 'document', 'text': document.text}) + '\n')
    source_bytes = source.read_bytes()
    kwargs = {'config': config(), 'retention': 'caller_managed'}
    uninterrupted = curate_resumable(source, tmp_path / 'uninterrupted', **kwargs)
    checkpoint = _Lifecycle.checkpoint
    def interrupt(self, name, paths):
        if name == 'publish':
            raise KeyboardInterrupt
        return checkpoint(self, name, paths)
    monkeypatch.setattr(_Lifecycle, 'checkpoint', interrupt)
    with pytest.raises(KeyboardInterrupt):
        curate_resumable(source, tmp_path / 'resumed', **kwargs)
    assert not (tmp_path / 'resumed' / 'manifest.json').exists()
    monkeypatch.setattr(_Lifecycle, 'checkpoint', checkpoint)
    resumed = curate_resumable(source, tmp_path / 'resumed', **kwargs)
    assert resumed.counts == uninterrupted.counts == {'input': 1, 'kept': 1, 'rejected': 0}
    assert (tmp_path / 'resumed' / 'kept.jsonl').read_bytes() == (tmp_path / 'uninterrupted' / 'kept.jsonl').read_bytes()
    assert curate_resumable(source, tmp_path / 'resumed', **kwargs).counts == resumed.counts
    assert source.read_bytes() == source_bytes
    for name in ('manifest.json', 'manifest.state.json', 'decisions.jsonl', 'kept.jsonl'):
        assert 'review.private@example.test' not in (tmp_path / 'resumed' / name).read_text()


@pytest.mark.parametrize('processing,expected,seen', [
    ('legacy', 'Contact marker-dependent-output', 'Contact <EMAIL>'),
    ('unified', 'Contact <EMAIL>', 'Contact review.private@example.test'),
])
def test_marker_dependent_callback_migration(processing, expected, seen):
    calls = []
    def callback(text):
        calls.append(text)
        return text.replace('<EMAIL>', 'marker-dependent-output')
    cleaner = TextCleaner(replace(config(custom_pipeline_steps=(callback,)), pii_processing=processing))
    result = cleaner.process('Contact review.private@example.test')
    assert calls == [seen]
    assert result.lm_text == expected
    if processing == 'unified':
        assert result.metadata['findings_v2'][0]['mapping_quality'] == 'unavailable'
        with pytest.raises(ValueError, match='source mapping'):
            TextCleaner(replace(cleaner.cfg, source_mapping='require_source')).process('Contact review.private@example.test')


@pytest.mark.parametrize('boundary', ['extract', 'document', 'direct'])
def test_public_error_trace_does_not_expose_provider_or_callback_values(boundary, extracted):
    sensitive = 'private-source-content@example.test'
    def fail(_):
        raise RuntimeError(sensitive)
    class Provider:
        extract = staticmethod(fail)
    cleaner = TextCleaner(config(custom_pipeline_steps=(fail,)))
    with pytest.raises(ValueError) as error:
        if boundary == 'extract':
            extract_document(Path('private.pdf'), provider=Provider())
        elif boundary == 'document':
            clean_document(extracted, cleaner)
        else:
            cleaner.process(extracted.text)
    assert sensitive not in str(error.value)
    assert sensitive not in ''.join(traceback.format_exception(error.value))
    assert error.value.__cause__ is None
    assert error.value.__suppress_context__
