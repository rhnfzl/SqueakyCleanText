"""Caller schemas select exact fields without guessing their meaning."""
from dataclasses import replace
import json

import pytest

from sct import TextCleaner, TextCleanerConfig
from sct.adapters import process_json, process_rag_chunks, RAGChunk
from sct.detectors import DetectorCapabilities, DetectorRegistry, DetectorSpec
from sct.findings import Detection, DetectionContext


def config(**kwargs):
    return TextCleanerConfig(**(dict(pii_processing='unified', check_ner_process=False,
        check_detect_language=False, check_remove_isolated_letters=False,
        check_to_ascii_unicode=False, check_remove_isolated_special_symbols=False,
        json_key_cleaning='off') | kwargs))


def schema(path=('name',), mode='protect_value', types=('PERSON',)):
    from sct.schema import FieldRule, FieldSchema
    return FieldSchema('caller', '1', (FieldRule(path, types, mode),))


def test_exact_field_protection_copies_input_without_name_heuristics():
    cleaner = TextCleaner(config())
    source = {'name': 'May', 'status': 'May', 'items': ('May', 7, None)}
    assert process_json(cleaner, source).value == {'name': 'May', 'status': 'May', 'items': ['May', 7, None]}
    result = process_json(cleaner, source, schema=schema())
    assert result.value == {'name': '<PERSON>', 'status': 'May', 'items': ['May', 7, None]}
    assert source['name'] == 'May' and isinstance(source['items'], tuple)
    assert result.sensitive
    assert result.signals['counts_v2'] == {'version': 2, 'detector_hits': 1, 'resolved_replacements': 1}


def context_registry(cfg):
    registry = DetectorRegistry()
    class Detector:
        capabilities = DetectorCapabilities('context', entity_types=('PERSON',), supports_context=True)
        def detect(self, text, *, language, context):
            if context.path == ('name',) and context.declared_entity_types == ('PERSON',):
                return (Detection('PERSON', 0, len(text), 'context', .9, 'canonical'),)
            return ()
    registry.register('context', lambda cfg, options: Detector())
    return registry


def test_hint_reaches_context_provider_and_missing_fields_are_optional():
    cfg = config(detector_specs=(DetectorSpec('context'),))
    cleaner = TextCleaner(cfg, detector_registry=context_registry(cfg))
    result = process_json(cleaner, [{'name': 'May'}, {'status': 'May'}],
        schema=schema((0, 'name'), 'hint'))
    # Absolute paths apply to one record, with no implicit wildcard.
    assert result.value == [{'name': 'May'}, {'status': 'May'}]
    assert process_json(cleaner, {'name': 'May'}, schema=schema(mode='hint')).value == {'name': '<PERSON>'}
    assert process_json(cleaner, {}, schema=schema()).value == {}


def test_required_hint_without_capable_provider_is_rejected():
    with pytest.raises(ValueError, match='context|hint'):
        process_json(TextCleaner(config()), {'name': 'May'}, schema=schema(mode='hint'))


@pytest.mark.parametrize('rule', [dict(action='keep'), dict(min_score=.8, unknown_score_action='error')])
def test_full_value_protection_rejects_conflicting_policy(rule):
    from sct.policy_v2 import UnifiedPrivacyPolicy, HandlingRule
    cleaner = TextCleaner(config(), unified_policy=UnifiedPrivacyPolicy('caller', '1', {'PERSON': HandlingRule(**rule)}))
    with pytest.raises(ValueError, match='protect_value|protection'):
        process_json(cleaner, {'name': 'May'}, schema=schema())


@pytest.mark.parametrize('value', [{'child': 'May'}, ['May']])
def test_matched_unsupported_values_are_rejected_without_echoing_path(value):
    with pytest.raises(ValueError) as error:
        process_json(TextCleaner(config()), {'private-name': value}, schema=schema(('private-name',)))
    assert 'private-name' not in str(error.value)


def test_schema_validation_rejects_conflicting_rules_and_mutability():
    from sct.schema import FieldRule, FieldSchema
    with pytest.raises(ValueError):
        FieldSchema('caller', '1', (FieldRule(('name',), ('PERSON',)), FieldRule(('name',), ('EMAIL',))))
    with pytest.raises(ValueError):
        FieldRule(['name'], ('PERSON',))


def test_legacy_context_refusal_and_existing_subclass_overrides():
    cleaner = TextCleaner(replace(config(), pii_processing='legacy'))
    with pytest.raises(ValueError, match='unified'):
        process_json(cleaner, {'name': 'May'}, schema=schema())
    class ExistingCleaner(TextCleaner):
        def process(self, text):
            return super().process(text)
    assert process_json(ExistingCleaner(cleaner.cfg), {'name': 'May'}).value == {'name': 'May'}


def test_rag_schema_covers_text_and_metadata_without_changing_id():
    result = process_rag_chunks(TextCleaner(config()), [RAGChunk('id', 'May', {'name': 'May', 'n': 2})],
        schema=schema(('metadata', 'name')))[0]
    assert result.id == 'id' and result.text == 'May' and result.metadata['name'] == '<PERSON>'
    assert result.sensitive


def test_unified_mcp_summary_and_full_findings_exclude_opted_in_values():
    from sct.mcp_server import sanitize_text
    cleaner = TextCleaner(config(), include_entities=True)
    output = sanitize_text(cleaner, 'alice@example.com', 'full')
    assert output['signals']['pii_count'] == 1
    assert output['signals']['counts_v2']['resolved_replacements'] == 1
    assert 'No PII' not in output['summary']
    assert output['findings_v2'][0]['entity_type'] == 'EMAIL'
    assert 'alice@example.com' not in json.dumps(output)


def test_explicit_numeric_boolean_protection_and_null_preservation():
    cleaner = TextCleaner(config())
    assert process_json(cleaner, {'name': 12345}, schema=schema()).value == {'name': '<PERSON>'}
    assert process_json(cleaner, {'name': True}, schema=schema()).value == {'name': '<PERSON>'}
    assert process_json(cleaner, {'name': None}, schema=schema()).value == {'name': None}
    assert process_json(cleaner, {'name': 12345}).value == {'name': 12345}
    with pytest.raises(ValueError):
        process_json(cleaner, {'name': float('inf')}, schema=schema())


def test_full_protection_refuses_allow_term_conflict():
    cleaner = TextCleaner(config(custom_allow_terms=('May',)))
    with pytest.raises(ValueError, match='protect_value|protection'):
        process_json(cleaner, {'name': 'May'}, schema=schema())


def test_threshold_with_unknown_score_protection_is_compatible():
    from sct.policy_v2 import UnifiedPrivacyPolicy, HandlingRule
    cleaner = TextCleaner(config(), unified_policy=UnifiedPrivacyPolicy('caller', '1',
        {'PERSON': HandlingRule(min_score=.9, unknown_score_action='protect')}))
    assert process_json(cleaner, {'name': 'May'}, schema=schema()).value == {'name': '<PERSON>'}


def test_json_full_unified_findings_are_versioned_and_value_free():
    from sct.mcp_server import sanitize_json
    output = sanitize_json(TextCleaner(config(), include_entities=True), {'name': 'May'}, 'full', schema=schema())
    assert output['findings_v2'][0]['entity_type'] == 'PERSON'
    assert 'May' not in json.dumps(output)


def test_json_full_unified_findings_preserve_safe_key_and_value_paths():
    from sct.mcp_server import sanitize_json
    cleaner = TextCleaner(config(json_key_cleaning='pii'), include_entities=True)
    output = sanitize_json(cleaner, {'alice@example.com': 'bob@example.com',
                                     'private-field': 'eve@example.com'}, 'full')
    findings = output['findings_v2']
    assert [item['start'] for item in findings] == [0, 0, 0]
    assert [item['path'] for item in findings] == [['@key', 0], ['@value', 0], ['@value', 1]]
    assert all('value' not in item for item in findings)
    assert 'private-field' not in json.dumps(findings)
    assert 'example.com' not in json.dumps(findings)


def test_presidio_analysis_receives_only_explicit_declared_context():
    from types import SimpleNamespace
    from sct.utils.ner import GeneralNER
    ner = object.__new__(GeneralNER)
    ner.cfg = config(ner_backend='presidio_gliner', gliner_model='local-test')
    ner._allow_regex = None
    ner._presidio_types = {}
    class Analyzer:
        supported_languages = ['en']
        def analyze(self, *, text, language, context=None):
            if context == ['PERSON']:
                return [SimpleNamespace(entity_type='PERSON', start=0, end=len(text), score=.9)]
            return []
    ner._analyzer = Analyzer()
    hits = ner.analyze('May', language=None, context=DetectionContext(declared_entity_types=('PERSON',)))
    assert len(hits) == 1 and hits[0].entity_type == 'PERSON'
    assert ner.analyze('May', language=None, context=DetectionContext()) == ()


def test_cli_registered_data_schema_and_v2_summary():
    from io import StringIO
    from sct.cli import main
    declared = {'id': 'caller', 'version': '1', 'rules': [
        {'path': ['name'], 'entity_types': ['PERSON'], 'mode': 'protect_value'}]}
    output = StringIO()
    main(['--no-ner', '--processing', 'unified', '--json-input', '--signals',
          '--schema', json.dumps(declared)], stdin=StringIO('{"name":"May"}'), stdout=output,
         schemas={'caller': schema()})
    payload = json.loads(output.getvalue())
    assert payload['value'] == {'name': '<PERSON>'}
    assert payload['signals']['counts_v2']['resolved_replacements'] == 1
    assert 'No PII' not in payload['summary']
    assert 'May' not in output.getvalue()
    with pytest.raises(SystemExit):
        main(['--no-ner', '--processing', 'unified', '--json-input', '--schema', json.dumps(declared)],
             stdin=StringIO('{"name":"May"}'), stdout=StringIO())


def test_framework_handle_factory_pickle_rebuilds_and_schema_protects():
    import pickle
    from sct.integrations.hf import SCTMap
    mapper = SCTMap(columns=('name',), config=config(), schema=schema(), registry_factory=context_registry)
    assert mapper({'name': ['May', None, 12]}) == {'name': ['<PERSON>', None, '<PERSON>']}
    restored = pickle.loads(pickle.dumps(mapper))  # noqa: S301
    assert restored.handle._cleaner is None
    assert restored({'name': ['May']}) == {'name': ['<PERSON>']}
    assert restored.handle.get() is not mapper.handle.get()


def test_dataframe_numeric_schema_preserves_copies_and_multiindex():
    pd = pytest.importorskip('pandas')
    from sct.adapters import process_dataframe
    frame = pd.DataFrame({('contact', 'id'): [12345, 67890], ('other', 'n'): [1, 2]},
                         index=pd.MultiIndex.from_tuples([('a', 1), ('b', 2)]))
    result = process_dataframe(TextCleaner(config()), frame, columns=(('contact', 'id'),),
        schema=schema((('a', 1), ('contact', 'id'))))
    assert result.frame.iat[0, 0] == '<PERSON>'
    assert result.frame.iat[1, 0] == 67890
    assert frame.iat[0, 0] == 12345 and frame.iloc[:, 0].dtype.kind in 'iu'
    assert result.frame.iloc[:, 1].dtype == frame.iloc[:, 1].dtype


def test_haystack_roundtrip_keeps_schema_context_and_trusted_factory():
    pytest.importorskip('haystack')
    from haystack import Document, Pipeline
    from sct.integrations.haystack import SCTDocumentCleaner
    component = SCTDocumentCleaner(columns=('name',), config=config(), schema=schema(),
        context=DetectionContext(document_id='private-id'), registry_factory=context_registry)
    pipeline = Pipeline()
    pipeline.add_component('cleaner', component)
    restored = Pipeline.loads(pipeline.dumps(), allowed_modules=["sct.integrations.haystack"])
    result = restored.run({'cleaner': {'documents': [Document(content='May', meta={'name': 'May'})]}})
    assert result['cleaner']['documents'][0].meta['name'] == '<PERSON>'


def test_hint_unsupported_provider_never_receives_context():
    cfg = config(detector_specs=(DetectorSpec('plain'), DetectorSpec('context')))
    registry = context_registry(cfg)
    class Plain:
        capabilities = DetectorCapabilities('plain', entity_types=('PERSON',))
        def detect(self, text, *, language, context):
            if context != DetectionContext():
                raise RuntimeError('Context leaked to an unsupported provider')
            return ()
    registry.register('plain', lambda cfg, options: Plain())
    cleaner = TextCleaner(cfg, detector_registry=registry)
    assert process_json(cleaner, {'name': 'May'}, schema=schema(mode='hint')).value == {'name': '<PERSON>'}


def test_haystack_data_rejects_unregistered_factory_reference():
    pytest.importorskip('haystack')
    from sct.integrations.haystack import SCTDocumentCleaner
    data = SCTDocumentCleaner(config=config()).to_dict()
    data['init_parameters']['registry_factory_reference'] = ['untrusted.module', 'factory']
    with pytest.raises(ValueError, match='registered|trusted'):
        SCTDocumentCleaner.from_dict(data)


def test_real_hf_dataset_schema_numeric_columns_change_output_type():
    datasets = pytest.importorskip('datasets')
    from sct.integrations.hf import clean_dataset
    source = datasets.Dataset.from_dict({'name': [12345, None], 'n': [1, 2]})
    output = clean_dataset(source, columns=('name',), config=config(), schema=schema(),
                           registry_factory=context_registry)
    assert output['name'] == ['<PERSON>', None]
    assert source['name'] == [12345, None] and output['n'] == [1, 2]


@pytest.mark.parametrize('framework', ['langchain', 'datatrove'])
def test_real_document_frameworks_forward_hints_and_trusted_factory(framework):
    cfg = config(detector_specs=(DetectorSpec('context'),))
    if framework == 'langchain':
        pytest.importorskip('langchain_core')
        from langchain_core.documents import Document
        from sct.integrations.langchain import SCTDocumentTransformer
        source = Document(page_content='May', metadata={'name': 'May', 'n': 7})
        transformer = SCTDocumentTransformer(columns=('name',), config=cfg, schema=schema(mode='hint'),
                                             registry_factory=context_registry)
        output = transformer.transform_documents([source])[0]
        assert source.metadata['name'] == 'May'
        assert output.metadata['name'] == '<PERSON>' and output.page_content == 'May'
    else:
        pytest.importorskip('datatrove')
        from datatrove.data import Document
        from sct.integrations.datatrove import SCTCleaner
        source = Document(text='May', id='id', metadata={'name': 'May', 'n': 7})
        output = list(SCTCleaner(columns=('name',), config=cfg, schema=schema(mode='hint'),
                                 registry_factory=context_registry).run([source]))[0]
        assert output.text == 'May' and output.metadata['name'] == '<PERSON>'
    assert output.metadata['n'] == 7 and output.metadata['sct']['counts_v2']['detector_hits'] == 1


def test_real_mcp_schema_registration_refuses_request_factory_paths():
    import asyncio
    pytest.importorskip('fastmcp')
    from fastmcp import Client
    from sct.mcp_server import create_mcp_server
    server = create_mcp_server(TextCleaner(config(), include_entities=True), schemas={'caller': schema()})
    declared = schema().to_dict()
    async def run():
        async with Client(server) as client:
            result = await client.call_tool('clean_json', {'value': {'name': 'May'}, 'schema': declared, 'detail': 'full'})
            payload = result.structured_content
            assert payload['value'] == {'name': '<PERSON>'}
            assert payload['signals']['counts_v2']['resolved_replacements'] == 1
            assert 'May' not in json.dumps(payload)
            with pytest.raises(Exception, match='registered'):
                await client.call_tool('clean_json', {'value': {'name': 'May'}, 'schema': declared | {'id': 'other'}})
    asyncio.run(run())


def test_hard_protection_uses_renderer_redaction_synthetic_and_core_reversible_map():
    from sct.policy_v2 import UnifiedPrivacyPolicy, HandlingRule
    from sct.schema import process_value
    for mode in ('placeholder', 'synthetic', 'reversible'):
        cleaner = TextCleaner(config(replacement_mode=mode))
        result = process_value(cleaner, 'May', ('name',), schema=schema())
        assert result.lm_text != 'May'
        assert result.metadata['findings_v2'][0]['score'] is None
        if mode == 'reversible':
            assert result.metadata['anon_map'].deanonymize(result.lm_text) == 'May'
    redactor = TextCleaner(config(), unified_policy=UnifiedPrivacyPolicy(
        'caller', '1', {'PERSON': HandlingRule(action='redact')}))
    assert process_json(redactor, {'name': 'May'}, schema=schema()).value == {'name': '<REDACTED>'}


def test_protected_disabled_tag_is_rejected():
    cleaner = TextCleaner(config(replace_with_email=None))
    with pytest.raises(ValueError, match='protection|protect_value'):
        process_json(cleaner, {'name': 'May'}, schema=schema(types=('EMAIL',)))


def test_real_presidio_context_boost_reaches_general_ner_analysis():
    pytest.importorskip('presidio_analyzer')
    import spacy
    from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer, RecognizerRegistry
    from presidio_analyzer.nlp_engine import SpacyNlpEngine
    from sct.utils.ner import GeneralNER
    engine = SpacyNlpEngine()
    engine.nlp = {'en': spacy.blank('en')}
    recognizer = PatternRecognizer(supported_entity='PERSON', patterns=[Pattern('scalar', '.+', .1)], context=['person'])
    registry = RecognizerRegistry(recognizers=[recognizer], supported_languages=['en'])
    analyzer = AnalyzerEngine(registry=registry, nlp_engine=engine, default_score_threshold=.3)
    ner = object.__new__(GeneralNER)
    ner.cfg = config(ner_backend='presidio_gliner', gliner_model='unused-offline')
    ner._allow_regex = None
    ner._presidio_types = {}
    ner._analyzer = analyzer
    assert ner.analyze('May', language=None, context=DetectionContext()) == ()
    hits = ner.analyze('May', language=None, context=DetectionContext(declared_entity_types=('PERSON',)))
    assert len(hits) == 1 and hits[0].entity_type == 'PERSON' and hits[0].score >= .3


def test_default_json_key_modes_and_original_path_selection():
    source = {'alice@example.com': 'May', 'status': 'May', ('part', 1): ('May', 7)}
    cleaner = TextCleaner(config(json_key_cleaning='pii'))
    result = process_json(cleaner, source, schema=schema(('alice@example.com',)))
    assert result.value == {'<EMAIL>': '<PERSON>', 'status': 'May', ('part', 1): ['May', 7]}
    assert source['alice@example.com'] == 'May'
    with pytest.raises(ValueError, match='duplicate'):
        process_json(cleaner, {'alice@example.com': 'May', 'bob@example.com': 'June'})
    for mode in ('off', 'pii', 'pii_ner', 'full'):
        assert process_json(TextCleaner(config(json_key_cleaning=mode)), {'name': 'May', 'status': 'May'}).value == {
            'name': 'May', 'status': 'May'}


def test_schema_hf_workers_keep_reference_only_factory_state():
    datasets = pytest.importorskip('datasets')
    from sct.integrations.hf import clean_dataset
    source = datasets.Dataset.from_dict({'name': ['May', 'June', 'July', 'August']})
    result = clean_dataset(source, columns=('name',), config=config(detector_specs=(DetectorSpec('context'),)),
                           schema=schema(mode='hint'), registry_factory=context_registry, num_proc=2)
    assert result['name'] == ['<PERSON>'] * 4


def test_request_schema_cannot_load_factory_or_guess_schema_by_id():
    from sct.schema import FieldSchema, registered_schema
    value = schema().to_dict()
    with pytest.raises(ValueError, match='Invalid'):
        FieldSchema.from_dict(value | {'factory': 'arbitrary.module.factory'})
    with pytest.raises(ValueError, match='registered'):
        registered_schema(value | {'rules': [{'path': ['other'], 'entity_types': ['PERSON']}]}, {'caller': schema()})


def test_sensitive_keys_context_and_labels_never_enter_v2_findings_or_signals():
    from sct.adapters import process_dataframe
    pd = pytest.importorskip('pandas')
    key = 'alice@example.com'
    context = DetectionContext(document_id=key, path=(key,))
    result = process_json(TextCleaner(config()), {key: 'May'}, schema=schema((key,)), context=context)
    assert result.findings_v2[0].path == ('@value', 0)
    assert key not in repr(result.findings_v2) and key not in json.dumps(result.signals)
    frame = pd.DataFrame({key: ['May']}, index=[key])
    output = process_dataframe(TextCleaner(config()), frame, columns=(key,), schema=schema((key, key)), context=context)
    assert output.findings_v2[0].path == (0, 0)
    assert key not in repr(output.findings_v2) and key not in json.dumps(output.signals)
    assert output.frame.index[0] == key and output.frame.columns[0] == key


def test_dataframe_explicit_numeric_protection_preserves_missing_cells():
    pd = pytest.importorskip('pandas')
    from sct.adapters import process_dataframe
    from sct.schema import FieldRule, FieldSchema
    frame = pd.DataFrame({'id': pd.Series([123, pd.NA], dtype='Int64')})
    declared = FieldSchema('caller', '1', tuple(FieldRule((i, 'id'), ('ID',), 'protect_value') for i in range(2)))
    output = process_dataframe(TextCleaner(config()), frame, columns=('id',), schema=declared)
    assert output.frame.iat[0, 0] == '<ID>' and pd.isna(output.frame.iat[1, 0])
    assert frame['id'].dtype == 'Int64' and pd.isna(frame.iat[1, 0])


def test_dataframe_declared_column_protects_all_numeric_cells_and_copies():
    pd = pytest.importorskip('pandas')
    from sct.adapters import process_dataframe
    frame = pd.DataFrame({'id': [12345, 67890], 'n': [1, 2]})
    result = process_dataframe(TextCleaner(config()), frame, columns=('id',), schema=schema(('id',), types=('ID',)))
    assert result.frame['id'].tolist() == ['<ID>', '<ID>']
    assert frame['id'].tolist() == [12345, 67890]
    assert result.frame['n'].dtype == frame['n'].dtype and result.frame['n'].tolist() == [1, 2]


def test_dataframe_overlapping_column_and_cell_rules_reject_conflicts():
    pd = pytest.importorskip('pandas')
    from sct.adapters import process_dataframe
    from sct.schema import FieldRule, FieldSchema
    declared = FieldSchema('caller', '1', (
        FieldRule(('name',), ('PERSON',), 'protect_value'), FieldRule((0, 'name'), ('PERSON',), 'hint')))
    with pytest.raises(ValueError, match='Conflicting'):
        process_dataframe(TextCleaner(config()), pd.DataFrame({'name': ['May']}), columns=('name',), schema=declared)


def test_json_regex_only_key_mode_does_not_run_external_models_on_keys():
    cfg = config(json_key_cleaning='pii', detector_specs=(DetectorSpec('all_text'),))
    registry = DetectorRegistry()
    class AllText:
        capabilities = DetectorCapabilities('all_text', entity_types=('PERSON',))
        def detect(self, text, *, language, context):
            return (Detection('PERSON', 0, len(text), 'all_text', .9, 'canonical'),)
    registry.register('all_text', lambda cfg, options: AllText())
    result = process_json(TextCleaner(cfg, detector_registry=registry), {'name': 'May'})
    assert result.value == {'name': '<PERSON>'}


def test_json_pii_ner_keys_keep_regex_protection_with_external_only_selection():
    cfg = config(json_key_cleaning='pii_ner', detector_specs=(DetectorSpec('context'),))
    result = process_json(TextCleaner(cfg, detector_registry=context_registry(cfg)), {'alice@example.com': 'May'})
    assert result.value == {'<EMAIL>': 'May'}


def test_mcp_application_config_registry_factory_and_pid_rebuild(monkeypatch):
    from sct.mcp_server import _cleaner_getter
    from sct.integrations import _runtime
    cfg = config(detector_specs=(DetectorSpec('context'),))
    get = _cleaner_getter(None, 'sct', registry_factory=context_registry, config=cfg)
    first = get(False)
    context = DetectionContext(path=('name',), declared_entity_types=('PERSON',))
    assert first.process('May', context=context).lm_text == '<PERSON>'
    monkeypatch.setattr(_runtime.os, 'getpid', lambda: -1)
    assert get(False) is not first


def test_mcp_rejects_conflicting_application_configuration():
    from sct.mcp_server import create_mcp_server
    pytest.importorskip('fastmcp')
    with pytest.raises(ValueError, match='config'):
        create_mcp_server(TextCleaner(config()), config=config())
    with pytest.raises(ValueError, match='preset'):
        create_mcp_server(config=config(), preset='training')


def test_empty_unified_structures_do_not_fall_back_to_legacy_no_pii_summary():
    from sct.mcp_server import sanitize_json
    for value in ({}, [], {'n': 7}):
        output = sanitize_json(TextCleaner(config()), value, 'full')
        assert output['signals']['version'] == 2
        assert output['signals']['counts_v2'] == {'version': 2, 'detector_hits': 0, 'resolved_replacements': 0}
        assert output['findings_v2'] == [] and 'No PII' not in output['summary']


@pytest.mark.parametrize('detector_id', ['date_provider', 'builtin_ner', 'builtin_regex'])
def test_non_pii_format_exclusion_is_limited_to_builtin_regex_origin(detector_id):
    registry = DetectorRegistry()
    class DateProvider:
        capabilities = DetectorCapabilities(detector_id, entity_types=('DATE',))
        def detect(self, text, *, language, context):
            return (Detection('DATE', 0, len(text), detector_id, .9, 'canonical'),)
    registry.register('date_provider', lambda cfg, options: DateProvider())
    cleaner = TextCleaner(config(detector_specs=(DetectorSpec('date_provider'),)), detector_registry=registry)
    result = cleaner.process('private birthday')
    assert result.lm_text == '<DATE>' and result.signals['sensitive']
    assert result.signals['pii_types'] == {'DATE': 1} and result.signals['pii_count'] == 1
    declared = process_json(TextCleaner(config()), {'name': 'private birthday'}, schema=schema(types=('DATE',)))
    assert declared.sensitive and declared.signals['pii_types'] == {'DATE': 1}
    formatting = TextCleaner(config()).process('2024-03-21')
    assert not formatting.signals['sensitive'] and formatting.signals['pii_count'] == 0


def test_covering_source_coverage_is_an_explicit_bound_not_an_exact_share():
    result = TextCleaner(config(check_remove_emoji=True)).process('Hi😀 alice@example.com')
    signals = result.signals
    assert signals['pii_chars'] is None and signals['pii_share'] is None
    assert signals['coverage_v2'] == {'version': 2, 'quality': 'covering', 'chars_lower': 0,
        'chars_upper': 21, 'share_lower': 0.0, 'share_upper': 1.0}
    assert 'source_coverage_covering' in signals['warnings']


def test_unavailable_source_coverage_is_not_zero_and_survives_merging():
    from sct.signals import merge_views
    exact = TextCleaner(config()).process('alice@example.com').signals
    unavailable = TextCleaner(config(custom_pipeline_steps=(str,))).process('Hi😀 alice@example.com').signals
    assert exact['pii_chars'] == 17 and exact['pii_share'] == 1.0
    assert exact['coverage_v2']['quality'] == 'exact'
    assert unavailable['pii_chars'] is None and unavailable['pii_share'] is None
    assert unavailable['coverage_v2'] == {'version': 2, 'quality': 'unavailable', 'chars_lower': 0,
        'chars_upper': 21, 'share_lower': 0.0, 'share_upper': 1.0}
    assert 'source_coverage_unavailable' in unavailable['warnings']
    merged = merge_views([exact, unavailable])
    assert merged['pii_chars'] is None and merged['pii_share'] is None
    assert merged['coverage_v2'] == {'version': 2, 'quality': 'unavailable', 'chars_lower': 17,
        'chars_upper': 38, 'share_lower': 17 / 38, 'share_upper': 1.0}


def test_covering_merge_retains_exact_lower_bound_and_readers_do_not_claim_a_measurement():
    from sct.signals import merge_views
    from sct.mcp_server import sanitize_text
    from sct.score import score_record, build_data_card
    covering_cleaner = TextCleaner(config(check_remove_emoji=True))
    exact = TextCleaner(config()).process('alice@example.com')
    covering = covering_cleaner.process('Hi😀 alice@example.com')
    merged = merge_views([exact.signals, covering.signals])
    assert merged['coverage_v2']['quality'] == 'covering'
    assert merged['coverage_v2']['chars_lower'] == 17 and merged['coverage_v2']['chars_upper'] == 38
    assert merged['pii_share'] is None
    output = sanitize_text(covering_cleaner, 'Hi😀 alice@example.com')
    assert output['signals']['pii_share'] is None and '100%' not in output['summary']
    row = score_record(covering)
    assert row['sct_pii_share'] is None
    assert build_data_card([row])['pii_share_histogram'] == {'unknown': 1}


def test_mixed_legacy_and_unified_views_do_not_promote_legacy_counts_to_exact_coordinates():
    from sct.signals import merge_views
    legacy = TextCleaner(replace(config(), pii_processing='legacy')).process('alice@example.com').signals
    unified = TextCleaner(config()).process('alice@example.com').signals
    merged = merge_views([legacy, unified])
    assert merged['coverage_v2'] == {'version': 2, 'quality': 'unavailable', 'chars_lower': 17,
        'chars_upper': 34, 'share_lower': .5, 'share_upper': 1.0}
    assert merged['pii_share'] is None and merged['pii_chars'] is None
    assert 'source_coverage_unavailable' in merged['warnings']


def test_cli_and_hf_score_serialize_coverage_uncertainty_without_a_fake_zero():
    import io
    from sct.cli import main
    from sct.integrations.hf import clean_dataset
    datasets = pytest.importorskip('datasets')
    # A request cannot install this transform; it is application-owned configuration.
    cfg = config(custom_pipeline_steps=(str,))
    ds = datasets.Dataset.from_dict({'text': ['alice@example.com']})
    output = clean_dataset(ds, config=cfg, score=True)
    assert output['sct_pii_share'] == [None]
    assert 'source_coverage_unavailable' in output['sct_warnings'][0]
    stream = io.StringIO()
    main(['--processing', 'unified', '--no-ner', '--signals'],
         stdin=io.StringIO('FranÃ§ais alice@example.com'), stdout=stream)
    payload = json.loads(stream.getvalue())
    assert payload['signals']['coverage_v2']['quality'] == 'covering'
    assert payload['signals']['pii_share'] is None and '100%' not in payload['summary']
