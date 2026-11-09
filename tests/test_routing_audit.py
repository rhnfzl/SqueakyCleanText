"""Saved observations must support scoring without executing a detector."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from scripts import benchmark_detector_routing as routing


@pytest.fixture
def frozen_context(tmp_path, monkeypatch):
    """Use real input/source bytes and dummy cache bytes, never instantiate a model."""
    context = routing.RunContext('v090-v2', tmp_path)
    original = routing.RunContext('v090-v2')
    paths = {Path(path) for path in routing.source_hashes(original)} | set(original.inputs.values())
    paths.update(routing.OUTPUT / f'routing-{name}-v1.jsonl' for name in ('test', 'mixed'))
    for path in paths:
        target = tmp_path / path.relative_to(routing.ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    snapshots, model_files = {}, {}
    for model in routing.MODELS.values():
        snapshot = tmp_path / 'cache' / ('models--' + model['id'].replace('/', '--')) / 'snapshots' / model['revision']
        snapshot.mkdir(parents=True)
        snapshots[model['id']] = snapshot
        for filename in ('config.json', 'model.onnx', 'tokenizer.json', 'tokenizer_config.json'):
            path = snapshot / filename
            path.write_bytes(b'non-executable cache fixture')
            model_files[str(path)] = routing.file_hash(path)
    import huggingface_hub
    def local_snapshot(repo_id, *, revision, local_files_only):
        assert local_files_only is True
        assert snapshots[repo_id].name == revision
        return str(snapshots[repo_id])
    monkeypatch.setattr(huggingface_hub, 'snapshot_download', local_snapshot)
    runtime = dict(python='fixture-python', executable='/fixture/bin/python', executable_sha256='a' * 64,
        prefix='/fixture', platform='Linux-fixture', cpu_affinity=[2, 3], environment=dict(routing.RUNTIME_ENV),
        providers={name: dict(version='fixture', path=f'/fixture/{name}/__init__.py', sha256='b' * 64, binaries={})
                   for name in ('onnxruntime', 'tokenizers', 'numpy', 'lingua-language-detector', 'huggingface-hub', 'onnx')})
    monkeypatch.setattr(routing, 'runtime_identity', lambda: deepcopy(runtime))
    monkeypatch.setattr(routing, 'require_runtime', lambda: None)
    calibrations = dict(schema_version=2, default_policy='always_run', model_identities=routing.MODELS,
        calibrations={c['id']: dict(routing.configuration(c).routing_calibration) for c in routing.candidate_grid()})
    routing.save(context.path('calibrations'), calibrations)
    manifest = dict(schema_version=2, experiment='v090-v2', source_root=str(tmp_path), seed=20261109,
        models=routing.MODELS, model_files=model_files, source_files=routing.source_hashes(context),
        inputs={str(path): routing.file_hash(path) for path in context.inputs.values()},
        input_counts={'dev': 220, 'test': 110, 'mixed': 6}, runtime=runtime, grid=routing.candidate_grid(),
        warmup_documents=2, measured_repeats=3,
        config_digest=routing.routing_config_digest(routing.configuration(), unified_policy=routing.POLICY),
        no_eligible_comparison='c99-m20-s256', timestamp_utc='2026-11-09T12:00:00Z',
        memory_method='Linux resource.getrusage(RUSAGE_SELF).ru_maxrss, KiB, cumulative process high-water mark. '
                      'All policies share resident models; these are not independent per-policy memory measurements.',
        calibrations_sha256=routing.file_hash(context.path('calibrations')))
    routing.save(context.path('manifest'), manifest)
    return context


@pytest.mark.parametrize('field,value', [
    ('schema_version', 2.0), ('experiment', 'v1'), ('seed', 0), ('models', {}), ('grid', []),
    ('config_digest', '0' * 64), ('input_counts', {}), ('warmup_documents', True), ('measured_repeats', 1),
    ('no_eligible_comparison', 'c90-m10-s256'), ('memory_method', 'independent'), ('timestamp_utc', 'invalid'),
])
def test_manifest_rejects_contradictory_fixed_identity(frozen_context, field, value):
    path = frozen_context.path('manifest')
    manifest = json.loads(path.read_text())
    manifest[field] = value
    routing.save(path, manifest)
    with pytest.raises(ValueError):
        routing.verify_frozen(frozen_context)


@pytest.mark.parametrize('inventory', ['source_files', 'inputs', 'model_files'])
@pytest.mark.parametrize('damage', ['empty', 'omit', 'extra', 'hash'])
def test_manifest_requires_complete_inventory_and_bytes(frozen_context, inventory, damage):
    path = frozen_context.path('manifest')
    manifest = json.loads(path.read_text())
    mapping = manifest[inventory]
    key = next(iter(mapping))
    if damage == 'empty':
        mapping.clear()
    elif damage == 'omit':
        mapping.pop(key)
    elif damage == 'extra':
        extra = frozen_context.root / 'unexpected.py'
        extra.write_text('fixture')
        mapping[str(extra)] = routing.file_hash(extra)
    else:
        mapping[key] = '0' * 64
    routing.save(path, manifest)
    with pytest.raises(ValueError):
        routing.verify_frozen(frozen_context)


@pytest.mark.parametrize('field', ['grid', 'model_files', 'runtime', 'calibrations_sha256'])
def test_manifest_rejects_omitted_fields(frozen_context, field):
    path = frozen_context.path('manifest')
    manifest = json.loads(path.read_text())
    manifest.pop(field)
    routing.save(path, manifest)
    with pytest.raises(ValueError):
        routing.verify_frozen(frozen_context)


@pytest.mark.parametrize('field,value', [('cpu_affinity', [0, 1]), ('environment', {}), ('platform', 'Darwin-fixture')])
def test_manifest_runtime_must_match_protocol_even_when_host_stub_agrees(frozen_context, monkeypatch, field, value):
    path = frozen_context.path('manifest')
    manifest = json.loads(path.read_text())
    manifest['runtime'][field] = value
    monkeypatch.setattr(routing, 'runtime_identity', lambda: manifest['runtime'])
    routing.save(path, manifest)
    with pytest.raises(ValueError):
        routing.verify_frozen(frozen_context)


def test_calibration_content_cannot_be_rebound_with_a_new_file_hash(frozen_context):
    path = frozen_context.path('calibrations')
    calibration = json.loads(path.read_text())
    calibration['calibrations']['c90-m10-s256']['max_chars'] = 999
    routing.save(path, calibration)
    manifest_path = frozen_context.path('manifest')
    manifest = json.loads(manifest_path.read_text())
    manifest['calibrations_sha256'] = routing.file_hash(path)
    routing.save(manifest_path, manifest)
    with pytest.raises(ValueError):
        routing.verify_frozen(frozen_context)


def test_valid_manifest_checks_actual_cache_but_pure_audit_does_not(frozen_context, monkeypatch):
    routing.verify_frozen(frozen_context)
    import huggingface_hub
    def forbidden_cache_access(*args, **kwargs):
        pytest.fail('Pure audit must not access the model cache')
    monkeypatch.setattr(huggingface_hub, 'snapshot_download', forbidden_cache_access)
    monkeypatch.setattr(routing, 'runtime_identity', forbidden_cache_access)
    routing.verify_frozen(frozen_context, inference=False)


def write_phase_observations(context, partition, candidate):
    policy = candidate['id'] if candidate else 'always_run'
    rows = routing.load_records(context.inputs[partition])
    manifest_hash = routing.file_hash(context.path('manifest'))
    for repeat in range(3):
        start = repeat * len(rows)
        batch = dict(schema_version=2, id=policy, partition=partition, repeat=repeat, manifest_sha256=manifest_hash,
            counter_start={'fast': start, 'full': start},
            counter_end={'fast': start + len(rows), 'full': start + len(rows)}, process_peak_rss_kib=1024,
            observations=[dict(id=row['id'], predictions=[{key: span[key] for key in ('label', 'start', 'end')}
                for span in row['spans']], duration_ms=1, actual_model_calls={'fast': 1, 'full': 1},
                skipped_stages=0, routing_reasons=[], language_source='detected') for row in rows])
        routing.save(context.raw_path(partition, policy, repeat), batch)


def write_development_phase(context):
    routing.consume(context.path('dev-consumed'), {'manifest_sha256': routing.file_hash(context.path('manifest'))})
    for candidate in [None, *routing.candidate_grid()]:
        write_phase_observations(context, 'dev', candidate)


def test_completed_observations_without_postflight_receipt_cannot_emit_summary(frozen_context):
    write_development_phase(frozen_context)
    with pytest.raises(ValueError, match='completion'):
        routing.audit_development(frozen_context, reconstruct=True)
    assert not frozen_context.path('development').exists()
    assert not frozen_context.path('dev-complete').exists()
    assert len(list(frozen_context.output.glob('routing-observations-dev-*'))) == 39


def test_postflight_failure_keeps_raw_observations_without_summary_or_receipt(frozen_context, monkeypatch):
    def fixture_measure(rows, candidate, detectors, *, context, partition):
        write_phase_observations(context, partition, candidate)
        if candidate and candidate['id'] == 'c99-m20-s512':
            with context.inputs['dev'].open('a') as stream:
                stream.write('\n')
        return {'id': candidate['id'] if candidate else 'always_run', 'median_ms': 1,
                'actual_model_calls': {'fast': 1, 'full': 1}, 'skipped_stages': 0}, []
    monkeypatch.setattr(routing, 'measure', fixture_measure)
    with pytest.raises(ValueError):
        routing.development({}, frozen_context)
    assert len(list(frozen_context.output.glob('routing-observations-dev-*'))) == 39
    assert not frozen_context.path('development').exists()
    assert not frozen_context.path('dev-complete').exists()
    with pytest.raises(ValueError, match='completion'):
        routing.audit_development(frozen_context)


@pytest.mark.parametrize('damage', ['missing_field', 'phase', 'status', 'manifest', 'consumption',
                                  'missing_observation', 'observation_hash', 'extra_observation', 'stale_raw'])
def test_bad_completion_receipts_cannot_reconstruct(frozen_context, damage):
    write_development_phase(frozen_context)
    routing.complete_phase(frozen_context, 'dev')
    target = frozen_context.path('dev-complete')
    receipt = json.loads(target.read_text())
    if damage == 'missing_field':
        receipt.pop('postflight')
    elif damage == 'phase':
        receipt['phase'] = 'test'
    elif damage == 'status':
        receipt['postflight'] = True
    elif damage == 'manifest':
        receipt['manifest_sha256'] = '0' * 64
    elif damage == 'consumption':
        receipt['consumption_sha256'] = '0' * 64
    elif damage == 'missing_observation':
        receipt['observations'].pop(next(iter(receipt['observations'])))
    elif damage == 'observation_hash':
        receipt['observations'][next(iter(receipt['observations']))] = '0' * 64
    elif damage == 'extra_observation':
        receipt['observations']['unexpected.json'] = '0' * 64
    else:
        path = frozen_context.raw_path('dev', 'always_run', 0)
        path.write_text(path.read_text() + '\n')
    routing.save(target, receipt)
    with pytest.raises(ValueError, match='completion'):
        routing.audit_development(frozen_context)
    assert not frozen_context.path('development').exists()


def test_valid_receipt_allows_postprocessing_recovery_without_cache_or_runtime(frozen_context, monkeypatch):
    def fixture_measure(rows, candidate, detectors, *, context, partition):
        write_phase_observations(context, partition, candidate)
    monkeypatch.setattr(routing, 'measure', fixture_measure)
    real_score = routing.development_from_observations
    def broken_postprocessing(*args):
        raise ValueError('postprocessing failed')
    monkeypatch.setattr(routing, 'development_from_observations', broken_postprocessing)
    with pytest.raises(ValueError, match='postprocessing failed'):
        routing.development({}, frozen_context)
    assert frozen_context.path('dev-complete').is_file()
    assert not frozen_context.path('development').exists()
    import huggingface_hub
    def forbidden(*args, **kwargs):
        pytest.fail('Recovery must not access the model cache or current runtime')
    monkeypatch.setattr(huggingface_hub, 'snapshot_download', forbidden)
    monkeypatch.setattr(routing, 'runtime_identity', forbidden)
    monkeypatch.setattr(routing, 'development_from_observations', real_score)
    report = routing.audit_development(frozen_context, reconstruct=True)
    assert report['selected'] == 'c90-m10-s256'


def test_completion_receipt_requires_complete_raw_inventory(frozen_context):
    write_development_phase(frozen_context)
    frozen_context.raw_path('dev', 'c99-m20-s512', 2).unlink()
    with pytest.raises(ValueError, match='completion'):
        routing.complete_phase(frozen_context, 'dev')
    assert not frozen_context.path('dev-complete').exists()


def test_receipted_copied_tree_audits_without_the_original_model_cache(frozen_context, monkeypatch):
    write_development_phase(frozen_context)
    routing.complete_phase(frozen_context, 'dev')
    destination = frozen_context.root.with_name(frozen_context.root.name + '-audit-copy')
    shutil.copytree(frozen_context.root, destination, ignore=shutil.ignore_patterns('cache'))
    copied = routing.RunContext('v090-v2', destination)
    import huggingface_hub
    def forbidden(*args, **kwargs):
        pytest.fail('Copied-tree audit cannot require cached weights or the current runtime')
    monkeypatch.setattr(huggingface_hub, 'snapshot_download', forbidden)
    monkeypatch.setattr(routing, 'runtime_identity', forbidden)
    report = routing.audit_development(copied, reconstruct=True)
    assert report['selected'] == 'c90-m10-s256'
    assert not (destination / 'cache').exists()


def test_preflight_detects_new_actual_cache_files_omitted_from_manifest(frozen_context):
    manifest = json.loads(frozen_context.path('manifest').read_text())
    snapshot = Path(next(iter(manifest['model_files']))).parent
    (snapshot / 'additional-config.json').write_text('{}')
    with pytest.raises(ValueError, match='Complete model cache'):
        routing.verify_frozen(frozen_context)


def test_postflight_runtime_drift_cannot_create_completion_receipt(frozen_context, monkeypatch):
    write_development_phase(frozen_context)
    runtime = routing.runtime_identity()
    runtime['python'] = 'changed runtime fixture'
    monkeypatch.setattr(routing, 'runtime_identity', lambda: runtime)
    with pytest.raises(ValueError, match='Runtime'):
        routing.complete_phase(frozen_context, 'dev')
    assert not frozen_context.path('dev-complete').exists()
    assert not frozen_context.path('development').exists()


def records():
    return [dict(id='fixture-a', template_id='template-a', language='en', domain='support',
                 text='Ada a@example.test', spans=[dict(label='PERSON', start=0, end=3),
                                                     dict(label='EMAIL', start=4, end=18)])]


def observations(policy='always_run', full_calls=2):
    return [dict(schema_version=2, id=policy, partition='dev', repeat=repeat, manifest_sha256='a' * 64,
                 counter_start={'fast': repeat, 'full': repeat * full_calls},
                 counter_end={'fast': repeat + 1, 'full': (repeat + 1) * full_calls},
                 process_peak_rss_kib=1024, observations=[dict(
                     id='fixture-a', predictions=records()[0]['spans'], duration_ms=[1, 9, 5][repeat],
                     actual_model_calls={'fast': 1, 'full': full_calls}, skipped_stages=0,
                     routing_reasons=['always_run'], language_source='detected')]) for repeat in range(3)]


def test_raw_audit_rebuilds_scores_quantiles_and_actual_calls():
    report, counts = routing.audit_observations(records(), observations(), 'always_run', 'dev', 'a' * 64)
    assert report['median_ms'] == 5
    assert report['p95_ms'] == 9
    assert report['actual_model_calls'] == {'fast': 3, 'full': 6}
    assert report['full']['overall']['recall'] == 1
    assert report['shared']['overall']['gold'] == 1
    assert counts[0]['docs_exposed'] == 0
    assert 'Ada' not in json.dumps(report)
    assert 'example.test' not in json.dumps(report)


@pytest.mark.parametrize('damage', ['missing_repeat', 'duplicate_repeat', 'wrong_id', 'missing_doc',
    'nonfinite', 'negative_duration', 'bool_calls', 'counter_total', 'extra_stage', 'repeat_prediction',
    'span_bounds', 'text_leak', 'source_leak', 'manifest', 'policy', 'partition'])
def test_corrupt_observations_are_rejected(damage):
    raw = observations()
    item = raw[0]['observations'][0]
    if damage == 'missing_repeat':
        raw.pop()
    elif damage == 'duplicate_repeat':
        raw[1]['repeat'] = 0
    elif damage == 'wrong_id':
        item['id'] = 'foreign'
    elif damage == 'missing_doc':
        raw[0]['observations'] = []
    elif damage == 'nonfinite':
        item['duration_ms'] = float('nan')
    elif damage == 'negative_duration':
        item['duration_ms'] = -1
    elif damage == 'bool_calls':
        item['actual_model_calls']['full'] = True
    elif damage == 'counter_total':
        raw[0]['counter_end']['full'] = 3
    elif damage == 'extra_stage':
        item['actual_model_calls']['other'] = 0
    elif damage == 'repeat_prediction':
        raw[1]['observations'][0]['predictions'] = []
    elif damage == 'span_bounds':
        item['predictions'][0]['end'] = 999
    elif damage == 'text_leak':
        item['predictions'][0]['text'] = 'Ada'
    elif damage == 'source_leak':
        raw[0]['text'] = 'Ada'
    elif damage == 'manifest':
        raw[0]['manifest_sha256'] = 'b' * 64
    elif damage == 'policy':
        raw[0]['id'] = 'different'
    else:
        raw[0]['partition'] = 'test'
    with pytest.raises(ValueError):
        routing.audit_observations(records(), raw, 'always_run', 'dev', 'a' * 64)


def test_shared_exposure_is_not_hidden_by_full_taxonomy_exposure():
    rows = records()
    base = {'fixture-a': [rows[0]['spans'][0]]}
    candidate = {'fixture-a': []}
    views = routing.exposure_views(rows, base, candidate)
    assert views['shared']['language:en'] == 1
    assert views['full']['language:en'] == 0
    assert views['full']['entity:PERSON'] == 1


def test_context_isolates_current_outputs_and_blocks_legacy_execution(tmp_path):
    context = routing.RunContext('v090-v2', tmp_path)
    assert context.path('manifest') == tmp_path / 'evaluation/improvements/routing-v090-v2/routing-manifest-v2.json'
    context.require_current()
    with pytest.raises(ValueError, match='historical'):
        routing.RunContext('v1', tmp_path).require_current()


def test_exclusive_consumption_cannot_be_replaced(tmp_path):
    target = tmp_path / 'consumed.json'
    routing.consume(target, {'id': 'first'})
    with pytest.raises(FileExistsError):
        routing.consume(target, {'id': 'second'})
    assert json.loads(target.read_text()) == {'id': 'first'}


def test_saved_development_rejects_edited_summary_and_preserves_negative_selection():
    raw = {'always_run': observations()}
    for candidate in routing.candidate_grid():
        raw[candidate['id']] = observations(candidate['id'], full_calls=1)
        for repeat in raw[candidate['id']]:
            repeat['observations'][0]['predictions'] = []
    report = routing.development_from_observations(records(), raw, 'a' * 64)
    assert report['selected'] is None
    assert report['comparison'] == 'c99-m20-s256'
    corrupt = deepcopy(report)
    corrupt['candidates'][0]['median_ms'] = .01
    with pytest.raises(ValueError, match='saved observations'):
        routing.validate_development_audit(corrupt, records(), raw, 'a' * 64)


def test_heldout_audit_uses_actual_repeats_and_no_eligible_cannot_promote():
    candidate = routing.candidate_grid()[0]
    raw = {}
    for partition in ('test', 'mixed'):
        for policy, calls in [('always_run', 2), (candidate['id'], 1)]:
            raw[partition, policy] = observations(policy, calls)
            for batch in raw[partition, policy]:
                batch['partition'] = partition
                if policy != 'always_run':
                    batch['observations'][0]['duration_ms'] /= 2
    report = routing.heldout_from_observations(records(), records(), raw, candidate,
        selected=candidate['id'], manifest_hash='a' * 64, development_hash='b' * 64, seed=20261109)
    assert report['promotion']['status'] == 'pass'
    assert report['promotion']['paired_95_percent_interval'] == [0, 0]
    assert report['candidate_result']['actual_model_calls']['full'] == 3
    report = routing.heldout_from_observations(records(), records(), raw, candidate,
        selected=None, manifest_hash='a' * 64, development_hash='b' * 64, seed=20261109)
    assert report['promotion']['status'] == 'fail'
    assert report['promotion']['no_eligible_development_candidate']


def test_runtime_constraints_fail_before_loading_models(monkeypatch):
    monkeypatch.setattr(routing.sys, 'platform', 'linux')
    monkeypatch.setattr(routing.os, 'sched_getaffinity', lambda pid: {2, 3}, raising=False)
    for key, value in routing.RUNTIME_ENV.items():
        monkeypatch.setenv(key, value)
    routing.require_runtime()
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    with pytest.raises(ValueError, match='runtime'):
        routing.require_runtime()


def test_existing_repeat_blocks_measurement_before_warmup(tmp_path):
    context = routing.RunContext('v090-v2', tmp_path)
    context.output.mkdir(parents=True)
    context.raw_path('test', 'always_run', 0).write_text('{}')
    with pytest.raises(ValueError, match='already'):
        routing.measure(records(), None, {}, context=context, partition='test')


def test_current_cohort_has_codepoint_gold_and_eleven_independent_templates():
    context = routing.RunContext('v090-v2')
    partitions = routing.validate_current_inputs(context)
    assert {key: len(rows) for key, rows in partitions.items()} == {'dev': 220, 'test': 110, 'mixed': 6}
    assert len({row['template_id'] for row in partitions['test']}) == 11
    for row in partitions['test']:
        assert {span['label'] for span in row['spans']} == set(routing.TYPES)
        assert len(row['spans']) == 6
        assert all(row['text'][span['start']:span['end']] for span in row['spans'])


def test_new_cohort_uses_new_typed_entity_values():
    context = routing.RunContext('v090-v2')
    current = routing.load_records(context.inputs['test'])
    previous = [*routing.load_records(context.inputs['dev']),
                *routing.load_records(routing.OUTPUT / 'routing-test-v1.jsonl'),
                *routing.load_records(routing.OUTPUT / 'routing-mixed-v1.jsonl')]
    def values(rows):
        return {(span['label'], row['text'][span['start']:span['end']]) for row in rows for span in row['spans']}
    assert not values(current) & values(previous)


def test_raw_trace_cannot_embed_source_text():
    raw = observations()
    raw[0]['observations'][0]['routing_reasons'] = ['Ada a@example.test']
    with pytest.raises(ValueError):
        routing.audit_observations(records(), raw, 'always_run', 'dev', 'a' * 64)


def test_measure_persists_repeats_without_running_postprocessing(tmp_path, monkeypatch):
    context = routing.RunContext('v090-v2', tmp_path)
    context.output.mkdir(parents=True)
    context.path('manifest').write_text('{}')
    detectors = {stage: SimpleNamespace(calls=[0]) for stage in ('fast', 'full')}
    class FixtureCleaner:
        def process(self, text):
            detectors['fast'].calls[0] += 1
            detectors['full'].calls[0] += 2
            return SimpleNamespace(metadata=dict(language_source='detected', findings_v2=[
                dict(entity_type=span['label'], start=span['start'], end=span['end'], action='redact')
                for span in records()[0]['spans']]))
    monkeypatch.setattr(routing, 'make_cleaner', lambda candidate, actual: FixtureCleaner())
    original_audit = routing.audit_observations
    def broken_postprocessing(*args):
        raise ValueError('postprocessing failed')
    monkeypatch.setattr(routing, 'audit_observations', broken_postprocessing)
    routing.measure(records(), None, detectors, context=context, partition='test')
    raw = routing.read_observations(context, 'test', ['always_run'])['always_run']
    report, _ = original_audit(records(), raw, 'always_run', 'test', routing.file_hash(context.path('manifest')))
    assert report['actual_model_calls'] == {'fast': 3, 'full': 6}
    assert report['full']['overall']['recall'] == 1


def test_receipted_development_reconstructs_without_test_observations(frozen_context):
    context = frozen_context
    write_development_phase(context)
    routing.complete_phase(context, 'dev')
    report = routing.audit_development(context, reconstruct=True)
    assert report['selected'] == 'c90-m10-s256'
    report['candidates'][0]['actual_model_calls']['full'] = 0
    routing.save(context.path('development'), report)
    with pytest.raises(ValueError, match='saved observations'):
        routing.heldout({}, context)
    assert not context.path('test-consumed').exists()


def test_heldout_postflight_failure_never_persists_a_pass(frozen_context, monkeypatch):
    context = frozen_context
    write_development_phase(context)
    routing.complete_phase(context, 'dev')
    routing.audit_development(context)
    def fixture_measure(rows, candidate, detectors, *, context, partition):
        write_phase_observations(context, partition, candidate)
        if candidate and partition == 'mixed':
            manifest = json.loads(context.path('manifest').read_text())
            path = Path(next(iter(manifest['model_files'])))
            path.write_bytes(b'changed cache fixture after observations')
    monkeypatch.setattr(routing, 'measure', fixture_measure)
    with pytest.raises(ValueError, match='model cache'):
        routing.heldout({}, context)
    assert len(list(context.output.glob('routing-observations-test-*'))) == 6
    assert len(list(context.output.glob('routing-observations-mixed-*'))) == 6
    assert not context.path('heldout').exists()
    assert not context.path('test-complete').exists()
    with pytest.raises(ValueError, match='completion'):
        routing.audit_heldout(context)


def test_heldout_postprocessing_recovers_only_after_durable_receipt(frozen_context, monkeypatch):
    context = frozen_context
    write_development_phase(context)
    routing.complete_phase(context, 'dev')
    routing.audit_development(context)
    def fixture_measure(rows, candidate, detectors, *, context, partition):
        write_phase_observations(context, partition, candidate)
    monkeypatch.setattr(routing, 'measure', fixture_measure)
    real_score = routing.heldout_from_observations
    def broken_postprocessing(*args, **kwargs):
        raise ValueError('held-out postprocessing failed')
    monkeypatch.setattr(routing, 'heldout_from_observations', broken_postprocessing)
    with pytest.raises(ValueError, match='postprocessing'):
        routing.heldout({}, context)
    assert context.path('test-complete').is_file()
    assert not context.path('heldout').exists()
    monkeypatch.setattr(routing, 'heldout_from_observations', real_score)
    import huggingface_hub
    def forbidden(*args, **kwargs):
        pytest.fail('Numeric recovery must not load models or recheck current runtime')
    monkeypatch.setattr(huggingface_hub, 'snapshot_download', forbidden)
    monkeypatch.setattr(routing, 'runtime_identity', forbidden)
    report = routing.audit_heldout(context)
    assert report['baseline']['full']['overall']['docs'] == 110
    assert report['promotion']['status'] == 'fail'
    assert not report['promotion']['gates']['fewer_expensive_model_calls']
