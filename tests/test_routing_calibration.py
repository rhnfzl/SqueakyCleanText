"""Calibration keeps test data out of selection and rejects unsafe promotion."""
import importlib.util
import json
from collections import Counter
from dataclasses import replace
import os

import pytest


def api():
    assert importlib.util.find_spec('scripts.benchmark_detector_routing'), 'Routing benchmark is missing'
    from scripts import benchmark_detector_routing
    return benchmark_detector_routing


def row(identifier='x', language='en', domain='support'):
    return dict(id=identifier, text='Alice alice@example.org', language=language, domain=domain,
                spans=[dict(label='PERSON', start=0, end=5), dict(label='EMAIL', start=6, end=23)])


def test_grid_is_finite_and_each_threshold_combination_is_unique():
    grid = api().candidate_grid()
    assert len(grid) == 12
    assert len({(c['confidence'], c['margin'], c['max_chars']) for c in grid}) == 12
    assert {c['confidence'] for c in grid} == {.90, .95, .99}
    assert {c['margin'] for c in grid} == {.10, .20}
    assert {c['max_chars'] for c in grid} == {256, 512}


def test_metrics_keep_undefined_support_and_ignore_unannotated_types():
    module = api()
    records = [row()]
    predictions = {'x': [dict(label='EMAIL', start=6, end=23), dict(label='DATE', start=0, end=5)]}
    report, _ = module.score(records, predictions)
    full = report['full']
    assert full['overall']['recall'] == .5
    assert full['overall']['precision'] == 1
    assert full['overall']['document_exposure'] == 1
    assert full['entity']['PHONE']['recall'] is None
    assert full['entity']['PHONE']['document_exposure'] is None
    assert full['entity']['PHONE']['gold'] == 0
    assert report['shared']['types'] == ['PERSON', 'LOCATION', 'ORGANISATION']


def test_overlapping_native_predictions_without_scores_merge_before_scoring():
    record = row()
    predictions = {'x': record['spans'] + [dict(label='PERSON', start=0, end=5)]}
    report, _ = api().score([record], predictions)
    assert report['full']['overall']['predicted'] == 2
    assert report['full']['overall']['recall'] == 1


def test_added_exposure_on_one_slice_cannot_be_cancelled_by_another():
    module = api()
    records = [row('a'), row('b', 'de', 'hr')]
    base = {'a': records[0]['spans'], 'b': []}
    candidate = {'a': [], 'b': records[1]['spans']}
    _, baseline = module.score(records, base)
    _, result = module.score(records, candidate)
    assert module.added_exposure_slices(records, baseline, result)['language:en'] == 1
    assert not module.eligible(records, baseline, result)


def test_entity_exposure_regression_counts_when_document_already_exposed():
    module = api()
    records = [row()]
    _, baseline = module.score(records, {'x': [records[0]['spans'][1]]})
    _, candidate = module.score(records, {'x': []})
    assert module.added_exposure_slices(records, baseline, candidate)['entity:EMAIL'] == 1
    assert not module.eligible(records, baseline, candidate)


def test_l3_acceptance():
    module = api()
    partitions = [module.load_records(path) for path in module.INPUTS.values()]
    module.validate_partitions(*partitions)
    assert list(map(len, partitions)) == [220, 110, 6]
    from sct.routing import RoutingPolicy, routing_config_digest
    cfg = module.configuration(module.candidate_grid()[0])
    assert cfg.routing_calibration['config_digest'] == routing_config_digest(cfg, unified_policy=module.POLICY)
    assert RoutingPolicy(cfg, unified_policy=module.POLICY).identity.startswith('experimental:')


def test_latency_selection_ties_prefer_fewer_skips_then_stable_id():
    module = api()
    candidates = [dict(id='b', eligible=True, median_ms=2, skipped_stages=5),
                  dict(id='a', eligible=True, median_ms=2, skipped_stages=5),
                  dict(id='unsafe', eligible=False, median_ms=1, skipped_stages=1),
                  dict(id='c', eligible=True, median_ms=2, skipped_stages=4)]
    assert module.select_candidate(candidates)['id'] == 'c'
    assert module.select_candidate(candidates[:2])['id'] == 'a'
    assert module.select_candidate([candidates[2]]) is None


@pytest.mark.parametrize('upper,mixed,calls,latency,status', [
    (0, True, 9, 9, 'pass'), (.01, True, 9, 9, 'inconclusive'),
    (0, False, 9, 9, 'fail'), (0, True, 10, 9, 'fail'), (0, True, 9, 10, 'fail'),
])
def test_promotion_requires_all_four_gates(upper, mixed, calls, latency, status):
    result = api().promotion_gate([-0.01, upper], mixed, calls, 10, latency, 10, slice_regressions={})
    assert result['status'] == status
    assert result['default_policy'] == 'always_run'


def test_positive_interval_is_failure_not_inconclusive():
    assert api().promotion_gate([.01, .02], True, 9, 10, 9, 10, slice_regressions={})['status'] == 'fail'


def test_actual_bootstrap_promotion_result_can_be_saved_as_json():
    module = api()
    rows = [('x', 'x', 'en', Counter(docs=1, docs_with_gold=1, docs_exposed=0))]
    interval = module.exposure_interval(rows, rows)
    result = module.promotion_gate(interval, True, 9, 10, 9, 10, slice_regressions={})
    assert json.loads(json.dumps(result))['status'] == 'pass'


def test_heldout_entity_regression_rejects_promotion_despite_zero_document_interval():
    module = api()
    records = [row()]
    _, baseline = module.score(records, {'x': [records[0]['spans'][1]]})
    _, candidate = module.score(records, {'x': []})
    interval = module.exposure_interval(module.paired_rows(records, candidate), module.paired_rows(records, baseline))
    assert interval == [0, 0]
    gate = module.promotion_gate(interval, True, 9, 10, 9, 10,
                                 slice_regressions=module.added_exposure_slices(records, baseline, candidate))
    assert gate['status'] == 'fail'
    assert not gate['gates']['no_added_slice_exposure']


def development_fixture(module, manifest_hash):
    measured = dict(median_ms=2, p95_ms=3, latency_samples=660,
                    actual_model_calls={'fast': 1320, 'full': 1320}, skipped_stages=0)
    slices = {f'{key}:{value}': 0 for key, values in
              (('language', module.LANGUAGES), ('domain', module.DOMAINS), ('entity', module.TYPES)) for value in values}
    candidates = [measured | dict(id=c['id'], median_ms=1 + i, eligible=True,
                                  added_exposure_documents_by_slice=slices.copy())
                  for i, c in enumerate(module.candidate_grid())]
    return dict(schema_version=1, baseline=measured | dict(id='always_run'), candidates=candidates,
                selected='c90-m10-s256', comparison='c90-m10-s256', selection_uses='development_only',
                manifest_sha256=manifest_hash)


def test_valid_development_selection_and_negative_fallback_are_accepted():
    module = api()
    report = development_fixture(module, 'a' * 64)
    module.validate_development_report(report, 'a' * 64)
    for candidate in report['candidates']:
        candidate['eligible'] = False
        candidate['added_exposure_documents_by_slice']['entity:PERSON'] = 1
    report.update(selected=None, comparison='c99-m20-s256')
    module.validate_development_report(report, 'a' * 64)


@pytest.mark.parametrize('change', ['manifest', 'missing', 'duplicate', 'selection', 'comparison', 'nonfinite',
                                   'negative_calls', 'eligibility'])
def test_invalid_development_report_never_consumes_or_reads_test(tmp_path, monkeypatch, change):
    module = api()
    monkeypatch.setattr(module, 'OUTPUT', tmp_path)
    manifest = tmp_path / 'routing-manifest-v1.json'
    manifest.write_text('{}')
    report = development_fixture(module, module.file_hash(manifest))
    if change == 'manifest':
        report['manifest_sha256'] = '0' * 64
    elif change == 'missing':
        report['candidates'].pop()
    elif change == 'duplicate':
        report['candidates'][-1]['id'] = report['candidates'][0]['id']
    elif change == 'selection':
        report['selected'] = None
    elif change == 'comparison':
        report['comparison'] = 'c99-m20-s256'
    elif change == 'nonfinite':
        report['candidates'][0]['median_ms'] = float('nan')
    elif change == 'negative_calls':
        report['candidates'][0]['actual_model_calls'] = {'fast': 1, 'full': -1}
    else:
        report['candidates'][0]['eligible'] = 'true'
    (tmp_path / 'routing-development-v1.json').write_text(json.dumps(report))
    def forbidden(*args):
        pytest.fail('Invalid selection must be rejected before reading or measuring test')
    monkeypatch.setattr(module, 'load_records', forbidden)
    monkeypatch.setattr(module, 'measure', forbidden)
    with pytest.raises(ValueError):
        module.validate_development_report(report, module.file_hash(manifest))
    assert not (tmp_path / 'routing-test-consumed-v1.json').exists()


def test_frozen_sources_bind_gold_normalization_dependency(tmp_path, monkeypatch):
    module = api()
    assert hasattr(module, 'source_hashes'), 'A testable source freeze is required'
    hashes = module.source_hashes()
    labels = module.ROOT / 'benchmarks/labels.py'
    assert str(labels) in hashes
    copied = tmp_path / 'labels.py'
    copied.write_bytes(labels.read_bytes())
    expected = {str(copied): hashes[str(labels)]}
    module.verify_hashes(expected)
    copied.write_text('# changed normalization\n')
    with pytest.raises(ValueError, match='identity'):
        module.verify_hashes(expected)


def test_input_overlap_rejected_before_inference():
    with pytest.raises(ValueError, match='overlap'):
        api().validate_partitions([row()], [row('different-id')], [row('fixture')])


def test_input_bad_span_rejected_before_inference():
    record = row()
    record['spans'][0]['end'] = 999
    with pytest.raises(ValueError, match='span'):
        api().validate_records([record])


def test_paired_interval_uses_matched_documents():
    module = api()
    a = [('a', 'a', 'en', Counter(docs=1, docs_with_gold=1, docs_exposed=1))]
    b = [('a', 'a', 'en', Counter(docs=1, docs_with_gold=1, docs_exposed=0))]
    assert module.exposure_interval(a, b) == [1, 1]


def test_frozen_inputs_cannot_change_after_manifest(tmp_path):
    module = api()
    source = tmp_path / 'input.jsonl'
    source.write_text('original')
    expected = {str(source): module.file_hash(source)}
    module.verify_hashes(expected)
    source.write_text('changed')
    with pytest.raises(ValueError, match='identity'):
        module.verify_hashes(expected)


@pytest.mark.skipif(os.environ.get('SCT_ROUTING_NATIVE') != '1', reason='Explicit qualified CPU model runtime required')
def test_installed_native_models_count_actual_inference_and_skip_full():
    module = api()
    from sct.routing import routing_config_digest
    detectors = {stage: module.ModelDetector(stage) for stage in ('fast', 'full')}
    for stage, detector in detectors.items():
        assert set(detector.ner.cfg.ner_models.values()) == {module.MODELS[stage]['id']}
        assert set(detector.ner.cfg.ner_model_revisions.values()) == {module.MODELS[stage]['revision']}
    baseline = module.make_cleaner(None, detectors)
    result = baseline.process('John Smith works in Berlin. Email john.smith@example.org for details.')
    assert {hit['detector_id'] for hit in result.metadata['findings_v2']} >= {'routing_fast', 'routing_full'}
    assert all(detector.calls[0] > 0 for detector in detectors.values())
    candidate = module.make_cleaner(module.candidate_grid()[0], detectors)
    cfg = replace(candidate.cfg, language='en', routing_calibration=None)
    artifact = dict(candidate.cfg.routing_calibration)
    artifact['config_digest'] = routing_config_digest(cfg, unified_policy=module.POLICY)
    from sct import TextCleaner
    cleaner = TextCleaner(replace(cfg, routing_calibration=artifact),
                          detector_registry=candidate._detector_registry, unified_policy=module.POLICY)
    before = {stage: detector.calls[0] for stage, detector in detectors.items()}
    result = cleaner.process('John Smith works in Berlin. Email john.smith@example.org for details.')
    assert result.metadata['routing_v1']['skipped_ids'] == ['full']
    assert detectors['fast'].calls[0] > before['fast']
    assert detectors['full'].calls[0] == before['full']
