"""Measure experimental detector routing on frozen development and test inputs.

Run freeze, dev, then test. Test is consumed once, after development selection.
Model files must already exist in the cache; execution is offline and CPU only.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, fields, replace
import hashlib
import importlib
import importlib.metadata
import itertools
import json
import math
import os
from pathlib import Path
import platform
import resource
import statistics
import sys
import time
from typing import Any

from benchmarks.compare.scorer import covered, doc_stats, merge_same_type, paired_bootstrap
from evaluation.improvements.measure_baseline import instrument_sessions
from sct import TextCleaner, TextCleanerConfig
from sct.detectors import DetectorCapabilities, DetectorRegistry, DetectorSpec
from sct.detectors_builtin import BuiltinRegexDetector
from sct.policy_v2 import UnifiedPrivacyPolicy
from sct.routing import capabilities_digest, routing_config_digest
from sct.utils.atomic_file import write_text_atomic

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'evaluation/improvements'
TYPES = ('PERSON', 'LOCATION', 'ORGANISATION', 'EMAIL', 'PHONE', 'IP')
SHARED = TYPES[:3]
LANGUAGES = ('en', 'nl', 'de', 'es', 'zh', 'ja', 'th', 'ar', 'hi', 'bn', 'ta')
DOMAINS = ('support', 'hr', 'security', 'office', 'billing')
SEED = 20261107
REPEATS = 3
WARMUP = 2
RUNTIME_ENV = {'HF_HUB_OFFLINE': '1', 'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2',
               'TOKENIZERS_PARALLELISM': 'false', 'CUDA_VISIBLE_DEVICES': ''}
PROVIDERS = (('onnxruntime', 'onnxruntime'), ('tokenizers', 'tokenizers'), ('numpy', 'numpy'),
             ('lingua-language-detector', 'lingua'), ('huggingface-hub', 'huggingface_hub'), ('onnx', 'onnx'))
MODEL_FILES = ('config.json', 'model.onnx', 'tokenizer.json', 'tokenizer_config.json')
MEMORY_METHOD = ('Linux resource.getrusage(RUSAGE_SELF).ru_maxrss, KiB, cumulative process high-water mark. '
                 'All policies share resident models; these are not independent per-policy memory measurements.')
ROUTING_REASONS = frozenset(('always_run', 'fast_stage_incomplete', 'missing_capabilities', 'missing_calibration',
    'unknown_language', 'uncalibrated_language', 'language_evidence', 'text_size', 'unsupported_context',
    'unknown_entity_request', 'unsupported_entity_coverage', 'calibrated_fast_coverage'))
POLICY = UnifiedPrivacyPolicy('routing-benchmark', '1', {})
MODELS: dict[str, dict[str, Any]] = {
    'fast': {'id': 'protectai/bert-base-NER-onnx', 'revision': '84d28ab2bc28401b4353b26033eb043c8c8a0739',
             'license': None, 'languages': ('ENGLISH',)},
    'full': {'id': 'rhnfzl/wikineural-multilingual-ner-onnx', 'revision': '74191432b21c956b5b04b4778c0e92e569b1f80d',
             'license': None,
             'languages': ('ENGLISH', 'DUTCH', 'GERMAN', 'SPANISH', 'FRENCH', 'ITALIAN', 'PORTUGUESE', 'POLISH', 'RUSSIAN')},
}
INPUTS = {'dev': ROOT / 'evaluation/multilingual_pii.jsonl',
          'test': OUTPUT / 'routing-test-v1.jsonl', 'mixed': OUTPUT / 'routing-mixed-v1.jsonl'}


@dataclass(frozen=True)
class RunContext:
    """Keep current experiments separate from consumed historical evidence."""

    experiment: str = 'v1'
    root: Path = ROOT

    @property
    def output(self):
        base = self.root / 'evaluation/improvements'
        return base / 'routing-v090-v2' if self.experiment == 'v090-v2' else base

    @property
    def seed(self):
        return 20261109 if self.experiment == 'v090-v2' else SEED

    @property
    def inputs(self):
        return {'dev': self.root / 'evaluation/multilingual_pii.jsonl',
                'test': self.path('test', '.jsonl'), 'mixed': self.path('mixed', '.jsonl')}

    def path(self, name, suffix='.json'):
        version = 'v2' if self.experiment == 'v090-v2' else 'v1'
        return self.output / f'routing-{name}-{version}{suffix}'

    def raw_path(self, partition, policy, repeat):
        return self.path(f'observations-{partition}-{policy}-r{repeat}')

    def require_current(self):
        if self.experiment != 'v090-v2':
            raise ValueError('v1 is historical and consumed; use its frozen archive for read-only inspection')


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_hashes(expected):
    for name, digest in expected.items():
        if file_hash(name) != digest:
            raise ValueError(f'Frozen identity changed: {name}')


def canonical(data):
    return json.dumps(data, sort_keys=True, separators=(',', ':'), allow_nan=False)


def require_equal(actual, expected, label):
    if canonical(actual) != canonical(expected):
        raise ValueError(f'{label} identity mismatch')


def save(path, data):
    write_text_atomic(Path(path), json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + '\n')
    with Path(path).open('rb') as stream:
        os.fsync(stream.fileno())
    descriptor = os.open(Path(path).parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def consume(path, data):
    """Persist an exclusive claim before any inference, including after a crash."""
    payload = json.dumps(data, sort_keys=True, allow_nan=False) + '\n'
    with Path(path).open('x') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    descriptor = os.open(Path(path).parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def candidate_grid():
    return [dict(id=f'c{int(confidence * 100)}-m{int(margin * 100)}-s{size}',
                 confidence=confidence, margin=margin, max_chars=size)
            for confidence, margin, size in itertools.product((.90, .95, .99), (.10, .20), (256, 512))]


def validate_records(records):
    seen, texts = set(), set()
    for row in records:
        if (not isinstance(row['id'], str) or not row['id'] or row['id'] in seen
                or not isinstance(row['text'], str) or not row['text'] or row['text'] in texts):
            raise ValueError('Input IDs and texts must be unique and nonempty')
        seen.add(row['id'])
        texts.add(row['text'])
        if row['language'] not in LANGUAGES or row['domain'] not in DOMAINS:
            raise ValueError('Undeclared input slice')
        for span in row['spans']:
            if (span['label'] not in TYPES or type(span['start']) is not int or type(span['end']) is not int
                    or not 0 <= span['start'] < span['end'] <= len(row['text'])):
                raise ValueError('Invalid input span')


def validate_partitions(*partitions):
    seen_ids: set[str] = set()
    seen_text: set[str] = set()
    for records in partitions:
        validate_records(records)
        identifiers = {row['id'] for row in records}
        texts = {hashlib.sha256(row['text'].encode()).hexdigest() for row in records}
        if identifiers & seen_ids or texts & seen_text:
            raise ValueError('Input partition overlap')
        seen_ids.update(identifiers)
        seen_text.update(texts)


def load_records(path):
    records = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    for row in records:
        row['spans'] = row.pop('entities')
    validate_records(records)
    return records


def validate_current_inputs(context):
    context.require_current()
    partitions = {key: load_records(value) for key, value in context.inputs.items()}
    previous = [load_records(context.root / f'evaluation/improvements/routing-{name}-v1.jsonl')
                for name in ('test', 'mixed')]
    validate_partitions(*partitions.values(), *previous)
    if {key: len(rows) for key, rows in partitions.items()} != {'dev': 220, 'test': 110, 'mixed': 6}:
        raise ValueError('Preregistered partition counts differ')
    cohort = partitions['test']
    balance = Counter({(lang, domain): 2 for lang in LANGUAGES for domain in DOMAINS})
    if Counter((row['language'], row['domain']) for row in cohort) != balance:
        raise ValueError('Preregistered language/domain balance differs')
    for row in cohort:
        if (Counter(s['label'] for s in row['spans']) != Counter(TYPES)
                or row['template_id'] != 'route-v090-v2-body-' + row['language']):
            raise ValueError('Preregistered types or template clusters differ')
    provenance = json.loads(context.path('input-provenance').read_text())
    verify_hashes({context.root / path: digest for path, digest in provenance['partition_sha256'].items()})
    return partitions


def summarize(counters):
    counts: Counter[str] = sum(counters, Counter())
    return {'docs': counts['docs'], 'gold': counts['gold'], 'predicted': counts['pred'],
            'true_positive': counts['strict_tp'], 'docs_with_gold': counts['docs_with_gold'],
            'exposed_docs': counts['docs_exposed'],
            'precision': counts['strict_tp'] / counts['pred'] if counts['pred'] else None,
            'recall': counts['strict_tp'] / counts['gold'] if counts['gold'] else None,
            'document_exposure': counts['docs_exposed'] / counts['docs_with_gold'] if counts['docs_with_gold'] else None}


def score(records, predictions, *, count_view='full'):
    """Score only annotated types; absence of support remains explicitly undefined."""
    if set(predictions) != {row['id'] for row in records}:
        raise ValueError('Predictions must cover every input document')
    reports, full_counts = {}, []
    for view, types in (('shared', SHARED), ('full', TYPES)):
        mapping = {kind: kind for kind in types}
        counts = []
        by_entity: dict[str, list[Counter[str]]] = {kind: [] for kind in TYPES}
        for row in records:
            spans = merge_same_type([dict(p, score=p.get('score'))
                                     for p in predictions[row['id']] if p['label'] in TYPES])
            count = doc_stats(row, spans, mapping, utility=False)
            counts.append(count)
            for kind in TYPES:
                entity = doc_stats(row, spans, {kind: kind} if kind in types else {}, utility=False)
                by_entity[kind].append(entity)
                count['entity_exposed:' + kind] = entity['docs_exposed']
            masked = covered(len(row['text']), spans)
            for index, gold in enumerate(row['spans']):
                count[f'gold_exposed:{index}'] = int(any(not masked[i] and row['text'][i].isalnum()
                                                        for i in range(gold['start'], gold['end'])))
        reports[view] = {'types': list(types), 'overall': summarize(counts),
                        'language': {lang: summarize([c for r, c in zip(records, counts, strict=True)
                                                     if r['language'] == lang]) for lang in LANGUAGES},
                        'domain': {domain: summarize([c for r, c in zip(records, counts, strict=True)
                                                     if r['domain'] == domain]) for domain in DOMAINS},
                        'entity': {kind: summarize(values) for kind, values in by_entity.items()}}
        if view == count_view:
            full_counts = counts
    return reports, full_counts


def added_exposure_slices(records, baseline, candidate):
    result = {}
    for label, values in (('language', LANGUAGES), ('domain', DOMAINS), ('entity', TYPES)):
        for value in values:
            indices = [i for i, row in enumerate(records) if
                       (any(span['label'] == value for span in row['spans']) if label == 'entity' else row[label] == value)]
            key = 'entity_exposed:' + value if label == 'entity' else 'docs_exposed'
            result[f'{label}:{value}'] = sum(max(0, candidate[i][key] - baseline[i][key])
                                            for i in indices) if indices else None
    return result


def eligible(records, baseline, candidate):
    return not any(value for value in added_exposure_slices(records, baseline, candidate).values())


def exposure_views(records, baseline, candidate):
    result = {}
    for view in ('shared', 'full'):
        _, base_counts = score(records, baseline, count_view=view)
        _, counts = score(records, candidate, count_view=view)
        result[view] = added_exposure_slices(records, base_counts, counts)
    return result


def select_candidate(candidates):
    choices = [row for row in candidates if row['eligible']]
    return min(choices, key=lambda row: (row['median_ms'], row['skipped_stages'], row['id'])) if choices else None


def exposure_interval(candidate, baseline, *, seed=SEED):
    result = paired_bootstrap(candidate, baseline, names=('doc_leak_rate',), samples=1000, seed=seed)
    return [float(value) for value in result['metrics']['doc_leak_rate']['diff_ci']]


def promotion_gate(interval, mixed_ok, calls, baseline_calls, median_ms, baseline_median, *, slice_regressions):
    gates = {'exposure_upper_nonpositive': interval[1] <= 0, 'mixed_no_regression': mixed_ok,
             'fewer_expensive_model_calls': calls < baseline_calls, 'lower_median_latency': median_ms < baseline_median,
             'no_added_slice_exposure': not any(slice_regressions.values())}
    status = 'pass' if all(gates.values()) else 'fail'
    if not gates['exposure_upper_nonpositive'] and interval[0] <= 0 and all(list(gates.values())[1:]):
        status = 'inconclusive'
    return {'status': status, 'gates': gates, 'paired_95_percent_interval': interval, 'default_policy': 'always_run'}


def capabilities(stage):
    model = MODELS[stage]
    return DetectorCapabilities('routing_' + stage, version='benchmark-1', entity_types=SHARED,
                                languages=model['languages'],
                                model_identities=((model['id'], model['revision'], model['license']),))


def configuration(candidate=None):
    values: dict[str, Any] = {field.name: False for field in fields(TextCleanerConfig) if field.name.startswith('check_')}
    cfg = TextCleanerConfig(**(values | dict(pii_processing='unified', check_detect_language=True,
        check_replace_emails=True, check_replace_phone_numbers=True, check_replace_ips=True,
        replace_ips_public_only=False, language_policy='abstain',
        routing_entity_types=TYPES,
        routing_fast_specs=(DetectorSpec('builtin_regex'), DetectorSpec('routing_fast', MODELS['fast'])),
        routing_full_specs=(DetectorSpec('routing_full', MODELS['full']),),
        routing_policy='calibrated_cascade' if candidate else 'always_run')))
    if candidate:
        calibration = dict(schema_version=1, config_digest=routing_config_digest(cfg, unified_policy=POLICY),
            fast_capabilities_digest=capabilities_digest((BuiltinRegexDetector(cfg).capabilities, capabilities('fast'))),
            language_min_confidence=candidate['confidence'], language_min_margin=candidate['margin'],
            max_chars=candidate['max_chars'], languages=['ENGLISH'], entity_types=list(TYPES))
        cfg = replace(cfg, routing_calibration=calibration)
    return cfg


class ModelDetector:
    """Bind each application factory to one pinned model and count real session.run calls."""

    def __init__(self, stage):
        from sct.config import DEFAULT_NER_MODELS
        from sct.utils.ner import GeneralNER
        model = MODELS[stage]
        cfg = TextCleanerConfig(ner_device='cpu', ner_models={k: model['id'] for k in DEFAULT_NER_MODELS},
                               ner_model_revisions={k: model['revision'] for k in DEFAULT_NER_MODELS},
                               onnx_intra_op_threads=2, onnx_inter_op_threads=1)
        self.ner = GeneralNER(cfg)
        self.capabilities = capabilities(stage)
        self.calls = instrument_sessions(self.ner._pipelines.values())

    def detect(self, text, *, language, context):
        return tuple(replace(hit, detector_id=self.capabilities.id)
                     for hit in self.ner.analyze(text, language=language, context=context) if hit.entity_type in SHARED)


def make_cleaner(candidate, detectors):
    registry = DetectorRegistry()
    for stage, detector in detectors.items():
        def factory(cfg, options, detector=detector):
            return detector
        registry.register('routing_' + stage, factory)
    return TextCleaner(configuration(candidate), detector_registry=registry, unified_policy=POLICY)


def audit_observations(records, raw, policy, partition, manifest_hash):
    """Recompute every score and timing quantity from source-free repeat observations."""
    try:
        json.dumps(raw, allow_nan=False)
        validate_records(records)
        if not records or len(raw) != REPEATS:
            raise ValueError('Exactly three complete repeats required')
        durations = []
        predictions: dict[str, list[dict[str, Any]]] = {}
        languages: Counter[str] = Counter()
        reasons: Counter[str] = Counter()
        calls: Counter[str] = Counter()
        skipped = 0
        for repeat, batch in enumerate(raw):
            if (set(batch) != {'schema_version', 'id', 'partition', 'repeat', 'manifest_sha256',
                              'counter_start', 'counter_end', 'process_peak_rss_kib', 'observations'}
                    or batch['schema_version'] != 2 or batch['id'] != policy or batch['partition'] != partition
                    or type(batch['repeat']) is not int or batch['repeat'] != repeat
                    or batch['manifest_sha256'] != manifest_hash):
                raise ValueError('Observation identity mismatch')
            if type(batch['process_peak_rss_kib']) is not int or batch['process_peak_rss_kib'] <= 0:
                raise ValueError('Invalid peak RSS')
            if [item['id'] for item in batch['observations']] != [row['id'] for row in records]:
                raise ValueError('Observation document coverage or order differs')
            totals: Counter[str] = Counter()
            for row, item in zip(records, batch['observations'], strict=True):
                if set(item) != {'id', 'predictions', 'duration_ms', 'actual_model_calls',
                                 'skipped_stages', 'routing_reasons', 'language_source'}:
                    raise ValueError('Unexpected observation fields')
                duration = item['duration_ms']
                if type(duration) not in (int, float) or duration < 0:
                    raise ValueError('Invalid duration')
                for span in item['predictions']:
                    if (set(span) != {'label', 'start', 'end'} or span['label'] not in TYPES
                            or type(span['start']) is not int or type(span['end']) is not int
                            or not 0 <= span['start'] < span['end'] <= len(row['text'])):
                        raise ValueError('Invalid positional prediction')
                if repeat and item['predictions'] != predictions[row['id']]:
                    raise ValueError('Inference changed across frozen repeats')
                predictions[row['id']] = item['predictions']
                deltas = item['actual_model_calls']
                if set(deltas) != {'fast', 'full'} or any(type(n) is not int or n < 0 for n in deltas.values()):
                    raise ValueError('Invalid actual model-call deltas')
                if type(item['skipped_stages']) is not int or not 0 <= item['skipped_stages'] <= 1:
                    raise ValueError('Invalid skipped stages')
                if (not isinstance(item['routing_reasons'], list)
                        or any(reason not in ROUTING_REASONS for reason in item['routing_reasons'])
                        or item['language_source'] not in ('detected', 'abstained', 'pinned', 'fallback', 'none')):
                    raise ValueError('Invalid routing or language trace')
                totals.update(deltas)
                durations.append(duration)
                reasons.update(item['routing_reasons'])
                skipped += item['skipped_stages']
                if not repeat:
                    languages[item['language_source']] += 1
            for key in ('counter_start', 'counter_end'):
                if (set(batch[key]) != {'fast', 'full'}
                        or any(type(n) is not int or n < 0 for n in batch[key].values())):
                    raise ValueError('Invalid cumulative model counters')
            if repeat and batch['counter_start'] != raw[repeat - 1]['counter_end']:
                raise ValueError('Model counters are not contiguous')
            if any(batch['counter_end'][stage] - batch['counter_start'][stage] != totals[stage] for stage in MODELS):
                raise ValueError('Actual model counter totals disagree')
            calls.update(totals)
        report, counts = score(records, predictions)
        report['language_model_support'] = {
            lang: {'fast': lang == 'en', 'full': lang in ('en', 'nl', 'de', 'es'),
                   'documents': sum(row['language'] == lang for row in records),
                   'interpretation': 'within declared training languages' if lang in ('en', 'nl', 'de', 'es')
                   else 'model extrapolation; no declared neural language support'} for lang in LANGUAGES}
        report.update(id=policy, median_ms=statistics.median(durations),
                      p95_ms=sorted(durations)[math.ceil(.95 * len(durations)) - 1],
                      latency_samples=len(durations), actual_model_calls=dict(calls),
                      skipped_stages=skipped, routing_reasons=dict(reasons), language_sources=dict(languages),
                      language_coverage=languages['detected'] / len(records),
                      process_peak_rss_kib=max(batch['process_peak_rss_kib'] for batch in raw))
        return report, counts
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError('Invalid saved observations') from exc


def first_predictions(raw):
    return {item['id']: item['predictions'] for item in raw[0]['observations']}


def development_from_observations(records, observations, manifest_hash):
    expected = {'always_run', *(candidate['id'] for candidate in candidate_grid())}
    if set(observations) != expected:
        raise ValueError('Development requires the complete unique candidate grid')
    baseline, _ = audit_observations(records, observations['always_run'], 'always_run', 'dev', manifest_hash)
    candidates = []
    for candidate in candidate_grid():
        policy = candidate['id']
        result, _ = audit_observations(records, observations[policy], policy, 'dev', manifest_hash)
        slices = exposure_views(records, first_predictions(observations['always_run']),
                                first_predictions(observations[policy]))
        result.update(eligible=not any(n for view in slices.values() for n in view.values()),
                      added_exposure_documents_by_view=slices)
        candidates.append(result)
    chosen = select_candidate(candidates)
    return dict(schema_version=2, baseline=baseline, candidates=candidates, selected=chosen['id'] if chosen else None,
                comparison=chosen['id'] if chosen else 'c99-m20-s256', selection_uses='development_only',
                manifest_sha256=manifest_hash)


def validate_development_audit(report, records, observations, manifest_hash):
    rebuilt = development_from_observations(records, observations, manifest_hash)
    if report != rebuilt:
        raise ValueError('Development report disagrees with saved observations')


def measure(records, candidate, detectors, *, context, partition):
    context.require_current()
    policy = candidate['id'] if candidate else 'always_run'
    if any(context.raw_path(partition, policy, repeat).exists() for repeat in range(REPEATS)):
        raise ValueError('Observations already exist; use audit without inference')
    manifest_hash = file_hash(context.path('manifest'))
    cleaner = make_cleaner(candidate, detectors)
    for row in records[:WARMUP]:
        cleaner.process(row['text'])
    for repeat in range(REPEATS):
        starts = {stage: detector.calls[0] for stage, detector in detectors.items()}
        observations = []
        for row in records:
            before = {stage: detector.calls[0] for stage, detector in detectors.items()}
            started = time.perf_counter()
            result = cleaner.process(row['text'])
            duration = (time.perf_counter() - started) * 1000
            found = [dict(label=hit['entity_type'], start=hit['start'], end=hit['end'])
                     for hit in result.metadata['findings_v2'] if hit['action'] != 'keep']
            route = result.metadata.get('routing_v1', {})
            observations.append(dict(id=row['id'], predictions=found, duration_ms=duration,
                actual_model_calls={stage: detector.calls[0] - before[stage] for stage, detector in detectors.items()},
                skipped_stages=len(route.get('skipped_ids', [])), routing_reasons=route.get('reasons', []),
                language_source=result.metadata['language_source']))
        batch = dict(schema_version=2, id=policy, partition=partition, repeat=repeat, manifest_sha256=manifest_hash,
                     counter_start=starts, counter_end={stage: detector.calls[0] for stage, detector in detectors.items()},
                     process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, observations=observations)
        save(context.raw_path(partition, policy, repeat), batch)


def runtime_identity():
    providers = {}
    for distribution, module in PROVIDERS:
        loaded = importlib.import_module(module)
        if loaded.__file__ is None:
            raise ValueError('Runtime module has no source path')
        path = Path(loaded.__file__).resolve()
        binaries = sorted(path.parent.rglob('*.so'))
        providers[distribution] = dict(version=importlib.metadata.version(distribution), path=str(path),
            sha256=file_hash(path), binaries={str(p): file_hash(p) for p in binaries})
    return dict(python=sys.version, executable=sys.executable, executable_sha256=file_hash(sys.executable),
                prefix=sys.prefix, platform=platform.platform(),
                cpu_affinity=sorted(getattr(os, 'sched_getaffinity')(0)), providers=providers,
                environment={key: os.environ.get(key) for key in RUNTIME_ENV})


def require_runtime():
    if (sys.platform != 'linux' or any(os.environ.get(key) != value for key, value in RUNTIME_ENV.items())
            or sorted(getattr(os, 'sched_getaffinity')(0)) != [2, 3]):
        raise ValueError('Run with the preregistered offline Linux CPU runtime and affinity')


def source_hashes(context=RunContext()):
    root = context.root
    sources = list((root / 'sct').rglob('*.py')) + [root / 'scripts/benchmark_detector_routing.py',
        root / 'benchmarks/compare/scorer.py', root / 'benchmarks/score.py', root / 'benchmarks/labels.py',
        root / 'evaluation/improvements/measure_baseline.py', context.output / 'routing-protocol.md',
        context.path('input-provenance'), root / 'pyproject.toml', root / 'uv.lock', root / 'pylock.toml']
    return {str(path): file_hash(path) for path in sources}


def fixed_identity(context):
    return dict(schema_version=2, experiment=context.experiment, seed=context.seed, models=MODELS,
        input_counts={'dev': 220, 'test': 110, 'mixed': 6}, grid=candidate_grid(),
        warmup_documents=WARMUP, measured_repeats=REPEATS,
        config_digest=routing_config_digest(configuration(), unified_policy=POLICY),
        no_eligible_comparison='c99-m20-s256', memory_method=MEMORY_METHOD)


def expected_calibrations():
    return dict(schema_version=2, default_policy='always_run', model_identities=MODELS,
        calibrations={c['id']: dict(configuration(c).routing_calibration) for c in candidate_grid()})


def cached_model_hashes():
    from huggingface_hub import snapshot_download
    files = {}
    for model in MODELS.values():
        snapshot = Path(snapshot_download(model['id'], revision=model['revision'], local_files_only=True))
        files.update({str(path): file_hash(path) for path in sorted(snapshot.rglob('*')) if path.is_file()})
    return files


def validate_hash_inventory(inventory):
    if not isinstance(inventory, dict) or not inventory:
        raise ValueError('Missing frozen inventory')
    for path, digest in inventory.items():
        if (not isinstance(path, str) or not Path(path).is_absolute() or '..' in Path(path).parts
                or not isinstance(digest, str) or len(digest) != 64
                or any(char not in '0123456789abcdef' for char in digest)):
            raise ValueError('Invalid frozen inventory path or digest')


def validate_model_declarations(files):
    validate_hash_inventory(files)
    snapshots = []
    for model in MODELS.values():
        suffix = '/models--' + model['id'].replace('/', '--') + '/snapshots/' + model['revision'] + '/model.onnx'
        matches = [Path(path).parent for path in files if path.endswith(suffix)]
        if len(matches) != 1 or any(str(matches[0] / name) not in files for name in MODEL_FILES):
            raise ValueError('Incomplete pinned model declaration')
        snapshots.extend(matches)
    if any(not any(Path(path).is_relative_to(snapshot) for snapshot in snapshots) for path in files):
        raise ValueError('Unexpected model inventory path')


def validate_runtime_declaration(runtime):
    require_equal(runtime['cpu_affinity'], [2, 3], 'Protocol CPU affinity')
    require_equal(runtime['environment'], RUNTIME_ENV, 'Protocol environment')
    if set(runtime) != {'python', 'executable', 'executable_sha256', 'prefix', 'platform',
                        'cpu_affinity', 'environment', 'providers'} or not runtime['platform'].startswith('Linux-'):
        raise ValueError('Invalid Linux runtime declaration')
    if any(not isinstance(runtime[key], str) or not runtime[key] for key in ('python', 'executable', 'prefix', 'platform')):
        raise ValueError('Incomplete runtime declaration')
    validate_hash_inventory({runtime['executable']: runtime['executable_sha256']})
    require_equal(sorted(runtime['providers']), sorted(name for name, _ in PROVIDERS), 'Runtime providers')
    for provider in runtime['providers'].values():
        if set(provider) != {'version', 'path', 'sha256', 'binaries'} or not isinstance(provider['version'], str):
            raise ValueError('Invalid provider declaration')
        validate_hash_inventory({provider['path']: provider['sha256']})
        if not isinstance(provider['binaries'], dict):
            raise ValueError('Invalid provider binary inventory')
        if provider['binaries']:
            validate_hash_inventory(provider['binaries'])


def freeze(context=RunContext()):
    context.require_current()
    require_runtime()
    path = context.path('manifest')
    if path.exists() or context.path('calibrations').exists() or context.path('test-consumed').exists():
        raise ValueError('Manifest already frozen or experiment consumed')
    validate_current_inputs(context)
    manifest = fixed_identity(context) | dict(source_root=str(context.root), model_files=cached_model_hashes(),
        source_files=source_hashes(context),
        inputs={str(p): file_hash(p) for p in context.inputs.values()},
        runtime=runtime_identity(), timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    validate_model_declarations(manifest['model_files'])
    validate_runtime_declaration(manifest['runtime'])
    consume(context.path('freeze-consumed'), {'source_files': manifest['source_files']})
    save(context.path('calibrations'), expected_calibrations())
    manifest['calibrations_sha256'] = file_hash(context.path('calibrations'))
    save(path, manifest)
    print(json.dumps({'phase': 'frozen', 'input_counts': manifest['input_counts'],
                      'manifest_sha256': file_hash(path)}), flush=True)


def verify_frozen(context=RunContext(), *, inference=True):
    context.require_current()
    try:
        manifest = json.loads(context.path('manifest').read_text())
        fixed = fixed_identity(context)
        extras = {'source_root', 'source_files', 'inputs', 'model_files', 'runtime', 'timestamp_utc', 'calibrations_sha256'}
        if set(manifest) != set(fixed) | extras:
            raise ValueError('Incomplete or unexpected manifest fields')
        require_equal({key: manifest[key] for key in fixed}, fixed, 'Preregistered experiment')
        stamp = time.strptime(manifest['timestamp_utc'], '%Y-%m-%dT%H:%M:%SZ')
        require_equal(manifest['timestamp_utc'], time.strftime('%Y-%m-%dT%H:%M:%SZ', stamp), 'Timestamp')
        source_root = Path(manifest['source_root'])
        if not source_root.is_absolute() or '..' in source_root.parts:
            raise ValueError('Invalid frozen source root')
        expected = {'source_files': source_hashes(context),
                    'inputs': {str(path): file_hash(path) for path in context.inputs.values()}}
        for name, inventory in expected.items():
            validate_hash_inventory(manifest[name])
            relocated = {str(context.root / Path(path).relative_to(source_root)): digest
                         for path, digest in manifest[name].items()}
            require_equal(relocated, inventory, name)
        validate_current_inputs(context)
        validate_model_declarations(manifest['model_files'])
        validate_runtime_declaration(manifest['runtime'])
        require_equal(file_hash(context.path('calibrations')), manifest['calibrations_sha256'], 'Calibration file')
        require_equal(json.loads(context.path('calibrations').read_text()), expected_calibrations(), 'Calibration contents')
        if inference:
            require_runtime()
            require_equal(str(context.root), manifest['source_root'], 'Inference source root')
            require_equal(cached_model_hashes(), manifest['model_files'], 'Complete model cache')
            require_equal(runtime_identity(), manifest['runtime'], 'Runtime')
        return manifest
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError('Invalid frozen manifest') from exc


def paired_rows(records, counts):
    return [(row['id'], row.get('template_id', row['id']), row['language'], count)
            for row, count in zip(records, counts, strict=True)]


def read_observations(context, partition, policies):
    return {policy: [json.loads(context.raw_path(partition, policy, repeat).read_text()) for repeat in range(REPEATS)]
            for policy in policies}


def completion_identity(context, phase):
    """Bind every expected repeat and the consumed phase to one frozen manifest."""
    if phase not in ('dev', 'test'):
        raise ValueError('Unknown completion phase')
    try:
        consumed_path = context.path(f'{phase}-consumed')
        consumed = json.loads(consumed_path.read_text())
        manifest_hash = file_hash(context.path('manifest'))
        if phase == 'dev':
            require_equal(consumed, {'manifest_sha256': manifest_hash}, 'Phase consumption')
            partitions, policies = ['dev'], ['always_run', *(c['id'] for c in candidate_grid())]
        else:
            if (set(consumed) != {'candidate', 'manifest_sha256', 'development_sha256', 'timestamp_utc'}
                    or consumed['candidate'] not in {c['id'] for c in candidate_grid()}
                    or consumed['manifest_sha256'] != manifest_hash
                    or consumed['development_sha256'] != file_hash(context.path('development'))):
                raise ValueError('Invalid test consumption')
            time.strptime(consumed['timestamp_utc'], '%Y-%m-%dT%H:%M:%SZ')
            partitions, policies = ['test', 'mixed'], ['always_run', consumed['candidate']]
        paths = [context.raw_path(partition, policy, repeat)
                 for partition in partitions for policy in policies for repeat in range(REPEATS)]
        return dict(schema_version=2, experiment=context.experiment, phase=phase,
            postflight='source_inputs_model_cache_runtime_verified', manifest_sha256=manifest_hash,
            consumption_sha256=file_hash(consumed_path), observations={path.name: file_hash(path) for path in paths})
    except (FileNotFoundError, KeyError, TypeError) as exc:
        raise ValueError('Incomplete phase completion evidence') from exc


def complete_phase(context, phase):
    """Only the inference path records postflight success before any final report."""
    target = context.path(f'{phase}-complete')
    if target.exists():
        raise ValueError('Phase completion already recorded')
    receipt = completion_identity(context, phase)
    verify_frozen(context, inference=True)
    require_equal(completion_identity(context, phase), receipt, 'Postflight completion inputs')
    save(target, receipt)


def verify_completion(context, phase):
    try:
        receipt = json.loads(context.path(f'{phase}-complete').read_text())
    except FileNotFoundError as exc:
        raise ValueError('Phase completion receipt is missing; inference remains incomplete') from exc
    require_equal(receipt, completion_identity(context, phase), 'Phase completion receipt')
    verify_frozen(context, inference=False)


def development(detectors, context=RunContext()):
    context.require_current()
    verify_frozen(context)
    if context.path('development').exists():
        raise ValueError('Development results already exist')
    consume(context.path('dev-consumed'), {'manifest_sha256': file_hash(context.path('manifest'))})
    records = load_records(context.inputs['dev'])
    for candidate in [None, *candidate_grid()]:
        measure(records, candidate, detectors, context=context, partition='dev')
        print(json.dumps({'id': candidate['id'] if candidate else 'always_run', 'saved_repeats': REPEATS}), flush=True)
    complete_phase(context, 'dev')
    return audit_development(context)


def audit_development(context, *, reconstruct=False):
    verify_completion(context, 'dev')
    records = load_records(context.inputs['dev'])
    raw = read_observations(context, 'dev', ['always_run', *(c['id'] for c in candidate_grid())])
    report = development_from_observations(records, raw, file_hash(context.path('manifest')))
    target = context.path('development')
    if target.exists():
        validate_development_audit(json.loads(target.read_text()), records, raw, file_hash(context.path('manifest')))
    else:
        save(target, report)
    if reconstruct:
        print(json.dumps({'audit': 'development', 'selected': report['selected'], 'inference_replayed': False}), flush=True)
    return report


def validate_development_report(report, manifest_hash):
    """Reject stale, partial or inconsistent selection before consuming test inputs."""
    try:
        json.dumps(report, allow_nan=False)
        if (report['schema_version'] != 1 or report['manifest_sha256'] != manifest_hash
                or report['selection_uses'] != 'development_only' or report['baseline']['id'] != 'always_run'):
            raise ValueError('Development identity mismatch')
        candidates = report['candidates']
        ids = [candidate['id'] for candidate in candidates]
        if len(ids) != 12 or set(ids) != {candidate['id'] for candidate in candidate_grid()}:
            raise ValueError('Development requires the complete unique candidate grid')
        expected_slices = {f'{key}:{value}' for key, values in
                           (('language', LANGUAGES), ('domain', DOMAINS), ('entity', TYPES)) for value in values}
        for measured in [report['baseline'], *candidates]:
            for key in ('median_ms', 'p95_ms'):
                if type(measured[key]) not in (int, float) or measured[key] < 0:
                    raise ValueError('Invalid development latency')
            for key in ('latency_samples', 'skipped_stages'):
                if type(measured[key]) is not int or measured[key] < (1 if key == 'latency_samples' else 0):
                    raise ValueError('Invalid development execution counts')
            calls = measured['actual_model_calls']
            if set(calls) != {'fast', 'full'} or any(type(n) is not int or n < 0 for n in calls.values()):
                raise ValueError('Invalid development model calls')
            if measured['latency_samples'] != report['baseline']['latency_samples']:
                raise ValueError('Development measurement coverage differs')
        for candidate in candidates:
            slices = candidate['added_exposure_documents_by_slice']
            if set(slices) != expected_slices or any(type(n) is not int or n < 0 for n in slices.values()):
                raise ValueError('Invalid development exposure slices')
            if type(candidate['eligible']) is not bool or candidate['eligible'] != (not any(slices.values())):
                raise ValueError('Development eligibility disagrees with exposure slices')
        selected = select_candidate(candidates)
        expected = selected['id'] if selected else None
        if report['selected'] != expected or report['comparison'] != (expected or 'c99-m20-s256'):
            raise ValueError('Development selection is inconsistent')
    except (KeyError, TypeError) as exc:
        raise ValueError('Invalid development report') from exc


def heldout_from_observations(records, mixed, raw, candidate, *, selected, manifest_hash, development_hash, seed):
    baseline, base_counts = audit_observations(records, raw['test', 'always_run'], 'always_run', 'test', manifest_hash)
    result, counts = audit_observations(records, raw['test', candidate['id']], candidate['id'], 'test', manifest_hash)
    mixed_baseline, mixed_base = audit_observations(mixed, raw['mixed', 'always_run'], 'always_run', 'mixed', manifest_hash)
    mixed_result, mixed_counts = audit_observations(
        mixed, raw['mixed', candidate['id']], candidate['id'], 'mixed', manifest_hash)
    intervals = {}
    for view in ('shared', 'full'):
        _, base_view = score(records, first_predictions(raw['test', 'always_run']), count_view=view)
        _, candidate_view = score(records, first_predictions(raw['test', candidate['id']]), count_view=view)
        intervals[view] = exposure_interval(paired_rows(records, candidate_view), paired_rows(records, base_view), seed=seed)
    mixed_ok = not any(candidate_count[key] > baseline_count[key]
                       for baseline_count, candidate_count in zip(mixed_base, mixed_counts, strict=True)
                       for key in baseline_count if key.startswith('gold_exposed:'))
    slices = exposure_views(records, first_predictions(raw['test', 'always_run']),
                            first_predictions(raw['test', candidate['id']]))
    gate = promotion_gate(intervals['full'], mixed_ok,
        result['actual_model_calls']['full'], baseline['actual_model_calls']['full'],
        result['median_ms'], baseline['median_ms'],
        slice_regressions={f'{view}:{key}': value for view, values in slices.items() for key, value in values.items()})
    gate['paired_intervals_by_view'] = intervals
    gate['gates']['shared_exposure_upper_nonpositive'] = intervals['shared'][1] <= 0
    if not gate['gates']['shared_exposure_upper_nonpositive'] and gate['status'] == 'pass':
        gate['status'] = 'inconclusive' if intervals['shared'][0] <= 0 else 'fail'
    if selected is None:
        gate['status'] = 'fail'
        gate['no_eligible_development_candidate'] = True
    return dict(schema_version=2, candidate=candidate, test_evaluations=1, baseline=baseline, candidate_result=result,
        promotion=gate, mixed_baseline=mixed_baseline, mixed_candidate=mixed_result,
        added_exposure_documents_by_view=slices,
        bootstrap=dict(samples=1000, seed=seed, groups=len({row['template_id'] for row in records}),
                       method='paired template-group percentile, language macro, candidate minus baseline'),
        manifest_sha256=manifest_hash, development_sha256=development_hash)


def heldout(detectors, context=RunContext()):
    context.require_current()
    verify_frozen(context)
    development_report = audit_development(context)
    candidate = next(c for c in candidate_grid() if c['id'] == development_report['comparison'])
    if context.path('heldout').exists():
        raise ValueError('Held-out results already exist')
    consume(context.path('test-consumed'), dict(candidate=candidate['id'],
        development_sha256=file_hash(context.path('development')), manifest_sha256=file_hash(context.path('manifest')),
        timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
    for partition in ('test', 'mixed'):
        records = load_records(context.inputs[partition])
        for policy in (None, candidate):
            measure(records, policy, detectors, context=context, partition=partition)
    complete_phase(context, 'test')
    return audit_heldout(context)


def audit_heldout(context):
    verify_completion(context, 'test')
    development_report = audit_development(context)
    candidate = next(c for c in candidate_grid() if c['id'] == development_report['comparison'])
    consumed = json.loads(context.path('test-consumed').read_text())
    manifest_hash, development_hash = file_hash(context.path('manifest')), file_hash(context.path('development'))
    if (consumed['candidate'] != candidate['id'] or consumed['manifest_sha256'] != manifest_hash
            or consumed['development_sha256'] != development_hash):
        raise ValueError('Consumption identity mismatch')
    raw = {(partition, policy): batches for partition in ('test', 'mixed')
           for policy, batches in read_observations(context, partition, ['always_run', candidate['id']]).items()}
    report = heldout_from_observations(load_records(context.inputs['test']), load_records(context.inputs['mixed']), raw,
        candidate, selected=development_report['selected'], manifest_hash=manifest_hash,
        development_hash=development_hash, seed=context.seed)
    target = context.path('heldout')
    if target.exists():
        if json.loads(target.read_text()) != report:
            raise ValueError('Held-out report disagrees with saved observations')
    else:
        save(target, report)
    print(json.dumps({'audit': 'heldout', 'inference_replayed': False, 'promotion': report['promotion']}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('freeze', 'dev', 'test', 'audit-dev', 'audit-test', 'validate-inputs'))
    parser.add_argument('--experiment', choices=('v1', 'v090-v2'), default='v1')
    args = parser.parse_args()
    context = RunContext(args.experiment)
    context.require_current()
    if args.phase == 'validate-inputs':
        partitions = validate_current_inputs(context)
        print(json.dumps({'input_counts': {key: len(rows) for key, rows in partitions.items()}, 'inference': False}))
        return
    if args.phase == 'freeze':
        freeze(context)
        return
    inference = args.phase in ('dev', 'test')
    verify_frozen(context, inference=inference)
    if not inference:
        if args.phase == 'audit-dev':
            audit_development(context, reconstruct=True)
        else:
            audit_heldout(context)
        return
    detectors = {stage: ModelDetector(stage) for stage in MODELS}
    if args.phase == 'dev':
        development(detectors, context)
    else:
        heldout(detectors, context)


if __name__ == '__main__':
    main()
