"""Native DataTrove stages with private staging and explicit retention."""
from contextlib import closing
from dataclasses import asdict
import json
import os
import math
import re
from pathlib import Path
import sqlite3
import struct
from importlib.metadata import version

from datatrove.utils.hashing import HashConfig, create_hash_func
from datatrove.data import Document
from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.dedup import (
    MinhashConfig, MinhashDedupSignature, MinhashDedupBuckets, MinhashDedupCluster, MinhashDedupFilter,
)

from sct.config import TextCleanerConfig, TOKEN_MODEL_BACKENDS, TORCH_BACKENDS, GLINER_BACKENDS, GLICLASS_DEFAULT_MODEL
from sct.corpus.manifest import RunIdentity, RunManifest, config_digest, file_digest, _require_bundled_gliner
from sct.corpus.types import CorpusRecord, SelectionConfig
from sct.integrations._runtime import CleanerHandle
from sct.schema import process_value
from sct.score import score_record


class _StageRecords(PipelineStep):
    name = 'Private corpus staging'
    type = 'SCT'

    def __init__(self, source, database, handle, selection, schema, context, scorer):
        super().__init__()
        self.source, self.database, self.handle = source, database, handle
        self.selection, self.schema, self.context, self.scorer = selection, schema, context, scorer

    def run(self, data=None, rank=0, world_size=1):
        cleaner = self.handle.get()
        with closing(sqlite3.connect(self.database)) as db, self.source.open(encoding='utf-8') as stream:
            db.execute('CREATE TABLE records (id TEXT PRIMARY KEY, cleaned TEXT, basis TEXT, scores TEXT, quality INTEGER)')
            for line in stream:
                try:
                    row = json.loads(line)
                    record = CorpusRecord(row['id'], row['text'])
                except (ValueError, TypeError, KeyError):
                    raise ValueError('Invalid corpus record') from None
                result = process_value(cleaner, record.text, ('text',), schema=self.schema, context=self.context)
                quality = None
                if self.scorer is not None:
                    values = self.scorer([record.text])
                    if (len(values) != 1 or not isinstance(values[0], (int, float))
                            or not math.isfinite(values[0]) or not 0 <= values[0] <= 1):
                        raise ValueError('Quality scorer must return one finite score in [0,1]')
                    quality = float(values[0])
                scores = score_record(result, original=record.text, quality=quality)
                passed = self.selection.quality_min is None or quality >= self.selection.quality_min
                basis = record.text if self.selection.basis == 'pre_pii' else result.lm_text
                try:
                    db.execute('INSERT INTO records VALUES (?,?,?,?,?)',
                               (record.id, result.lm_text, basis, json.dumps(scores), int(passed)))
                except sqlite3.IntegrityError:
                    raise ValueError('Duplicate stable record identity') from None
            db.commit()
        return None


class _ReadBasis(PipelineStep):
    name = 'Private basis reader'
    type = 'SCT'

    def __init__(self, database):
        super().__init__()
        self.database = database

    def run(self, data=None, rank=0, world_size=1):
        with closing(sqlite3.connect(self.database)) as db:
            for record_id, basis in db.execute('SELECT id,basis FROM records WHERE quality=1 ORDER BY id'):
                yield Document(text=basis, id=record_id)


class _KeepIDs(PipelineStep):
    name = 'Native kept identities'
    type = 'SCT'

    def __init__(self, database):
        super().__init__()
        self.database = database

    def run(self, data, rank=0, world_size=1):
        with closing(sqlite3.connect(self.database)) as db:
            db.execute('CREATE TABLE native_kept (id TEXT PRIMARY KEY)')
            db.executemany('INSERT INTO native_kept VALUES (?)', ((doc.id,) for doc in data))
            db.commit()
        return None


class _PrivateStage:
    def __init__(self, stage):
        self.stage = stage

    def __repr__(self):
        return '<private corpus stage>'


class _SafeStage(PipelineStep):
    name = 'Private corpus operation'
    type = 'SCT'

    def __init__(self, stage):
        super().__init__()
        self.private = _PrivateStage(stage)
        self.operation = type(stage).__name__

    def run(self, data=None, rank=0, world_size=1):
        failed = False
        try:
            result = self.private.stage(data, rank, world_size)
            if result is not None:
                iterator = iter(result)
                while True:
                    try:
                        yield next(iterator)
                    except StopIteration:
                        break
        except Exception:
            failed = True
        if failed:
            raise ValueError(f'Corpus stage failed ({self.operation}); private state remains incomplete') from None


def _execute(pipeline, output, name, *, tasks=1, workers=1, lifecycle=None, paths=()):
    if lifecycle is not None and lifecycle.reuse(name, paths):
        return
    LocalPipelineExecutor([_SafeStage(stage) for stage in pipeline], tasks=tasks, workers=workers,
                          logging_dir=str(output / 'logs' / name),
                          skip_completed=False, randomize_start_duration=0).run()
    if lifecycle is not None:
        lifecycle.checkpoint(name, paths)


def _cluster_keepers(path: Path, eligible_count: int) -> dict[int, int]:
    members: dict[int, list[int]] = {}
    seen = set()
    if path.exists():
        with path.open('rb') as stream:
            while chunk := stream.read(8):
                if len(chunk) != 8:
                    raise ValueError('Invalid native cluster file length')
                index, cluster_id = struct.unpack('<II', chunk)
                if index >= eligible_count or index in seen:
                    raise ValueError('Invalid native cluster ordinal')
                seen.add(index)
                members.setdefault(cluster_id, []).append(index)
    return {index: min(group) for group in members.values() for index in group}


def _model_identities(config: TextCleanerConfig) -> tuple[tuple[str, str], ...]:
    models: list[tuple[str, str | None, str | None]] = []
    ner_active = config.check_ner_process
    if config.pii_processing == 'unified' and config.detector_specs:
        ner_active = any(spec.name == 'builtin_ner' for spec in config.detector_specs)
    if ner_active:
        if config.ner_backend == 'presidio_gliner':
            raise ValueError('presidio_gliner does not enforce immutable full NLP provenance in this recipe')
        if config.ner_backend in TOKEN_MODEL_BACKENDS:
            model_ids = config.torch_ner_models if config.ner_backend in TORCH_BACKENDS else config.ner_models
            revisions = (config.torch_ner_model_revisions if config.ner_backend in TORCH_BACKENDS
                         else config.ner_model_revisions)
            models.extend((f'ner:{key}', model_id, (revisions or {}).get(key))
                          for key, model_id in (model_ids or {}).items())
        if config.ner_backend in GLINER_BACKENDS:
            models.append(('gliner', config.gliner_model, config.gliner_revision))
    if config.check_classify_document:
        models.append(('classification', config.gliclass_model or GLICLASS_DEFAULT_MODEL, config.gliclass_revision))
    identities = []
    for kind, model_id, revision in models:
        if (not isinstance(revision, str) or not re.fullmatch('[a-f0-9]{40}', revision)
                or not isinstance(model_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+', model_id)
                or Path(model_id).exists()):
            raise ValueError('Active models require an immutable model revision and a remote repository ID')
        if kind == 'gliner' and config.gliner_variant == 'gliner':
            _require_bundled_gliner(model_id, revision)
        identities.append((kind, model_id + '@' + revision))
    return tuple(sorted(identities))


def _identity_for_run(source, output, *, retention, config=None, selection=SelectionConfig(),
                      native_config=None, language='eng', workers=1, registry_factory=None,
                      schema=None, context=None, scorer=None, component_identities=()):
    if not isinstance(source, Path) or not isinstance(output, Path):
        raise TypeError('Corpus paths must be pathlib.Path objects')
    if (retention != 'caller_managed' or output.is_symlink()
            or not source.is_file() or source.is_symlink()
            or source.resolve().is_relative_to(output.resolve())):
        raise ValueError('Require a new separate output directory and explicit caller-managed retention')
    if type(workers) is not int or workers < 1:
        raise ValueError('workers must be a positive integer')
    if not isinstance(selection, SelectionConfig) or selection.similarity_threshold != .9:
        raise ValueError('Native matching uses native_config, not similarity_threshold')
    if selection.quality_min is not None and scorer is None:
        raise ValueError('Quality selection requires an explicitly supplied scorer')
    native = native_config or MinhashConfig(hash_config=HashConfig(hash_fc='sha1'))
    if not isinstance(native, MinhashConfig):
        raise TypeError('native_config must be a DataTrove MinhashConfig')
    if any(type(value) is not int or value < 1 for value in (native.n_grams, native.num_buckets, native.hashes_per_bucket)):
        raise ValueError('Native signature dimensions must be positive integers')
    try:
        create_hash_func(native.hash_config)('SCT native hash preflight')
    except (TypeError, ValueError):
        raise ValueError('Native hash algorithm is unavailable for text; choose a supported hash algorithm') from None
    if config is None:
        raise ValueError('Corpus recipe requires an explicit cleaner configuration or preset')
    if version('datatrove') != '0.10.1':
        raise ValueError('Native corpus format requires DataTrove 0.10.1')
    handle = CleanerHandle(config, registry_factory=registry_factory)
    model_identities = _model_identities(handle.config)
    if model_identities:
        from huggingface_hub import constants as hub_constants
        if not hub_constants.HF_HUB_OFFLINE:
            raise ValueError('Pre-cache pinned model artifacts, then start the recipe with HF_HUB_OFFLINE=1')
    identities = dict(component_identities)
    if ((registry_factory is not None and 'registry' not in identities)
            or (scorer is not None and 'quality' not in identities)):
        raise ValueError('Runtime components require explicit registry and quality version identities')
    cleaner_config = {**handle.config._plain_state(), 'supported_languages': sorted(handle.config.supported_languages)}
    fingerprint = config_digest({'cleaner': cleaner_config, 'selection': asdict(selection),
                                 'native': asdict(native), 'language': language,
                                 'schema': None if schema is None else asdict(schema),
                                 'context': None if context is None else asdict(context),
                                 'components': component_identities})
    RunIdentity(output.name, 'baseline', 'package', (), fingerprint, component_identities, native.seed)
    input_digest = file_digest(source)
    record_ids = []
    try:
        with source.open(encoding='utf-8') as stream:
            for line in stream:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError('Invalid corpus record')
                record_ids.append(CorpusRecord(row['id'], row['text']).id)
    except (ValueError, TypeError, KeyError):
        raise ValueError('Corpus stage failed (_StageRecords); invalid input') from None
    stable_ids = tuple(sorted(record_ids))
    identity = RunIdentity(output.name, '0.8.3/8afef61c74ea738a2c904374776dea34d9c1c39e', version('SqueakyCleanText'),
                           (('input', input_digest, stable_ids),), fingerprint,
                           (('datatrove', version('datatrove')), *model_identities,
                            ('model_configuration', config_digest({key: value for key, value in cleaner_config.items()
                                                                 if 'model' in key or 'revision' in key})),
                            ('detector_configuration', config_digest(cleaner_config['detector_specs'])),
                            ('policy_configuration', config_digest({'pii_processing': handle.config.pii_processing})),
                            ('native_minhash', json.dumps(asdict(native), sort_keys=True)),
                            ('native_format', 'datatrove-0.10.1/rank0/clusters-le-u32-u32/remove-le-u32'),
                            *component_identities), native.seed)
    return identity, handle, native


def curate_datatrove(
    source: Path, output: Path, *, retention: str,
    config: TextCleanerConfig | str | None = None, selection: SelectionConfig = SelectionConfig(),
    native_config: MinhashConfig | None = None, language: str = 'eng', workers: int = 1,
    registry_factory=None, schema=None, context=None, scorer=None,
    component_identities: tuple[tuple[str, str], ...] = (),
    _lifecycle=None,
) -> RunManifest:
    """Write a separate cleaned corpus using native approximate bucket matching.

    Private state contains source text and identifying hashes. Caller-managed
    retention requires the caller to protect and eventually remove the run directory.
    D1's exact similarity threshold is unsupported here; native configuration controls matching.
    """
    if not isinstance(source, Path) or not isinstance(output, Path):
        raise TypeError('Corpus paths must be pathlib.Path objects')
    if _lifecycle is None and (output.exists() or output.is_symlink()):
        raise ValueError('Require a new separate output directory and explicit caller-managed retention')
    identity, handle, native = _identity_for_run(
        source, output, retention=retention, config=config, selection=selection,
        native_config=native_config, language=language, workers=workers, registry_factory=registry_factory,
        schema=schema, context=context, scorer=scorer, component_identities=component_identities,
    )
    input_digest = identity.input_partitions[0][1]
    if _lifecycle is None:
        output.mkdir(mode=0o700, parents=False)
    else:
        identity = _lifecycle.prepare(identity)
        if _lifecycle.completed is not None:
            return _lifecycle.completed
    database = output / 'private.sqlite'
    _execute([_StageRecords(source, database, handle, selection, schema, context, scorer)],
             output, 'prepare', workers=workers, lifecycle=_lifecycle, paths=(database,))
    signatures, buckets, clusters = (output / name for name in ('signatures', 'buckets', 'clusters'))
    _execute([_ReadBasis(database), MinhashDedupSignature(str(signatures), config=native, language=language)],
             output, 'signatures', workers=workers, lifecycle=_lifecycle, paths=(signatures,))
    _execute([MinhashDedupBuckets(str(signatures), str(buckets), config=native)],
             output, 'buckets', tasks=native.num_buckets, workers=workers, lifecycle=_lifecycle, paths=(buckets,))
    reuse_clusters = _lifecycle is not None and _lifecycle.reuse('clusters', (clusters,))
    if not reuse_clusters:
        _execute([MinhashDedupCluster(str(buckets), str(clusters), config=native, save_cluster_id=True)],
                 output, 'clusters', workers=workers)
        # Native cluster roots depend on union order. Sorted input plus minimum index fixes the keeper.
        with closing(sqlite3.connect(database)) as db:
            eligible_count = db.execute('SELECT COUNT(*) FROM records WHERE quality=1').fetchone()[0]
        keeper_indexes = _cluster_keepers(clusters / '000000.clusters', eligible_count)
        clusters.mkdir(exist_ok=True, mode=0o700)
        with (clusters / '000000.remove').open('wb') as removal_stream:
            for index in sorted(i for i, keeper in keeper_indexes.items() if i != keeper):
                removal_stream.write(struct.pack('<I', index))
        if _lifecycle is not None:
            _lifecycle.checkpoint('clusters', (clusters,))
    with closing(sqlite3.connect(database)) as db:
        eligible_count = db.execute('SELECT COUNT(*) FROM records WHERE quality=1').fetchone()[0]
    keeper_indexes = _cluster_keepers(clusters / '000000.clusters', eligible_count)
    kept_database = output / 'kept.sqlite'
    _execute([_ReadBasis(database), MinhashDedupFilter(str(clusters)), _KeepIDs(kept_database)],
             output, 'filter', workers=workers,
             lifecycle=_lifecycle, paths=(kept_database,))
    if _lifecycle is not None:
        _lifecycle.reuse('publish', tuple(output / name for name in
                                       ('kept.pending', 'decisions.pending', 'kept.jsonl', 'decisions.jsonl',
                                        'manifest.pending', 'native-manifest.pending', 'native-manifest.json')))
    with closing(sqlite3.connect(database)) as db:
        db.execute('ATTACH DATABASE ? AS selected', (str(kept_database),))
        stable_ids = tuple(row[0] for row in db.execute('SELECT id FROM records ORDER BY id'))
        eligible = tuple(row[0] for row in db.execute('SELECT id FROM records WHERE quality=1 ORDER BY id'))
        counts = {'input': len(stable_ids), 'kept': 0, 'rejected': 0}
        duplicate_keepers = {eligible[index]: eligible[keeper] for index, keeper in keeper_indexes.items()}
        with (
            (output / 'kept.pending').open('x', encoding='utf-8') as kept,
            (output / 'decisions.pending').open('x', encoding='utf-8') as audit,
        ):
            for record_id, cleaned, scores, quality, native_kept in db.execute(
                'SELECT r.id,r.cleaned,r.scores,r.quality,k.id IS NOT NULL FROM records r '
                'LEFT JOIN selected.native_kept k ON k.id=r.id ORDER BY r.id'
            ):
                retain = selection.dedup_mode == 'report' or bool(quality and native_kept)
                counts['kept' if retain else 'rejected'] += 1
                decision = {'id': record_id, 'keeper_id': duplicate_keepers.get(record_id, record_id),
                            'quality_pass': bool(quality), 'duplicate': bool(quality and not native_kept),
                            'kept': retain, 'basis': selection.basis, 'scores': json.loads(scores),
                            'reasons': (['quality_below_threshold'] if not quality else
                                        ['native_duplicate'] if not native_kept else [])}
                audit.write(json.dumps(decision, sort_keys=True) + '\n')
                if retain:
                    kept.write(json.dumps({'id': record_id, 'text': cleaned}, sort_keys=True) + '\n')
            for output_stream in (kept, audit):
                output_stream.flush()
                os.fsync(output_stream.fileno())
    for stem in ('kept', 'decisions'):
        (output / f'{stem}.pending').rename(output / f'{stem}.jsonl')
    if file_digest(source) != input_digest:
        raise ValueError('Input changed during corpus processing')
    shards = tuple((name, file_digest(output / name)) for name in ('kept.jsonl', 'decisions.jsonl'))
    manifest = RunManifest(identity, shards, counts)
    for path in output.rglob('*'):
        path.chmod(0o700 if path.is_dir() else 0o600)
    manifest.publish(output / ('manifest.json' if _lifecycle is None else 'native-manifest.json'))
    if _lifecycle is not None:
        _lifecycle.checkpoint('publish', tuple(output / name for name in
                                             ('kept.jsonl', 'decisions.jsonl', 'native-manifest.json')))
    return manifest
