"""Resume private native checkpoints and report caller-reference matches.

Run directories, hashes and stable IDs are private identifying state. Execution
can repeat after interruption. Caller-managed retention includes incomplete state.
"""
from collections.abc import Iterable
from contextlib import closing
from dataclasses import asdict, dataclass
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import sqlite3
import unicodedata

from sct.corpus.manifest import RunIdentity, RunManifest, config_digest, file_digest, _open_regular
from sct.corpus.types import CorpusRecord
from sct.score import minhash_tokens
import sct.score as score_module


class RunIdentityMismatch(ValueError):
    """The requested computation differs from the stored private run."""


class InvalidCheckpoint(ValueError):
    """A completed checkpoint cannot be verified."""


class CorpusRunFailed(ValueError):
    """Bounded execution failed; private reason codes remain available."""


@dataclass(frozen=True, repr=False)
class FailureRecord:
    stable_id: str
    reason: str
    attempt: int
    scope: str = 'run'


@dataclass(frozen=True, repr=False)
class ResumeState:
    identity: RunIdentity
    complete: bool
    completed_stages: tuple[str, ...] = ()
    manifest: RunManifest | None = None
    failures: tuple[FailureRecord, ...] = ()


_STAGES = {
    'prepare': ('private.sqlite',), 'signatures': ('signatures',), 'buckets': ('buckets',),
    'clusters': ('clusters',), 'filter': ('kept.sqlite',),
    'publish': ('kept.jsonl', 'decisions.jsonl', 'native-manifest.json'),
}
_REPLAY_PATHS = {**_STAGES, 'publish': ('kept.pending', 'decisions.pending', 'kept.jsonl', 'decisions.jsonl',
                                      'manifest.pending', 'native-manifest.pending', 'native-manifest.json'),
                 'finalize': ('manifest.pending',)}


def _identity(row):
    return RunIdentity(row['run_id'], row['baseline_version'], row['package_version'],
                       tuple((name, digest, tuple(ids)) for name, digest, ids in row['input_partitions']),
                       row['config_digest'], tuple(tuple(item) for item in row['component_identities']), row['random_seed'])


def _read(path):
    if path.is_symlink():
        raise InvalidCheckpoint('Symlink checkpoint refused')
    try:
        with _open_regular(path) as stream:
            return json.load(stream)
    except (OSError, ValueError):
        raise InvalidCheckpoint('Checkpoint is missing or invalid') from None


def _manifest(path, expected):
    try:
        row = _read(path)
        identity = _identity(row)
        if identity != expected:
            raise RunIdentityMismatch('Run identity differs; no state was reused')
        if not isinstance(row['output_shards'], dict):
            raise ValueError('Invalid output shard inventory')
        manifest = RunManifest(identity, tuple(row['output_shards'].items()), row['counts'],
                               row['retention'], row['completion_status'], row['schema_version'])
        if not manifest.output_shards:
            raise ValueError('Missing output checksums')
        for name, digest in manifest.output_shards:
            shard = path.parent / name
            if shard.is_symlink() or file_digest(shard) != digest:
                raise InvalidCheckpoint('Completed output checksum mismatch')
        return manifest
    except RunIdentityMismatch:
        raise
    except (OSError, ValueError, TypeError, KeyError):
        raise InvalidCheckpoint('Completed manifest or output is invalid') from None


def _inventory(root, names, *, sync=False):
    inventory = {}
    for name in names:
        path = root / name
        if path.is_symlink() or not path.exists():
            raise InvalidCheckpoint('Completed stage artifact is missing or unsafe')
        paths = (path, *sorted(path.rglob('*'))) if path.is_dir() else (path,)
        for item in paths:
            if item.is_symlink() or not (item.is_file() or item.is_dir()):
                raise InvalidCheckpoint('Unsafe checkpoint artifact')
            relative = item.relative_to(root).as_posix()
            if sync:
                item.chmod(0o700 if item.is_dir() else 0o600)
                descriptor = os.open(item, os.O_RDONLY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            inventory[relative] = 'directory' if item.is_dir() else file_digest(item)
    return inventory


def _state_path(path):
    return path.with_suffix('.state.json')


def resume_run(manifest_path: Path, *, expected: RunIdentity) -> ResumeState:
    """Check identity before accepting any completed output or native stage."""
    if not isinstance(manifest_path, Path) or not isinstance(expected, RunIdentity):
        raise TypeError('Require a Path and validated RunIdentity')
    if manifest_path.parent.is_symlink():
        raise InvalidCheckpoint('Symlink run directory refused')
    if manifest_path.exists() or manifest_path.is_symlink():
        manifest = _manifest(manifest_path, expected)
        return ResumeState(expected, True, manifest=manifest)
    try:
        row = _read(_state_path(manifest_path))
        if _identity(row['identity']) != expected:
            raise RunIdentityMismatch('Run identity differs; no state was reused')
        if row['schema_version'] != 1:
            raise ValueError('Unsupported state')
        stages = row['checkpoints']
        if not isinstance(stages, dict) or any(name not in _STAGES for name in stages):
            raise ValueError('Invalid stage')
        for name, inventory in stages.items():
            if _inventory(manifest_path.parent, _STAGES[name]) != inventory:
                raise InvalidCheckpoint('Completed stage checksum mismatch')
        failures = tuple(FailureRecord(**failure) for failure in row['failures'])
        return ResumeState(expected, False, tuple(stages), failures=failures)
    except RunIdentityMismatch:
        raise
    except (OSError, ValueError, TypeError, KeyError):
        raise InvalidCheckpoint('Private resume state is invalid') from None


def _atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.writing')
    # Only this runner's private pending file is replaced after a process crash.
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, 'O_NOFOLLOW', 0)
    descriptor = os.open(temporary, flags, 0o600)
    with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class _Lifecycle:
    def __init__(self, output, expected):
        self.output, self.expected = output, expected
        self.path = output / 'manifest.json'
        self.completed = None
        self.lock = None

    def __enter__(self):
        if self.output.is_symlink():
            raise InvalidCheckpoint('Symlink run directory refused')
        if self.output.exists():
            resume_run(self.path, expected=self.expected)
        else:
            self.output.mkdir(mode=0o700)
            _atomic_json(_state_path(self.path), {'schema_version': 1, 'identity': self.expected.to_dict(),
                                                'checkpoints': {}, 'failures': []})
        lock_path = self.output / 'runner-lock.sqlite'
        if lock_path.is_symlink():
            raise InvalidCheckpoint('Unsafe run lock')
        self.lock = sqlite3.connect(lock_path, timeout=0)
        lock_path.chmod(0o600)
        try:
            self.lock.execute('BEGIN IMMEDIATE')
        except sqlite3.OperationalError:
            self.lock.close()
            raise InvalidCheckpoint('Run is already active') from None
        self.row = _read(_state_path(self.path))
        return self

    def __exit__(self, *exc):
        if self.lock is not None:
            self.lock.close()

    def prepare(self, identity):
        if identity != self.expected:
            raise RunIdentityMismatch('Run identity differs; no state was reused')
        resume_run(self.path, expected=identity)
        native = self.output / 'native-manifest.json'
        if 'publish' in self.row['checkpoints']:
            self.completed = _manifest(native, identity)
        return identity

    def reuse(self, name, paths):
        if name not in _REPLAY_PATHS or tuple(path.name for path in paths) != _REPLAY_PATHS[name]:
            raise InvalidCheckpoint('Invalid owned stage paths')
        if (self.output / 'logs').is_symlink():
            raise InvalidCheckpoint('Symlink log directory refused')
        if name in self.row['checkpoints']:
            if _inventory(self.output, _STAGES[name]) != self.row['checkpoints'][name]:
                raise InvalidCheckpoint('Completed stage checksum mismatch')
            return True
        # Preserve partial artifacts, but replay only this runner's exact stage paths.
        for path in (*paths, self.output / 'logs' / name):
            if path.parent != self.output and path != self.output / 'logs' / name:
                raise InvalidCheckpoint('Unsafe stage path')
            if path.is_symlink():
                raise InvalidCheckpoint('Symlink stage artifact refused')
            if path.exists():
                number = 0
                while (target := self.output / f'incomplete-{name}-{path.name}-{number}').exists():
                    number += 1
                path.rename(target)
        return False

    def checkpoint(self, name, paths):
        if tuple(path.name for path in paths) != _STAGES[name]:
            raise InvalidCheckpoint('Invalid stage inventory')
        self.row['checkpoints'][name] = _inventory(self.output, _STAGES[name], sync=True)
        _atomic_json(_state_path(self.path), self.row)

    def failure(self, attempt):
        for _, _, ids in self.expected.input_partitions:
            self.row['failures'].extend(asdict(FailureRecord(record_id, 'native_stage_failed', attempt)) for record_id in ids)
        _atomic_json(_state_path(self.path), self.row)


@dataclass(frozen=True)
class ReferenceConfig:
    mode: str = 'report'
    basis: str | None = None
    threshold: float = .9
    shingle: int = 3
    min_tokens: int = 5

    def __post_init__(self):
        if self.mode not in ('report', 'apply') or self.basis not in (None, 'pre_pii', 'pii_normalized'):
            raise ValueError('Invalid reference mode or basis')
        if self.mode == 'apply' and self.basis is None:
            raise ValueError('Reference removal requires an explicit basis')
        if not math.isfinite(self.threshold) or not 0 < self.threshold <= 1:
            raise ValueError('Reference threshold must be in (0,1]')
        if any(type(value) is not int or value < 1 for value in (self.shingle, self.min_tokens)):
            raise ValueError('Reference token limits must be positive integers')


@dataclass(frozen=True, repr=False)
class ReferenceMatch:
    reference_ids: tuple[str, ...]
    reason: str


def _shingles(text, config):
    tokens = minhash_tokens(text)
    if len(tokens) < max(config.min_tokens, config.shingle):
        return set()
    return {hashlib.sha256(json.dumps(tokens[i:i + config.shingle], ensure_ascii=False).encode()).hexdigest()
            for i in range(len(tokens) - config.shingle + 1)}


def _tokenizer_identity():
    return config_digest({'source': file_digest(Path(score_module.__file__)),
                          'regex': version('regex'), 'unicode': unicodedata.unidata_version})


class ReferenceIndex:
    """Disk-backed exact shingle Jaccard, separate from native MinHash files.

    Caller references must already use the intended text basis. This index never
    guesses reference fields, normalizes PII, downloads models or stores raw text.
    """
    def __init__(self, path: Path):
        if path.is_symlink() or not path.is_file():
            raise ValueError('Require a regular reference index')
        self.path = path
        if any(Path(str(path) + suffix).exists() for suffix in ('-wal', '-journal')):
            raise ValueError('Reference indexes must be fully committed without sidecar journals')
        self.database = sqlite3.connect(path.resolve().as_uri() + '?mode=ro&immutable=1', uri=True)
        try:
            row = self.database.execute('SELECT config,format,tokenizer FROM identity').fetchone()
            if row is None or row[1] != 'sct-exact-shingles-v1' or row[2] != _tokenizer_identity():
                raise ValueError('Incompatible reference format')
            self._config = ReferenceConfig(**json.loads(row[0]))
            digest = hashlib.sha256()
            for query in ('SELECT id,size FROM refs ORDER BY id', 'SELECT hash,id FROM shingles ORDER BY hash,id'):
                for values in self.database.execute(query):
                    digest.update(json.dumps(values, ensure_ascii=False, separators=(',', ':')).encode() + b'\n')
            self._identity = config_digest({'content': digest.hexdigest(), 'config': asdict(self.config),
                                           'format': row[1], 'tokenizer': row[2]})
        except (sqlite3.Error, ValueError, TypeError):
            self.database.close()
            raise ValueError('Invalid reference index') from None
        self.digest = file_digest(path)

    @property
    def config(self) -> ReferenceConfig:
        return self._config

    @property
    def identity(self) -> str:
        return self._identity

    @classmethod
    def build(cls, path: Path, references: Iterable[CorpusRecord], *, config=ReferenceConfig()):
        if not isinstance(config, ReferenceConfig) or path.exists() or path.is_symlink():
            raise ValueError('Require a new reference index and ReferenceConfig')
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(descriptor)
        try:
            with closing(sqlite3.connect(path)) as database:
                database.executescript('CREATE TABLE identity(config TEXT,format TEXT,tokenizer TEXT); '
                                       'CREATE TABLE refs(id TEXT PRIMARY KEY,size INTEGER); '
                                       'CREATE TABLE shingles(hash TEXT,id TEXT); '
                                       'CREATE INDEX by_hash ON shingles(hash);')
                database.execute('INSERT INTO identity VALUES (?,?,?)',
                                 (json.dumps(asdict(config), sort_keys=True), 'sct-exact-shingles-v1', _tokenizer_identity()))
                for reference in references:
                    if not isinstance(reference, CorpusRecord):
                        raise ValueError('References must be CorpusRecord values')
                    shingles = _shingles(reference.text, config)
                    database.execute('INSERT INTO refs VALUES (?,?)', (reference.id, len(shingles)))
                    database.executemany('INSERT INTO shingles VALUES (?,?)',
                                         ((item, reference.id) for item in sorted(shingles)))
                database.commit()
        except (sqlite3.Error, ValueError, TypeError):
            raise ValueError('Reference index construction failed; incomplete private index retained') from None
        return cls(path)

    def match(self, text: str) -> ReferenceMatch:
        shingles = _shingles(text, self.config)
        if not shingles:
            return ReferenceMatch((), 'reference_short_input')
        # Keep only candidate identities in memory, never the reference documents.
        intersections: dict[str, int] = {}
        for shingle in shingles:
            for (reference_id,) in self.database.execute('SELECT id FROM shingles WHERE hash=?', (shingle,)):
                intersections[reference_id] = intersections.get(reference_id, 0) + 1
        matches = []
        for reference_id, intersection in intersections.items():
            size = self.database.execute('SELECT size FROM refs WHERE id=?', (reference_id,)).fetchone()[0]
            if intersection / (len(shingles) + size - intersection) >= self.config.threshold:
                matches.append(reference_id)
        return ReferenceMatch(tuple(sorted(matches)), 'reference_match' if matches else 'reference_no_match')

    def verify(self):
        if (self.path.is_symlink() or file_digest(self.path) != self.digest
                or any(Path(str(self.path) + suffix).exists() for suffix in ('-wal', '-journal'))):
            raise InvalidCheckpoint('Reference index changed during execution')

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.database.close()


def _reference_outputs(output, manifest, references):
    references.verify()
    counts = {'input': 0, 'kept': 0, 'rejected': 0}
    paths = ('selected.jsonl', 'reference-decisions.jsonl')
    for name in paths:
        for path in (output / name, output / (name + '.pending')):
            if path.is_symlink():
                raise InvalidCheckpoint('Unsafe reference output')
    with (closing(sqlite3.connect(output / 'private.sqlite')) as database,
          (output / 'decisions.jsonl').open(encoding='utf-8') as decisions,
          (output / (paths[0] + '.pending')).open('w', encoding='utf-8') as kept,
          (output / (paths[1] + '.pending')).open('w', encoding='utf-8') as audit):
        for line in decisions:
            decision = json.loads(line)
            cleaned, basis = database.execute('SELECT cleaned,basis FROM records WHERE id=?', (decision['id'],)).fetchone()
            # pre_pii requires the native recipe to have explicitly staged that same basis.
            match = references.match(basis if references.config.basis == 'pre_pii' else cleaned)
            decision['reference_ids'] = match.reference_ids
            decision['reference_reason'] = match.reason
            if match.reference_ids:
                decision['reasons'].append('reference_match')
                if references.config.mode == 'apply':
                    decision['kept'] = False
            counts['input'] += 1
            counts['kept' if decision['kept'] else 'rejected'] += 1
            audit.write(json.dumps(decision, sort_keys=True) + '\n')
            if decision['kept']:
                kept.write(json.dumps({'id': decision['id'], 'text': cleaned}, sort_keys=True) + '\n')
        for stream in (kept, audit):
            stream.flush()
            os.fsync(stream.fileno())
    references.verify()
    for name in paths:
        pending = output / (name + '.pending')
        pending.chmod(0o600)
        pending.replace(output / name)
    return RunManifest(manifest.identity, tuple((name, file_digest(output / name)) for name in paths), counts)


def curate_resumable(source: Path, output: Path, *, retention: str, references: ReferenceIndex | None = None,
                     max_attempts: int = 1, **recipe_options) -> RunManifest:
    """Resume verified native stages; replay incomplete work into owned sinks.

    max_attempts bounds retries in this invocation. Failure attribution is run
    scope, not evidence that each listed record caused the native stage failure.
    No raw rejection samples are retained by this API.
    """
    from sct.corpus.datatrove import _identity_for_run, curate_datatrove
    if type(max_attempts) is not int or not 1 <= max_attempts <= 100:
        raise ValueError('max_attempts must be an integer in [1,100]')
    if '_lifecycle' in recipe_options:
        raise ValueError('Lifecycle is runner-owned')
    if references is not None:
        references.verify()
        if references.config.basis == 'pre_pii':
            selection = recipe_options.get('selection')
            if selection is None or selection.basis != 'pre_pii':
                raise ValueError('pre_pii references require an explicit matching native selection basis')
        components = recipe_options.get('component_identities', ())
        recipe_options['component_identities'] = (*components, ('reference_index', references.identity))
    expected, _, _ = _identity_for_run(source, output, retention=retention, **recipe_options)
    with _Lifecycle(output, expected) as lifecycle:
        state = resume_run(output / 'manifest.json', expected=expected)
        if state.complete:
            if state.manifest is None:
                raise InvalidCheckpoint('Missing completed manifest')
            return state.manifest
        for attempt in range(1, max_attempts + 1):
            failed = False
            try:
                manifest = curate_datatrove(source, output, retention=retention, _lifecycle=lifecycle, **recipe_options)
                lifecycle.reuse('finalize', (output / 'manifest.pending',))
                if references is not None:
                    manifest = _reference_outputs(output, manifest, references)
                if source.is_symlink() or file_digest(source) != expected.input_partitions[0][1]:
                    raise InvalidCheckpoint('Input changed before final publication')
                manifest.publish(output / 'manifest.json')
                return manifest
            except (RunIdentityMismatch, InvalidCheckpoint):
                raise
            except Exception:
                failed = True
            if failed:
                lifecycle.failure(attempt)
        raise CorpusRunFailed('Native corpus execution exhausted bounded attempts; inspect private reason codes') from None
