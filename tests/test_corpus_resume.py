"""Private resume guards and native replay acceptance."""
import importlib.util

import json
from dataclasses import replace

import pytest

from sct.corpus.manifest import RunIdentity, RunManifest, config_digest, file_digest


def identity():
    return RunIdentity('run', 'base', 'package', (('input', 'a' * 64, ('private-id',)),),
                       config_digest({}), (('detector', 'v1'), ('native_format', 'v1')), 1)


def test_completed_checksums_and_identity_guard(tmp_path):
    from sct.corpus.runner import resume_run, RunIdentityMismatch, InvalidCheckpoint
    shard = tmp_path / 'kept.jsonl'
    shard.write_text('{}\n')
    expected = identity()
    manifest = RunManifest(expected, (('kept.jsonl', file_digest(shard)),), {'input': 1, 'kept': 1, 'rejected': 0})
    path = tmp_path / 'manifest.json'
    manifest.publish(path)
    assert resume_run(path, expected=expected).complete
    for changed in (replace(expected, random_seed=2), replace(expected, config_digest='b' * 64),
                    replace(expected, input_partitions=(('changed', 'a' * 64, ('private-id',)),)),
                    replace(expected, component_identities=(('detector', 'v2'),))):
        with pytest.raises(RunIdentityMismatch):
            resume_run(path, expected=changed)
    shard.write_text('corrupt')
    with pytest.raises(InvalidCheckpoint):
        resume_run(path, expected=expected)
    shard.unlink()
    with pytest.raises(InvalidCheckpoint):
        resume_run(path, expected=expected)


requires_native = pytest.mark.skipif(
    any(importlib.util.find_spec(name) is None for name in ('datatrove', 'spacy', 'orjson')),
    reason='install corpus extra for native replay',
)


@requires_native
@pytest.mark.parametrize('crash_stage', ['signatures', 'publish'])
def test_d3_acceptance(tmp_path, monkeypatch, crash_stage):
    from sct.config import TextCleanerConfig
    from sct.corpus.runner import curate_resumable, _Lifecycle
    source = tmp_path / 'source.jsonl'
    text = 'Careful scientific research uses measured results and records all observations across repeated experiments.'
    source.write_text('\n'.join(json.dumps({'id': record_id, 'text': text}) for record_id in ('b', 'a')))
    kwargs = {'config': TextCleanerConfig(check_ner_process=False), 'retention': 'caller_managed'}
    baseline = curate_resumable(source, tmp_path / 'baseline', **kwargs)
    original = _Lifecycle.checkpoint
    def crash(self, name, paths):
        if name == crash_stage:
            raise KeyboardInterrupt('private interruption')
        return original(self, name, paths)
    monkeypatch.setattr(_Lifecycle, 'checkpoint', crash)
    with pytest.raises(KeyboardInterrupt):
        curate_resumable(source, tmp_path / 'run', **kwargs)
    assert not (tmp_path / 'run' / 'manifest.json').exists()
    monkeypatch.setattr(_Lifecycle, 'checkpoint', original)
    resumed = curate_resumable(source, tmp_path / 'run', **kwargs)
    assert resumed.counts == baseline.counts
    assert (tmp_path / 'run' / 'kept.jsonl').read_bytes() == (tmp_path / 'baseline' / 'kept.jsonl').read_bytes()
    rows = [json.loads(line)['id'] for line in (tmp_path / 'run' / 'kept.jsonl').read_text().splitlines()]
    assert sorted(rows) == ['a', 'b'] and len(rows) == len(set(rows))
    assert curate_resumable(source, tmp_path / 'run', **kwargs).counts == resumed.counts
    assert source.exists()


@requires_native
def test_bounded_failure_quarantine_contains_only_codes(tmp_path):
    from sct.config import TextCleanerConfig
    from sct.corpus.runner import curate_resumable, CorpusRunFailed
    source = tmp_path / 'source'
    sensitive = 'private raw document content and a secret callback message'
    source.write_text(json.dumps({'id': 'stable-record', 'text': sensitive}))
    def fail(texts):
        raise RuntimeError(sensitive)
    with pytest.raises(CorpusRunFailed) as error:
        curate_resumable(source, tmp_path / 'run', config=TextCleanerConfig(check_ner_process=False),
                         scorer=fail, component_identities=(('quality', 'fixed-v1'),),
                         retention='caller_managed', max_attempts=2)
    state = json.loads((tmp_path / 'run' / 'manifest.state.json').read_text())
    assert [(row['stable_id'], row['reason'], row['attempt'], row['scope']) for row in state['failures']] == [
        ('stable-record', 'native_stage_failed', 1, 'run'), ('stable-record', 'native_stage_failed', 2, 'run')]
    assert sensitive not in str(error.value) + json.dumps(state)
    assert not (tmp_path / 'run' / 'manifest.json').exists()
    assert all(sensitive not in path.read_text() for path in (tmp_path / 'run' / 'logs').rglob('*') if path.is_file())


@requires_native
@pytest.mark.parametrize('change', ['text', 'ids', 'config', 'seed', 'signature', 'component'])
def test_incomplete_run_refuses_changed_computation(tmp_path, monkeypatch, change):
    from sct.config import TextCleanerConfig
    from sct.corpus.runner import curate_resumable, _Lifecycle, RunIdentityMismatch
    from datatrove.pipeline.dedup import MinhashConfig
    from datatrove.utils.hashing import HashConfig
    source = tmp_path / 'source'
    source.write_text(json.dumps({'id': 'stable', 'text': 'one two three four five six seven eight nine ten'}))
    config = TextCleanerConfig(check_ner_process=False)
    original = _Lifecycle.checkpoint
    def crash(self, name, paths):
        if name == 'signatures':
            raise KeyboardInterrupt
        return original(self, name, paths)
    monkeypatch.setattr(_Lifecycle, 'checkpoint', crash)
    with pytest.raises(KeyboardInterrupt):
        curate_resumable(source, tmp_path / 'run', config=config, retention='caller_managed')
    before = (tmp_path / 'run' / 'manifest.state.json').read_bytes()
    kwargs = {'config': config}
    if change == 'text':
        source.write_text(json.dumps({'id': 'stable', 'text': 'changed document'}))
    elif change == 'ids':
        source.write_text(json.dumps({'id': 'other', 'text': 'one two three four five six seven eight nine ten'}))
    elif change == 'config':
        kwargs['config'] = TextCleanerConfig(check_ner_process=False, check_replace_emails=False)
    elif change in ('seed', 'signature'):
        kwargs['native_config'] = MinhashConfig(hash_config=HashConfig(hash_fc='sha1'),
                                                **({'seed': 9} if change == 'seed' else {'n_grams': 3}))
    else:
        kwargs['component_identities'] = (('detector', 'immutable-v2'),)
    with pytest.raises(RunIdentityMismatch):
        curate_resumable(source, tmp_path / 'run', retention='caller_managed', **kwargs)
    assert (tmp_path / 'run' / 'manifest.state.json').read_bytes() == before


@requires_native
@pytest.mark.parametrize('damage', ['missing', 'corrupt', 'extra'])
def test_completed_native_inventory_refuses_damage(tmp_path, monkeypatch, damage):
    from sct.config import TextCleanerConfig
    from sct.corpus.runner import curate_resumable, _Lifecycle, InvalidCheckpoint
    source = tmp_path / 'source'
    source.write_text(json.dumps({'id': 'id', 'text': 'one two three four five six seven eight nine ten'}))
    original = _Lifecycle.checkpoint
    def crash(self, name, paths):
        original(self, name, paths)
        if name == 'signatures':
            raise KeyboardInterrupt
    monkeypatch.setattr(_Lifecycle, 'checkpoint', crash)
    with pytest.raises(KeyboardInterrupt):
        curate_resumable(source, tmp_path / 'run', config=TextCleanerConfig(check_ner_process=False),
                         retention='caller_managed')
    checkpoint = tmp_path / 'run' / 'private.sqlite'
    if damage == 'missing':
        checkpoint.unlink()
    elif damage == 'corrupt':
        checkpoint.write_bytes(b'bad bytes')
    else:
        (tmp_path / 'run' / 'signatures' / 'extra').write_text('unexpected')
    with pytest.raises(InvalidCheckpoint):
        curate_resumable(source, tmp_path / 'run', config=TextCleanerConfig(check_ner_process=False),
                         retention='caller_managed')


def test_private_lock_and_incomplete_stage_state(tmp_path):
    from sct.corpus.runner import _Lifecycle, resume_run, InvalidCheckpoint
    expected = identity()
    output = tmp_path / 'run'
    with _Lifecycle(output, expected) as lifecycle:
        with pytest.raises(InvalidCheckpoint, match='already active'):
            with _Lifecycle(output, expected):
                pass
        artifact = output / 'private.sqlite'
        artifact.write_bytes(b'private-checkpoint')
        lifecycle.checkpoint('prepare', (artifact,))
        assert resume_run(output / 'manifest.json', expected=expected).completed_stages == ('prepare',)
    artifact.write_bytes(b'changed')
    with pytest.raises(InvalidCheckpoint):
        resume_run(output / 'manifest.json', expected=expected)


@requires_native
def test_crash_before_final_manifest_rename_replays_publication(tmp_path, monkeypatch):
    from sct.config import TextCleanerConfig
    from sct.corpus.runner import curate_resumable
    from pathlib import Path
    source = tmp_path / 'source'
    source.write_text(json.dumps({'id': 'id', 'text': 'one two three four five six seven eight nine ten'}))
    original = Path.rename
    def crash(self, target):
        if self.name == 'manifest.pending':
            raise KeyboardInterrupt
        return original(self, target)
    monkeypatch.setattr(Path, 'rename', crash)
    with pytest.raises(KeyboardInterrupt):
        curate_resumable(source, tmp_path / 'run', config=TextCleanerConfig(check_ner_process=False),
                         retention='caller_managed')
    assert (tmp_path / 'run' / 'manifest.pending').exists()
    monkeypatch.setattr(Path, 'rename', original)
    result = curate_resumable(source, tmp_path / 'run', config=TextCleanerConfig(check_ner_process=False),
                             retention='caller_managed')
    assert result.counts == {'input': 1, 'kept': 1, 'rejected': 0}


@pytest.mark.skipif(not hasattr(__import__('os'), 'mkfifo'), reason='requires POSIX FIFO support')
@pytest.mark.parametrize('artifact', ['manifest', 'state', 'shard'])
def test_fifo_artifacts_fail_before_opening(tmp_path, monkeypatch, artifact):
    import os
    from pathlib import Path
    from sct.corpus.runner import resume_run, InvalidCheckpoint
    expected = identity()
    path = tmp_path / 'manifest.json'
    shard = tmp_path / 'kept.jsonl'
    shard.write_text('{}')
    manifest = RunManifest(expected, (('kept.jsonl', file_digest(shard)),), {'input': 1, 'kept': 1, 'rejected': 0})
    if artifact == 'shard':
        manifest.publish(path)
        shard.unlink()
    fifo = {'manifest': path, 'state': tmp_path / 'manifest.state.json', 'shard': shard}[artifact]
    os.mkfifo(fifo, 0o600)
    path_open, os_open = Path.open, os.open
    def guarded_path_open(self, *args, **kwargs):
        assert self != fifo, 'FIFO was opened instead of rejected'
        return path_open(self, *args, **kwargs)
    def guarded_os_open(target, *args, **kwargs):
        assert Path(target) != fifo, 'FIFO was opened instead of rejected'
        return os_open(target, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', guarded_path_open)
    monkeypatch.setattr(os, 'open', guarded_os_open)
    with pytest.raises(InvalidCheckpoint):
        resume_run(path, expected=expected)
