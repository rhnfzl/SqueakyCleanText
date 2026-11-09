"""Persisted corpus state must fail privately before any output can be reused."""
from contextlib import closing
import json
from pathlib import Path
import sqlite3

import pytest

from sct.corpus import CorpusRecord, InvalidCheckpoint, ReferenceIndex, RunIdentity, RunManifest, resume_run
from sct.corpus.manifest import config_digest, file_digest


PRIVATE = 'synthetic-private-source-content'


def snapshot(root):
    return {
        str(path.relative_to(root)): ('symlink', str(path.readlink())) if path.is_symlink() else
        (path.stat().st_mode, path.read_bytes() if path.is_file() else None)
        for path in root.rglob('*')
    }


@pytest.fixture
def run_state(tmp_path):
    output = tmp_path / 'run'
    output.mkdir()
    identity = RunIdentity('run', 'base', 'package', (('input', 'a' * 64, ('record',)),),
                           config_digest({}), (), 1)
    shard = output / 'kept.jsonl'
    shard.write_text(json.dumps({'id': 'record', 'text': PRIVATE}))
    manifest = RunManifest(identity, ((shard.name, file_digest(shard)),), {'input': 1, 'kept': 1, 'rejected': 0})
    return output, identity, manifest


@pytest.mark.parametrize('field,value', [
    ('counts', {'input': 1, 'kept': 0, 'rejected': 0}),
    ('counts', {'input': True, 'kept': 1, 'rejected': 0}),
    ('counts', ['input', 'kept', 'rejected']),
    ('completion_status', PRIVATE),
    ('output_shards', {}),
    ('output_shards', []),
    ('output_shards', {'../' + PRIVATE: 'a' * 64}),
])
def test_completed_manifest_rejects_invalid_persisted_contract_without_changes(run_state, field, value):
    output, identity, manifest = run_state
    path = output / 'manifest.json'
    manifest.publish(path)
    assert resume_run(path, expected=identity).complete
    row = json.loads(path.read_text())
    row[field] = value
    path.write_text(json.dumps(row))
    before = snapshot(output.parent)
    with pytest.raises(InvalidCheckpoint) as error:
        resume_run(path, expected=identity)
    assert PRIVATE not in str(error.value)
    assert snapshot(output.parent) == before


@pytest.mark.parametrize('damage', ['changed_shard', 'existing_destination'])
def test_manifest_publication_cannot_publish_stale_checksums_or_replace_output(run_state, damage):
    output, _, manifest = run_state
    path = output / 'manifest.json'
    if damage == 'changed_shard':
        (output / 'kept.jsonl').write_text(PRIVATE + '-changed')
    else:
        path.write_text(PRIVATE + '-existing-manifest')
    before = snapshot(output)
    with pytest.raises(ValueError) as error:
        manifest.publish(path)
    assert PRIVATE not in str(error.value)
    assert snapshot(output) == before


@pytest.fixture
def checkpoint(run_state):
    output, identity, _ = run_state
    signatures = output / 'signatures'
    signatures.mkdir()
    artifact = signatures / 'shard.bin'
    artifact.write_bytes(PRIVATE.encode())
    path = output / 'manifest.state.json'
    row = {'schema_version': 1, 'identity': identity.to_dict(), 'failures': [],
           'checkpoints': {'signatures': {'signatures': 'directory', 'signatures/shard.bin': file_digest(artifact)}}}
    path.write_text(json.dumps(row))
    assert resume_run(output / 'manifest.json', expected=identity).completed_stages == ('signatures',)
    return output, identity, path, row


@pytest.mark.parametrize('damage', ['schema', 'stage_type', 'unknown_stage', 'inventory', 'failure', 'json'])
def test_incomplete_checkpoint_rejects_malformed_state_without_replay(checkpoint, damage):
    output, identity, path, row = checkpoint
    if damage == 'schema':
        row['schema_version'] = 2
    elif damage == 'stage_type':
        row['checkpoints'] = []
    elif damage == 'unknown_stage':
        row['checkpoints'] = {PRIVATE: {}}
    elif damage == 'inventory':
        row['checkpoints']['signatures']['signatures/shard.bin'] = 'a' * 64
    elif damage == 'failure':
        row['failures'] = [{'stable_id': PRIVATE, 'reason': PRIVATE}]
    path.write_text(PRIVATE if damage == 'json' else json.dumps(row))
    before = snapshot(output.parent)
    with pytest.raises(InvalidCheckpoint) as error:
        resume_run(output / 'manifest.json', expected=identity)
    assert PRIVATE not in str(error.value)
    assert snapshot(output.parent) == before


@pytest.mark.parametrize('location', ['run', 'manifest', 'state', 'stage', 'nested_artifact'])
def test_resume_rejects_unsafe_checkpoint_locations_without_following_or_mutating(checkpoint, location):
    output, identity, state_path, _ = checkpoint
    external = output.parent / 'external'
    external.write_text(PRIVATE)
    manifest_path = output / 'manifest.json'
    if location == 'run':
        linked_run = output.parent / 'linked-run'
        linked_run.symlink_to(output, target_is_directory=True)
        manifest_path = linked_run / 'manifest.json'
    elif location == 'manifest':
        manifest_path.symlink_to(external)
    else:
        path = {'state': state_path, 'stage': output / 'signatures',
                'nested_artifact': output / 'signatures/shard.bin'}[location]
        path.rename(path.with_name(path.name + '.retained'))
        path.symlink_to(external)
    before = snapshot(output.parent)
    with pytest.raises(InvalidCheckpoint) as error:
        resume_run(manifest_path, expected=identity)
    assert PRIVATE not in str(error.value)
    assert snapshot(output.parent) == before


@pytest.mark.parametrize('damage', ['format', 'missing_identity', 'mode', 'threshold', 'token_limits', 'journal'])
def test_reference_index_rejects_incompatible_persisted_identity_without_changes(tmp_path, damage):
    path = tmp_path / 'references.sqlite'
    with ReferenceIndex.build(path, [CorpusRecord('reference', PRIVATE)]):
        pass
    if damage == 'journal':
        Path(str(path) + '-journal').write_text(PRIVATE)
    else:
        with closing(sqlite3.connect(path)) as database, database:
            if damage == 'missing_identity':
                database.execute('DELETE FROM identity')
            elif damage == 'format':
                database.execute('UPDATE identity SET format = ?', (PRIVATE,))
            else:
                config = json.loads(database.execute('SELECT config FROM identity').fetchone()[0])
                config.update({'mode': PRIVATE} if damage == 'mode' else
                              {'threshold': 0} if damage == 'threshold' else {'shingle': True})
                database.execute('UPDATE identity SET config = ?', (json.dumps(config),))
    before = snapshot(tmp_path)
    with pytest.raises(ValueError) as error:
        ReferenceIndex(path)
    assert PRIVATE not in str(error.value)
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize('damage', ['changed_bytes', 'active_journal'])
def test_live_reference_verification_refuses_changed_or_uncommitted_inputs(tmp_path, damage):
    path = tmp_path / 'references.sqlite'
    with ReferenceIndex.build(path, [CorpusRecord('reference', PRIVATE)]) as index:
        index.verify()
        if damage == 'changed_bytes':
            with path.open('ab') as stream:
                stream.write(PRIVATE.encode())
        else:
            Path(str(path) + '-journal').write_text(PRIVATE)
        before = snapshot(tmp_path)
        with pytest.raises(InvalidCheckpoint) as error:
            index.verify()
        assert PRIVATE not in str(error.value)
        assert snapshot(tmp_path) == before
