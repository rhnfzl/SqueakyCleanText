"""Malformed run identities cannot reuse state or replace caller artifacts."""
from dataclasses import replace
import json
import sqlite3
import sys
from contextlib import closing
from types import ModuleType

import pytest

from sct.corpus import CorpusRecord, ReferenceIndex, RunIdentity, RunManifest, resume_run
from sct.corpus.manifest import config_digest, file_digest
from sct.corpus.runner import RunIdentityMismatch, InvalidCheckpoint, CorpusRunFailed, ReferenceConfig, curate_resumable

PRIVATE = 'synthetic-private-source-value'


@pytest.fixture
def retained_input(tmp_path):
    path = tmp_path / 'retained-input.jsonl'
    original = PRIVATE.encode()
    path.write_bytes(original)
    return path, original


@pytest.fixture
def identity():
    return RunIdentity('run', 'base', 'package', (), config_digest({}), (), 1)


def rejected_privately(error, retained_input):
    path, original = retained_input
    assert path.read_bytes() == original
    assert PRIVATE not in str(error.value)


@pytest.mark.parametrize('value', [{1: PRIVATE}, {'value': object()}, {'value': float('nan')}])
def test_invalid_config_identity_preserves_caller_input(retained_input, value):
    with pytest.raises(ValueError) as error:
        config_digest(value)
    rejected_privately(error, retained_input)


@pytest.mark.parametrize('changes', [
    {'baseline_version': ''},
    {'random_seed': True},
    {'config_digest': PRIVATE},
    {'input_partitions': []},
    {'input_partitions': (('input', 'a' * 64, ('same', 'same')),)},
    {'component_identities': ((PRIVATE, ''),)},
])
def test_unusable_run_identity_preserves_caller_input(identity, retained_input, changes):
    with pytest.raises(ValueError) as error:
        replace(identity, **changes)
    rejected_privately(error, retained_input)


@pytest.mark.parametrize('shards', [[], ((PRIVATE,),)])
def test_invalid_shard_inventory_cannot_publish(identity, retained_input, tmp_path, shards):
    destination = tmp_path / 'manifest.json'
    with pytest.raises(ValueError) as error:
        manifest = RunManifest(identity, shards, {'input': 0, 'kept': 0, 'rejected': 0})
        manifest.publish(destination)
    assert not destination.exists()
    rejected_privately(error, retained_input)


def test_changed_interrupted_identity_refuses_reuse_without_mutation(identity, retained_input, tmp_path):
    path = tmp_path / 'manifest.state.json'
    path.write_text(json.dumps({'schema_version': 1, 'identity': identity.to_dict(),
                                'checkpoints': {}, 'failures': []}))
    original = path.read_bytes()
    with pytest.raises(RunIdentityMismatch) as error:
        resume_run(tmp_path / 'manifest.json', expected=replace(identity, random_seed=2))
    assert path.read_bytes() == original
    assert not (tmp_path / 'manifest.json').exists()
    rejected_privately(error, retained_input)


def test_resume_requires_validated_arguments_without_touching_input(identity, retained_input):
    with pytest.raises(TypeError) as error:
        resume_run(PRIVATE, expected=identity)
    rejected_privately(error, retained_input)


def test_reference_publication_never_replaces_an_existing_destination(retained_input):
    destination, original = retained_input
    with pytest.raises(ValueError) as error:
        ReferenceIndex.build(destination, [CorpusRecord('reference', 'one two three four five')])
    assert destination.read_bytes() == original
    rejected_privately(error, retained_input)


def test_invalid_reference_keeps_source_and_private_incomplete_output(retained_input, tmp_path):
    destination = tmp_path / 'incomplete.sqlite'
    with pytest.raises(ValueError) as error:
        ReferenceIndex.build(destination, [PRIVATE])
    assert destination.is_file()
    assert PRIVATE.encode() not in destination.read_bytes()
    rejected_privately(error, retained_input)


REFERENCE = 'alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu'
OTHER = 'different unrelated words describe remote farming and rivers beneath cloudy skies'


@pytest.fixture
def native_boundary(monkeypatch):
    """Replace only the unavailable native adapter, keeping real core publication."""
    state = {'calls': 0, 'failures': 0, 'pending': False}
    adapter = ModuleType('sct.corpus.datatrove')

    def run_identity(source, output, **options):
        value = RunIdentity('run', 'base', 'package',
                            ((source.name, file_digest(source), ('a', 'b')),),
                            config_digest({}), options.get('component_identities', ()), 1)
        return value, None, None

    def native(source, output, *, _lifecycle, **options):
        state['calls'] += 1
        value, _, _ = run_identity(source, output, **options)
        _lifecycle.prepare(value)
        if state['calls'] <= state['failures']:
            raise RuntimeError(PRIVATE)
        records = [('a', REFERENCE), ('b', OTHER)]
        with closing(sqlite3.connect(output / 'private.sqlite')) as database, database:
            database.execute('CREATE TABLE IF NOT EXISTS records(id TEXT, cleaned TEXT, basis TEXT)')
            database.executemany('INSERT INTO records VALUES (?,?,?)',
                                 [(record_id, text, text) for record_id, text in records])
        (output / 'decisions.jsonl').write_text(''.join(
            json.dumps({'id': record_id, 'kept': True, 'reasons': []}) + '\n'
            for record_id, _ in records))
        kept = output / 'kept.jsonl'
        kept.write_text(''.join(json.dumps({'id': record_id, 'text': text}) + '\n'
                                for record_id, text in records))
        if state['pending']:
            (output / 'manifest.pending').write_bytes(b'retained incomplete publication')
        return RunManifest(value, (('kept.jsonl', file_digest(kept)),),
                           {'input': 2, 'kept': 2, 'rejected': 0})

    adapter._identity_for_run = run_identity
    adapter.curate_datatrove = native
    monkeypatch.setitem(sys.modules, adapter.__name__, adapter)
    return state


def test_bounded_native_failure_preserves_input_and_records_only_reason_codes(
        native_boundary, retained_input, tmp_path):
    native_boundary['failures'] = 5
    source, original = retained_input
    output = tmp_path / 'run'
    with pytest.raises(CorpusRunFailed) as error:
        curate_resumable(source, output, retention='caller_managed', max_attempts=2)
    assert native_boundary['calls'] == 2
    state = json.loads((output / 'manifest.state.json').read_text())
    assert {(row['attempt'], row['reason'], row['scope']) for row in state['failures']} == {
        (1, 'native_stage_failed', 'run'), (2, 'native_stage_failed', 'run')}
    assert PRIVATE not in json.dumps(state) + str(error.value)
    assert not (output / 'manifest.json').exists()
    assert source.read_bytes() == original


def test_successful_retry_reuses_completed_publication_without_native_execution(
        native_boundary, retained_input, tmp_path):
    native_boundary['failures'] = 1
    native_boundary['pending'] = True
    source, original = retained_input
    output = tmp_path / 'run'
    result = curate_resumable(source, output, retention='caller_managed', max_attempts=2)
    assert native_boundary['calls'] == 2
    assert result.counts == {'input': 2, 'kept': 2, 'rejected': 0}
    retained = output / 'incomplete-finalize-manifest.pending-0'
    assert retained.read_bytes() == b'retained incomplete publication'
    artifacts = {name: (output / name).read_bytes() for name in ('kept.jsonl', 'manifest.json')}
    assert curate_resumable(source, output, retention='caller_managed').counts == result.counts
    assert native_boundary['calls'] == 2
    assert all((output / name).read_bytes() == data for name, data in artifacts.items())
    assert source.read_bytes() == original


@pytest.mark.parametrize('mode,kept', [('report', 2), ('apply', 1)])
def test_reference_publication_uses_explicit_basis_and_preserves_inputs(
        native_boundary, retained_input, tmp_path, mode, kept):
    source, original = retained_input
    references_path = tmp_path / 'references.sqlite'
    with ReferenceIndex.build(references_path, [CorpusRecord('reference', REFERENCE)],
                              config=ReferenceConfig(mode=mode, basis='pii_normalized')) as references:
        reference_bytes = references_path.read_bytes()
        output = tmp_path / 'run'
        result = curate_resumable(source, output, references=references, retention='caller_managed')
        assert result.counts == {'input': 2, 'kept': kept, 'rejected': 2 - kept}
        decisions = [json.loads(row) for row in (output / 'reference-decisions.jsonl').read_text().splitlines()]
        assert decisions[0]['reference_ids'] == ['reference']
        assert decisions[1]['reference_ids'] == []
        assert len((output / 'selected.jsonl').read_text().splitlines()) == kept
        assert references_path.read_bytes() == reference_bytes
        assert source.read_bytes() == original


@pytest.mark.parametrize('options', [{'max_attempts': True}, {'max_attempts': 0}, {'_lifecycle': object()}])
def test_invalid_public_runner_controls_cannot_touch_artifacts(
        native_boundary, retained_input, tmp_path, options):
    source, original = retained_input
    output = tmp_path / 'run'
    with pytest.raises(ValueError) as error:
        curate_resumable(source, output, retention='caller_managed', **options)
    assert native_boundary['calls'] == 0
    assert not output.exists()
    assert source.read_bytes() == original
    assert PRIVATE not in str(error.value)


def test_pre_pii_reference_requires_matching_declared_selection_before_output(
        native_boundary, retained_input, tmp_path):
    source, original = retained_input
    reference_path = tmp_path / 'references.sqlite'
    with ReferenceIndex.build(reference_path, [CorpusRecord('reference', REFERENCE)],
                              config=ReferenceConfig(basis='pre_pii')) as references:
        before = reference_path.read_bytes()
        with pytest.raises(ValueError):
            curate_resumable(source, tmp_path / 'run', references=references, retention='caller_managed')
        assert native_boundary['calls'] == 0
        assert not (tmp_path / 'run').exists()
        assert reference_path.read_bytes() == before
        assert source.read_bytes() == original


def test_active_run_lock_refuses_reentry_without_replacing_completed_outputs(
        native_boundary, retained_input, tmp_path):
    source, original = retained_input
    output = tmp_path / 'run'
    curate_resumable(source, output, retention='caller_managed')
    manifest = (output / 'manifest.json').read_bytes()
    kept = (output / 'kept.jsonl').read_bytes()
    with closing(sqlite3.connect(output / 'runner-lock.sqlite')) as lock:
        lock.execute('BEGIN IMMEDIATE')
        with pytest.raises(InvalidCheckpoint):
            curate_resumable(source, output, retention='caller_managed')
    assert native_boundary['calls'] == 1
    assert (output / 'manifest.json').read_bytes() == manifest
    assert (output / 'kept.jsonl').read_bytes() == kept
    assert source.read_bytes() == original
