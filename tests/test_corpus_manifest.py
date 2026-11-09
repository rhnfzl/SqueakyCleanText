"""Native corpus recipe acceptance checks."""
import json
import importlib.util

import pytest

from sct.config import TextCleanerConfig
from sct.corpus import SelectionConfig


requires_native = pytest.mark.skipif(
    any(importlib.util.find_spec(name) is None for name in ('datatrove', 'spacy', 'orjson')),
    reason='install the corpus extra for native DataTrove execution',
)

@requires_native
def test_d2_acceptance(tmp_path):
    import sct.corpus.datatrove as recipe
    source = tmp_path / 'input.jsonl'
    text = 'This is a sufficiently long document about scientific data and careful research methods.'
    source.write_text('\n'.join(json.dumps({'id': i, 'text': text}) for i in ['b', 'a']))
    before = source.read_bytes()
    result = recipe.curate_datatrove(
        source, tmp_path / 'run', config=TextCleanerConfig(check_ner_process=False),
        selection=SelectionConfig(dedup_mode='apply', dedup_basis='pre_pii'), retention='caller_managed',
    )
    assert result.counts == {'input': 2, 'kept': 1, 'rejected': 1}
    assert source.read_bytes() == before
    assert json.loads((tmp_path / 'run' / 'kept.jsonl').read_text())['id'] == 'a'
    assert json.loads((tmp_path / 'run' / 'manifest.json').read_text())['schema_version'] == 1
    assert len((tmp_path / 'run' / 'decisions.jsonl').read_text().splitlines()) == 2


@requires_native
def test_report_retains_all(tmp_path):
    from sct.corpus.datatrove import curate_datatrove
    source = tmp_path / 'in.jsonl'
    text = 'same words repeated across this long enough document for native shingles'
    source.write_text('\n'.join(json.dumps({'id': i, 'text': text}) for i in ['a', 'b']))
    result = curate_datatrove(source, tmp_path / 'run', config=TextCleanerConfig(check_ner_process=False),
                             retention='caller_managed')
    assert result.counts['kept'] == 2


def test_identity_is_private_and_sensitive_config_affects_digest():
    from sct.corpus.manifest import config_digest, RunIdentity
    sensitive_literal = 'private-sensitive-literal'
    assert config_digest({'terms': [sensitive_literal]}) != config_digest({'terms': []})
    identity = RunIdentity('run', 'base', 'package', (('partition', 'a' * 64, ('id',)),),
                           config_digest({'terms': [sensitive_literal]}), (), 1)
    assert sensitive_literal not in repr(identity)
    assert sensitive_literal not in json.dumps(identity.to_dict())
    with pytest.raises(ValueError):
        RunIdentity('../bad', 'base', 'package', (), 'bad', (), 1)


@requires_native
def test_paths_cannot_overwrite_input(tmp_path):
    from sct.corpus.datatrove import curate_datatrove
    source = tmp_path / 'input'
    source.write_text('{}')
    with pytest.raises(ValueError):
        curate_datatrove(source, source, retention='caller_managed')

@pytest.mark.parametrize('basis,expected', [('pre_pii', 2), ('pii_normalized', 1)])
@requires_native
def test_explicit_basis_changes_native_decisions(tmp_path, basis, expected):
    from datatrove.pipeline.dedup import MinhashConfig
    from datatrove.utils.hashing import HashConfig
    from sct.corpus.datatrove import curate_datatrove
    source = tmp_path / 'input.jsonl'
    source.write_text('\n'.join(json.dumps({'id': i, 'text': address}) for i, address in
                                [('a', 'first@example.org'), ('b', 'second@example.net')]))
    result = curate_datatrove(source, tmp_path / 'run', retention='caller_managed',
                             config=TextCleanerConfig(check_ner_process=False),
                             selection=SelectionConfig(dedup_mode='apply', dedup_basis=basis),
                             native_config=MinhashConfig(n_grams=1, hash_config=HashConfig(hash_fc='sha1')))
    assert result.counts['kept'] == expected
    assert 'first@example.org' not in (tmp_path / 'run' / 'manifest.json').read_text()
    assert (tmp_path / 'run').stat().st_mode & 0o777 == 0o700
    assert all(p.stat().st_mode & 0o077 == 0 for p in (tmp_path / 'run').rglob('*'))


@requires_native
def test_failed_worker_does_not_log_sensitive_values(tmp_path, capsys):
    from sct.corpus.datatrove import curate_datatrove
    sensitive_literal = 'sensitive-document-contents'
    source = tmp_path / 'in.jsonl'
    source.write_text(json.dumps({'id': 'a', 'text': sensitive_literal}) + '\n' + '{invalid')
    with pytest.raises(ValueError, match='Corpus stage failed'):
        curate_datatrove(source, tmp_path / 'run', retention='caller_managed',
                         config=TextCleanerConfig(check_ner_process=False))
    assert sensitive_literal not in capsys.readouterr().err
    assert not (tmp_path / 'run' / 'manifest.json').exists()
    assert all(sensitive_literal not in p.read_text() for p in (tmp_path / 'run' / 'logs').rglob('*') if p.is_file())


@requires_native
def test_native_threshold_is_not_silently_ignored(tmp_path):
    from sct.corpus.datatrove import curate_datatrove
    source = tmp_path / 'in'
    source.write_text('')
    with pytest.raises(ValueError, match='native_config'):
        curate_datatrove(source, tmp_path / 'run', retention='caller_managed',
                         selection=SelectionConfig(similarity_threshold=.8))


@requires_native
def test_caller_hash_algorithm_is_never_substituted(tmp_path):
    pytest.importorskip('xxhash')
    from datatrove.pipeline.dedup import MinhashConfig
    from sct.corpus.datatrove import curate_datatrove
    from datatrove.utils.hashing import create_hash_func
    native = MinhashConfig()
    try:
        create_hash_func(native.hash_config)('probe')
    except TypeError:
        source = tmp_path / 'in'
        source.write_text('')
        with pytest.raises(ValueError, match='hash algorithm'):
            curate_datatrove(source, tmp_path / 'run', retention='caller_managed', native_config=native)
        assert not (tmp_path / 'run').exists()
    else:
        pytest.skip('Installed xxhash supports text, so incompatible-runtime branch does not apply')


def test_malformed_identity_rejects_wrong_digest_type():
    from sct.corpus.manifest import RunIdentity
    with pytest.raises(ValueError):
        RunIdentity('run', 'base', 'package', (('input', 42, ('a',)),), 'a' * 64, (), 1)


@requires_native
def test_explicit_cleaner_configuration_is_required(tmp_path):
    from sct.corpus.datatrove import curate_datatrove
    source = tmp_path / 'in'
    source.write_text('')
    with pytest.raises(ValueError, match='explicit cleaner'):
        curate_datatrove(source, tmp_path / 'run', retention='caller_managed')
    assert not (tmp_path / 'run').exists()

@pytest.mark.parametrize('payload', [b'x', b'\x02\x00\x00\x00\x00\x00\x00\x00'])
@requires_native
def test_native_cluster_ordinals_are_validated(tmp_path, payload):
    import sct.corpus.datatrove as recipe
    path = tmp_path / 'clusters'
    path.write_bytes(payload)
    assert hasattr(recipe, '_cluster_keepers')
    with pytest.raises(ValueError, match='cluster'):
        recipe._cluster_keepers(path, 2)


def test_digest_supports_frozen_configuration_without_revealing_terms():
    from types import MappingProxyType
    from sct.corpus.manifest import config_digest
    assert config_digest({'terms': frozenset({'private-value'})}) == config_digest({'terms': ['private-value']})
    assert config_digest(MappingProxyType({'nested': MappingProxyType({'revision': 'one'})})) != config_digest(
        {'nested': {'revision': 'two'}})


def _quality_fixture(texts):
    return [.1 if text.startswith('low') else .9 for text in texts]


@requires_native
def test_quality_rejection_cannot_remove_all_eligible_cluster_members(tmp_path):
    from datatrove.pipeline.dedup import MinhashConfig
    from datatrove.utils.hashing import HashConfig
    from sct.corpus.datatrove import curate_datatrove
    source = tmp_path / 'input'
    source.write_text('\n'.join(json.dumps({'id': i, 'text': text}) for i, text in
                                [('a', 'low@example.org'), ('b', 'high@example.org'), ('c', 'other@example.org')]))
    output = tmp_path / 'run'
    manifest = curate_datatrove(
        source, output, retention='caller_managed', config=TextCleanerConfig(check_ner_process=False),
        selection=SelectionConfig(quality_min=.5, dedup_mode='apply', dedup_basis='pii_normalized'),
        native_config=MinhashConfig(n_grams=1, num_buckets=2, hash_config=HashConfig(hash_fc='sha1')),
        scorer=_quality_fixture, component_identities=(('quality', 'fixture-v1'),),
    )
    assert manifest.counts == {'input': 3, 'kept': 1, 'rejected': 2}
    assert json.loads((output / 'kept.jsonl').read_text())['id'] == 'b'
    decisions = [json.loads(line) for line in (output / 'decisions.jsonl').read_text().splitlines()]
    assert decisions[0]['reasons'] == ['quality_below_threshold']
    assert decisions[2]['keeper_id'] == 'b'


def test_manifest_rejects_duplicate_output_shards():
    from sct.corpus.manifest import RunIdentity, RunManifest
    identity = RunIdentity('run', 'base', 'package', (), 'a' * 64, (), 1)
    with pytest.raises(ValueError):
        RunManifest(identity, (('kept.jsonl', 'a' * 64), ('kept.jsonl', 'b' * 64)),
                    {'input': 0, 'kept': 0, 'rejected': 0})


@pytest.mark.parametrize('options', [
    {'ner_models': {'ENGLISH': 'caller/model', 'MULTILINGUAL': 'caller/multilingual'}},
    {'ner_backend': 'torch', 'torch_ner_models': {'ENGLISH': 'caller/model', 'MULTILINGUAL': 'caller/multilingual'}},
    {'ner_backend': 'gliner', 'gliner_model': 'caller/model'},
    {'ner_backend': 'gliner', 'gliner_variant': 'gliner2', 'gliner_model': 'caller/model'},
    {'ner_backend': 'ensemble_onnx', 'gliner_model': 'caller/model'},
    {'ner_backend': 'ensemble_torch', 'gliner_model': 'caller/model'},
    {'check_ner_process': False, 'check_classify_document': True, 'gliclass_model': 'caller/model'},
])
@requires_native
def test_unpinned_active_models_fail_before_output_creation(tmp_path, options):
    from sct.corpus.datatrove import curate_datatrove
    source = tmp_path / 'input'
    source.write_text('')
    with pytest.raises(ValueError, match='immutable model revision'):
        curate_datatrove(source, tmp_path / 'run', retention='caller_managed', config=TextCleanerConfig(**options))
    assert not (tmp_path / 'run').exists()


@requires_native
def test_different_immutable_model_artifacts_change_identity_without_loading():
    from dataclasses import replace
    import sct.corpus.datatrove as recipe
    config = TextCleanerConfig(ner_models={'ENGLISH': 'caller/model', 'MULTILINGUAL': 'caller/multilingual'},
                               ner_model_revisions={'ENGLISH': 'a' * 40, 'MULTILINGUAL': 'b' * 40})
    assert hasattr(recipe, '_model_identities')
    first = recipe._model_identities(config)
    second = recipe._model_identities(replace(config, ner_model_revisions={**config.ner_model_revisions, 'ENGLISH': 'c' * 40}))
    assert first != second
    assert ('ner:ENGLISH', 'caller/model@' + 'a' * 40) in first
    assert recipe._model_identities(replace(config, check_ner_process=False)) == ()


def test_manifest_requires_immutable_outer_shard_collection():
    from sct.corpus.manifest import RunIdentity, RunManifest
    identity = RunIdentity('run', 'base', 'package', (), 'a' * 64, (), 1)
    with pytest.raises(ValueError):
        RunManifest(identity, [('kept.jsonl', 'a' * 64)], {'input': 0, 'kept': 0, 'rejected': 0})


@requires_native
def test_active_models_require_actual_offline_hub_runtime(tmp_path, monkeypatch):
    from huggingface_hub import constants
    from sct.corpus.datatrove import curate_datatrove
    monkeypatch.setattr(constants, 'HF_HUB_OFFLINE', False)
    source = tmp_path / 'input'
    source.write_text('')
    with pytest.raises(ValueError, match='HF_HUB_OFFLINE=1'):
        curate_datatrove(source, tmp_path / 'run', retention='caller_managed', config=TextCleanerConfig())
    assert not (tmp_path / 'run').exists()


@requires_native
def test_unified_selected_detector_specs_determine_active_model_routes():
    from sct.corpus.datatrove import _model_identities
    from sct.detectors import DetectorSpec
    inactive = TextCleanerConfig(pii_processing='unified', check_ner_process=True,
                                 detector_specs=(DetectorSpec('builtin_regex'),),
                                 ner_models={'ENGLISH': 'caller/unpinned', 'MULTILINGUAL': 'caller/other'})
    assert _model_identities(inactive) == ()
    active = TextCleanerConfig(pii_processing='unified', check_ner_process=False,
                               detector_specs=(DetectorSpec('builtin_ner'),))
    assert _model_identities(active)


@requires_native
def test_presidio_route_cannot_claim_revision_enforcement(tmp_path):
    from sct.corpus.datatrove import curate_datatrove
    source = tmp_path / 'input'
    source.write_text('')
    with pytest.raises(ValueError, match='does not enforce immutable'):
        curate_datatrove(source, tmp_path / 'run', retention='caller_managed',
                         config=TextCleanerConfig(ner_backend='presidio_gliner', gliner_model='caller/model',
                                                  gliner_revision='a' * 40))
    assert not (tmp_path / 'run').exists()


def _private_failure_fixture(texts):
    raise RuntimeError('private-exception-literal:' + texts[0])


@requires_native
def test_scorer_failure_does_not_log_private_exception_or_custom_terms(tmp_path, capsys):
    from sct.corpus.datatrove import curate_datatrove
    source = tmp_path / 'in'
    source.write_text(json.dumps({'id': 'a', 'text': 'private-record-literal'}))
    output = tmp_path / 'run'
    with pytest.raises(ValueError, match='Corpus stage failed'):
        curate_datatrove(
            source, output, retention='caller_managed',
            config=TextCleanerConfig(check_ner_process=False, custom_allow_terms=('private-allow-literal',)),
            scorer=_private_failure_fixture, component_identities=(('quality', 'immutable-fixture-v1'),),
        )
    observed = capsys.readouterr().err + '\n'.join(p.read_text() for p in (output / 'logs').rglob('*') if p.is_file())
    for literal in ('private-exception-literal', 'private-record-literal', 'private-allow-literal'):
        assert literal not in observed
    assert not (output / 'manifest.json').exists()
