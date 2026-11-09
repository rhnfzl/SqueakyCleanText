"""Corpus identities must cover every model component before creating output."""
import json

import pytest

from sct import TextCleanerConfig

@pytest.fixture
def bundled_model(tmp_path, monkeypatch):
    files = {
        'gliner_config.json': {'encoder_config': {'model_type': 'deberta-v2'}},
        'tokenizer_config.json': {'tokenizer_class': 'DebertaV2Tokenizer'},
        'tokenizer.json': {'model': {'type': 'Unigram', 'vocab': [['test', 0]]}},
    }
    calls = []
    def cached(repo_id, filename, *, revision, local_files_only):
        assert repo_id == 'caller/model' and revision == 'a' * 40 and local_files_only is True
        calls.append(filename)
        if filename not in files:
            raise OSError('Missing prepared artifact')
        path = tmp_path / filename
        path.write_text(json.dumps(files[filename]))
        return str(path)
    monkeypatch.setattr('huggingface_hub.hf_hub_download', cached)
    return files, calls


@pytest.mark.parametrize('missing', ['encoder_config', 'tokenizer_config.json', 'tokenizer.json', 'auxiliary_encoder'])
def test_unbundled_pinned_gliner_fails_before_output_creation(tmp_path, bundled_model, missing):
    pytest.importorskip('datatrove')
    from sct.corpus.datatrove import curate_datatrove
    files, _ = bundled_model
    if missing == 'encoder_config':
        files['gliner_config.json'].pop('encoder_config')
    elif missing == 'auxiliary_encoder':
        files['gliner_config.json']['labels_encoder'] = 'caller/parent'
    else:
        files.pop(missing)
    source = tmp_path / 'input.jsonl'
    source.write_text('')
    with pytest.raises(ValueError, match='self-contained GLiNER'):
        curate_datatrove(source, tmp_path / 'output', retention='caller_managed',
                        config=TextCleanerConfig(ner_backend='gliner', gliner_model='caller/model',
                                                 gliner_revision='a' * 40))
    assert not (tmp_path / 'output').exists()


def test_bundled_pinned_gliner_identity_requires_only_cached_metadata(bundled_model):
    pytest.importorskip('datatrove')
    from sct.corpus.datatrove import _model_identities
    _, calls = bundled_model
    config = TextCleanerConfig(ner_backend='gliner', gliner_model='caller/model', gliner_revision='a' * 40)
    assert _model_identities(config) == (('gliner', 'caller/model@' + 'a' * 40),)
    assert set(calls) == {'gliner_config.json', 'tokenizer_config.json', 'tokenizer.json'}


def test_bundled_gliner_validation_requires_only_cached_metadata(bundled_model):
    from sct.corpus.manifest import _require_bundled_gliner
    _, calls = bundled_model
    _require_bundled_gliner('caller/model', 'a' * 40)
    assert set(calls) == {'gliner_config.json', 'tokenizer_config.json', 'tokenizer.json'}


@pytest.mark.parametrize(('filename', 'metadata'), [
    ('gliner_config.json', {}),
    ('gliner_config.json', {'encoder_config': {}}),
    ('gliner_config.json', {'encoder_config': {'model_type': 'deberta-v2'}, 'labels_encoder': 'caller/parent'}),
    ('gliner_config.json', {'encoder_config': {'model_type': 'deberta-v2'}, 'labels_decoder': 'caller/parent'}),
    ('tokenizer_config.json', {}),
    ('tokenizer.json', {'model': None}),
    ('tokenizer.json', []),
])
def test_gliner_metadata_requires_complete_self_contained_components(bundled_model, filename, metadata):
    from sct.corpus.manifest import _require_bundled_gliner
    files, _ = bundled_model
    files[filename] = metadata
    with pytest.raises(ValueError, match='self-contained GLiNER'):
        _require_bundled_gliner('caller/model', 'a' * 40)


def test_missing_cached_gliner_metadata_fails_without_network(bundled_model):
    from sct.corpus.manifest import _require_bundled_gliner
    files, _ = bundled_model
    files.pop('tokenizer.json')
    with pytest.raises(ValueError, match='self-contained GLiNER'):
        _require_bundled_gliner('caller/model', 'a' * 40)
