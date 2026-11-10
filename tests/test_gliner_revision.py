"""The GLiNER2 SDK must receive the requested snapshot, even when it ignores kwargs."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from sct.utils.gliner_adapter import GLiNERAdapter


@pytest.fixture
def model_files(tmp_path, monkeypatch):
    default = tmp_path / 'default'
    pinned = tmp_path / 'pinned'
    for path, content in ((default, 'default weights'), (pinned, 'pinned weights')):
        path.mkdir()
        (path / 'weights').write_text(content)
    class SDK:
        @staticmethod
        def from_pretrained(repo_or_dir, **kwargs):
            source = Path(repo_or_dir) if Path(repo_or_dir).is_dir() else default
            return SimpleNamespace(weights=(source / 'weights').read_text())
    monkeypatch.setitem(sys.modules, 'gliner2', SimpleNamespace(GLiNER2=SDK))
    return pinned


def test_gliner2_revision_selects_pinned_snapshot_not_sdk_default(model_files, monkeypatch):
    def snapshot(repo_id, *, revision):
        assert (repo_id, revision) == ('caller/model', 'a' * 40)
        return str(model_files)
    monkeypatch.setattr('huggingface_hub.snapshot_download', snapshot)
    adapter = GLiNERAdapter('caller/model', revision='a' * 40, variant='gliner2')
    assert adapter.model.weights == 'pinned weights'


@pytest.mark.parametrize('local', [False, True])
def test_gliner2_unpinned_and_explicit_local_loads_do_not_resolve_hub_snapshots(model_files, monkeypatch, local):
    def forbidden(*args, **kwargs):
        raise AssertionError('This load must not resolve a Hub snapshot')
    monkeypatch.setattr('huggingface_hub.snapshot_download', forbidden)
    adapter = GLiNERAdapter(str(model_files) if local else 'caller/model',
                            revision='a' * 40 if local else None, variant='gliner2')
    assert adapter.model.weights == ('pinned weights' if local else 'default weights')


def test_gliner2_missing_revision_never_falls_back_to_default_weights(model_files, monkeypatch):
    def missing(*args, **kwargs):
        raise OSError('Requested snapshot is not available')
    monkeypatch.setattr('huggingface_hub.snapshot_download', missing)
    with pytest.raises(OSError, match='Requested snapshot'):
        GLiNERAdapter('caller/model', revision='a' * 40, variant='gliner2')
