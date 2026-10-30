"""ONNX models with external data load from a sharded huggingface-hub blob store.

huggingface-hub 1.x can cache each blob in its own directory (``hub/blobs/<xx>/<sha256>``),
so ``model.onnx`` and ``model.onnx_data`` resolve to different directories and ONNX Runtime
refuses the data file ("External data path escapes model directory").
"""

import hashlib
import json
import os
import shutil
import struct

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

import sct.utils.onnx_pipeline as onnx_pipeline

REPO = "test/tiny-external-onnx"
WEIGHTS = struct.pack("<2f", -1.0, 1.0)


def _varint(n):
    out = b""
    while True:
        byte, n = n & 0x7F, n >> 7
        out += bytes([byte | (0x80 if n else 0)])
        if not n:
            return out


def _f(num, val):
    """One protobuf field: ints as varints, str/bytes length-delimited."""
    if isinstance(val, int):
        return _varint(num << 3) + _varint(val)
    val = val.encode() if isinstance(val, str) else val
    return _varint(num << 3 | 2) + _varint(len(val)) + val


def _value_info(name, elem_type, dims):
    dim_fields = b"".join(_f(1, _f(1, d) if isinstance(d, int) else _f(2, d)) for d in dims)
    return _f(1, name) + _f(2, _f(1, _f(1, elem_type) + _f(2, dim_fields)))


def _model_bytes():
    """logits[b, s, :] = float(input_ids[b, s]) * W, with W (2 floats) in model.onnx_data."""
    weight = (
        _f(1, 2) + _f(2, 1) + _f(8, "W") + _f(14, 1)  # dims [2], FLOAT, EXTERNAL
        + _f(13, _f(1, "location") + _f(2, "model.onnx_data"))
        + _f(13, _f(1, "offset") + _f(2, "0"))
        + _f(13, _f(1, "length") + _f(2, str(len(WEIGHTS))))
    )
    axes = _f(1, 1) + _f(2, 7) + _f(8, "axes") + _f(9, struct.pack("<q", 2))
    nodes = (
        _f(1, _f(1, "input_ids") + _f(2, "x") + _f(4, "Cast")
           + _f(5, _f(1, "to") + _f(3, 1) + _f(20, 2)))
        + _f(1, _f(1, "x") + _f(1, "axes") + _f(2, "xu") + _f(4, "Unsqueeze"))
        + _f(1, _f(1, "xu") + _f(1, "W") + _f(2, "logits") + _f(4, "Mul"))
    )
    graph = (
        nodes + _f(2, "g") + _f(5, weight) + _f(5, axes)
        + _f(11, _value_info("input_ids", 7, ["b", "s"]))
        + _f(11, _value_info("attention_mask", 7, ["b", "s"]))
        + _f(12, _value_info("logits", 1, ["b", "s", 2]))
    )
    return _f(1, 8) + _f(8, _f(2, 13)) + _f(7, graph)


def _hub_cache(tmp_path):
    """Snapshot symlinks -> repo blobs -> sharded shared store, as huggingface-hub 1.32 lays it out."""
    hub = tmp_path / "hub"
    repo = hub / "models--test--tiny-external-onnx"
    snap = repo / "snapshots" / "rev1"
    snap.mkdir(parents=True)
    (repo / "blobs").mkdir()
    for name, data in (("model.onnx", _model_bytes()), ("model.onnx_data", WEIGHTS)):
        digest = hashlib.sha256(data).hexdigest()
        shard = hub / "blobs" / digest[:2]
        shard.mkdir(parents=True, exist_ok=True)
        (shard / digest).write_bytes(data)
        os.symlink(f"../../blobs/{digest[:2]}/{digest}", repo / "blobs" / digest)
        os.symlink(f"../../blobs/{digest}", snap / name)
    (snap / "config.json").write_text(json.dumps({"id2label": {"0": "O", "1": "B-PER"}}))
    tok = Tokenizer(models.WordLevel({"<pad>": 0, "john": 2}))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.save(str(snap / "tokenizer.json"))
    assert (snap / "model.onnx").resolve().parent != (snap / "model.onnx_data").resolve().parent
    return snap


@pytest.fixture
def snapshot(tmp_path, monkeypatch):
    snap = _hub_cache(tmp_path)

    def fake_download(repo_id, filename, **kwargs):
        assert repo_id == REPO
        if not (snap / filename).exists():
            raise FileNotFoundError(filename)
        return str(snap / filename)

    monkeypatch.setattr(onnx_pipeline, "hf_hub_download", fake_download)
    return snap


def _linked(tmp_path):
    return tmp_path / "cache" / "sct_linked" / "test__tiny-external-onnx"


def test_sharded_blob_store_loads_and_runs(tmp_path, snapshot):
    pipe, _ = onnx_pipeline.load_onnx_ner_model(REPO, cache_dir=str(tmp_path / "cache"))
    assert pipe(["john"]) == [[{
        "entity_group": "PER", "score": pytest.approx(0.9820, abs=1e-4),
        "word": "john", "start": 0, "end": 4,
    }]]
    linked = _linked(tmp_path)
    for name in ("model.onnx", "model.onnx_data"):
        assert os.path.samefile(linked / name, snapshot / name)  # hard links, no copy
    assert sorted(p.name for p in linked.parent.iterdir()) == [linked.name]

    # Second load reuses the directory.
    inode = (linked / "model.onnx_data").stat().st_ino
    onnx_pipeline.load_onnx_ner_model(REPO, cache_dir=str(tmp_path / "cache"))
    assert (linked / "model.onnx_data").stat().st_ino == inode


def test_copies_when_hard_link_fails(tmp_path, snapshot, monkeypatch):
    def no_link(*args):
        raise OSError("cross-device link")

    monkeypatch.setattr(onnx_pipeline.os, "link", no_link)
    pipe, _ = onnx_pipeline.load_onnx_ner_model(REPO, cache_dir=str(tmp_path / "cache"))
    assert pipe(["john"])[0][0]["entity_group"] == "PER"
    assert (_linked(tmp_path) / "model.onnx_data").read_bytes() == WEIGHTS


def test_failed_copy_leaves_no_partial_directory(tmp_path, snapshot, monkeypatch):
    def no_link(*args):
        raise OSError("cross-device link")

    def half_copy(src, dst):
        with open(dst, "wb") as f:
            f.write(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(onnx_pipeline.os, "link", no_link)
    monkeypatch.setattr(onnx_pipeline.shutil, "copyfile", half_copy)
    with pytest.raises(OSError, match="disk full"):
        onnx_pipeline.load_onnx_ner_model(REPO, cache_dir=str(tmp_path / "cache"))
    assert list(_linked(tmp_path).parent.iterdir()) == []


def test_flat_layout_is_loaded_in_place(tmp_path, monkeypatch):
    snap = _hub_cache(tmp_path)
    for name in ("model.onnx", "model.onnx_data"):
        real = (snap / name).resolve()
        (snap / name).unlink()
        shutil.copyfile(real, snap / name)
    seen = []
    real_session = onnx_pipeline.ort.InferenceSession
    monkeypatch.setattr(onnx_pipeline, "hf_hub_download",
                        lambda repo_id, filename, **kw: str(snap / filename))
    monkeypatch.setattr(onnx_pipeline.ort, "InferenceSession",
                        lambda path, **kw: seen.append(path) or real_session(path, **kw))
    onnx_pipeline.load_onnx_ner_model(REPO, cache_dir=str(tmp_path / "cache"))
    assert seen == [str(snap / "model.onnx")]
    assert not (tmp_path / "cache").exists()


def test_copied_cache_is_reused(tmp_path, snapshot, monkeypatch):
    copies = []
    real_copy = shutil.copyfile

    def no_link(*args):
        raise OSError("cross-device link")

    monkeypatch.setattr(onnx_pipeline.os, "link", no_link)
    monkeypatch.setattr(onnx_pipeline.shutil, "copyfile",
                        lambda src, dst: copies.append(src) or real_copy(src, dst))
    for _ in range(2):
        onnx_pipeline.load_onnx_ner_model(REPO, cache_dir=str(tmp_path / "cache"))
    assert len(copies) == 2  # second load reuses the copies


def test_concurrent_loader_accepts_winners_copy(tmp_path, snapshot, monkeypatch):
    def no_link(*args):
        raise OSError("cross-device link")

    real_rename = os.rename

    def winner_first(src, dst):
        shutil.copytree(src, dst)  # another process published its copy first
        return real_rename(src, dst)

    monkeypatch.setattr(onnx_pipeline.os, "link", no_link)
    monkeypatch.setattr(onnx_pipeline.os, "rename", winner_first)
    pipe, _ = onnx_pipeline.load_onnx_ner_model(REPO, cache_dir=str(tmp_path / "cache"))
    assert pipe(["john"])[0][0]["entity_group"] == "PER"
    assert sorted(p.name for p in _linked(tmp_path).parent.iterdir()) == [_linked(tmp_path).name]


def test_copy_from_older_blob_is_replaced(tmp_path, snapshot, monkeypatch):
    def no_link(*args):
        raise OSError("cross-device link")

    monkeypatch.setattr(onnx_pipeline.os, "link", no_link)
    onnx_pipeline.load_onnx_ner_model(REPO, cache_dir=str(tmp_path / "cache"))
    copied = _linked(tmp_path) / "model.onnx_data"
    copied.write_bytes(struct.pack("<2f", 5.0, 5.0))  # same size, stale content
    os.utime(copied, (0, 0))  # older than the blob, as if the snapshot moved to a new blob
    pipe, _ = onnx_pipeline.load_onnx_ner_model(REPO, cache_dir=str(tmp_path / "cache"))
    assert copied.read_bytes() == WEIGHTS
