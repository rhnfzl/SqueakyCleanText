"""Real worker-process tests for the adapters (decisions 13 and 15).

Linux only: run by the ``linux-workers`` CI job with ``SCT_MP_START`` set to
``fork`` (Python 3.13) or ``forkserver`` (3.14's Linux default). Every test
uses the small test NER model, never the default ensemble.
"""

import functools
import gc
import json
import multiprocessing
import os
import sys

import pytest

from tests.conftest import TEST_NER_MODELS, TEST_NER_REVISIONS

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="real fork/forkserver path is Linux only")

START = os.environ.get("SCT_MP_START", "fork")
TIMEOUT = 300  # a deadlocked fork would otherwise hang CI forever

TEXTS = [
    "John Smith lives in London and his email is john.smith@example.com.",
    "Call Maria Garcia at +34 612 345 678 tomorrow in Madrid.",
    "Jan de Vries werkt bij Philips in Eindhoven, mail jan@vries.nl.",
    "Angela Merkel besuchte Berlin am 3. Oktober 2020.",
    "Server 192.168.1.20 belongs to Acme Corp in New York.",
    "Visit https://example.org/page?id=42 for details from Paul Jones.",
    "Le président Emmanuel Macron était à Paris hier.",
    "Mario Rossi abita a Roma, telefono 06 1234 5678.",
    "Pedro Alvarez trabaja en Telefonica desde 2015.",
    "राहुल शर्मा दिल्ली में रहते हैं और rahul@example.in पर ईमेल करें।",
    "Sarah Connor paid $1,200 to Cyberdyne Systems on 12/05/2021.",
    "Peter Parker works at the Daily Bugle in Manhattan.",
    "Kontakt: anna.schmidt@firma.de, Telefon 030 1234567.",
    "Lisa Jansen woont in Utrecht sinds 1999.",
    "Contact support@company.com or call 555-123-4567.",
    "Tom Hanks met Steven Spielberg in Los Angeles.",
    "Juan Perez envió 500 euros a Ana Lopez.",
    "Marie Curie worked in Paris at the Sorbonne.",
    "Plain text with no personal data at all.",
    "",
]


def _config():
    from sct.presets import preset_config

    return preset_config("training", ner_models=TEST_NER_MODELS, ner_model_revisions=TEST_NER_REVISIONS)


def test_hf_num_proc_matches_single_process():
    datasets = pytest.importorskip("datasets")
    hf = pytest.importorskip("sct.integrations.hf")

    ds = datasets.Dataset.from_dict({"text": TEXTS, "id": list(range(len(TEXTS)))})
    single = hf.clean_dataset(ds, config=_config(), batch_size=4, num_proc=None)
    multi = hf.clean_dataset(ds, config=_config(), batch_size=4, num_proc=2)

    assert multi["id"] == single["id"]
    assert multi["text"] == single["text"]
    assert single["text"] != TEXTS  # the cleaner really ran


def _read_shard(data, rank, world_size):
    # Plain callable steps keep the test off DataTrove's optional IO deps (orjson, fsspec writers).
    from datatrove.data import Document

    for i, text in enumerate(TEXTS):
        if i % world_size == rank:
            yield Document(text=text, id=str(i))


def _write_rows(out_dir, data, rank, world_size):
    rows = [{"id": d.id, "text": d.text, "sct": d.metadata.get("sct")} for d in data]
    (out_dir / f"{rank}.json").write_text(json.dumps(rows))
    return iter(())


def _run_datatrove(tmp_path, name, tasks, workers):
    from datatrove.executor import LocalPipelineExecutor

    from sct.integrations.datatrove import SCTCleaner

    out = tmp_path / name
    out.mkdir()
    LocalPipelineExecutor(
        pipeline=[_read_shard, SCTCleaner(config=_config()), functools.partial(_write_rows, out)],
        tasks=tasks,
        workers=workers,
        logging_dir=str(tmp_path / f"logs-{name}"),
        start_method=START,
    ).run()
    rows = [r for f in sorted(out.glob("*.json")) for r in json.loads(f.read_text())]
    return sorted((r["id"], r["text"], json.dumps(r["sct"], sort_keys=True)) for r in rows)


def test_datatrove_two_workers_match_in_process(tmp_path):
    pytest.importorskip("datatrove")
    pytest.importorskip("sct.integrations.datatrove")

    in_process = _run_datatrove(tmp_path, "one", tasks=1, workers=1)
    two_workers = _run_datatrove(tmp_path, "two", tasks=2, workers=2)

    assert len(in_process) == len(TEXTS)
    assert two_workers == in_process
    assert all(sct != "null" for _, _, sct in two_workers)


_HANDLE = None


def _init_worker(handle):
    global _HANDLE
    _HANDLE = handle


def _clean_in_worker(text):
    cleaner = _HANDLE.get()
    gc.collect()  # a long-lived worker collects the dropped inherited cleaner sooner or later
    # The parent tagged its cleaner; a forked child that reuses it was never rebuilt.
    return os.getpid(), getattr(cleaner, "_sct_parent_built", False), cleaner.process(text).lm_text


def test_raw_pool_children_rebuild_parent_cleaner():
    runtime = pytest.importorskip("sct.integrations._runtime")

    handle = runtime.CleanerHandle(_config())
    parent_cleaner = handle.get()  # built before the pool forks
    parent_cleaner._sct_parent_built = True
    expected = [parent_cleaner.process(t).lm_text for t in TEXTS]
    parent_id = id(parent_cleaner)
    # Only the handle keeps it alive now, so a child rebuild drops the last reference to an
    # inherited ONNX session: destroying it after fork crashed (exit 139) on Linux.
    del parent_cleaner

    # fork passes initargs by inheritance, forkserver/spawn by pickling: both must rebuild.
    with multiprocessing.get_context(START).Pool(2, initializer=_init_worker, initargs=(handle,)) as pool:
        results = pool.map_async(_clean_in_worker, TEXTS, chunksize=1).get(timeout=TIMEOUT)

    assert {pid for pid, _, _ in results}.isdisjoint({os.getpid()})
    assert not any(reused for _, reused, _ in results)
    assert [text for _, _, text in results] == expected
    assert id(handle.get()) == parent_id  # same process, no rebuild


def test_native_corpus_two_workers_match_private_staging(tmp_path):
    """Native tasks recreate schema-aware cleaners inside framework workers."""
    pytest.importorskip('datatrove')
    pytest.importorskip('spacy')
    pytest.importorskip('orjson')
    from datatrove.pipeline.dedup import MinhashConfig
    from datatrove.utils.hashing import HashConfig
    from sct.config import TextCleanerConfig
    from sct.corpus.datatrove import curate_datatrove
    from sct.corpus import SelectionConfig
    from sct.schema import FieldRule, FieldSchema
    from tests.test_schema_context import context_registry
    source = tmp_path / 'input.jsonl'
    source.write_text('\n'.join(json.dumps({'id': value, 'text': 'May'}) for value in ['b', 'a']))
    before = source.read_bytes()
    outputs = []
    for workers in (1, 2):
        output = tmp_path / f'workers{workers}'
        manifest = curate_datatrove(
            source, output, retention='caller_managed', workers=workers,
            config=TextCleanerConfig(check_ner_process=False, pii_processing='unified'),
            schema=FieldSchema('caller', '1', (FieldRule(('text',), ('PERSON',), mode='protect_value'),)),
            registry_factory=context_registry, component_identities=(('registry', 'test-context-v1'),),
            selection=SelectionConfig(dedup_mode='apply', dedup_basis='pii_normalized'),
            native_config=MinhashConfig(n_grams=1, num_buckets=2, hash_config=HashConfig(hash_fc='sha1')),
        )
        assert manifest.counts == {'input': 2, 'kept': 1, 'rejected': 1}
        outputs.append((output / 'kept.jsonl').read_bytes())
    assert outputs[0] == outputs[1]
    assert json.loads(outputs[0]) == {'id': 'a', 'text': '<PERSON>'}
    assert source.read_bytes() == before
