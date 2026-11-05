"""DataTrove, LangChain, Haystack and Hugging Face adapters (each test skips without its package)."""

import asyncio

import pytest

from sct.presets import preset_config
import dataclasses
from sct import TextCleanerConfig

FAST = preset_config("training", check_ner_process=False)
RAW = "mail john@example.com now"
CLEAN = "mail <EMAIL> now"


def test_datatrove_step():
    pytest.importorskip("datatrove")
    from datatrove.data import Document

    from sct.integrations.datatrove import SCTCleaner

    step = SCTCleaner(columns=("text", "author"), config=FAST, batch_size=2)
    docs = [Document(text=RAW, id=str(i), metadata={"author": RAW}) for i in range(3)]
    out = list(step.run(iter(docs)))
    assert [d.text for d in out] == [CLEAN] * 3
    assert out[0].metadata["author"] == CLEAN and out[0].metadata["sct"]["pii_count"] == 2
    stats = step.stats.stats
    assert stats["pii"].total == 6 and stats["pii_EMAIL"].total == 6 and stats["sensitive"].total == 3


def test_langchain_transformer():
    pytest.importorskip("langchain_core")
    from langchain_core.documents import Document

    from sct.integrations.langchain import SCTDocumentTransformer

    doc = Document(page_content=RAW, metadata={"src": "a"})
    out = SCTDocumentTransformer(config=FAST).transform_documents([doc])
    assert out[0].page_content == CLEAN and out[0].metadata["src"] == "a" and "sct" in out[0].metadata
    assert doc.page_content == RAW and "sct" not in doc.metadata
    new = asyncio.run(SCTDocumentTransformer(config=FAST, output="new_column").atransform_documents([doc]))
    assert new[0].page_content == RAW and new[0].metadata["page_content_clean"] == CLEAN
    both = Document(page_content="plain text", metadata={"author": RAW})
    merged = SCTDocumentTransformer(columns=("page_content", "author"), config=FAST).transform_documents([both])
    assert merged[0].metadata["sct"]["pii_types"] == {"EMAIL": 1}


def test_haystack_component():
    pytest.importorskip("haystack")
    from haystack import Document

    from sct.integrations.haystack import SCTDocumentCleaner

    doc = Document(content=RAW, meta={"src": "a"})
    out = SCTDocumentCleaner(config=FAST).run([doc, Document(content=None)])["documents"]
    assert out[0].content == CLEAN and out[0].meta["src"] == "a" and "sct" in out[0].meta
    assert out[0].id != doc.id and out[1].content is None
    assert doc.content == RAW
    both = Document(content="plain text", meta={"author": RAW})
    merged = SCTDocumentCleaner(columns=("content", "author"), config=FAST).run([both])["documents"]
    assert merged[0].meta["sct"]["pii_types"] == {"EMAIL": 1}


RAW_ALICE = "mail alice@example.com"


# --- a str column name is one column, not its characters ---
@pytest.mark.parametrize("module, cls, column", [
    ("langchain_core", "langchain", "page_content"),
    ("haystack", "haystack", "content"),
    ("datatrove", "datatrove", "text"),
    ("datasets", "hf", "text"),
])
def test_adapters_take_a_single_column_name(module, cls, column):
    pytest.importorskip(module)
    mod = __import__(f"sct.integrations.{cls}", fromlist=["x"])
    adapter = {"langchain": "SCTDocumentTransformer", "haystack": "SCTDocumentCleaner",
               "datatrove": "SCTCleaner", "hf": "SCTMap"}[cls]
    assert getattr(mod, adapter)(columns=column, config=FAST).columns == (column,)


def test_langchain_str_column_cleans():
    pytest.importorskip("langchain_core")
    from langchain_core.documents import Document

    from sct.integrations.langchain import SCTDocumentTransformer

    out = SCTDocumentTransformer(columns="page_content", config=FAST).transform_documents(
        [Document(page_content=RAW_ALICE)])
    assert out[0].page_content == "mail <EMAIL>"


# --- Haystack pipelines keep the cleaner config through to_dict/from_dict ---
def test_haystack_pipeline_round_trip_keeps_config():
    pytest.importorskip("haystack")
    from haystack import Document, Pipeline

    from sct.integrations.haystack import SCTDocumentCleaner

    cfg = TextCleanerConfig(check_ner_process=False, replace_with_email="<MAIL>",
                            positional_tags=("PER",))
    pipe = Pipeline()
    pipe.add_component("clean", SCTDocumentCleaner(columns="content", config=cfg))
    restored = Pipeline.loads(pipe.dumps(), allowed_modules=["sct.integrations.haystack"])
    comp = restored.get_component("clean")
    assert comp.handle.config.check_ner_process is False
    assert comp.handle.config.positional_tags == ("PER",)
    out = restored.run({"clean": {"documents": [Document(content=RAW_ALICE)]}})
    assert out["clean"]["documents"][0].content == "mail <MAIL>"
    named = Pipeline()
    named.add_component("clean", SCTDocumentCleaner(config="training"))
    assert named.to_dict()["components"]["clean"]["init_parameters"]["config"] == "training"


# --- Haystack serializes nested config sets (custom stopwords) ---
def test_haystack_serializes_custom_stopwords():
    pytest.importorskip("haystack")
    from haystack import Pipeline

    from sct.integrations.haystack import SCTDocumentCleaner

    cfg = TextCleanerConfig(check_ner_process=False, custom_stopwords={"ENGLISH": frozenset({"foo"})})
    pipe = Pipeline()
    pipe.add_component("clean", SCTDocumentCleaner(config=cfg))
    restored = Pipeline.loads(pipe.dumps(), allowed_modules=["sct.integrations.haystack"])
    assert restored.get_component("clean").handle.config.custom_stopwords["ENGLISH"] == frozenset({"foo"})


# --- remove_columns with output="new_column" really drops the raw column ---
@pytest.mark.parametrize("iterable", [False, True])
def test_clean_dataset_remove_columns_drops_raw_text(iterable):
    datasets = pytest.importorskip("datasets")
    from sct.integrations.hf import clean_dataset

    ds = datasets.Dataset.from_dict({"text": ["mail alice@example.com"], "id": [7]})
    if iterable:
        ds = ds.to_iterable_dataset()
    out = clean_dataset(ds, output="new_column", config=FAST, remove_columns=["text"])
    rows = list(out)
    assert rows == [{"id": 7, "text_clean": "mail <EMAIL>"}]


# --- datasets output columns are declared, so a first null batch cannot fix a null type ---
@pytest.mark.parametrize("data, kwargs", [
    ({"text": ["plain", "alice@example.com"]}, {"score": True}),
    ({"text": [None, "hello world"]}, {"score": True}),
    ({"text": [None, "alice@example.com"]}, {"output": "new_column"}),
])
def test_clean_dataset_first_null_batch(data, kwargs):
    datasets = pytest.importorskip("datasets")
    from sct.integrations.hf import clean_dataset
    out = clean_dataset(datasets.Dataset.from_dict(data), config=FAST, batch_size=1, **kwargs)
    assert len(out) == 2
    if kwargs.get("score"):
        assert out.features["sct_pii_types"].feature.dtype == "string"


# --- a Haystack round trip keeps the PII label map derived ---
@pytest.mark.parametrize("legacy", [False, True])
def test_haystack_round_trip_rederives_pii_map(legacy):
    pytest.importorskip("haystack")
    from sct.integrations.haystack import SCTDocumentCleaner

    cfg = TextCleanerConfig(ner_mode="pii", legacy_type_names=legacy)
    loaded = SCTDocumentCleaner.from_dict(SCTDocumentCleaner(config=cfg).to_dict())._config
    flipped = dataclasses.replace(loaded, legacy_type_names=not legacy)
    direct = dataclasses.replace(cfg, legacy_type_names=not legacy)
    assert dict(flipped.gliner_label_map) == dict(direct.gliner_label_map)
    assert flipped.positional_tags == direct.positional_tags
