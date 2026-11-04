"""DataTrove, LangChain and Haystack adapters (each skips without its package)."""

import asyncio

import pytest

from sct.presets import preset_config

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
    assert out[0].metadata["author"] == CLEAN and out[0].metadata["sct"]["pii_count"] == 1
    stats = step.stats.stats
    assert stats["pii"].total == 3 and stats["pii_EMAIL"].total == 3 and stats["sensitive"].total == 3


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


def test_haystack_component():
    pytest.importorskip("haystack")
    from haystack import Document

    from sct.integrations.haystack import SCTDocumentCleaner

    doc = Document(content=RAW, meta={"src": "a"})
    out = SCTDocumentCleaner(config=FAST).run([doc, Document(content=None)])["documents"]
    assert out[0].content == CLEAN and out[0].meta["src"] == "a" and "sct" in out[0].meta
    assert out[0].id != doc.id and out[1].content is None
    assert doc.content == RAW
