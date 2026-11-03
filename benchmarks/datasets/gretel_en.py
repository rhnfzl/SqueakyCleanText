"""Gretel PII masking EN v1: synthetic English docs (47 domains, ~1k document types), test split.

Gold is entity strings with types, no offsets: each string is placed at every exact
occurrence in the text (all test entities occur exactly once at the pinned revision).
``group`` is the document type, so dev/test splits never share a document template.
nvidia/gliner-PII reports evaluation on it; bardsai v3 and OpenMed trained on its train split.
"""

import ast

from benchmarks.datasets import hf_file, sample_per_lang, valid_spans

NAME = "gretel_en"
HF_ID = "gretelai/gretel-pii-masking-en-v1"
REVISION = "e06eb1499ca8d54470f085021cd8e54f9efac7fd"
LICENCE = "Apache-2.0"
PUBLISH = "scores-and-ids"
SUITE = "pii"


def entity_spans(text: str, entities: list[dict]) -> list[dict]:
    """Every exact, non-overlapping occurrence of each entity string, one span per type."""
    spans = []
    for entity in entities:
        value, start = entity["entity"], 0
        while value and (start := text.find(value, start)) >= 0:
            spans.extend({"start": start, "end": start + len(value), "label": t} for t in entity["types"])
            start += len(value)
    return valid_spans(text, spans)


def load(n: int | None, seed: int) -> list[dict]:
    import pyarrow.parquet as pq

    table = pq.read_table(hf_file(HF_ID, "data/test-00000-of-00001.parquet", REVISION),
                          columns=["uid", "document_type", "entities", "text"])
    rows = ({"id": r["uid"], "lang": "en", "text": r["text"], "group": r["document_type"],
             "spans": entity_spans(r["text"], ast.literal_eval(r["entities"]))}
            for r in table.to_pylist())
    return sample_per_lang(rows, n, seed)
