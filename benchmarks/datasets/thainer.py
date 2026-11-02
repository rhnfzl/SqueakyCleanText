"""Thai NER v2.2 (PyThaiNLP): real Thai text, word tokens + BIO. Test split.

Words include the space tokens, so the text is the words joined with no separator.
The id is the row index at the pinned revision.
"""

import json

from benchmarks.datasets import bio_to_spans, hf_file, sample_per_lang

NAME = "thainer"
HF_ID = "pythainlp/thainer-corpus-v2.2"
REVISION = "e516176cf83d96526e407a7652205d6620837600"
LICENCE = "CC-BY-3.0 (HF tag)"
PUBLISH = "scores-and-ids"
SUITE = "pii"


def load(n: int | None, seed: int) -> list[dict]:
    import pyarrow.parquet as pq

    path = hf_file(HF_ID, "data/test-00000-of-00001.parquet", REVISION)
    names = json.loads(pq.read_schema(path).metadata[b"huggingface"])["info"]["features"]["ner"]["feature"]["names"]
    records = []
    for index, row in enumerate(pq.read_table(path).to_pylist()):
        text, spans = bio_to_spans(row["words"], [names[t] for t in row["ner"]], sep="")
        records.append({"id": str(index), "lang": "th", "text": text, "spans": spans})
    return sample_per_lang(records, n, seed)
