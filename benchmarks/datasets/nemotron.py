"""Nemotron-PII (NVIDIA): English PII and PHI, structured and unstructured docs, test split.

Each uid appears twice (us and intl locale, different text), so the id is ``uid:locale``.
Spans are scored by offsets (1.5% have span text != text[start:end]).
"""

import ast

from benchmarks.datasets import hf_file, sample_per_lang, valid_spans

NAME = "nemotron"
HF_ID = "nvidia/Nemotron-PII"
REVISION = "b70ffaf5ff39e079776134c5bf4381f00a9fd1ed"
LICENCE = "CC-BY-4.0"
PUBLISH = "scores-and-ids"
SUITE = "pii"


def load(n: int | None, seed: int) -> list[dict]:
    import pyarrow.parquet as pq

    table = pq.read_table(hf_file(HF_ID, "data/test-00000-of-00001.parquet", REVISION),
                          columns=["uid", "locale", "document_type", "text", "spans"])
    rows = ({"id": f"{r['uid']}:{r['locale']}", "lang": "en", "text": r["text"], "group": r["document_type"],
             "spans": valid_spans(r["text"], [{"start": s["start"], "end": s["end"], "label": s["label"]}
                                              for s in ast.literal_eval(r["spans"])])}
            for r in table.to_pylist())
    return sample_per_lang(rows, n, seed)
