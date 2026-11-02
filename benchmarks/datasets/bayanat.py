"""Bayanat Arabic PII eval: template-based Arabic regex PII (phones, IBANs, national ids). Test split.

The ``lang`` is the Arabic variety (msa, gulf, egyptian, mixed_en).
"""

import json

from benchmarks.datasets import hf_file, sample_per_lang, valid_spans

NAME = "bayanat"
HF_ID = "Bayanat-Labs/arabic-pii-eval"
REVISION = "c59d164dc2ed1f0e1030edb9af748fd3eccb3715"
LICENCE = "CC-BY-4.0"
PUBLISH = "scores-and-ids"
SUITE = "pii"


def _records(path):
    with path.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            spans = [{"start": s["start"], "end": s["end"], "label": s["entity_type"]} for s in row["spans"]]
            yield {"id": row["id"], "lang": f"ar-{row['variety']}", "text": row["text"],
                   "spans": valid_spans(row["text"], spans)}


def load(n: int | None, seed: int) -> list[dict]:
    return sample_per_lang(_records(hf_file(HF_ID, "jsonl/test.jsonl", REVISION)), n, seed)
