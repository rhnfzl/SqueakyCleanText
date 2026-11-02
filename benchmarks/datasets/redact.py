"""REDACT PII benchmark: locked 1,000-record split, 25 language variants. Gated (auto).

Terms allow benchmarking but limit redistribution: publish aggregate scores only.
Entities whose ``entity_string`` differs from ``text[start:end]`` are dropped.
"""

import ast
import json

from benchmarks.datasets import hf_file, sample_per_lang, valid_spans

NAME = "redact"
HF_ID = "guneeshv/REDACT-PII-Benchmark"
REVISION = "1d7fef611cbaad42ba19666e7d7238ed45b264dd"
LICENCE = "REDACT Dataset Terms v1.0 (benchmarking allowed; scores only)"
PUBLISH = "scores-only"
SUITE = "pii"


def _value(raw):
    return ast.literal_eval(raw) if isinstance(raw, str) else raw


def load(n: int | None, seed: int) -> list[dict]:
    rows = json.loads(hf_file(HF_ID, "pii_benchmark_sample1000.json", REVISION).read_text(encoding="utf-8"))
    records = []
    for index, row in enumerate(rows):  # record_id is not unique: prefix the row index
        text = row["text"]
        spans = [{"start": e["start"], "end": e["end"], "label": e["entity_type"]} for e in _value(row["entities"])
                 if text[e["start"]:e["end"]] == e["entity_string"]]
        records.append({"id": f"{index}:{row['record_id']}", "lang": _value(row["axes"])["language"].lower(),
                        "text": text, "spans": valid_spans(text, spans)})
    return sample_per_lang(records, n, seed)
