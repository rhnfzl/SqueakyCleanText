"""RedactionBench test split: 200 long English docs (101 rewritten real public forms, 99 synthetic), untyped gold.

Spans carry only ``mandatory`` or ``contextual``, no entity type. ``mandatory`` maps to ``labels.UNTYPED``,
so only the type-agnostic measures are scored (masked recall/precision, character scores, leaks); the typed
strict/partial measures are not defined here (``scorer.metrics`` reports None). ``contextual`` is out of scope:
neither a miss nor a false positive. 49 source docs are "All rights reserved": aggregate scores only.
"""

import json

from benchmarks.datasets import hf_file, valid_spans

NAME = "redactionbench"
HF_ID = "RedactionBench/RedactionBench"
REVISION = "d45e9cec89bc49c69355e252fec29cc0229982f6"
LICENCE = "CC-BY-4.0 (49 source docs all rights reserved: aggregate scores only)"
PUBLISH = "scores-only"
SUITE = "long"


def load(n: int | None, seed: int) -> list[dict]:
    import pyarrow.parquet as pq

    table = pq.read_table(hf_file(HF_ID, "data/test-00000-of-00001.parquet", REVISION), columns=["raw_text", "spans"])
    records = []
    for index, row in enumerate(table.to_pylist()):
        spans = json.loads(row["spans"]) if isinstance(row["spans"], str) else row["spans"] or ()
        text = row["raw_text"]
        records.append({"id": str(index), "lang": "en", "text": text,
                        "spans": valid_spans(text, [{"start": s["start"], "end": s["end"], "label": s["label"]}
                                                    for s in spans])})
    # Long docs are few: keep all unless n caps them (first n by row, as TAB does).
    return records if n is None else records[:n]
