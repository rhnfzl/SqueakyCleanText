"""newmean-pii-dataset public test: 17,058 synthetic texts in 54 languages (Claude Sonnet/Opus 5.5), one split.

Double-tagged (blind re-tag by a second model). 40% of rows have no PII (clean docs for over-masking).
Licence CC BY-NC 4.0: non-commercial evaluation only, so aggregate scores only, no ids, no predictions.
No source split: dev/test come from ``compare.split``. SECRET and USERNAME have no SCT type.
"""

import json

from benchmarks.datasets import hf_file, sample_per_lang, valid_spans

NAME = "newmean"
HF_ID = "newmean-ai/pii-dataset"
REVISION = "062c374e34f9805b9344a0ef06c5385e616642ae"
LICENCE = "CC-BY-NC-4.0 (non-commercial; credit newmean.ai)"
PUBLISH = "scores-only"
SUITE = "pii"


def _records(path):
    with path.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            spans = [{"start": s["start"], "end": s["end"], "label": s["label"]} for s in row["spans"]]
            yield {"id": row["id"], "lang": row["lang"], "text": row["text"], "spans": valid_spans(row["text"], spans)}


def load(n: int | None, seed: int) -> list[dict]:
    return sample_per_lang(_records(hf_file(HF_ID, "data/test.jsonl", REVISION)), n, seed)
