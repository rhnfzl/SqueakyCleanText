"""OpenPII 1.5M (ai4privacy): multilingual PII, 30 languages, validation split."""

import ast
import json

from benchmarks.datasets import hf_file, sample_per_lang, valid_spans

NAME = "openpii"
HF_ID = "ai4privacy/pii-masking-openpii-1.5m"
REVISION = "a785eb528e28be2693c3718a27e066970de5dadb"
LICENCE = "CC-BY-4.0 (Ai4Privacy / Ai Suisse SA)"
PUBLISH = "scores-and-ids"
SUITE = "pii"


def _records(path):
    with path.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            mask = row["privacy_mask"]
            mask = ast.literal_eval(mask) if isinstance(mask, str) else mask
            spans = [{"start": m["start"], "end": m["end"], "label": m["label"]} for m in mask]
            yield {"id": str(row["uid"]), "lang": row["language"], "text": row["source_text"],
                   "spans": valid_spans(row["source_text"], spans)}


def load(n: int | None, seed: int) -> list[dict]:
    return sample_per_lang(_records(hf_file(HF_ID, "data/validation.jsonl", REVISION)), n, seed)
