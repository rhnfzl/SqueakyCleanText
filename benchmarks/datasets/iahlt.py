"""IAHLT Arabic NER: real news text, manual annotation, nested spans allowed. Test split."""

import json

from benchmarks.datasets import hf_file, sample_per_lang, valid_spans

NAME = "iahlt"
HF_ID = "HebArabNlpProject/arabic-iahlt-NER"
REVISION = "f3b43f40dcef6ea30c46aee455ede12f8333601a"
LICENCE = "CC-BY-4.0"
PUBLISH = "scores-and-ids"
SUITE = "pii"


def _records(path):
    with path.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            spans = [{"start": s, "end": e, "label": label} for s, e, label in row["label"]]
            yield {"id": str(row["id"]), "lang": "ar", "text": row["text"], "spans": valid_spans(row["text"], spans)}


def load(n: int | None, seed: int) -> list[dict]:
    return sample_per_lang(_records(hf_file(HF_ID, "iahlt_ner_test.jsonl", REVISION)), n, seed)
