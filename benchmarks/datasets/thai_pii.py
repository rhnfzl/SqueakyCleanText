"""Thai PII Synthetic: Thai regex PII (phone, Thai ID, email) with hard negatives. Test split."""

import json

from benchmarks.datasets import hf_file, sample_per_lang, valid_spans

NAME = "thai_pii"
HF_ID = "DuckerMaster/Thai-PII-Synthetic"
REVISION = "dcf1b695e4be330c97c200e0a86a7bd8696da4f1"
LICENCE = "CC-BY-4.0"
PUBLISH = "scores-and-ids"
SUITE = "pii"


def load(n: int | None, seed: int) -> list[dict]:
    rows = json.loads(hf_file(HF_ID, "data/test.json", REVISION).read_text(encoding="utf-8"))
    records = ({"id": r["id"], "lang": "th", "text": r["text"],
                "spans": valid_spans(r["text"], [{"start": e["start"], "end": e["end"], "label": e["label"]}
                                                 for e in r["entities"]])} for r in rows)
    return sample_per_lang(records, n, seed)
