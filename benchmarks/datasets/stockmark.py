"""Stockmark Japanese NER (Wikipedia). No split: all 5,343 sentences are test. Id is ``row:curid``."""

import json

from benchmarks.datasets import hf_file, sample_per_lang, valid_spans

NAME = "stockmark"
HF_ID = "stockmark/ner-wikipedia-dataset"
REVISION = "d6c4d4821885ed656b9fbc7b2283dc7d165916d1"
LICENCE = "CC-BY-SA-3.0"
PUBLISH = "scores-and-ids"
SUITE = "pii"


def load(n: int | None, seed: int) -> list[dict]:
    rows = json.loads(hf_file(HF_ID, "ner.json", REVISION).read_text(encoding="utf-8"))
    records = ({"id": f"{index}:{r['curid']}", "lang": "ja", "text": r["text"],
                "spans": valid_spans(r["text"], [{"start": e["span"][0], "end": e["span"][1], "label": e["type"]}
                                                 for e in r["entities"]])}
               for index, r in enumerate(rows))
    return sample_per_lang(records, n, seed)
