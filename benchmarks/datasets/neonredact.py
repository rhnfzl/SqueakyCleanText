"""NeonRedact-TR Bench v1: 400 fictional Turkish documents (LLM-drafted, hand-checked), own dev/test split.

The set's split is kept (``record["part"]``): ``dev.jsonl`` (100) is ``dev``, ``test.jsonl`` (300) is ``test``.
The 7 labels the author does not score (SICIL_NO, RUHSAT_NO, ...) and the non-SCT ones (MESLEK, SAGLIK, ...)
are out of scope. Written by the author of a Turkish PII product (home-field risk noted on the card).
"""

import json

from benchmarks.datasets import hf_file, sample_per_lang, sha256_checked, valid_spans

NAME = "neonredact"
HF_ID = "neondijital/neonredact-tr-bench"
REVISION = "72e799fefb3506f815b88c4baf0dcc0477d1dfcb"
SHA256 = {"test.jsonl": "c8b37f6ef4c58da5ea60044ba3bcf13ee47dbe675adb41e55e023572c0644dd9",  # SHORTLIST section 3
          "dev.jsonl": "17910e59de0bf24c587a51349318f477578bf9010261c9762ae122b222c8e8d3"}
LICENCE = "CC-BY-4.0 (cite the card)"
PUBLISH = "scores-only"  # SHORTLIST section 3: no per-example predictions for any round-2 set
SUITE = "pii"


def _records():
    for part in ("dev", "test"):
        name = f"{part}.jsonl"
        with sha256_checked(hf_file(HF_ID, name, REVISION), SHA256.get(name)).open(encoding="utf-8") as source:
            for line in source:
                row = json.loads(line)
                spans = [{"start": e["start"], "end": e["end"], "label": e["label"], "form": e.get("text")}
                         for e in row["entities"]]
                yield {"id": f"{part}:{row['id']}", "lang": "tr", "text": row["text"], "part": part,
                       "spans": valid_spans(row["text"], spans)}


def load(n: int | None, seed: int) -> list[dict]:
    return sample_per_lang(_records(), n, seed)
