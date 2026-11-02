"""Gretel synthetic PII finance multilingual: regex PII and long financial docs, test split.

Gold comes from an LLM + GLiNER, so it is noisy: precision is a lower bound.
"""

import json

from benchmarks.datasets import hf_file, sample_per_lang, valid_spans

NAME = "gretel"
HF_ID = "gretelai/synthetic_pii_finance_multilingual"
REVISION = "7b844d16738527a04264f50214cb426a4cea0897"
LICENCE = "Apache-2.0"
PUBLISH = "scores-and-ids"
SUITE = "pii"
# The language column says "France" for French.
LANGS = {"English": "en", "Spanish": "es", "Swedish": "sv", "German": "de", "Italian": "it",
         "Dutch": "nl", "France": "fr", "French": "fr"}


def load(n: int | None, seed: int) -> list[dict]:
    import pyarrow.parquet as pq

    table = pq.read_table(hf_file(HF_ID, "data/test-00000-of-00001.parquet", REVISION),
                          columns=["index", "language", "generated_text", "pii_spans"])
    rows = ({"id": str(r["index"]), "lang": LANGS[r["language"]], "text": r["generated_text"],
             "spans": valid_spans(r["generated_text"], json.loads(r["pii_spans"]))}
            for r in table.to_pylist())
    return sample_per_lang(rows, n, seed)
