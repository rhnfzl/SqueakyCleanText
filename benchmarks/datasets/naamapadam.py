"""Naamapadam (AI4Bharat): human-labelled hi, bn, ta NER test sets, BIO tokens in zips.

The id is ``lang:row index`` at the pinned revision.
"""

import json
import zipfile

from benchmarks.datasets import bio_to_spans, hf_file, sample_per_lang

NAME = "naamapadam"
HF_ID = "ai4bharat/naamapadam"
REVISION = "9d4f21ac57d11ed4f9ea64854fdc9f5618e61acc"
LICENCE = "CC0-1.0 per card (loader script says CC-BY-NC-4.0)"
PUBLISH = "scores-and-ids"
SUITE = "pii"
LANGS = ("hi", "bn", "ta")


def _records():
    for lang in LANGS:
        with zipfile.ZipFile(hf_file(HF_ID, f"data/{lang}_IndicNER_v1.0.zip", REVISION)) as archive:
            lines = archive.read(f"{lang}_test.json").decode("utf-8").splitlines()
        for index, line in enumerate(lines):
            row = json.loads(line)
            text, spans = bio_to_spans(row["words"], row["ner"])
            yield {"id": f"{lang}:{index}", "lang": lang, "text": text, "spans": spans}


def load(n: int | None, seed: int) -> list[dict]:
    return sample_per_lang(_records(), n, seed)
