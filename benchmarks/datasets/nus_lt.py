"""NUS-LT-PII corpus, ``validacija`` (HF test split): 123 real Lithuanian docs (science, news, government).

Licence NewGenLTU Open RAIL-D v1.0: evaluation allowed, but text, spans and per-example outputs are
derivatives with real personal data: aggregate scores only, no ids, delete prediction caches after scoring.
``MISC`` mixes e-mails, URLs and other strings: a gold MISC span with ``@`` becomes ``MISC:email``, one that
starts with a scheme or ``www.`` becomes ``MISC:url``, the rest stays out of scope. No train data is used.
"""

import json

from benchmarks.datasets import hf_file, sample_per_lang, valid_spans

NAME = "nus_lt"
HF_ID = "VytautoDidziojoUniversitetas/NUS-LT-PII-corpus"
REVISION = "3097156f9841c41e071aeb2f3a7448b2455eb29c"
LICENCE = "NewGenLTU Open RAIL-D v1.0 (aggregate scores only)"
PUBLISH = "scores-only"
DELETE_PREDICTIONS = True  # licence: no prediction caches kept after scoring (runner)
SUITE = "pii"


def gold_label(text: str, span: dict) -> str:
    if span["label"] != "MISC":
        return span["label"]
    value = text[span["start"]:span["end"]].strip().lower()
    if "@" in value:
        return "MISC:email"
    return "MISC:url" if value.startswith(("http://", "https://", "www.")) else "MISC"


def _records(path):
    with path.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            text = row["text"]
            spans = [{"start": s["start"], "end": s["end"], "label": gold_label(text, s)} for s in row["label"]]
            yield {"id": row["id"], "lang": "lt", "text": text, "spans": valid_spans(text, spans)}


def load(n: int | None, seed: int) -> list[dict]:
    return sample_per_lang(_records(hf_file(HF_ID, "validacija/data.jsonl", REVISION)), n, seed)
