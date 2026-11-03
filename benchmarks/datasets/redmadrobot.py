"""red_mad_robot PII benchmark (test.csv): 2,841 Russian sentences, production logs with synthetic PII swapped in.

Gold is tokens + BIO tags. Tokens are found in the text in order; a token whose case was changed by the
set's normalisation (``www`` -> ``WWW``) is found case-insensitively. There is no ORG, DATE or IBAN gold.
"""

import csv
import json

from benchmarks.datasets import hf_file, sample_per_lang, valid_spans

NAME = "redmadrobot"
HF_ID = "redmadrobot-rnd/pii_benchmark"
REVISION = "f77ea831274daf980cc45c61a93c226be9d978d6"
LICENCE = "MIT (card tag; cite the repository)"
PUBLISH = "scores-only"  # SHORTLIST section 3: no per-example predictions for any round-2 set
SUITE = "pii"


def token_spans(text: str, tokens: list[str], tags: list[str]) -> list[dict]:
    """BIO tags on tokens -> char spans in *text*. A token not found ends the current span and is skipped."""
    lower = text.lower() if len(text.lower()) == len(text) else None
    spans, at, current = [], 0, None
    for token, tag in zip(tokens, tags, strict=True):
        start = text.find(token, at)
        if start < 0 and lower is not None:
            start = lower.find(token.lower(), at)
        if start < 0:
            current = None
            continue
        end = at = start + len(token)
        kind = tag[2:] if tag[:2] in ("B-", "I-") else None
        if kind and tag.startswith("I-") and current and current["label"] == kind:
            current["end"] = end
        elif kind:
            current = {"start": start, "end": end, "label": kind}
            spans.append(current)
        else:
            current = None
    return spans


def load(n: int | None, seed: int) -> list[dict]:
    with hf_file(HF_ID, "test.csv", REVISION).open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    records = ({"id": str(i), "lang": "ru", "text": r["text"],
                "spans": valid_spans(r["text"], token_spans(r["text"], json.loads(r["tokens"]), json.loads(r["ner_tags"])))}
               for i, r in enumerate(rows))
    return sample_per_lang(records, n, seed)
