"""pl-pii-bench v1.0.0: the public synthetic Polish corpus (code generator), 7 lanes, no dev/test split.

Documents (``core``, ``identifiers``, ``address``, ``negative``, ``robustness``, ``pdf``) are one record each;
``inflection`` is one sentence per line. ``group`` keeps a ``robustness`` variant with its source document and
the case forms of one inflected name together. Entities marked ``protection: keep`` are relabelled
``<LABEL>:keep`` (out of scope: neither a miss nor a false positive). The authors ask not to train on it.
"""

import json

from benchmarks.datasets import hf_files, sample_per_lang, valid_spans

NAME = "pl_pii"
HF_ID = "pl-pii-bench/pl-pii-bench"
REVISION = "bef947d33bc37ec3f8f759ba251bf27208de239f"
LICENCE = "CC-BY-4.0 (data; harness Apache-2.0)"
PUBLISH = "scores-only"  # SHORTLIST section 3: no per-example predictions for any round-2 set
SUITE = "pii"


def entity_spans(text: str, entities: list[dict]) -> list[dict]:
    return valid_spans(text, [{"start": e["start"], "end": e["end"], "form": e.get("text"),
                               "label": e["label"] if e.get("protection", "protect") == "protect" else f"{e['label']}:keep"}
                              for e in entities])


def _records(root):
    for path in sorted((root / "data").rglob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        yield {"id": doc["doc_id"], "lang": "pl", "text": doc["text"], "group": doc["doc_id"].split("-robustness-")[0],
               "spans": entity_spans(doc["text"], doc.get("entities") or [])}
    for path in sorted((root / "data").rglob("*.jsonl")):
        with path.open(encoding="utf-8") as source:
            for index, line in enumerate(source):
                row = json.loads(line)
                yield {"id": f"{path.stem}:{index}", "lang": "pl", "text": row["text"],
                       "group": f"{path.stem}:{row.get('entity_id', index)}",
                       "spans": entity_spans(row["text"], [{**row, "text": None}])}  # its text is the sentence


def load(n: int | None, seed: int) -> list[dict]:
    return sample_per_lang(_records(hf_files(HF_ID, ["data/*.json", "data/*.jsonl"], REVISION)), n, seed)
