"""Tonic PrivacyBench v2.5: the 6 human-annotated workspaces (synthetic Tonic Fabricate workplace data, English).

Messages (email, Slack) and document pages are records. ``group`` keeps a Slack thread or all pages of one
document together. USERNAME has no SCT type. Cite the card's BibTeX; Tonic's scorer code is not reused.
"""

import json

from benchmarks.datasets import hf_file, sample_per_lang, valid_spans

NAME = "tonic"
HF_ID = "TonicAI/Privacy-Bench"
REVISION = "a79d9e80c5ee68f5dd79da8452e5eb1e1ba97a1e"
LICENCE = "CC-BY-4.0 (cite the card)"
PUBLISH = "scores-only"  # SHORTLIST section 3: no per-example predictions for any round-2 set
SUITE = "pii"
WORKSPACES = ("daniel_lincoln_financial_group", "isabella_chipotle", "megan_donovan_eli_lilly",
              "morgan_morgan_stanley", "nora_caterpillar", "renee_tyson_foods")


def row_record(workspace: str, kind: str, index: int, row: dict) -> dict:
    meta = row.get("meta") or {}
    unit = meta.get("filename") if kind == "doc" else meta.get("thread_ts") or meta.get("row_id")
    spans = [{"start": s["start"], "end": s["end"], "label": s["label"]} for s in row["ground_truth_spans"]]
    return {"id": f"{workspace}:{kind}:{index}", "lang": "en", "text": row["text"],
            "group": f"{workspace}:{kind}:{unit if unit is not None else index}",
            "spans": valid_spans(row["text"], spans)}


def _records():
    for workspace in WORKSPACES:
        for kind, suffix in (("msg", ""), ("doc", "_documents")):
            path = hf_file(HF_ID, f"human_annotations/{workspace}{suffix}.jsonl", REVISION)
            with path.open(encoding="utf-8") as source:
                for index, line in enumerate(source):
                    yield row_record(workspace, kind, index, json.loads(line))


def load(n: int | None, seed: int) -> list[dict]:
    return sample_per_lang(_records(), n, seed)
