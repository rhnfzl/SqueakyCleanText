"""Greek clinical de-identification benchmark: 500 long synthetic discharge notes (Gemini 3.1 Pro), one split.

Gold: ``annotations/gold_standard.json`` (dual review, arbitrated; labels are its ``fine`` names), not the
superseded ``phi`` ``expect: redact`` pre-annotation. ``record["survive"]`` keeps the must-survive clinical
probes (drug names, lab values, headers; values without offsets) for the over-masking view: a probe survives
when its value still stands verbatim in the masked output (``scorer.survival``). Cite doi 10.5281/zenodo.22824895.
"""

import json

from benchmarks.datasets import hf_file, sample_per_lang, sha256_checked, valid_spans

NAME = "greek_deid"
HF_ID = "prorris/greek-deid-benchmark"
REVISION = "dd82ef67dd4d647c58a2dbba342c87203d65a13f"
CORPUS, GOLD = "data/deid_corpus_gemini500.jsonl", "annotations/gold_standard.json"
SHA256 = {CORPUS: "8faad9b64830636775ced18917d93ea01a9aae6c08a411a3329e10e19420c010",  # SHORTLIST section 3
          GOLD: "872e8b8abf3e8f8eaef7d3218fea007e99b5fdb7aeabdab51934c28d02098746"}
LICENCE = "CC-BY-4.0 (cite doi 10.5281/zenodo.22824895)"
PUBLISH = "scores-only"  # SHORTLIST section 3: no per-example predictions for any round-2 set
SUITE = "long"


def note_record(row: dict, gold: list[dict]) -> dict:
    spans = [{"start": s["start"], "end": s["end"], "label": s["fine"], "form": s.get("text")} for s in gold]
    return {"id": row["id"], "lang": "el", "text": row["text"], "spans": valid_spans(row["text"], spans),
            "survive": [p["value"] for p in row.get("phi") or () if p.get("expect") == "survive" and p.get("value")]}


def load(n: int | None, seed: int) -> list[dict]:
    path = {name: sha256_checked(hf_file(HF_ID, name, REVISION), SHA256.get(name)) for name in (CORPUS, GOLD)}
    notes = json.loads(path[GOLD].read_text(encoding="utf-8"))["notes"]
    with path[CORPUS].open(encoding="utf-8") as source:
        rows = [json.loads(line) for line in source]
    if {r["id"] for r in rows} != set(notes):  # a missing key would score a note as PII-free
        raise ValueError(f"gold ids do not match corpus ids ({len(rows)} notes, {len(notes)} gold entries)")
    return sample_per_lang((note_record(r, notes[r["id"]]) for r in rows), n, seed)
