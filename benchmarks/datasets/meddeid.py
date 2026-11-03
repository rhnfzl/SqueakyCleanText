"""MedDeID English + Dutch synthetic clinical benchmarks (GPT-written on Synthea, human-checked): 300 + 300 docs.

The cards allow final evaluation only (no tuning, rule development or model selection), so every record is
``part="test"``: the dev split is empty. ``lang`` is the locale (``en-gb``, ``en-us``; ``nl`` when a row names none).
Gold labels are the span ``category``, refined from the set's own structure: ``Age_Birthdate`` is a birth date
(``Age_Birthdate:date``) when its subannotations hold a day, month or year, else a bare age (no SCT type);
``Contactdetails`` is ``:email`` when it holds ``@``, else ``:phone``. Profession has no SCT type.
"""

import json

from benchmarks.datasets import hf_file, sample_per_lang, sha256_checked, valid_spans

NAME = "meddeid"
HF_ID = "stighellemans/meddeid-english-synthetic-benchmark + stighellemans/meddeid-dutch-synthetic-benchmark"
REPOS = {"en": ("stighellemans/meddeid-english-synthetic-benchmark", "288c5718e5112e75c857ddf511a30e6eb6b32cdb"),
         "nl": ("stighellemans/meddeid-dutch-synthetic-benchmark", "45c601bccaf5e0ba5c5fabf55384a9b4c5b5346a")}
REVISION = "288c5718e5112e75c857ddf511a30e6eb6b32cdb+45c601bccaf5e0ba5c5fabf55384a9b4c5b5346a"
SHA256 = {"en": "c6089f2e36cbca90571941ce3a7f29954aea14e1c6487cb950c157999589be7e",  # data/test.jsonl (SHORTLIST)
          "nl": "f1fc1a0447d1c447d742a1d32c1bb18393840665cc44d591eccc95e8cf3f5d01"}
LICENCE = "CC-BY-4.0 (final evaluation only; cite Zenodo 22689857/22689856 + arXiv 2609.10049)"
PUBLISH = "scores-only"  # SHORTLIST section 3: no per-example predictions for any round-2 set
SUITE = "pii"
DATE_PARTS = frozenset({"day", "month", "year"})


def gold_label(span: dict) -> str:
    category = span["category"]
    if category == "Age_Birthdate":
        parts = {s["category"] for s in span.get("subannotations") or ()}
        return "Age_Birthdate:date" if parts & DATE_PARTS else "Age_Birthdate:age"
    if category == "Contactdetails":
        return "Contactdetails:email" if "@" in span["text"] else "Contactdetails:phone"
    return category


def doc_record(row: dict, fallback_lang: str) -> dict:
    meta = json.loads(row.get("metadata_json") or "{}")
    lang = str(meta.get("lang") or fallback_lang).lower().replace("_", "-")
    spans = [{"start": s["begin"], "end": s["end"], "label": gold_label(s), "form": s.get("text")} for s in row["spans"]]
    return {"id": f"{lang}:{row['document_id']}", "lang": lang, "text": row["text"], "part": "test",
            "spans": valid_spans(row["text"], spans)}


def _records():
    for code, (repo, revision) in REPOS.items():
        with sha256_checked(hf_file(repo, "data/test.jsonl", revision), SHA256.get(code)).open(encoding="utf-8") as source:
            for line in source:
                yield doc_record(json.loads(line), code)


def load(n: int | None, seed: int) -> list[dict]:
    return sample_per_lang(_records(), n, seed)
