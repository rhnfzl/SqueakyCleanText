"""KDPII v1 (Zenodo 10968609): human-written Korean chat sentences with writer-invented PII.

A record is one sentence (the set's own unit); ``group`` is its dialog. The set's own split is kept:
``test.json`` is ``test``, ``valid.json`` is ``dev`` (``record["part"]``). Cite Fei et al., IEEE Access 2024.
"""

import json

from benchmarks.datasets import md5_checked, sample_per_lang, url_file, valid_spans

NAME = "kdpii"
HF_ID = "zenodo:10968609"
REVISION = "08eebd98593c1bad23fbf29053b3f9d2"  # md5 of test.json; valid.json is pinned in FILES
LICENCE = "CC-BY-4.0 (cite Fei et al., IEEE Access 12, 2024)"
PUBLISH = "scores-only"  # SHORTLIST section 3: no per-example predictions for any round-2 set
SUITE = "pii"
FILES = {"test": ("test.json", REVISION), "dev": ("valid.json", "6f89923f873c840ca773d255d058150c")}


def sentence_record(row: dict, part: str) -> dict:
    spans = [{"start": p["begin"], "end": p["end"], "label": p["label"], "form": p.get("form")} for p in row["PII_set"] or ()]
    sid = row["sent_idx"]
    return {"id": f"{part}:{sid}", "lang": "ko", "text": row["sentence"], "group": sid.rsplit("_", 2)[0],
            "part": part, "spans": valid_spans(row["sentence"], spans)}


def load(n: int | None, seed: int) -> list[dict]:
    records = []
    for part, (name, md5) in FILES.items():
        path = md5_checked(url_file(f"https://zenodo.org/records/10968609/files/{name}?download=1",
                                    f"kdpii/10968609/{name}"), md5)
        records += [sentence_record(row, part) for row in json.loads(path.read_text(encoding="utf-8"))]
    return sample_per_lang(records, n, seed)
