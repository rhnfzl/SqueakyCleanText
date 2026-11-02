"""TAB, Text Anonymization Benchmark (ECHR court cases): long English docs, 1-10 annotators each. Test split.

Typed spans come from the first quality-checked annotator. ``annotators`` keeps every
annotator's mentions with DIRECT / QUASI / NO_MASK for the masking recall in ``score.py``.
"""

import json

from benchmarks.datasets import url_file, valid_spans

NAME = "tab"
HF_ID = "github:NorskRegnesentral/text-anonymization-benchmark"
REVISION = "558e09e26d6b36f5f78440074e6a233946d98bd9"
LICENCE = "MIT"
PUBLISH = "scores-and-ids"
SUITE = "long"
URL = f"https://raw.githubusercontent.com/NorskRegnesentral/text-anonymization-benchmark/{REVISION}/echr_test.json"


def load(n: int | None, seed: int) -> list[dict]:
    docs = json.loads(url_file(URL, f"tab/{REVISION}/echr_test.json").read_text(encoding="utf-8"))
    records = []
    for doc in docs:
        text = doc["text"]
        annotators = {
            name: valid_spans(text, [{"start": m["start_offset"], "end": m["end_offset"], "label": m["entity_type"],
                                      "identifier": m["identifier_type"]} for m in value["entity_mentions"]])
            for name, value in doc["annotations"].items()
        }
        first = (doc.get("quality_checked") or sorted(annotators))[0]
        records.append({"id": doc["doc_id"], "lang": "en", "text": text,
                        "spans": [{k: m[k] for k in ("start", "end", "label")} for m in annotators[first]],
                        "annotators": annotators})
    # Long docs are few: keep all unless n caps them (first n by doc id, no randomness needed).
    records.sort(key=lambda r: r["id"])
    return records if n is None else records[:n]
