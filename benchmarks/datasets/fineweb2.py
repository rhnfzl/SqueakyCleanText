"""FineWeb-2 test splits: web text per language/script for speed and injected-noise cleaning.

One directory per script by default (FineWeb-2 has no English). ``lang`` is the
directory name; ids are FineWeb-2's own doc ids. Emails and IPs are already anonymised.
"""

import random

from benchmarks.datasets import hf_file

NAME = "fineweb2"
HF_ID = "HuggingFaceFW/fineweb-2"
REVISION = "af9c13333eb981300149d5ca60a8e9d659b276b9"
LICENCE = "ODC-By-1.0 (and CommonCrawl terms of use)"
PUBLISH = "scores-and-ids"
SUITE = "speed"
DIRS = ("nld_Latn", "rus_Cyrl", "ell_Grek", "arb_Arab", "heb_Hebr", "hin_Deva", "tha_Thai",
        "cmn_Hani", "jpn_Jpan", "kor_Hang")


def load(n: int | None, seed: int) -> list[dict]:
    import pyarrow.parquet as pq

    records = []
    for name in DIRS:
        table = pq.read_table(hf_file(HF_ID, f"data/{name}/test/000_00000.parquet", REVISION), columns=["id", "text"])
        rows = range(table.num_rows)
        picked = rows if n is None else sorted(random.Random(f"{seed}:{name}").sample(rows, min(n, len(rows))))  # noqa: S311
        ids, texts = table.column("id").to_pylist(), table.column("text").to_pylist()
        records += [{"id": ids[i], "lang": name, "text": texts[i], "spans": []} for i in picked]
    return records
