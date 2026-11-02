"""WiLI-2018: paragraph-level language ID, 235 classes, test split. Id is the row index.

Labels are mostly ISO 639-3. Labels that are not a Lingua language (Alemannic ``als``,
``hbs``, ``zh-yue``, ...) are out-of-set and only feed the false-positive rate.
"""

import json

from benchmarks.datasets import hf_file, sample_per_lang

NAME = "wili"
HF_ID = "MartinThoma/wili_2018"
REVISION = "12c3a519f58673e4051a5d9db51c1fcd94a94c75"
LICENCE = "ODbL-1.0"
PUBLISH = "scores-and-ids"
SUITE = "lid"


def load(n: int | None, seed: int) -> list[dict]:
    import pyarrow.parquet as pq

    path = hf_file(HF_ID, "WiLI-2018 dataset/test-00000-of-00001.parquet", REVISION)
    names = json.loads(pq.read_schema(path).metadata[b"huggingface"])["info"]["features"]["label"]["names"]
    table = pq.read_table(path)
    rows = ({"id": str(i), "lang": names[label], "text": text, "spans": []}
            for i, (text, label) in enumerate(zip(table.column("sentence").to_pylist(),
                                                  table.column("label").to_pylist(), strict=True)))
    return sample_per_lang(rows, n, seed)
