"""Argilla PII (argilla/textcat-tokencat-pii-per-domain): 2,096 English records, one split.

The "Argilla PII" set on the nvidia/gliner-PII card (it links only the Argilla org; this is
the org's only PII dataset). It is ``ai4privacy/pii-masking-200k`` train[:1%] loaded into
Argilla; spans are the source ``privacy_mask`` (synthetic, status "pending", not reviewed).
No licence on the repo; the upstream Ai4Privacy licence forbids redistributing derivatives,
so publish aggregate scores only, no ids. No source split: dev/test come from ``compare.split``.
"""

from benchmarks.datasets import hf_file, sample_per_lang, valid_spans

NAME = "argilla_pii"
HF_ID = "argilla/textcat-tokencat-pii-per-domain"
REVISION = "683954a23877151a1d05e7515d5e6cf87203ffd2"
LICENCE = "none on repo; source ai4privacy/pii-masking-200k (Ai4Privacy licence, no redistribution)"
PUBLISH = "scores-only"
SUITE = "pii"


def load(n: int | None, seed: int) -> list[dict]:
    import pyarrow.parquet as pq

    table = pq.read_table(hf_file(HF_ID, "data/train-00000-of-00001.parquet", REVISION),
                          columns=["id", "language", "source-text", "pii.suggestion"])
    rows = ({"id": r["id"], "lang": r["language"], "text": r["source-text"],
             "spans": valid_spans(r["source-text"], [{"start": s["start"], "end": s["end"], "label": s["label"]}
                                                     for s in r["pii.suggestion"] or ()])}
            for r in table.to_pylist())
    return sample_per_lang(rows, n, seed)
