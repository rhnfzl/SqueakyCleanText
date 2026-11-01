"""Hugging Face ``datasets`` adapter (``pip install squeakycleantext[datasets]``)."""

from typing import Any, Optional, Sequence, Union

import sct
from sct.config import TextCleanerConfig
from sct.integrations._runtime import CleanerHandle, as_columns, clean_columns


class SCTMap:
    """Picklable batched map function; pickles only its config, never the cleaner."""

    def __init__(
        self,
        columns: Sequence[str] = ("text",),
        output: str = "overwrite",
        config: Union[TextCleanerConfig, str, None] = None,
        score: bool = False,
    ):
        self.columns = as_columns(columns)
        self.output = output
        self.score = score
        self.handle = CleanerHandle(config)

    def __call__(self, batch: dict) -> dict:
        out = clean_columns(self.handle, batch, self.columns, self.output, score=self.score)
        # HF merges the returned columns back after remove_columns, so return only the
        # columns SCT wrote: pass-through columns would resurrect dropped raw text, and
        # dropping a written column would keep a stale same-named input column.
        produced = [c if self.output == "overwrite" else f"{c}_clean" for c in self.columns]
        if self.score:
            from sct.score import SCORE_A_COLUMNS
            produced += SCORE_A_COLUMNS
        return {k: out[k] for k in produced if k in out}


def _output_features(features: Any, fn: SCTMap, remove_columns: Any) -> Any:
    from datasets import Features, Sequence as HFSequence, Value
    removed = {remove_columns} if isinstance(remove_columns, str) else set(remove_columns or ())
    out = Features({k: v for k, v in features.items() if k not in removed})
    for col in fn.columns:
        # Non-str cells pass through, so a written column keeps its source type.
        out[col if fn.output == "overwrite" else f"{col}_clean"] = features[col]
    if fn.score:
        out.update({
            "sct_language": Value("string"), "sct_language_score": Value("float64"),
            "sct_pii_count": Value("int64"), "sct_pii_share": Value("float64"),
            "sct_pii_types": HFSequence(Value("string")), "sct_warnings": HFSequence(Value("string")),
        })
    return out


def clean_dataset(
    ds: Any,
    *,
    columns: Sequence[str] = ("text",),
    output: str = "overwrite",
    config: Union[TextCleanerConfig, str, None] = None,
    batch_size: int = 64,
    num_proc: Optional[int] = None,
    score: bool = False,
    **map_kwargs: Any,
) -> Any:
    """Return ``ds`` with ``columns`` cleaned (``Dataset`` or ``IterableDataset``).

    Run it under ``if __name__ == "__main__":`` when ``num_proc`` > 1.
    """
    if num_proc is not None:
        map_kwargs["num_proc"] = num_proc
    fn = SCTMap(columns, output, config, score)
    if getattr(ds, "features", None) is not None and "features" not in map_kwargs:
        # Declare the written columns, or datasets types a first batch of None or []
        # as null and fails on the next batch.
        map_kwargs["features"] = _output_features(ds.features, fn, map_kwargs.get("remove_columns"))
    if getattr(ds, "_fingerprint", None) and "new_fingerprint" not in map_kwargs:
        # datasets hashes frozensets in pickle order, which changes per process
        # (PYTHONHASHSEED), so the cache would never hit; hash a sorted view instead.
        from datasets.fingerprint import Hasher
        state = {
            k: sorted(v) if isinstance(v, (set, frozenset)) else v
            for k, v in fn.handle.config.__getstate__()[0].items()
        }
        map_kwargs["new_fingerprint"] = Hasher.hash([
            ds._fingerprint, sct.__version__, fn.columns, output, score, batch_size,
            sorted(state.items()), sorted(map_kwargs.items()),
        ])
    return ds.map(fn, batched=True, batch_size=batch_size, **map_kwargs)
