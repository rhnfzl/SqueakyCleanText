"""Hugging Face ``datasets`` adapter (``pip install squeakycleantext[datasets]``)."""

from typing import Any
from collections.abc import Sequence
from uuid import uuid4

import sct
from sct.config import TextCleanerConfig
from sct.schema import FieldSchema
from sct.findings import DetectionContext
from sct.detectors import DetectorRegistry
from collections.abc import Callable
from sct.integrations._runtime import CleanerHandle, as_columns, clean_columns, output_column
from sct.score import SCORE_A_COLUMNS, SCORE_A_TYPES


class SCTMap:
    """Picklable batched map function; pickles only its config, never the cleaner."""

    def __init__(
        self,
        columns: Sequence[str] = ("text",),
        output: str = "overwrite",
        config: TextCleanerConfig | str | None = None,
        score: bool = False,
        *, schema: FieldSchema | None = None, context: DetectionContext | None = None,
        registry_factory: Callable[[TextCleanerConfig], DetectorRegistry] | None = None,
    ):
        self.columns = as_columns(columns)
        self.output = output
        self.score = score
        self.schema, self.context = schema, context
        self.handle = CleanerHandle(config, registry_factory=registry_factory)

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        out = clean_columns(self.handle, batch, self.columns, self.output,
                            score=self.score, schema=self.schema, context=self.context)
        # HF merges the returned columns back after remove_columns, so return only the
        # columns SCT wrote: pass-through columns would resurrect dropped raw text, and
        # dropping a written column would keep a stale same-named input column.
        produced = [output_column(c, self.output) for c in self.columns]
        if self.score:
            produced += SCORE_A_COLUMNS
        return {k: out[k] for k in produced if k in out}


def _output_features(features: Any, fn: SCTMap, remove_columns: Any) -> Any:
    from datasets import Features, Sequence as HFSequence, Value
    removed = {remove_columns} if isinstance(remove_columns, str) else set(remove_columns or ())
    out = Features({k: v for k, v in features.items() if k not in removed})
    for col in fn.columns:
        # Non-str cells pass through, so a written column keeps its source type.
        rule = fn.schema.rule_at((col,)) if fn.schema is not None else None
        out[output_column(col, fn.output)] = Value('string') if rule and rule.mode == 'protect_value' else features[col]
    if fn.score:
        out.update({
            name: HFSequence(Value(kind[0])) if isinstance(kind, list) else Value(kind)
            for name, kind in SCORE_A_TYPES.items()
        })
    return out


def clean_dataset(
    ds: Any,
    *,
    columns: Sequence[str] = ("text",),
    output: str = "overwrite",
    config: TextCleanerConfig | str | None = None,
    batch_size: int = 64,
    num_proc: int | None = None,
    score: bool = False,
    schema: FieldSchema | None = None, context: DetectionContext | None = None,
    registry_factory: Callable[[TextCleanerConfig], DetectorRegistry] | None = None,
    **map_kwargs: Any,
) -> Any:
    """Return ``ds`` with ``columns`` cleaned (``Dataset`` or ``IterableDataset``).

    Run it under ``if __name__ == "__main__":`` when ``num_proc`` > 1.
    Custom registries bypass cache reuse unless ``new_fingerprint`` supplies a
    caller-owned identity for the complete input, configuration, provider and model behavior.
    """
    if num_proc is not None:
        map_kwargs["num_proc"] = num_proc
    fn = SCTMap(columns, output, config, score, schema=schema, context=context, registry_factory=registry_factory)
    if getattr(ds, "features", None) is not None and "features" not in map_kwargs:
        # Declare the written columns, or datasets types a first batch of None or []
        # as null and fails on the next batch.
        map_kwargs["features"] = _output_features(ds.features, fn, map_kwargs.get("remove_columns"))
    if getattr(ds, "_fingerprint", None) and registry_factory is not None and map_kwargs.get("new_fingerprint") is None:
        # Factory names cannot identify provider behavior; fresh output also invalidates downstream caches.
        map_kwargs["new_fingerprint"] = uuid4().hex
        map_kwargs["load_from_cache_file"] = False
    if getattr(ds, "_fingerprint", None) and "new_fingerprint" not in map_kwargs:
        # datasets hashes frozensets in pickle order, which changes per process
        # (PYTHONHASHSEED), so the cache would never hit; hash a sorted view instead.
        from datasets.fingerprint import Hasher
        state = {
            k: sorted(v) if isinstance(v, (set, frozenset)) else v
            for k, v in fn.handle.config._plain_state().items()
        }
        map_kwargs["new_fingerprint"] = Hasher.hash([
            ds._fingerprint, sct.__version__, fn.columns, output, score, batch_size,
            sorted(state.items()), sorted(map_kwargs.items()), schema, context, fn.handle._registry_factory_reference,
        ])
    return ds.map(fn, batched=True, batch_size=batch_size, **map_kwargs)
