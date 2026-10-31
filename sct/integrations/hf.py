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
        return clean_columns(self.handle, batch, self.columns, self.output, score=self.score)


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
