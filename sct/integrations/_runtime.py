"""Shared adapter runtime: each adapter owns one cleaner per process.

The cleaner is built on first use and rebuilt when the process ID changes,
so a handle pickled to (or forked into) a worker never reuses the parent's
ONNX sessions. Pickling keeps only the config.
"""

import os
from typing import Any, Mapping, Optional, Sequence, Union

from sct.config import TextCleanerConfig
from sct.presets import preset_config
from sct.sct import TextCleaner

OUTPUTS = ("overwrite", "new_column")


def resolve_config(config: Union[TextCleanerConfig, str, None]) -> TextCleanerConfig:
    """``None`` -> ``"training"`` preset, a str -> that preset, a config -> itself."""
    if config is None:
        config = "training"
    if isinstance(config, str):
        config = preset_config(config)
    if not isinstance(config, TextCleanerConfig):
        raise TypeError(f"config must be a TextCleanerConfig, a preset name or None, got {type(config)}")
    if config.replacement_mode == "reversible":
        raise ValueError(
            "Adapters do not support reversible mode. "
            "Process each text directly and retain its token map."
        )
    return config


class CleanerHandle:
    """Picklable owner of one ``TextCleaner`` per process."""

    def __init__(self, config: Union[TextCleanerConfig, str, None] = None):
        self.config = resolve_config(config)
        self._cleaner: Optional[TextCleaner] = None
        self._pid: Optional[int] = None

    def get(self) -> TextCleaner:
        # shortcut: no lock, two threads racing on first use both build; fine for process workers.
        pid = os.getpid()
        if self._cleaner is None or self._pid != pid:
            self._cleaner = TextCleaner(self.config)
            self._pid = pid
        return self._cleaner

    def __getstate__(self):
        return {"config": self.config}

    def __setstate__(self, state):
        self.config = state["config"]
        self._cleaner = None
        self._pid = None


def _clean(handle: CleanerHandle, batch: Mapping[str, Sequence], columns, output: str):
    """Return (new batch, ProcessResult-or-None per row of the first column)."""
    if output not in OUTPUTS:
        raise ValueError(f"output must be one of {OUTPUTS}, got {output!r}")
    columns = (columns,) if isinstance(columns, str) else tuple(columns)
    if not columns:
        raise ValueError("columns must name at least one column")
    missing = [c for c in columns if c not in batch]
    if missing:
        raise KeyError(f"columns not in batch: {missing}")
    cleaner = handle.get()
    out = dict(batch)
    first = None
    for col in columns:
        values = list(batch[col])
        idx = [i for i, v in enumerate(values) if isinstance(v, str)]
        results: list = [None] * len(values)
        for i, result in zip(idx, cleaner.process_batch([values[i] for i in idx])):
            results[i] = result
        out[col if output == "overwrite" else f"{col}_clean"] = [
            v if r is None else r.lm_text for v, r in zip(values, results)
        ]
        if first is None:
            first = results
    return out, first


def clean_columns(
    handle: CleanerHandle,
    batch: Mapping[str, Sequence],
    columns: Sequence[str] = ("text",),
    output: str = "overwrite",
    *,
    score: bool = False,
) -> dict[str, list]:
    """Clean ``columns`` of a column batch; non-str cells and other columns pass through.

    ``output="new_column"`` writes ``<col>_clean`` and keeps the raw text.
    ``score=True`` adds ``sct.score.SCORE_A_COLUMNS`` for the first column.
    """
    out, results = _clean(handle, batch, columns, output)
    if score:
        from sct.score import SCORE_A_COLUMNS, score_record
        first = columns if isinstance(columns, str) else columns[0]
        records = [
            None if r is None else score_record(r, original=t)
            for r, t in zip(results, batch[first])
        ]
        for col in SCORE_A_COLUMNS:
            out[col] = [None if rec is None else rec[col] for rec in records]
    return out


def clean_records(
    handle: CleanerHandle,
    records: Sequence[tuple[Optional[str], Mapping[str, Any]]],
    text_key: str,
    columns: Sequence[str],
    output: str,
) -> list[tuple[Optional[str], dict]]:
    """Clean ``(text, metadata)`` pairs of document objects.

    A column equal to ``text_key`` is the document text, any other is a metadata
    key (missing keys pass through). Metadata is copied; ``metadata["sct"]``
    holds the public signals of the first column.
    """
    columns = (columns,) if isinstance(columns, str) else tuple(columns)
    batch = {c: [t if c == text_key else m.get(c) for t, m in records] for c in columns}
    out, results = _clean(handle, batch, columns, output)
    cleaned = []
    for i, (text, meta) in enumerate(records):
        meta = dict(meta)
        for col in columns:
            if batch[col][i] is None:
                continue
            key = col if output == "overwrite" else f"{col}_clean"
            if key == text_key:
                text = out[key][i]
            else:
                meta[key] = out[key][i]
        if results[i] is not None:
            meta["sct"] = results[i].signals
        cleaned.append((text, meta))
    return cleaned
