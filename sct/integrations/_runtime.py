"""Shared adapter runtime: each adapter owns one cleaner per process.

The cleaner is built on first use and rebuilt when the process ID changes,
so a handle pickled to (or forked into) a worker never reuses the parent's
ONNX sessions. Pickling keeps only the config.
"""

import os
import threading
from dataclasses import replace
from typing import Any
from collections.abc import Iterable, Mapping, Sequence

from sct.config import ONNX_BACKENDS, TextCleanerConfig
from sct.presets import preset_config
from sct.score import SCORE_A_COLUMNS, score_record
from sct.sct import TextCleaner
from sct.signals import merge_views

OUTPUTS = ("overwrite", "new_column")

# One lock for first builds: threads sharing a handle build one cleaner, not one each.
_BUILD_LOCK = threading.Lock()
if hasattr(os, "register_at_fork"):
    # A child forked while another thread held the lock must not inherit it locked.
    os.register_at_fork(after_in_child=lambda: globals().update(_BUILD_LOCK=threading.Lock()))


def resolve_config(config: TextCleanerConfig | str | None) -> TextCleanerConfig:
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


def _pin_loaded_revisions(cleaner: TextCleaner | None, config: TextCleanerConfig) -> TextCleanerConfig:
    """Pin unpinned ONNX models to the commits *cleaner* loaded, so workers skip the Hub."""
    ner = getattr(cleaner, "GeneralNER", None)
    if ner is None or config.ner_backend not in ONNX_BACKENDS:
        return config
    loaded = {key: p["revision"] for key, p in ner.provenance.items() if p.get("revision")}
    pinned = config.ner_model_revisions or {}
    if not loaded.keys() - pinned.keys():
        return config
    return replace(config, ner_model_revisions={**loaded, **pinned})


class CleanerHandle:
    """Picklable owner of one ``TextCleaner`` per process."""

    def __init__(self, config: TextCleanerConfig | str | None = None):
        self.config = resolve_config(config)
        self._cleaner: TextCleaner | None = None
        self._pid: int | None = None

    def get(self) -> TextCleaner:
        pid = os.getpid()
        if self._cleaner is None or self._pid != pid:
            with _BUILD_LOCK:
                if self._cleaner is None or self._pid != pid:
                    # Forked child: reuse the parent's resolved commits (no network after fork).
                    self.config = _pin_loaded_revisions(self._cleaner, self.config)
                    self._cleaner = TextCleaner(self.config)
                    self._pid = pid
        return self._cleaner

    def __getstate__(self):
        return {"config": _pin_loaded_revisions(self._cleaner, self.config)}

    def __setstate__(self, state):
        self.config = state["config"]
        self._cleaner = None
        self._pid = None


def as_columns(columns: str | Iterable[str]) -> tuple[str, ...]:
    """One column name or a sequence of them, as a tuple ("text" is not ("t", "e", "x", "t"))."""
    return (columns,) if isinstance(columns, str) else tuple(columns)


def output_column(col: str, output: str) -> str:
    """The column that ``output`` writes the cleaned ``col`` to."""
    return col if output == "overwrite" else f"{col}_clean"


def _clean(handle: CleanerHandle, batch: Mapping[str, Sequence], columns, output: str):
    """Return (new batch, {column: ProcessResult-or-None per row})."""
    if output not in OUTPUTS:
        raise ValueError(f"output must be one of {OUTPUTS}, got {output!r}")
    columns = as_columns(columns)
    if not columns:
        raise ValueError("columns must name at least one column")
    missing = [c for c in columns if c not in batch]
    if missing:
        raise KeyError(f"columns not in batch: {missing}")
    cleaner = handle.get()
    out = dict(batch)
    per_column: dict[str, list] = {}
    for col in columns:
        values = list(batch[col])
        idx = [i for i, v in enumerate(values) if isinstance(v, str)]
        results: list = [None] * len(values)
        for i, result in zip(idx, cleaner.process_batch([values[i] for i in idx])):
            results[i] = result
        out[output_column(col, output)] = [
            v if r is None else r.lm_text for v, r in zip(values, results)
        ]
        per_column[col] = results
    return out, per_column


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
    out, per_column = _clean(handle, batch, columns, output)
    if score:
        first = as_columns(columns)[0]
        records = [
            None if r is None else score_record(r, original=t)
            for r, t in zip(per_column[first], batch[first])
        ]
        for col in SCORE_A_COLUMNS:
            out[col] = [None if rec is None else rec[col] for rec in records]
    return out


def clean_records(
    handle: CleanerHandle,
    records: Sequence[tuple[str | None, Mapping[str, Any]]],
    text_key: str,
    columns: Sequence[str],
    output: str,
) -> list[tuple[str | None, dict]]:
    """Clean ``(text, metadata)`` pairs of document objects.

    A column equal to ``text_key`` is the document text, any other is a metadata
    key (missing keys pass through). Metadata is copied; ``metadata["sct"]``
    holds the public signals merged over all listed columns.
    """
    columns = as_columns(columns)
    batch = {c: [t if c == text_key else m.get(c) for t, m in records] for c in columns}
    out, per_column = _clean(handle, batch, columns, output)
    cleaned = []
    for i, (text, meta) in enumerate(records):
        meta = dict(meta)
        for col in columns:
            if batch[col][i] is None:
                continue
            key = output_column(col, output)
            if key == text_key:
                text = out[key][i]
            else:
                meta[key] = out[key][i]
        views = [results[i].signals for results in per_column.values() if results[i] is not None]
        # An empty cell has no view to add: it would only add warnings.
        views = [view for view in views if view["chars"]] or views[:1]
        if views:
            meta["sct"] = views[0] if len(views) == 1 else merge_views(views)
        cleaned.append((text, meta))
    return cleaned
