"""Boundary adapters for structured data and retrieval pipelines.

``findings`` stay empty unless the cleaner collects them (``include_entities=True``
or a privacy policy). ``sensitive`` and ``signals`` are always filled.
"""

from dataclasses import dataclass, field, replace
from typing import Any, cast
from collections.abc import Hashable, Iterable, Mapping

from sct.presets import PRESETS
from sct.privacy import EntityFinding
from sct.sct import TextCleaner
from sct.signals import Signals, is_sensitive_result, merge_views


def _no_signals() -> Signals:
    return merge_views(())


@dataclass(frozen=True)
class PathFinding:
    """Entity finding with its path in structured input."""

    path: tuple[Hashable, ...]
    finding: EntityFinding


@dataclass(frozen=True)
class StructuredResult:
    """Processed structured value and path-aware findings."""

    value: Any
    findings: tuple[PathFinding, ...]
    sensitive: bool
    # merge_views over every string leaf, keys included.
    signals: Signals = field(default_factory=_no_signals)


@dataclass(frozen=True)
class RAGChunk:
    """Text chunk before privacy processing."""

    id: str
    text: str
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class ProcessedRAGChunk:
    """Retrieval chunk with cleaned text and sensitivity metadata.

    ``findings`` and ``metadata["sct"]`` cover the text only. ``metadata_findings``
    covers the metadata; ``sensitive`` and ``signals`` cover both.
    """

    id: str
    text: str
    language: str | None
    metadata: Mapping[str, Any]
    findings: tuple[EntityFinding, ...]
    sensitive: bool = False
    signals: Signals = field(default_factory=_no_signals)
    metadata_findings: tuple[PathFinding, ...] = ()


@dataclass(frozen=True)
class DataFrameResult:
    """Copied DataFrame-like value with cell-aware findings."""

    frame: Any
    findings: tuple[PathFinding, ...]
    sensitive: bool = False
    # merge_views over every processed cell.
    signals: Signals = field(default_factory=_no_signals)


class _DuplicateKeyError(ValueError):
    """Two JSON keys became the same text after cleaning."""


def _require_adapter_mode(cleaner: TextCleaner) -> None:
    if cleaner.cfg.replacement_mode == "reversible":
        raise ValueError(
            "Structured adapters do not support reversible mode. "
            "Process each text directly and retain its token map."
        )


def _pii_only_cleaner(cleaner: TextCleaner, ner: bool = True) -> TextCleaner:
    """Return a view of *cleaner* that runs only its PII steps, sharing its NER models."""
    # Everything the training preset turns off (folding, letter/number removal, casefold...).
    off: dict[str, Any] = {name: value for name, value in PRESETS['training'].items() if value is False}
    if not ner:
        off['check_ner_process'] = False
    return cleaner._with_config(replace(cleaner.cfg, **off))


def process_json(cleaner: TextCleaner, value: Any) -> StructuredResult:
    """Process every string in a JSON-compatible value.

    Keys follow ``cfg.json_key_cleaning``: ``'pii'`` (default) replaces regex
    PII only (emails, phones, URLs...), so short keys such as ``"k"`` and
    common words such as ``"message"`` survive (NER tags lone words out of
    context); ``'pii_ner'`` also runs NER on keys; ``'full'`` uses the whole
    config; ``'off'`` keeps keys as written.
    """
    _require_adapter_mode(cleaner)
    findings: list[PathFinding] = []
    views: list[Signals] = []
    key_mode = cleaner.cfg.json_key_cleaning
    key_cleaner: TextCleaner | None = None

    def visit(item: Any, path: tuple[Hashable, ...], text_cleaner: TextCleaner = cleaner) -> Any:
        nonlocal key_cleaner
        if isinstance(item, str):
            result = text_cleaner.process(item)
            views.append(result.signals)
            findings.extend(
                PathFinding(path, finding)
                for finding in result.metadata.get("findings", ())
            )
            return result.lm_text
        if isinstance(item, dict):
            processed = {}
            for index, (key, child) in enumerate(item.items()):
                # The raw key may itself be PII, so its path uses the index.
                if isinstance(key, str) and key_mode != 'off':
                    if key_mode in ('pii', 'pii_ner') and key_cleaner is None:
                        key_cleaner = _pii_only_cleaner(cleaner, ner=key_mode == 'pii_ner')
                    processed_key = visit(
                        key,
                        path + ("@key", index),
                        cleaner if key_mode == 'full' else cast(TextCleaner, key_cleaner),
                    )
                else:
                    processed_key = key
                if processed_key in processed:
                    raise _DuplicateKeyError(
                        "Cleaning JSON keys produced a duplicate key. "
                        "Pass a JSON value whose keys stay unique after cleaning."
                    )
                processed[processed_key] = visit(
                    child,
                    path + (processed_key,),
                )
            return processed
        if isinstance(item, list):
            return [
                visit(child, path + (index,))
                for index, child in enumerate(item)
            ]
        return item

    cleaned = visit(value, ())
    signals = merge_views(views)
    return StructuredResult(
        value=cleaned,
        findings=tuple(findings),
        sensitive=signals["sensitive"],
        signals=signals,
    )


def process_rag_chunks(
    cleaner: TextCleaner,
    chunks: Iterable[RAGChunk],
) -> list[ProcessedRAGChunk]:
    """Process retrieval chunks without changing their identifiers.

    Writes ``metadata["sensitive"]`` and ``metadata["sct"]``, replacing caller keys
    with those names.
    """
    _require_adapter_mode(cleaner)
    processed = []
    for chunk in chunks:
        result = cleaner.process(chunk.text)
        signals = result.signals
        processed_metadata = process_json(
            cleaner,
            dict(chunk.metadata),
        )
        metadata = processed_metadata.value
        metadata["sensitive"] = (
            is_sensitive_result(result)
            or processed_metadata.sensitive
        )
        metadata["sct"] = signals
        # A value with no text has no view to add: its empty view would only add warnings.
        if processed_metadata.signals["chars"]:
            signals = merge_views([signals, processed_metadata.signals])
        processed.append(ProcessedRAGChunk(
            id=chunk.id,
            text=result.lm_text,
            language=result.language,
            metadata=metadata,
            findings=tuple(result.metadata.get("findings", ())),
            sensitive=metadata["sensitive"],
            signals=signals,
            metadata_findings=processed_metadata.findings,
        ))
    return processed


def process_dataframe(
    cleaner: TextCleaner,
    frame: Any,
    *,
    columns: tuple[str, ...] | str,
) -> DataFrameResult:
    """Process selected string columns in a copied DataFrame-like object."""
    _require_adapter_mode(cleaner)
    if isinstance(columns, str):  # "text" is not ("t", "e", "x", "t")
        columns = (columns,)
    unknown = set(columns).difference(frame.columns)
    if unknown:
        raise KeyError(f"Unknown columns: {sorted(unknown)}")

    output = frame.copy(deep=True)
    findings: list[PathFinding] = []
    views: list[Signals] = []
    # Positional access: a duplicate row or column label makes .at return a Series.
    selected = [j for j, name in enumerate(output.columns) if name in columns]
    for i, row in enumerate(output.index):
        for j in selected:
            value = output.iat[i, j]
            if not isinstance(value, str):
                continue
            result = cleaner.process(value)
            output.iat[i, j] = result.lm_text
            views.append(result.signals)
            findings.extend(
                PathFinding((row, output.columns[j]), finding)
                for finding in result.metadata.get("findings", ())
            )
    signals = merge_views(views)
    return DataFrameResult(output, tuple(findings), signals["sensitive"], signals)
