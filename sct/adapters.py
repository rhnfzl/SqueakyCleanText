"""Boundary adapters for structured data and retrieval pipelines."""

from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Mapping, Optional

from sct.presets import PRESETS
from sct.privacy import EntityFinding
from sct.sct import TextCleaner
from sct.signals import merge_views


@dataclass(frozen=True)
class PathFinding:
    """Entity finding with its path in structured input."""

    path: tuple[str | int, ...]
    finding: EntityFinding


@dataclass(frozen=True)
class StructuredResult:
    """Processed structured value and path-aware findings."""

    value: Any
    findings: tuple[PathFinding, ...]
    sensitive: bool
    # merge_views over every string leaf, keys included.
    signals: dict = field(default_factory=dict)


@dataclass(frozen=True)
class RAGChunk:
    """Text chunk before privacy processing."""

    id: str
    text: str
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class ProcessedRAGChunk:
    """Retrieval chunk with cleaned text and sensitivity metadata."""

    id: str
    text: str
    language: Optional[str]
    metadata: Mapping[str, Any]
    findings: tuple[EntityFinding, ...]


@dataclass(frozen=True)
class DataFrameResult:
    """Copied DataFrame-like value with cell-aware findings."""

    frame: Any
    findings: tuple[PathFinding, ...]


def _require_adapter_mode(cleaner: TextCleaner) -> None:
    if cleaner.cfg.replacement_mode == "reversible":
        raise ValueError(
            "Structured adapters do not support reversible mode. "
            "Process each text directly and retain its token map."
        )


def is_sensitive_result(result: Any) -> bool:
    """Return whether processing found an entity or made a regex replacement."""
    metadata = result.metadata or {}
    return bool(
        metadata.get("regex_counts")
        or metadata.get("entity_counts")
        or metadata.get("findings")
    )


def _pii_only_cleaner(cleaner: TextCleaner, ner: bool = True) -> TextCleaner:
    """Return a view of *cleaner* that runs only its PII steps, sharing its NER models."""
    # Everything the training preset turns off (folding, letter/number removal, casefold...).
    off = {name: value for name, value in PRESETS['training'].items() if value is False}
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
    findings = []
    views = []
    key_mode = cleaner.cfg.json_key_cleaning
    key_cleaner = None

    def visit(item: Any, path: tuple[str | int, ...], text_cleaner: TextCleaner = cleaner) -> Any:
        nonlocal key_cleaner
        if isinstance(item, str):
            result = text_cleaner.process(item)
            views.append(result.signals)
            if result.metadata:
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
                        cleaner if key_mode == 'full' else key_cleaner,
                    )
                else:
                    processed_key = key
                if processed_key in processed:
                    raise ValueError(
                        "Cleaning JSON keys produced a duplicate key"
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
    """Process retrieval chunks without changing their identifiers."""
    _require_adapter_mode(cleaner)
    processed = []
    for chunk in chunks:
        result = cleaner.process(chunk.text)
        findings = (
            tuple(result.metadata.get("findings", ()))
            if result.metadata else ()
        )
        processed_metadata = process_json(
            cleaner,
            dict(chunk.metadata),
        )
        metadata = processed_metadata.value
        metadata["sensitive"] = (
            is_sensitive_result(result)
            or processed_metadata.sensitive
        )
        metadata["sct"] = result.signals
        processed.append(ProcessedRAGChunk(
            id=chunk.id,
            text=result.lm_text,
            language=result.language,
            metadata=metadata,
            findings=findings,
        ))
    return processed


def process_dataframe(
    cleaner: TextCleaner,
    frame: Any,
    *,
    columns: tuple[str, ...],
) -> DataFrameResult:
    """Process selected string columns in a copied DataFrame-like object."""
    _require_adapter_mode(cleaner)
    unknown = set(columns).difference(frame.columns)
    if unknown:
        raise KeyError(f"Unknown columns: {sorted(unknown)}")

    output = frame.copy(deep=True)
    findings = []
    # Positional access: a duplicate row or column label makes .at return a Series.
    selected = [j for j, name in enumerate(output.columns) if name in columns]
    for i, row in enumerate(output.index):
        for j in selected:
            value = output.iat[i, j]
            if not isinstance(value, str):
                continue
            result = cleaner.process(value)
            output.iat[i, j] = result.lm_text
            if result.metadata:
                findings.extend(
                    PathFinding((row, output.columns[j]), finding)
                    for finding in result.metadata.get("findings", ())
                )
    return DataFrameResult(output, tuple(findings))
