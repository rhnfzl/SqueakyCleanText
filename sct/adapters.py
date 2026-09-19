"""Boundary adapters for structured data and retrieval pipelines."""

import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional

from sct.privacy import EntityFinding
from sct.sct import TextCleaner


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


def is_sensitive_result(
    cleaner: TextCleaner,
    source: str,
    result: Any,
) -> bool:
    """Return whether processing found or replaced sensitive content."""
    metadata = result.metadata or {}
    if metadata.get("findings"):
        return True
    if (
        cleaner.cfg.replacement_mode == "synthetic"
        and result.lm_text != source
    ):
        return True
    configured_markers = (
        cleaner.cfg.replace_with_url,
        cleaner.cfg.replace_with_html,
        cleaner.cfg.replace_with_email,
        cleaner.cfg.replace_with_years,
        cleaner.cfg.replace_with_dates,
        cleaner.cfg.replace_with_phone_numbers,
        cleaner.cfg.replace_with_numbers,
        cleaner.cfg.replace_with_currency_symbols,
    )
    if any(
        isinstance(marker, str)
        and marker
        and marker in result.lm_text
        and marker not in source
        for marker in configured_markers
    ):
        return True
    return bool(
        re.search(r"<[A-Z][A-Z0-9_]*>", result.lm_text)
        and not re.search(r"<[A-Z][A-Z0-9_]*>", source)
    )


def process_json(cleaner: TextCleaner, value: Any) -> StructuredResult:
    """Process every string in a JSON-compatible value."""
    _require_adapter_mode(cleaner)
    findings = []
    sensitive = False

    def visit(item: Any, path: tuple[str | int, ...]) -> Any:
        nonlocal sensitive
        if isinstance(item, str):
            result = cleaner.process(item)
            sensitive = sensitive or is_sensitive_result(
                cleaner,
                item,
                result,
            )
            if result.metadata:
                findings.extend(
                    PathFinding(path, finding)
                    for finding in result.metadata.get("findings", ())
                )
            return result.lm_text
        if isinstance(item, dict):
            processed = {}
            for key, child in item.items():
                processed_key = (
                    visit(key, path + ("@key", key))
                    if isinstance(key, str) else key
                )
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

    return StructuredResult(
        value=visit(value, ()),
        findings=tuple(findings),
        sensitive=sensitive,
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
            is_sensitive_result(
                cleaner,
                chunk.text,
                result,
            )
            or processed_metadata.sensitive
        )
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
    for row in output.index:
        for column in columns:
            value = output.at[row, column]
            if not isinstance(value, str):
                continue
            result = cleaner.process(value)
            output.at[row, column] = result.lm_text
            if result.metadata:
                findings.extend(
                    PathFinding((row, column), finding)
                    for finding in result.metadata.get("findings", ())
                )
    return DataFrameResult(output, tuple(findings))
