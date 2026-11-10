"""Boundary adapters for structured data and retrieval pipelines.

``findings`` stay empty unless the cleaner collects them (``include_entities=True``
or a privacy policy). ``sensitive`` and ``signals`` are always filled.
"""

from dataclasses import dataclass, field, replace
from typing import Any, cast
from collections.abc import Hashable, Iterable, Mapping

from sct.config import require_adapter_mode
from sct.detectors import DetectorSpec
from sct.findings import DetectionContext
from sct.schema import FieldSchema, process_value, validate_options
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
class PathFindingV2:
    """Unified finding with a positional output reference, never caller keys or labels."""

    path: tuple[Hashable, ...] = field(repr=False)
    finding: dict[str, Any]


@dataclass(frozen=True)
class StructuredResult:
    """Processed structured value and path-aware findings."""

    value: Any
    findings: tuple[PathFinding, ...]
    sensitive: bool
    # merge_views over every string leaf, keys included.
    signals: Signals = field(default_factory=_no_signals)
    findings_v2: tuple[PathFindingV2, ...] = field(default=(), repr=False)


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
    findings_v2: tuple[dict[str, Any], ...] = ()
    metadata_findings_v2: tuple[PathFindingV2, ...] = ()


@dataclass(frozen=True)
class DataFrameResult:
    """Copied DataFrame-like value with cell-aware findings."""

    frame: Any
    findings: tuple[PathFinding, ...]
    sensitive: bool = False
    # merge_views over every processed cell.
    signals: Signals = field(default_factory=_no_signals)
    findings_v2: tuple[PathFindingV2, ...] = ()


class _DuplicateKeyError(ValueError):
    """Two JSON keys became the same text after cleaning."""


def _pii_only_cleaner(cleaner: TextCleaner, ner: bool = True) -> TextCleaner:
    """Return a view of *cleaner* that runs only its PII steps, sharing its NER models."""
    # Everything the training preset turns off (folding, letter/number removal, casefold...).
    off: dict[str, Any] = {name: value for name, value in PRESETS['training'].items() if value is False}
    if not ner:
        off['check_ner_process'] = False
    if cleaner.cfg.pii_processing == 'unified':
        selected = (cleaner.cfg.routing_fast_specs + cleaner.cfg.routing_full_specs) or cleaner.cfg.detector_specs
        if cleaner.cfg.routing_fast_specs:
            # Derived key cleanup changes the calibrated configuration, so run the applicable providers.
            off.update(routing_policy='always_run', routing_fast_specs=(), routing_full_specs=(),
                       routing_entity_types=(), routing_calibration=None, detector_specs=selected)
        if not ner:
            off['detector_specs'] = (DetectorSpec('builtin_regex'),)
        elif selected and not any(spec.name == 'builtin_regex' for spec in selected):
            off['detector_specs'] = (DetectorSpec('builtin_regex'), *selected)
    return cleaner._with_config(replace(cleaner.cfg, **off))


def process_json(cleaner: TextCleaner, value: Any, *, schema: FieldSchema | None = None,
                 context: DetectionContext | None = None) -> StructuredResult:
    """Process every string in a JSON-compatible value.

    Tuples come back as lists and any ``Mapping`` as a dict. A tuple key stays a
    tuple, with the strings inside it cleaned as keys, since a list cannot be a key.

    Keys follow ``cfg.json_key_cleaning``: ``'pii'`` (default) replaces regex
    PII only (emails, phones, URLs...), so short keys such as ``"k"`` and
    common words such as ``"message"`` survive (NER tags lone words out of
    context); ``'pii_ner'`` also runs NER on keys; ``'full'`` uses the whole
    config; ``'off'`` keeps keys as written.
    """
    require_adapter_mode(cleaner.cfg)
    validate_options(cleaner, schema, context)
    findings: list[PathFinding] = []
    views: list[Signals] = []
    findings_v2: list[PathFindingV2] = []
    key_mode = cleaner.cfg.json_key_cleaning
    key_cleaner: TextCleaner | None = None

    def clean_key(key: Hashable, path: tuple[Hashable, ...], value_path: tuple[Hashable, ...]) -> Any:
        nonlocal key_cleaner
        if isinstance(key, tuple):
            return tuple(clean_key(part, path, value_path) for part in key)
        if not isinstance(key, str) or key_mode == 'off':
            return key
        if key_mode in ('pii', 'pii_ner') and key_cleaner is None:
            key_cleaner = _pii_only_cleaner(cleaner, ner=key_mode == 'pii_ner')
        return visit(key, path, cleaner if key_mode == 'full' else cast(TextCleaner, key_cleaner),
                     is_key=True, value_path=value_path)

    def visit(item: Any, path: tuple[Hashable, ...], text_cleaner: TextCleaner = cleaner,
              *, is_key: bool = False, schema_path: tuple[Hashable, ...] = (), value_path: tuple[Hashable, ...] = ()) -> Any:
        if not is_key and schema is not None and schema.rule_at(schema_path) is not None:
            selected_result = process_value(text_cleaner, item, schema_path, schema=schema, context=context)
            if selected_result is None:
                return item
            views.append(selected_result.signals)
            findings_v2.extend(PathFindingV2(value_path, dict(finding))
                               for finding in selected_result.metadata.get('findings_v2', ()))
            findings.extend(PathFinding(path, finding) for finding in selected_result.metadata.get("findings", ()))
            return selected_result.lm_text
        if isinstance(item, str):
            result = (text_cleaner.process(item) if is_key else
                      process_value(text_cleaner, item, schema_path, schema=schema, context=context))
            views.append(result.signals)
            findings_v2.extend(PathFindingV2(value_path, dict(finding))
                               for finding in result.metadata.get('findings_v2', ()))
            findings.extend(
                PathFinding(path, finding)
                for finding in result.metadata.get("findings", ())
            )
            return result.lm_text
        if isinstance(item, Mapping):
            processed = {}
            for index, (key, child) in enumerate(item.items()):
                # The raw key may itself be PII, so its path uses the index.
                processed_key = clean_key(key, path + ("@key", index), value_path + ("@key", index))
                if processed_key in processed:
                    raise _DuplicateKeyError(
                        "Cleaning JSON keys produced a duplicate key. "
                        "Pass a JSON value whose keys stay unique after cleaning."
                    )
                processed[processed_key] = visit(
                    child,
                    path + (processed_key,), schema_path=schema_path + (key,), value_path=value_path + ("@value", index),
                )
            return processed
        if isinstance(item, (list, tuple)):
            return [
                visit(child, path + (index,), schema_path=schema_path + (index,), value_path=value_path + (index,))
                for index, child in enumerate(item)
            ]
        return item

    cleaned = visit(value, ())
    signals = (cleaner.process('').signals if not views and cleaner.cfg.pii_processing == 'unified'
               else merge_views(views))
    return StructuredResult(
        value=cleaned,
        findings=tuple(findings),
        sensitive=signals["sensitive"],
        signals=signals,
        findings_v2=tuple(findings_v2),
    )


def process_rag_chunks(
    cleaner: TextCleaner,
    chunks: Iterable[RAGChunk],
    *, schema: FieldSchema | None = None, context: DetectionContext | None = None,
) -> list[ProcessedRAGChunk]:
    """Process retrieval chunks without changing their identifiers.

    Writes ``metadata["sensitive"]`` and ``metadata["sct"]``, replacing caller keys
    with those names.
    """
    require_adapter_mode(cleaner.cfg)
    validate_options(cleaner, schema, context)
    if schema is not None and any(not rule.path or rule.path[0] not in ('text', 'metadata')
                                  or (rule.path[0] == 'text' and len(rule.path) != 1) for rule in schema.rules):
        raise ValueError('RAG schemas must select text or a field below metadata')
    processed = []
    for chunk in chunks:
        result = process_value(cleaner, chunk.text, ("text",), schema=schema, context=context)
        signals = result.signals
        metadata_schema = (FieldSchema(schema.id, schema.version, tuple(
            replace(rule, path=rule.path[1:]) for rule in schema.rules
            if rule.path and rule.path[0] == 'metadata')) if schema is not None else None)
        metadata_context = (replace(context or DetectionContext(), path=(context.path if context else ()) + ('metadata',))
                            if context is not None or schema is not None else None)
        processed_metadata = (process_json(cleaner, dict(chunk.metadata))
                              if metadata_schema is None and metadata_context is None else
                              process_json(cleaner, dict(chunk.metadata), schema=metadata_schema, context=metadata_context))
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
            findings_v2=tuple(dict(finding) for finding in result.metadata.get('findings_v2', ())),
            metadata_findings_v2=processed_metadata.findings_v2,
        ))
    return processed


def process_dataframe(
    cleaner: TextCleaner,
    frame: Any,
    *,
    columns: tuple[Hashable, ...] | str,
    schema: FieldSchema | None = None, context: DetectionContext | None = None,
) -> DataFrameResult:
    """Process selected string columns in a copied DataFrame-like object."""
    require_adapter_mode(cleaner.cfg)
    validate_options(cleaner, schema, context)
    if isinstance(columns, str):  # "text" is not ("t", "e", "x", "t")
        columns = (columns,)
    unknown = set(columns).difference(frame.columns)
    if unknown:
        raise KeyError("Unknown columns in selection")

    if schema is not None and any(len(rule.path) not in (1, 2) or rule.path[-1] not in columns for rule in schema.rules):
        raise ValueError('DataFrame schemas require a selected column or a row and selected column')
    output = frame.copy(deep=True)
    missing_cells = output.isna() if schema is not None and hasattr(output, "isna") else None
    findings_v2: list[PathFindingV2] = []
    findings: list[PathFinding] = []
    views: list[Signals] = []
    # Positional access: a duplicate row or column label makes .at return a Series.
    selected = [j for j, name in enumerate(output.columns) if name in columns]
    for i, row in enumerate(output.index):
        for j in selected:
            value = output.iat[i, j]
            if missing_cells is not None and missing_cells.iat[i, j]:
                continue
            column_rule = schema.rule_at((output.columns[j],)) if schema is not None else None
            cell_rule = schema.rule_at((row, output.columns[j])) if schema is not None else None
            if column_rule is not None and cell_rule is not None and (
                    column_rule.mode != cell_rule.mode or column_rule.entity_types != cell_rule.entity_types):
                raise ValueError('Conflicting column and cell field rules')
            result = process_value(cleaner, value, (row, output.columns[j]), schema=schema,
                                   context=context, rule=cell_rule or column_rule)
            if result is None:
                continue
            if not isinstance(value, str) and hasattr(output, 'isetitem'):
                output.isetitem(j, output.iloc[:, j].astype(object))
            output.iat[i, j] = result.lm_text
            findings_v2.extend(PathFindingV2((i, j), dict(finding))
                               for finding in result.metadata.get('findings_v2', ()))
            views.append(result.signals)
            findings.extend(
                PathFinding((row, output.columns[j]), finding)
                for finding in result.metadata.get("findings", ())
            )
    signals = (cleaner.process('').signals if not views and cleaner.cfg.pii_processing == 'unified'
               else merge_views(views))
    return DataFrameResult(output, tuple(findings), signals["sensitive"], signals, tuple(findings_v2))
