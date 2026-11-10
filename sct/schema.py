"""Caller-owned field declarations select exact paths without guessing field names."""
from collections.abc import Hashable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Literal
import math

from numpy import generic as NumpyScalar

from sct.findings import DetectionContext


@dataclass(frozen=True)
class FieldRule:
    path: tuple[Hashable, ...] = field(repr=False)
    entity_types: tuple[str, ...]
    mode: Literal['hint', 'protect_value'] = 'hint'

    def __post_init__(self) -> None:
        if not isinstance(self.path, tuple):
            raise ValueError('Field paths must be immutable tuples')
        try:
            hash(self.path)
        except TypeError:
            raise ValueError('Field path components must be hashable') from None
        if not isinstance(self.entity_types, tuple) or not self.entity_types or any(
            not isinstance(kind, str) or not kind.strip() for kind in self.entity_types
        ):
            raise ValueError('Field entity types must be a nonempty tuple of nonempty strings')
        if len(set(self.entity_types)) != len(self.entity_types) or self.mode not in ('hint', 'protect_value'):
            raise ValueError('Invalid field rule')


@dataclass(frozen=True)
class FieldSchema:
    id: str
    version: str
    rules: tuple[FieldRule, ...] = field(repr=False)

    def __post_init__(self) -> None:
        if any(not isinstance(value, str) or not value.strip() for value in (self.id, self.version)):
            raise ValueError('Schema ID and version must be nonempty strings')
        if not isinstance(self.rules, tuple) or any(not isinstance(rule, FieldRule) for rule in self.rules):
            raise ValueError('Schema rules must be an immutable tuple of FieldRule values')
        by_path: dict[tuple[Hashable, ...], FieldRule] = {}
        for rule in self.rules:
            if rule.path in by_path and by_path[rule.path] != rule:
                raise ValueError('Conflicting rules select the same field')
            by_path[rule.path] = rule

    def rule_at(self, path: tuple[Hashable, ...]) -> FieldRule | None:
        return next((rule for rule in self.rules if rule.path == path), None)

    def to_dict(self) -> dict[str, Any]:
        """Serialize data-only schemas; richer Python paths remain in-process only."""
        if any(type(part) not in (str, int) for rule in self.rules for part in rule.path):
            raise ValueError('Data-only schemas require string or integer path components')
        return {'id': self.id, 'version': self.version, 'rules': [
            {'path': list(rule.path), 'entity_types': list(rule.entity_types), 'mode': rule.mode} for rule in self.rules]}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> 'FieldSchema':
        """Read data-only JSON paths. Python callers can use any hashable component."""
        try:
            if set(value) != {'id', 'version', 'rules'} or not isinstance(value['rules'], list):
                raise ValueError
            rules = []
            for rule in value['rules']:
                if set(rule) - {'path', 'entity_types', 'mode'} or not {'path', 'entity_types'} <= set(rule):
                    raise ValueError
                if not isinstance(rule['path'], list) or any(type(part) not in (str, int) for part in rule['path']):
                    raise ValueError
                if not isinstance(rule['entity_types'], list):
                    raise ValueError
                rules.append(FieldRule(tuple(rule['path']), tuple(rule['entity_types']), rule.get('mode', 'hint')))
            return cls(value['id'], value['version'], tuple(rules))
        except (TypeError, KeyError, ValueError):
            raise ValueError('Invalid data-only field schema') from None


def registered_schema(value: Mapping[str, Any] | None,
                      schemas: Mapping[str, FieldSchema] | None) -> FieldSchema | None:
    """A request may select only an exact application-registered schema."""
    if value is None:
        return None
    schema = FieldSchema.from_dict(value)
    if schemas is None or schemas.get(schema.id) != schema:
        raise ValueError('Field schema is not registered by the application')
    return schema


def validate_options(cleaner, schema: FieldSchema | None, context: DetectionContext | None) -> None:
    if schema is not None and not isinstance(schema, FieldSchema):
        raise ValueError('schema must be a FieldSchema')
    if context is not None and not isinstance(context, DetectionContext):
        raise ValueError('context must be a DetectionContext')
    if (schema is not None or context is not None) and cleaner.cfg.pii_processing != 'unified':
        raise ValueError('Field schemas and context require unified processing')


def process_value(cleaner, value, path=(), *, schema=None, context=None, rule=None):
    """Use the common core for hints and full-value protection. Missing paths are optional."""
    rule = rule or (schema.rule_at(path) if schema is not None else None)
    if rule is not None and rule.mode == 'protect_value' and isinstance(value, NumpyScalar):
        value = value.item()
    if rule is not None and value is not None and not isinstance(value, str):
        if rule.mode != 'protect_value' or type(value) not in (int, float, bool):
            raise ValueError('Selected fields require text or finite numeric and boolean scalars')
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError('Selected numeric scalars must be finite')
        value = str(value)
    if not isinstance(value, str):
        return None
    if rule is None and context is None:
        return cleaner.process(value)
    selected = replace(context or DetectionContext(), path=(context.path if context else ()) + path)
    if rule is not None:
        selected = replace(selected, schema_id=schema.id, declared_entity_types=rule.entity_types)
    if rule is not None and rule.mode == 'protect_value':
        return cleaner._process_unified_document(value, selected, protect_value=rule.entity_types).result
    return cleaner.process(value, context=selected)
