"""Select a chunker and validate its ranges into the text SCT supplied."""
from __future__ import annotations

from bisect import bisect_right
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal

from sct.findings import TextRange

ChunkStage = Literal['detection', 'output']
_STAGES = ('detection', 'output')
_RESERVED = ('auto', 'native')


class ChunkBudgetError(ValueError):
    """A range cannot fit the requested size budget."""


def _name(value: object) -> str:
    if type(value) is not str or not value.strip() or value.strip() != value:
        raise ValueError('Chunking names must be nonempty strings without surrounding whitespace')
    return value


def _names(values: Iterable[str]) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise ValueError('Chunking names must be a list or tuple')
    return tuple(_name(value) for value in values)


@dataclass(frozen=True)
class ChunkingRule:
    """Prefer a strategy when all supplied document conditions match."""

    strategy: str
    languages: tuple[str, ...] = ()
    formats: tuple[str, ...] = ()
    min_chars: int | None = None
    max_chars: int | None = None

    def __post_init__(self) -> None:
        if _name(self.strategy) == 'auto':
            raise ValueError('A chunking rule must select a named or native strategy')
        for key in ('languages', 'formats'):
            object.__setattr__(self, key, tuple(item.casefold() for item in _names(getattr(self, key))))
        bounds = (self.min_chars, self.max_chars)
        if any(value is not None and (type(value) is not int or value < 0) for value in bounds):
            raise ValueError('Chunking rule lengths must be nonnegative character counts')
        if self.min_chars is not None and self.max_chars is not None and self.min_chars > self.max_chars:
            raise ValueError('Chunking rule minimum exceeds its maximum')

    def matches(self, language: str, document_format: str, length: int) -> bool:
        return (
            (not self.languages or language in self.languages)
            and (not self.formats or document_format in self.formats)
            and (self.min_chars is None or length >= self.min_chars)
            and (self.max_chars is None or length <= self.max_chars)
        )


@dataclass(frozen=True)
class ChunkingPolicy:
    """A pinned strategy or an ordered automatic selection."""

    strategy: str = 'native'
    candidates: tuple[str, ...] = ()
    rules: tuple[ChunkingRule, ...] = ()

    def __post_init__(self) -> None:
        _name(self.strategy)
        object.__setattr__(self, 'candidates', _names(self.candidates))
        if not isinstance(self.rules, (list, tuple)):
            raise ValueError('Chunking rules must be a list or tuple')
        rules = tuple(ChunkingRule(**rule) if isinstance(rule, Mapping) else rule for rule in self.rules)
        if any(not isinstance(rule, ChunkingRule) for rule in rules):
            raise ValueError('Chunking rules must contain ChunkingRule values')
        object.__setattr__(self, 'rules', rules)
        if self.strategy != 'auto' and (self.candidates or rules):
            raise ValueError('Chunking candidates and rules require auto strategy')
        if 'auto' in self.candidates:
            raise ValueError('Chunking candidates must name a strategy or native')

    @classmethod
    def coerce(cls, value: ChunkingPolicy | str | Mapping | None) -> ChunkingPolicy | None:
        if isinstance(value, Mapping):
            return cls(**value)
        if isinstance(value, str):
            return cls(value)
        if value is not None and not isinstance(value, cls):
            raise ValueError('Expected a chunking policy, strategy name, or mapping')
        return value


@dataclass(frozen=True)
class ChunkRequest:
    """The exact stage text, with the size limit in units of count()."""

    text: str = field(repr=False)
    max_size: int
    count: Callable[[str], int] = field(repr=False)
    stage: ChunkStage
    language: str | None = None
    document_format: str | None = None
    elements: tuple[TextRange, ...] = ()
    allow_network: bool = False
    protected: tuple[TextRange, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not callable(self.count) or self.stage not in _STAGES:
            raise ValueError('Chunking needs text, a size counter and a valid stage')
        if type(self.max_size) is not int or self.max_size < 1:
            raise ValueError('Chunk size must be a positive integer')


Chunker = Callable[[ChunkRequest], Iterable[TextRange]]


def legal_cut(position: int, protected: tuple[TextRange, ...], starts: tuple[int, ...]) -> int:
    """Move a cut inside an indivisible range back to its start."""
    index = bisect_right(starts, position) - 1
    return starts[index] if index >= 0 and position < protected[index].end else position


def measured_size(count: Callable[[str], int], text: str) -> int:
    try:
        size = count(text)
    except Exception:
        raise ValueError('Chunk size counter failed') from None
    if type(size) is not int or size < 0:
        raise ValueError('Chunk size counters must return nonnegative integers')
    return size


@dataclass(frozen=True)
class _RegisteredChunker:
    split: Chunker = field(repr=False)
    stages: tuple[ChunkStage, ...]
    network_access: bool
    available: Callable[[], bool] | None = field(repr=False)


class ChunkerRegistry:
    """An application-owned registry shared by detector and output chunking."""

    def __init__(self) -> None:
        from sct.chunkers_builtin import register_builtin_chunkers

        self._providers: Mapping[str, _RegisteredChunker] = {}
        self._frozen = False
        register_builtin_chunkers(self)

    def register(self, name: str, chunker: Chunker, *, stages: tuple[ChunkStage, ...] = ('output',),
                 network_access: bool = False, available: Callable[[], bool] | None = None) -> None:
        if self._frozen:
            raise ValueError('Chunker registry is frozen')
        if _name(name) in _RESERVED or name in self._providers:
            raise ValueError('Chunker name is reserved or already registered')
        if not callable(chunker) or not stages or any(stage not in _STAGES for stage in stages):
            raise ValueError('Chunkers must be callable and declare their supported stages')
        if type(network_access) is not bool:
            raise ValueError('Chunker network access must be a boolean')
        if available is not None and not callable(available):
            raise ValueError('Chunker availability must be callable')
        self._providers = {**self._providers, name: _RegisteredChunker(chunker, tuple(stages), network_access, available)}

    def freeze(self) -> None:
        self._providers = MappingProxyType(dict(self._providers))
        self._frozen = True

    def _compatible(self, name: str, stage: ChunkStage, allow_network: bool) -> bool:
        if name == 'native':
            return True
        provider = self._providers.get(name)
        if provider is None:
            raise ValueError('Unknown chunker name')
        return (stage in provider.stages and (allow_network or not provider.network_access)
                and (provider.available is None or provider.available()))

    def resolve(self, policy: ChunkingPolicy | None, *, stage: ChunkStage, language: str | None,
                document_format: str | None, length: int, allow_network: bool = False) -> str:
        policy = policy or ChunkingPolicy()
        if policy.strategy != 'auto':
            if not self._compatible(policy.strategy, stage, allow_network):
                raise ValueError('Selected chunker is unavailable or incompatible with this stage or network setting')
            return policy.strategy
        language_key = language.casefold() if language else ''
        format_key = document_format.casefold() if document_format else ''
        preferences = (
            *(rule.strategy for rule in policy.rules if rule.matches(language_key, format_key, length)),
            *policy.candidates,
        )
        return next((name for name in preferences if self._compatible(name, stage, allow_network)), 'native')

    def plan(self, name: str, request: ChunkRequest) -> tuple[TextRange, ...]:
        if name in _RESERVED or not self._compatible(name, request.stage, request.allow_network):
            raise ValueError('Selected chunker is unavailable or incompatible with this stage or network setting')
        try:
            ranges = tuple(self._providers[name].split(request))
        except ChunkBudgetError:
            raise ChunkBudgetError('Chunker could not fit the size budget') from None
        except Exception:
            raise ValueError('Chunker failed while planning ranges') from None
        validate_chunk_ranges(ranges, request)
        return ranges


def validate_chunk_ranges(ranges: tuple[TextRange, ...], request: ChunkRequest) -> None:
    """Require advancing ranges covering all input and fitting the stage budget."""
    end = start = 0
    protected_starts = tuple(span.start for span in request.protected)
    for index, span in enumerate(ranges):
        if type(span) is not TextRange:
            raise ValueError('Chunkers must return TextRange values')
        span.__post_init__()
        if span.start > end or span.end <= end or span.end > len(request.text) or (index and span.start <= start):
            raise ValueError('Chunk ranges must cover the input in order')
        if request.stage == 'detection' and span.start != end:
            raise ValueError('Detection ranges must cover each character exactly once before overlap is applied')
        if any(legal_cut(pos, request.protected, protected_starts) != pos for pos in (span.start, span.end)):
            raise ValueError('Chunk range cuts an indivisible replacement')
        if measured_size(request.count, request.text[span.start:span.end]) > request.max_size:
            raise ChunkBudgetError('Chunk range exceeds the size budget')
        start, end = span.start, span.end
    if end != len(request.text):
        raise ValueError('Chunk ranges must cover the complete input')
