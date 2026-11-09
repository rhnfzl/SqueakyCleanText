"""Explicit application-owned detectors. Configuration contains data, never imports."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Protocol

from sct.findings import Detection, DetectionContext

if TYPE_CHECKING:
    from sct.config import TextCleanerConfig


class DetectorUnavailable(RuntimeError):
    """A selected detector cannot run with the installed dependencies or permissions."""


class _FrozenOptions(dict):
    __slots__ = ('_initialized',)

    def __init__(self, *args, **kwargs):
        if hasattr(self, '_initialized'):
            raise TypeError('Detector options are immutable')
        dict.__init__(self, *args, **kwargs)
        self._initialized = True

    def _immutable(self, *args, **kwargs):
        raise TypeError('Detector options are immutable')

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = __ior__ = _immutable

    def __reduce__(self):
        return (_FrozenOptions, (dict(self),))


def _freeze(value: Any) -> Any:
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise ValueError('Detector option keys must be strings')
        return _FrozenOptions({key: _freeze(item) for key, item in value.items()})
    if type(value) in (tuple, list):
        return tuple(_freeze(item) for item in value)
    raise ValueError('Detector options must contain finite JSON data only')


def _identifier(value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Detector identifiers must be nonempty strings')


@dataclass(frozen=True)
class DetectorSpec:
    name: str
    options: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        _identifier(self.name)
        if not isinstance(self.options, Mapping):
            raise ValueError('Detector options must be a mapping')
        object.__setattr__(self, 'options', _freeze(self.options))


@dataclass(frozen=True)
class DetectorCapabilities:
    id: str
    version: str = '1'
    entity_types: tuple[str, ...] = ()
    languages: tuple[str, ...] = ()
    coordinate_system: str = 'unicode_codepoint'
    supports_context: bool = False
    supports_windowed: bool = False
    network_access: bool = False
    model_identities: tuple[tuple[str, str | None, str | None], ...] = ()

    def __post_init__(self) -> None:
        _identifier(self.id)
        _identifier(self.version)
        if self.coordinate_system != 'unicode_codepoint':
            raise ValueError('Detectors must return Unicode code-point coordinates')
        for name in ('entity_types', 'languages'):
            values = tuple(getattr(self, name))
            for value in values:
                _identifier(value)
            object.__setattr__(self, name, values)
        for name in ('supports_context', 'supports_windowed', 'network_access'):
            if type(getattr(self, name)) is not bool:
                raise ValueError('Detector capability flags must be booleans')
        identities = tuple(tuple(identity) for identity in self.model_identities)
        for identity in identities:
            if len(identity) != 3:
                raise ValueError('Model identities require ID, revision and license')
            _identifier(identity[0])
            for value in identity[1:]:
                if value is not None:
                    _identifier(value)
        object.__setattr__(self, 'model_identities', identities)


class Detector(Protocol):
    @property
    def capabilities(self) -> DetectorCapabilities: ...

    def detect(self, text: str, *, language: str | None,
               context: DetectionContext) -> tuple[Detection, ...]: ...


DetectorFactory = Callable[['TextCleanerConfig', Mapping[str, Any]], Detector]


def _regex(cfg: TextCleanerConfig, options: Mapping[str, Any]) -> Detector:
    if options:
        raise ValueError('Builtin regex options belong in TextCleanerConfig')
    from sct.detectors_builtin import BuiltinRegexDetector
    return BuiltinRegexDetector(cfg)


def _ner(cfg: TextCleanerConfig, options: Mapping[str, Any]) -> Detector:
    if options:
        raise ValueError('Builtin NER options belong in TextCleanerConfig')
    from sct.detectors_builtin import BuiltinNERDetector
    from sct.utils.ner import GeneralNER, ModelLoadError
    try:
        return BuiltinNERDetector(GeneralNER(cfg=cfg))
    except ModelLoadError as exc:
        raise DetectorUnavailable("Builtin NER model or dependency is unavailable") from exc


class DetectorRegistry:
    """Factories belong to one application and create fresh process-owned instances."""

    def __init__(self) -> None:
        self._factories: Mapping[str, DetectorFactory] = {'builtin_regex': _regex, 'builtin_ner': _ner}
        self._network = {"builtin_ner"}
        self._frozen = False

    def register(self, name: str, factory: DetectorFactory, *, network_access: bool = False) -> None:
        _identifier(name)
        if self._frozen:
            raise ValueError('Detector registry is frozen')
        if name in self._factories:
            raise ValueError('Detector name is already registered')
        if not callable(factory):
            raise TypeError('Detector factory must be callable')
        if type(network_access) is not bool:
            raise ValueError("network_access must be a boolean")
        self._factories = {**self._factories, name: factory}
        if network_access:
            self._network.add(name)

    def freeze(self) -> None:
        self._factories = MappingProxyType(dict(self._factories))
        self._frozen = True

    @property
    def identity(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))

    def create(self, spec: DetectorSpec, cfg: TextCleanerConfig) -> Detector:
        if not isinstance(spec, DetectorSpec):
            raise TypeError('Expected DetectorSpec')
        factory = self._factories.get(spec.name)
        if factory is None:
            raise ValueError('Unknown detector name')
        if spec.name in self._network and not cfg.allow_network_detectors:
            raise DetectorUnavailable("Network detectors require allow_network_detectors=True")
        try:
            detector = factory(cfg, spec.options)
        except ImportError as exc:
            raise DetectorUnavailable('Selected detector requires an optional dependency') from exc
        if not callable(getattr(detector, "detect", None)):
            raise TypeError("Detector must expose a detect method")
        if not isinstance(getattr(detector, "capabilities", None), DetectorCapabilities):
            raise TypeError('Detector must expose DetectorCapabilities')
        if detector.capabilities.network_access and not cfg.allow_network_detectors:
            raise DetectorUnavailable('Network detectors require allow_network_detectors=True')
        return detector
