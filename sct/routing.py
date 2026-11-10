"""Callers can skip a detector stage using explicit, experimental calibration.

Stage completion describes execution, never proof that a document contains no PII.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
import re
from typing import TYPE_CHECKING

from sct.detectors import DetectorCapabilities, DetectorSpec, _freeze
from sct.language_policy import LanguageDecision
from sct.policy_v2 import HandlingRule, UnifiedPrivacyPolicy

if TYPE_CHECKING:
    from sct.config import TextCleanerConfig


@dataclass(frozen=True)
class RoutingContext:
    language: LanguageDecision
    requested_entity_types: tuple[str, ...]
    text_size: int
    requires_context: bool = False
    quality_score: float | None = None
    has_classification: bool = False

    def __post_init__(self):
        if not isinstance(self.language, LanguageDecision):
            raise ValueError('Routing requires a language decision')
        if type(self.text_size) is not int or self.text_size < 0:
            raise ValueError('Routing text_size must be a nonnegative integer')
        _types(self.requested_entity_types)
        if type(self.requires_context) is not bool or type(self.has_classification) is not bool:
            raise ValueError('Routing context flags must be booleans')
        if self.quality_score is not None:
            _score(self.quality_score)


@dataclass(frozen=True)
class DetectorRun:
    stage_id: str
    capabilities: tuple[DetectorCapabilities, ...]
    status: str = 'succeeded'
    calls: int = 0

    def __post_init__(self):
        if self.stage_id not in ('fast', 'full') or self.status not in ('succeeded', 'failed', 'skipped'):
            raise ValueError('Invalid detector stage or status')
        if not isinstance(self.capabilities, tuple) or any(
                not isinstance(item, DetectorCapabilities) for item in self.capabilities):
            raise ValueError('Detector runs require actual capability tuples')
        if type(self.calls) is not int or self.calls < 0:
            raise ValueError('Detector calls must be nonnegative integers')


@dataclass(frozen=True)
class RoutingDecision:
    selected_ids: tuple[str, ...] = ()
    skipped_ids: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    policy_identity: str | None = None


def _types(values):
    if not isinstance(values, tuple) or any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError('Routing entity types must be a tuple of nonempty strings')


def _score(value):
    if type(value) not in (int, float) or not 0 <= value <= 1:
        raise ValueError('Calibration scores must be finite numbers in [0, 1]')


def routing_config_digest(cfg: TextCleanerConfig, *, unified_policy: UnifiedPrivacyPolicy | None = None) -> str:
    """Bind normalized configuration and the actual declarative handling policy."""
    from sct.corpus.manifest import config_digest
    state = cfg._plain_state()
    for key in ('routing_calibration', 'routing_policy'):
        state.pop(key)
    policy = unified_policy if unified_policy is not None else UnifiedPrivacyPolicy('default', '1', {})
    if type(policy) is not UnifiedPrivacyPolicy or any(type(rule) is not HandlingRule for rule in policy.rules.values()):
        raise ValueError('Routing calibration requires declarative policy and rule classes')
    state['unified_policy'] = {'name': policy.name, 'version': policy.version,
                               'rules': {kind: asdict(rule) for kind, rule in policy.rules.items()}}
    return config_digest(state)


def capabilities_digest(capabilities: tuple[DetectorCapabilities, ...]) -> str:
    """Bind the ordered capability declarations, including declared model identities."""
    from sct.corpus.manifest import config_digest
    return config_digest([asdict(item) for item in capabilities])


def validate_routing_config(cfg: TextCleanerConfig) -> None:
    if cfg.routing_policy not in ('always_run', 'calibrated_cascade'):
        raise ValueError('routing_policy must be always_run or calibrated_cascade')
    for name in ('routing_fast_specs', 'routing_full_specs'):
        specs = tuple(DetectorSpec(**item) if isinstance(item, Mapping) else item for item in getattr(cfg, name))
        if any(not isinstance(item, DetectorSpec) for item in specs):
            raise ValueError('Routing specs must contain DetectorSpec values')
        object.__setattr__(cfg, name, specs)
    if not isinstance(cfg.routing_entity_types, (tuple, list)):
        raise ValueError('Routing entity types must be a tuple or list')
    _types(tuple(cfg.routing_entity_types))
    object.__setattr__(cfg, 'routing_entity_types', tuple(cfg.routing_entity_types))
    active = cfg.routing_policy != 'always_run' or cfg.routing_fast_specs or cfg.routing_full_specs
    if active:
        if cfg.pii_processing != 'unified':
            raise ValueError('Detector routing requires unified processing')
        if cfg.detector_specs or not cfg.routing_fast_specs or not cfg.routing_full_specs:
            raise ValueError('Routing requires explicit fast/full specs and no detector_specs')
        specs = cfg.routing_fast_specs + cfg.routing_full_specs
        if any(spec in specs[:index] for index, spec in enumerate(specs)):
            raise ValueError('Routing specs must not contain duplicate configurations')
    elif cfg.routing_calibration is not None or cfg.routing_entity_types:
        raise ValueError('Routing options require explicit fast/full specs')
    calibration = cfg.routing_calibration
    if calibration is None:
        return
    keys = {'schema_version', 'config_digest', 'fast_capabilities_digest', 'language_min_confidence',
            'language_min_margin', 'max_chars', 'languages', 'entity_types'}
    if not isinstance(calibration, Mapping) or set(calibration) != keys:
        raise ValueError('Invalid routing calibration fields')
    if type(calibration['schema_version']) is not int or calibration['schema_version'] != 1:
        raise ValueError('Unsupported routing calibration version')
    for key in ('config_digest', 'fast_capabilities_digest'):
        if not isinstance(calibration[key], str) or not re.fullmatch('[0-9a-f]{64}', calibration[key]):
            raise ValueError('Invalid routing calibration identity')
    for key in ('language_min_confidence', 'language_min_margin'):
        _score(calibration[key])
    if type(calibration['max_chars']) is not int or calibration['max_chars'] < 1:
        raise ValueError('Calibration max_chars must be a positive integer')
    for key in ('languages', 'entity_types'):
        if not isinstance(calibration[key], (tuple, list)) or not calibration[key]:
            raise ValueError('Calibration requires explicit languages and entity types')
        _types(tuple(calibration[key]))
    object.__setattr__(cfg, 'routing_calibration', _freeze(calibration))


@dataclass(frozen=True)
class RoutingPolicy:
    cfg: TextCleanerConfig = field(repr=False)
    unified_policy: UnifiedPrivacyPolicy | None = field(default=None, repr=False, kw_only=True)
    _identity: str | None = field(init=False, default=None, repr=False)
    _config_identity: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self):
        if self.cfg.routing_policy == 'always_run':
            return
        from sct.corpus.manifest import config_digest
        identity = routing_config_digest(self.cfg, unified_policy=self.unified_policy)
        artifact = self.cfg.routing_calibration
        if artifact is not None and artifact['config_digest'] != identity:
            raise ValueError('Routing calibration configuration identity mismatch')
        object.__setattr__(self, '_config_identity', identity)
        object.__setattr__(self, '_identity', 'experimental:calibrated_cascade:v1:' +
                           config_digest({'config_digest': identity, 'calibration': artifact}))

    @property
    def identity(self) -> str | None:
        if self._config_identity is not None and (
                routing_config_digest(self.cfg, unified_policy=self.unified_policy) != self._config_identity):
            raise ValueError('Routing calibration configuration identity changed')
        return self._identity

    def plan(self, *, context: RoutingContext, completed: tuple[DetectorRun, ...]) -> RoutingDecision:
        """Select at most two stages; completed stages cannot be retried or reordered."""
        stages = ('full',) if self.cfg.routing_policy == 'always_run' else ('fast', 'full')
        if not isinstance(completed, tuple) or any(not isinstance(run, DetectorRun) for run in completed):
            raise ValueError('Invalid completed detector stages')
        if len(completed) > len(stages) or tuple(run.stage_id for run in completed) != stages[:len(completed)]:
            raise ValueError('Detector stages cannot repeat or run out of order')
        if len(completed) == len(stages):
            return RoutingDecision(policy_identity=self.identity)
        if not completed:
            return RoutingDecision((stages[0],), policy_identity=self.identity)
        reason = self._full_reason(context, completed[0])
        if reason:
            return RoutingDecision(('full',), reasons=(reason,), policy_identity=self.identity)
        return RoutingDecision(skipped_ids=('full',), reasons=('calibrated_fast_coverage',), policy_identity=self.identity)

    def _full_reason(self, context: RoutingContext, run: DetectorRun) -> str | None:
        artifact = self.cfg.routing_calibration
        if run.status != 'succeeded':
            return 'fast_stage_incomplete'
        if not run.capabilities or run.calls != len(run.capabilities):
            return 'missing_capabilities'
        if artifact is None:
            return 'missing_calibration'
        if capabilities_digest(run.capabilities) != artifact['fast_capabilities_digest']:
            raise ValueError('Routing calibration capability identity mismatch')
        language = context.language
        if language.language is None:
            return 'unknown_language'
        if language.language not in artifact['languages']:
            return 'uncalibrated_language'
        if language.source != 'pinned' and (language.source != 'detected'
                or language.confidence is None or language.margin is None
                or type(language.confidence) not in (int, float) or type(language.margin) not in (int, float)
                or not artifact['language_min_confidence'] <= language.confidence <= 1
                or not artifact['language_min_margin'] <= language.margin <= 1):
            return 'language_evidence'
        if context.text_size > artifact['max_chars']:
            return 'text_size'
        if context.quality_score is not None or context.has_classification:
            return 'unsupported_context'
        if context.requires_context and any(not cap.supports_context for cap in run.capabilities):
            return 'unsupported_context'
        requested = set(context.requested_entity_types)
        if not requested:
            return 'unknown_entity_request'
        capable = {kind for cap in run.capabilities
                   if not cap.languages or language.language in cap.languages for kind in cap.entity_types}
        if not requested <= capable or not requested <= set(artifact['entity_types']):
            return 'unsupported_entity_coverage'
        return None
