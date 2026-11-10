"""Apply explicit handling rules to value-free findings from every detector."""
from collections.abc import Mapping
from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Literal

from sct.findings import Detection

Action = Literal['use_mode', 'keep', 'redact']


@dataclass(frozen=True)
class HandlingRule:
    action: Action = 'use_mode'
    min_score: float | None = None
    unknown_score_action: Literal['protect', 'error'] = 'protect'

    def __post_init__(self) -> None:
        if self.action not in ('use_mode', 'keep', 'redact'):
            raise ValueError('Unknown handling action')
        if self.unknown_score_action not in ('protect', 'error'):
            raise ValueError('Unknown score action must be protect or error')
        if self.min_score is not None and (
            type(self.min_score) not in (int, float) or not math.isfinite(self.min_score) or not 0 <= self.min_score <= 1
        ):
            raise ValueError('Minimum score must be finite and in [0, 1]')


@dataclass(frozen=True)
class FindingDecision:
    detection: Detection
    action: Action
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.detection, Detection) or self.action not in ('use_mode', 'keep', 'redact'):
            raise ValueError('Invalid finding decision')
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError('A finding decision requires a reason')


@dataclass(frozen=True)
class UnifiedPrivacyPolicy:
    name: str
    version: str
    rules: Mapping[str, HandlingRule]

    def __post_init__(self) -> None:
        if any(not isinstance(value, str) or not value.strip() for value in (self.name, self.version)):
            raise ValueError('Policy name and version must be nonempty strings')
        if not isinstance(self.rules, Mapping) or any(
            not isinstance(key, str) or not key.strip() or not isinstance(rule, HandlingRule)
            for key, rule in self.rules.items()
        ):
            raise ValueError('Policy rules must map entity types to HandlingRule values')
        object.__setattr__(self, 'rules', MappingProxyType(dict(self.rules)))

    def evaluate(self, detections: tuple[Detection, ...]) -> tuple[FindingDecision, ...]:
        decisions = []
        for detection in detections:
            rule = self.rules.get(detection.entity_type, HandlingRule())
            action, reason = rule.action, 'policy-rule'
            if rule.action != 'keep' and detection.score is None:
                if rule.unknown_score_action == 'error':
                    raise ValueError('Detector score is unknown under the selected policy')
                reason = 'unknown-score-protected'
            elif rule.min_score is not None and detection.score is not None and detection.score < rule.min_score:
                action, reason = 'keep', 'below-threshold'
            decisions.append(FindingDecision(detection, action, reason))
        return tuple(decisions)
