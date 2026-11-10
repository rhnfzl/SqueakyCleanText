"""Keep language evidence when callers choose to abstain from uncertain detection."""
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sct.config import TextCleanerConfig


@dataclass(frozen=True)
class LanguageDecision:
    language: str | None
    confidence: float | None
    margin: float | None
    source: str
    reason: str


def apply_language_policy(decision: LanguageDecision, policy: 'TextCleanerConfig') -> LanguageDecision:
    """Apply opt-in thresholds without changing pinned languages or score evidence."""
    if policy.language_policy == 'legacy' or decision.source == 'pinned':
        return decision
    reason = None
    if decision.source in ('none', 'fallback'):
        reason = 'no_language_evidence'
    elif policy.language_min_confidence is not None and (
            decision.confidence is None or decision.confidence < policy.language_min_confidence):
        reason = 'confidence_below_minimum'
    elif policy.language_min_margin is not None and (
            decision.margin is None or decision.margin < policy.language_min_margin):
        reason = 'margin_below_minimum'
    return replace(decision, language=None, source='abstained', reason=reason) if reason else decision
