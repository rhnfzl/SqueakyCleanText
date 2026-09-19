"""Policy and audit types for sensitive-data processing."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Optional, Protocol

from sct.utils.anonymization_map import AnonymizationMap


VALID_ENTITY_ACTIONS = frozenset({"placeholder", "redact", "keep"})


class TokenStore(Protocol):
    """Storage boundary for reversible anonymization maps."""

    def save(self, anon_map: AnonymizationMap) -> str:
        """Store a map and return an opaque reference."""

    def load(self, reference: str) -> Optional[AnonymizationMap]:
        """Load a map by reference."""

    def delete(self, reference: str) -> None:
        """Delete a stored map."""


class InMemoryTokenStore:
    """Process-local token store for tests and short-lived sessions."""

    def __init__(self) -> None:
        self._maps: dict[str, AnonymizationMap] = {}

    def save(self, anon_map: AnonymizationMap) -> str:
        self._maps[anon_map.session_id] = anon_map
        return anon_map.session_id

    def load(self, reference: str) -> Optional[AnonymizationMap]:
        return self._maps.get(reference)

    def delete(self, reference: str) -> None:
        self._maps.pop(reference, None)


@dataclass(frozen=True)
class DetectedEntity:
    """One entity produced by a detector."""

    entity_type: str
    score: float
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class EntityRule:
    """Policy behavior for one entity type."""

    action: str = "placeholder"
    min_score: float = 0.0

    def __post_init__(self) -> None:
        if self.action not in VALID_ENTITY_ACTIONS:
            raise ValueError(
                f"action must be one of {sorted(VALID_ENTITY_ACTIONS)}, "
                f"got {self.action!r}"
            )
        if not 0.0 <= self.min_score <= 1.0:
            raise ValueError("min_score must be between 0 and 1")


@dataclass(frozen=True)
class EntityFinding:
    """Auditable policy decision for one detected entity."""

    entity: DetectedEntity
    action: str
    reason: str


@dataclass(frozen=True)
class NERProcessingResult:
    """Text and policy findings returned by detailed NER processing."""

    text: str
    findings: tuple[EntityFinding, ...]


@dataclass(frozen=True)
class ProcessingMetrics:
    """Opt-in processing measurements without source text or entity values."""

    duration_ms: float
    language: Optional[str]
    backend: Optional[str]
    policy: Optional[str]
    entity_counts: Mapping[str, int]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "entity_counts",
            MappingProxyType(dict(self.entity_counts)),
        )


@dataclass(frozen=True)
class PrivacyPolicy:
    """Versioned rules for deciding how detected entities are handled."""

    name: str
    version: str
    rules: Mapping[str, EntityRule]
    allowlist: frozenset[str] = frozenset()
    default_rule: EntityRule = EntityRule()

    def __post_init__(self) -> None:
        object.__setattr__(self, "rules", MappingProxyType(dict(self.rules)))
        object.__setattr__(self, "allowlist", frozenset(self.allowlist))

    def evaluate(self, entities: tuple[DetectedEntity, ...]) -> tuple[EntityFinding, ...]:
        """Apply the policy to detector output."""
        findings = []
        for entity in entities:
            rule = self.rules.get(entity.entity_type, self.default_rule)
            if entity.text in self.allowlist:
                findings.append(EntityFinding(entity, "allow", "allowlist"))
            elif entity.score < rule.min_score:
                findings.append(EntityFinding(entity, "ignore", "below-threshold"))
            else:
                findings.append(EntityFinding(entity, rule.action, "policy-rule"))
        return tuple(findings)

    def transform(self, text: str, findings: tuple[EntityFinding, ...]) -> str:
        """Apply policy findings from right to left."""
        transforming = sorted(
            (
                finding for finding in findings
                if finding.action in {"placeholder", "redact"}
            ),
            key=lambda item: item.entity.start,
        )
        groups: list[list[EntityFinding]] = []
        for finding in transforming:
            if (
                groups
                and finding.entity.start
                < max(item.entity.end for item in groups[-1])
            ):
                groups[-1].append(finding)
            else:
                groups.append([finding])

        replacements = []
        for group in groups:
            start = min(item.entity.start for item in group)
            end = max(item.entity.end for item in group)
            if any(item.action == "redact" for item in group):
                replacement = "<REDACTED>"
            else:
                strongest = max(
                    group,
                    key=lambda item: item.entity.score,
                )
                replacement = f"<{strongest.entity.entity_type}>"
            replacements.append((start, end, replacement))

        for start, end, replacement in reversed(replacements):
            text = text[:start] + replacement + text[end:]
        return text
