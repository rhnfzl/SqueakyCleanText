"""Optional credential format scanning without models or retained matched values."""
from __future__ import annotations

from collections.abc import Sequence
import math

from sct.detectors import DetectorCapabilities, DetectorUnavailable
from sct.findings import Detection, DetectionContext

_TYPES = ("API_KEY", "BEARER_TOKEN", "JWT", "PRIVATE_KEY")


class DataFogSecretDetector:
    """Use the pinned rule scanner on explicitly selected credential categories."""

    def __init__(self, *, entities: Sequence[str] = _TYPES) -> None:
        if (not isinstance(entities, (list, tuple)) or not entities
                or any(type(entity) is not str or entity not in _TYPES for entity in entities)):
            raise ValueError("Select supported credential entity types")
        self._entities = tuple(dict.fromkeys(entities))
        try:
            from datafog_core import scan
        except ImportError:
            raise DetectorUnavailable("Credential scanning requires the secrets extra") from None
        self._scan = scan
        self._capabilities = DetectorCapabilities(
            "datafog_secrets", version="1.0.0", entity_types=self._entities,
        )

    @property
    def capabilities(self) -> DetectorCapabilities:
        return self._capabilities

    def detect(self, text: str, *, language: str | None,
               context: DetectionContext) -> tuple[Detection, ...]:
        try:
            results = self._scan(text, {"entities": list(self._entities)})
        except Exception:
            raise DetectorUnavailable("Credential scanner failed") from None
        hits = []
        try:
            for result in results:
                entity = result.entity_type
                if type(entity) is not str or entity not in self._entities:
                    raise ValueError
                start, end = result.codepoint_range.start, result.codepoint_range.end
                if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text):
                    raise ValueError
                # The 1.0.0 binding supplies both units; verify agreement without searching matched values.
                byte_start, byte_end = result.byte_range.start, result.byte_range.end
                if (type(byte_start) is not int or type(byte_end) is not int
                        or byte_start != len(text[:start].encode("utf-8"))
                        or byte_end != len(text[:end].encode("utf-8"))):
                    raise ValueError
                score = result.confidence
                if score is not None:
                    if type(score) not in (float, int) or not math.isfinite(score) or not 0 <= score <= 1:
                        raise ValueError
                    score = float(score)
                hits.append(Detection(entity, start, end, "datafog_secrets", score, "canonical"))
        except Exception:
            raise ValueError("Credential scanner returned invalid findings") from None
        return tuple(hits)
