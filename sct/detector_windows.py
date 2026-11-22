"""Run a detector on validated windows and project its findings to the document."""
from collections.abc import Callable
from dataclasses import replace

from sct.chunking import ChunkRequest, ChunkerRegistry, ChunkingPolicy, measured_size
from sct.detectors import Detector, DetectorWindowLimits
from sct.detectors_builtin import BuiltinNERDetector
from sct.findings import Detection, DetectionContext, TextRange


def _expand(core: TextRange, text: str, limits: DetectorWindowLimits) -> tuple[int, int]:
    """Keep as much nearby context as fits, even when the counter measures bytes or tokens."""
    def bounds(radius: int) -> tuple[int, int]:
        return max(0, core.start - radius), min(len(text), core.end + radius)

    lo, hi = 0, limits.overlap_chars
    start, end = bounds(hi)
    if not hi or measured_size(limits.count, text[start:end]) <= limits.max_size:
        return start, end
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        start, end = bounds(mid)
        if measured_size(limits.count, text[start:end]) <= limits.max_size:
            lo = mid
        else:
            hi = mid
    return bounds(lo)


def detect_with_chunking(
    detector: Detector, text: str, *, language: str | None, context: DetectionContext,
    document_format: str | None, policy: ChunkingPolicy | None, inherited: bool,
    registry: ChunkerRegistry, allow_network: bool,
) -> tuple[tuple[Detection, ...], str]:
    if policy is None or policy.strategy == 'native':
        return detector.detect(text, language=language, context=context), 'native'

    limits_for = getattr(detector, 'window_limits', None)
    builtin = detector if isinstance(detector, BuiltinNERDetector) else None
    supported = detector.capabilities.supports_windowed and (
        builtin.supports_custom_chunking if builtin is not None else callable(limits_for)
    )
    if not supported and not inherited and policy.strategy != 'auto':
        raise ValueError('Selected detector does not support external chunking')
    name = registry.resolve(
        policy if supported else None, stage='detection', language=language,
        document_format=document_format, length=len(text), allow_network=allow_network,
    )
    if name == 'native':
        return detector.detect(text, language=language, context=context), name

    used = False

    def split(budget: int, count: Callable[[str], int]):
        nonlocal used
        used = True
        request = ChunkRequest(text, budget, count, 'detection', language, document_format,
                               allow_network=allow_network)
        return registry.plan(name, request)

    if builtin is not None:
        found = builtin.detect_with_chunker(text, language=language, context=context, split=split)
        return found, name if used else 'native'

    if not callable(limits_for):
        raise ValueError('Detector does not provide window limits')
    limits = limits_for(language)
    if type(limits) is not DetectorWindowLimits:
        raise ValueError('Detector window_limits must return DetectorWindowLimits')
    if measured_size(limits.count, text) <= limits.max_size:
        return detector.detect(text, language=language, context=context), 'native'

    budget = limits.max_size - 2 * limits.overlap_chars
    best: dict[tuple[str, int, int], Detection] = {}
    for core in split(budget, limits.count):
        start, end = _expand(core, text, limits)
        window = text[start:end]
        found = detector.detect(window, language=language, context=context)
        if not isinstance(found, tuple):
            raise ValueError('Detector results must be a tuple')
        for hit in found:
            if not isinstance(hit, Detection):
                raise ValueError('Detector result must contain Detection values')
            hit.__post_init__()
            hit.validate(window, coordinate_space='canonical')
            if hit.detector_id != detector.capabilities.id or hit.entity_type not in detector.capabilities.entity_types:
                raise ValueError('Detector finding does not match its declared capabilities')
            if any(span.end > len(window) for span in hit.source_ranges):
                raise ValueError('Detector source range exceeds the window')
            projected = replace(hit, start=start + hit.start, end=start + hit.end,
                                source_ranges=(), mapping_quality='unavailable')
            key = (projected.entity_type, projected.start, projected.end)
            previous = best.get(key)
            if previous is None or (hit.score is not None and (previous.score is None or hit.score > previous.score)):
                best[key] = projected
    return tuple(best.values()), name
