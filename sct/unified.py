"""Collect findings on one text stage, then protect their resolved spans once."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import os
import re
import threading
import time
from typing import TYPE_CHECKING

from sct.coordinates import SourceMap
from sct.detectors import DetectorSpec, DetectorUnavailable
from sct.detectors_builtin import BuiltinRegexDetector
from sct.findings import Detection, DetectionContext
from sct.policy_v2 import FindingDecision, UnifiedPrivacyPolicy
from sct.privacy import ProcessingMetrics
from sct.utils import constants, normtext
from sct.utils.anonymization_map import AnonymizationMap
from sct.utils.markers import PLACEHOLDER_TOKEN
from sct.utils.process_result import ProcessMetadata, ProcessResult
from sct.utils.synthetic import FAKER_GENERATORS
from sct.utils.ner_spans import drop_allowed

if TYPE_CHECKING:
    from sct.sct import TextCleaner


@dataclass(frozen=True)
class UnifiedDocument:
    """Private document handoff: output map projects to the original input string."""
    result: ProcessResult = field(repr=False)
    source_map: SourceMap = field(repr=False)


def _whole_edit(mapping, transform):
    before = mapping._text
    after = transform(before)
    if after == before:
        return mapping
    return mapping.replace(0, len(before), after)


def _sub(mapping, pattern, replacement):
    matches = list(pattern.finditer(mapping._text))
    for match in reversed(matches):
        value = replacement(match) if callable(replacement) else replacement
        mapping = mapping.replace(match.start(), match.end(), value)
    return mapping


def _canonical(cleaner: TextCleaner, text: str) -> SourceMap:
    cfg = cleaner.cfg
    mapping = SourceMap.identity(text)
    if cfg.check_fix_bad_unicode:
        # Whole-stage repair has a proven cover, but no claimed character alignment.
        mapping = _whole_edit(mapping, cleaner._fix_bad_unicode)
    if cfg.check_to_ascii_unicode:
        transform = cleaner._text_normaliser.to_ascii_unicode
        parts = [transform(char) for char in mapping._text]
        if ''.join(parts) == transform(mapping._text):
            for i in reversed(range(len(parts))):
                mapping = mapping.replace(i, i + 1, parts[i])
        else:
            mapping = _whole_edit(mapping, transform)
    if cfg.check_remove_emoji:
        mapping = _whole_edit(mapping, cleaner._text_normaliser.remove_emoji)
    # Decode escaped brackets after the HTML step so escaped markup remains text.
    if cfg.check_replace_html:
        mapping = _sub(mapping, cleaner._html_regex, cfg.replace_with_html)
    if cfg.check_fix_bad_unicode:
        mapping = _sub(mapping, normtext._MARKUP_OR_ENTITY, lambda m: normtext.unescape_markup(m.group()))
    if cfg.check_remove_isolated_letters:
        mapping = _sub(mapping, constants.ISOLATED_LETTERS_REGEX, '')
    if cfg.check_remove_isolated_special_symbols:
        mapping = _whole_edit(mapping, cleaner._remove_isolated_special_symbols)
    if cfg.check_normalize_whitespace:
        mapping = _sub(mapping, re.compile(r'\s+'), ' ')
        mapping = _sub(mapping, re.compile(r'^ +| +$'), '')
    for callback in cfg.custom_pipeline_steps:
        if cfg.source_mapping == 'require_source':
            raise ValueError('Strict source mapping does not support opaque custom transforms')
        try:
            mapping = mapping.transform(callback)
        except Exception:
            raise ValueError('Custom transform failed in unified processing') from None
    return mapping


def _resolve(decisions: tuple[FindingDecision, ...]) -> list[tuple[int, int, FindingDecision]]:
    protected = sorted((d for d in decisions if d.action != 'keep'),
                       key=lambda d: (d.detection.start, d.detection.end, d.detection.detector_id, d.detection.entity_type))
    groups: list[list[FindingDecision]] = []
    ends: list[int] = []
    for decision in protected:
        hit = decision.detection
        if groups and hit.start < ends[-1]:
            groups[-1].append(decision)
            ends[-1] = max(ends[-1], hit.end)
        else:
            groups.append([decision])
            ends.append(hit.end)
    resolved = []
    for group, end in zip(groups, ends, strict=True):
        winner = min(group, key=lambda d: (
            d.action != 'redact', -(d.detection.end - d.detection.start),
            d.detection.start, d.detection.detector_id, d.detection.entity_type))
        resolved.append((group[0].detection.start, end, winner))
    return resolved


class UnifiedProcessor:
    """Keep provider instances on the process and thread that created them."""

    def __init__(self, cleaner: TextCleaner):
        self.cleaner = cleaner
        self._local = threading.local()
        self.specs = cleaner.cfg.detector_specs or (
            (DetectorSpec('builtin_regex'), DetectorSpec('builtin_ner')) if cleaner.cfg.check_ner_process
            else (DetectorSpec('builtin_regex'),))
        self.policy = cleaner._unified_policy or UnifiedPrivacyPolicy('default', '1', {})

    def _detectors(self):
        if getattr(self._local, 'pid', None) != os.getpid():
            try:
                detectors = tuple(self.cleaner._detector_registry.create(spec, self.cleaner.cfg) for spec in self.specs)
            except DetectorUnavailable:
                raise DetectorUnavailable("Selected detector unavailable; check dependencies and network permission") from None
            except Exception:
                raise ValueError("Selected detector could not initialize; check its configuration") from None
            self._local.detectors = detectors
            self._local.pid = os.getpid()
        return self._local.detectors

    def process(self, text: str, context: DetectionContext) -> UnifiedDocument:
        cleaner, cfg = self.cleaner, self.cleaner.cfg
        started = time.perf_counter()
        if not text or text.isspace():
            empty_metadata = cleaner._empty_metadata(text)
            empty_metadata['findings_v2'] = []
            empty_metadata['counts_v2'] = {'version': 2, 'detector_hits': 0, 'resolved_replacements': 0}
            if cleaner._include_entities:
                empty_metadata['findings_v2_values'] = []
            empty_metadata.pop('findings', None)
            empty_metadata['provenance_v2'] = {'version': 2, 'detectors': [],
                'policy': {'name': self.policy.name, 'version': self.policy.version}}
            return UnifiedDocument(
                ProcessResult('', '', None, empty_metadata), SourceMap.identity(text).replace(0, len(text), ''))
        classes = None
        if cleaner._gliclass is not None:
            try:
                classes = cleaner._gliclass.classify(text)
            except Exception:
                raise ValueError("Document classification failed") from None
        language, confidence, source, margin = cleaner._detect_language(text)
        cleaner._document_context.language = language
        cleaner._document_context.warnings = set()
        amap = AnonymizationMap() if cfg.replacement_mode == 'reversible' else None
        cleaner._document_context.anon_map = amap
        mapping = _canonical(cleaner, text)
        canonical = mapping._text
        if amap is not None:
            amap.reserved.update(PLACEHOLDER_TOKEN.findall(text))
            amap.reserved.update(PLACEHOLDER_TOKEN.findall(canonical))
            amap.reserved.update(tag for tag in cleaner._custom_markers if tag in text or tag in canonical)
        synthetic = cleaner._synthetic_replacer
        if synthetic:
            synthetic.reset_cache(text, language, cleaner._language_low_confidence(source, confidence, margin))
        hits = []
        provenance = []
        for detector in self._detectors():
            capabilities = detector.capabilities
            if context != DetectionContext() and not capabilities.supports_context:
                # Native detectors have no context-specific behavior; schema hints remain caller-owned.
                if capabilities.id not in ('builtin_regex', 'builtin_ner'):
                    raise ValueError('Selected detector does not support context')
            helper = detector._helper if isinstance(detector, BuiltinRegexDetector) else None
            if helper is not None:
                helper._document_context.warnings = cleaner._document_context.warnings
            try:
                found = detector.detect(canonical, language=language, context=context)
                if not isinstance(found, tuple):
                    raise ValueError('Detector results must be a tuple')
                for hit in found:
                    if not isinstance(hit, Detection):
                        raise ValueError('Detector result must contain Detection values')
                    hit.__post_init__()
                    hit.validate(canonical, coordinate_space='canonical')
                    if hit.detector_id != capabilities.id or hit.entity_type not in capabilities.entity_types:
                        raise ValueError('Detector finding does not match its declared capabilities')
                    if any(r.end > len(text) for r in hit.source_ranges):
                        raise ValueError('Detector source range exceeds the input')
                    projection = mapping.project(hit.start, hit.end)
                    hits.append(replace(hit, source_ranges=projection.ranges, mapping_quality=projection.mapping_quality))
            except DetectorUnavailable:
                raise DetectorUnavailable('Selected detector is unavailable during analysis') from None
            except Exception:
                raise ValueError('Detector failed or returned invalid findings') from None
            finally:
                if helper is not None:
                    helper._document_context.clear()
            provenance.append(asdict(capabilities))
        hits = drop_allowed(cleaner._allow_regex, canonical, hits, lambda hit: (hit.start, hit.end))
        if amap is not None:
            for kind in {hit.entity_type for hit in hits}:
                marker = re.compile(rf"<{re.escape(kind)}_[0-9]+>")
                amap.reserved.update(marker.findall(text))
                amap.reserved.update(marker.findall(canonical))
        decisions = self.policy.evaluate(tuple(hits))
        resolved = _resolve(decisions)
        replacements = []
        for start, end, decision in resolved:
            kind = decision.detection.entity_type
            original = canonical[start:end]
            if decision.action == 'redact':
                value = '<REDACTED>'
            elif amap is not None:
                value = amap.next_placeholder(kind)
                amap.add(value, original, kind, start, end)
            elif synthetic and kind in FAKER_GENERATORS:
                value = synthetic.get_replacement(kind, original)
            else:
                label = constants.LEGACY_TYPE_NAMES.get(kind, kind) if cfg.legacy_type_names else kind
                tag = cleaner._tags.get(label, f'<{label}>')
                value = tag if tag is not None else ''.join(constants.CURRENCIES.get(char, char) for char in original)
            replacements.append((start, end, value))
        for start, end, value in reversed(replacements):
            mapping = mapping.replace(start, end, value)
        metadata: ProcessMetadata = {
            'source_chars': len(text), 'regex_counts': {}, 'entity_counts': {},
            'language_confidence': confidence, 'language_source': source,
            'warnings': sorted(cleaner._document_context.warnings),
            'provenance': {'backend': None, 'models': []},
            'findings_v2': [asdict(decision.detection) | {
                'source_ranges': [asdict(r) for r in decision.detection.source_ranges],
                'action': decision.action, 'reason': decision.reason} for decision in decisions],
            'counts_v2': {'version': 2, 'detector_hits': len(hits), 'resolved_replacements': len(resolved)},
            'provenance_v2': {'version': 2, 'detectors': provenance,
                              'policy': {'name': self.policy.name, 'version': self.policy.version}},
        }
        if cleaner._include_entities:
            metadata['findings_v2_values'] = [
                {'finding_index': index, 'coordinate_space': 'canonical', 'value': canonical[hit.start:hit.end]}
                for index, hit in enumerate(hits)]
        if amap is not None:
            if cleaner._token_store is not None:
                metadata['anon_map_reference'] = cleaner._token_store.save(amap)
            else:
                metadata['anon_map'] = amap
        if classes is not None:
            metadata['classes'] = classes
        output = mapping._text
        stat = cleaner._build_stat_text(output, language) if cfg.check_statistical_model_processing else None
        if cleaner._metrics_callback:
            cleaner._metrics_callback(ProcessingMetrics(
                (time.perf_counter() - started) * 1000, language, None, self.policy.name, {}))
        return UnifiedDocument(ProcessResult(output, stat, language, metadata), mapping)
