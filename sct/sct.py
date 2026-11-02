"""
Comprehensive text cleaning and preprocessing pipeline.
"""
import asyncio
import logging
import os
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple

from sct.config import TextCleanerConfig, GLINER_BACKENDS, _config_from_module_globals
from sct.privacy import PrivacyPolicy, ProcessingMetrics, TokenStore
from sct.utils import constants, contact, datetime, ner, normtext, resources, special, stopwords
from sct.utils.anonymization_map import AnonymizationMap
from sct.utils.process_result import ProcessResult

logger = logging.getLogger(__name__)
# A _protect key (an emitted value hidden from later steps).
_KEY = re.compile('\U00100000[\U000F0000-\U0010FFFF]\U00100001')

class TextCleaner:

    def __init__(
        self,
        cfg: Optional[TextCleanerConfig] = None,
        *,
        privacy_policy: Optional[PrivacyPolicy] = None,
        token_store: Optional[TokenStore] = None,
        include_entities: bool = False,
        metrics_callback: Optional[Callable[[ProcessingMetrics], None]] = None,
    ):
        """Initialize the text cleaning pipeline.

        Args:
            cfg: Immutable configuration. If None, reads from module-level
                 config variables (backward compatible).
        """
        self.cfg = cfg or _config_from_module_globals()
        names = constants.LEGACY_TYPE_NAMES if self.cfg.legacy_type_names else {}
        self._phone_type, self._ip_type = names.get('PHONE', 'PHONE'), names.get('IP', 'IP')
        if (
            privacy_policy is not None
            and self.cfg.replacement_mode != 'placeholder'
        ):
            raise ValueError(
                "privacy_policy currently requires replacement_mode='placeholder'"
            )
        self._privacy_policy = privacy_policy
        self._token_store = token_store
        self._include_entities = include_entities
        self._metrics_callback = metrics_callback
        self._document_context = threading.local()
        # NER reads the placeholder-mode tags; fakes / indexed placeholders go in after NER.
        self._defer_tags = (self.cfg.ner_sees_tags and self.cfg.check_ner_process
                            and self.cfg.replacement_mode != 'placeholder')
        self._tags = {
            'HTML': self.cfg.replace_with_html, 'URL': self.cfg.replace_with_url,
            'EMAIL': self.cfg.replace_with_email, self._ip_type: self.cfg.replace_with_ips,
            'DATE': self.cfg.replace_with_dates, 'YEAR': self.cfg.replace_with_years,
            self._phone_type: self.cfg.replace_with_phone_numbers,
            'NUMBER': self.cfg.replace_with_numbers, 'CURRENCY': self.cfg.replace_with_currency_symbols,
            'IBAN': self.cfg.replace_with_iban, 'CARD': self.cfg.replace_with_cards,
            'ID': self.cfg.replace_with_identifiers,
        }
        # Markers that are not <TAG>-shaped ("EMAIL"): NER protects them like <EMAIL> (ner.regex_markers).
        self._custom_markers = tuple(sorted({
            tag for tag in self._tags.values() if tag and not ner._PLACEHOLDER_TOKEN.fullmatch(tag)}))
        self._marker_regex = re.compile('|'.join([ner._PLACEHOLDER_TOKEN.pattern, *map(
            re.escape, sorted(self._custom_markers, key=len, reverse=True))]))

        # Language detector: tuple restricts; else all 75 Lingua languages unless turned off
        if isinstance(self.cfg.language, tuple):
            self._detector = resources.build_detector(frozenset(self.cfg.language))
        elif self.cfg.detect_all_languages:
            self._detector = resources.build_all_detector()
        else:
            self._detector = resources.build_detector(self.cfg.supported_languages)

        # Instance-scoped stopwords (with custom override support)
        self.ProcessStopwords = stopwords.ProcessStopwords(
            supported_languages=self.cfg.supported_languages,
            custom_stopwords=(dict(self.cfg.custom_stopwords)
                              if self.cfg.custom_stopwords and self.cfg.check_remove_stext_custom_stop_words
                              else None),
        )

        # Instance-scoped datetime (with custom month names -> per-instance regex)
        custom_months = self.cfg.custom_month_names or {}
        slash_year = self.cfg.date_month_name_slash_year
        mn_dict = resources.configured_month_names(self.cfg)
        if mn_dict is not None:
            date_regex = constants.build_date_regex(mn_dict, slash_year=slash_year)
            fuzzy_all, fuzzy_by_lang = constants.build_fuzzy_vocabulary(
                mn_dict,
                {**constants._LANG_TO_VOCAB,
                 **{lang: resources.language_to_iso(lang) for lang in custom_months}},
            )
            self.ProcessDateTime = datetime.ProcessDateTime(
                date_regex=date_regex,
                fuzzy_vocabulary=fuzzy_all,
                fuzzy_vocabulary_by_lang=fuzzy_by_lang,
            )
        else:
            self.ProcessDateTime = datetime.ProcessDateTime()
        # fold_styled_characters: dates, like phones, read any space as " " and any dash as "-" (1987\u201105\u201112),
        # any dot form as "." (1990﹒ 1﹒ 15﹒), and a month-name date's day may be a circled number (⑮ January 1990).
        self._date_view = (
            contact.DigitView(constants.build_date_regex(mn_dict, styled=True, slash_year=slash_year),
                              contact.DATE_SEPARATOR_VIEW)
            if self.cfg.fold_styled_characters else self.ProcessDateTime._date_regex)

        self.ProcessContacts = contact.ProcessContacts()
        self._html_regex = constants.HTML_REGEX if self.cfg.html_tag_strict else constants.HTML_REGEX_ANY
        self._url_regex = constants.URL_REGEX if self.cfg.url_trailing_punct_outside else constants.URL_REGEX_LEGACY
        self.ProcessSpecialSymbols = special.ProcessSpecialSymbols()
        self.NormaliseText = normtext.NormaliseText()

        # Synthetic replacer: shared by regex pipeline steps + NER processor
        self._synthetic_replacer = None
        if self.cfg.replacement_mode == 'synthetic':
            from sct.utils.synthetic import SyntheticReplacer
            self._synthetic_replacer = SyntheticReplacer(
                seed=self.cfg.synthetic_seed, per_document=self.cfg.synthetic_per_document,
                locale=self.cfg.synthetic_locale,
                low_confidence_locale=self.cfg.synthetic_low_confidence_locale,
                collapse_period=self.cfg.synthetic_collapse_period,
                reference_date=self.cfg.synthetic_reference_date,
            )

        self.GeneralNER: Optional[ner.GeneralNER] = None
        if self.cfg.check_ner_process:
            # Build GLiNER config dict (if needed)
            gliner_config = None
            needs_gliner = self.cfg.ner_backend in GLINER_BACKENDS
            if needs_gliner:
                gliner_config = {
                    'model': self.cfg.gliner_model,
                    'revision': self.cfg.gliner_revision,
                    'variant': self.cfg.gliner_variant,
                    'labels': self.cfg.gliner_labels,
                    'threshold': self.cfg.effective_gliner_threshold,
                    'label_map': (
                        dict(self.cfg.gliner_label_map)
                        if self.cfg.gliner_label_map else None
                    ),
                    'label_descriptions': (
                        dict(self.cfg.gliner_label_descriptions)
                        if self.cfg.gliner_label_descriptions else None
                    ),
                    'onnx': self.cfg.gliner_onnx,
                    'legacy_type_names': self.cfg.legacy_type_names,
                }

            # Determine torch model names (if needed)
            torch_model_names = None
            if self.cfg.ner_backend in ('torch', 'ensemble_torch'):
                torch_model_names = dict(self.cfg.torch_ner_models) if self.cfg.torch_ner_models else None

            self.GeneralNER = ner.GeneralNER(
                device=self.cfg.ner_device,
                cuda_device_id=self.cfg.cuda_device_id,
                onnx_cuda_provider_options=(
                    dict(self.cfg.onnx_cuda_provider_options)
                    if self.cfg.onnx_cuda_provider_options else None
                ),
                model_names=dict(self.cfg.ner_models) if self.cfg.ner_models else None,
                model_revisions=(
                    dict(self.cfg.ner_model_revisions)
                    if self.cfg.ner_model_revisions else None
                ),
                ner_backend=self.cfg.ner_backend,
                gliner_config=gliner_config,
                torch_model_names=torch_model_names,
                torch_model_revisions=(
                    dict(self.cfg.torch_ner_model_revisions)
                    if self.cfg.torch_ner_model_revisions else None
                ),
                ner_batch_size=self.cfg.ner_batch_size,
                ner_subword_widen=self.cfg.ner_subword_widen,
                ner_subword_widen_same_script=self.cfg.ner_subword_widen_same_script,
                ensemble_models=dict(self.cfg.ner_ensemble) if self.cfg.ner_ensemble is not None else None,
                ensemble_default_keys=(
                    tuple(self.cfg.ner_ensemble_default_keys)
                    if self.cfg.ner_ensemble_default_keys is not None
                    else None
                ),
                replacement_mode=self.cfg.replacement_mode,
                synthetic_replacer=self._synthetic_replacer,
                regex_markers=self._custom_markers,
                onnx_intra_op_threads=self.cfg.onnx_intra_op_threads,
                onnx_inter_op_threads=self.cfg.onnx_inter_op_threads,
                ner_chunk_overlap=self.cfg.ner_chunk_overlap,
                ner_ensemble_vote=self.cfg.ner_ensemble_vote,
                ner_propagate_masked=self.cfg.ner_propagate_masked,
                ner_propagate_min_score=self.cfg.ner_propagate_min_score,
                ner_merge_location_gap=self.cfg.ner_merge_location_gap,
                ner_snap_word_edges=self.cfg.ner_snap_word_edges,
            )
        # GLiClass document-level pre-classification (optional, lazy-loaded)
        self._gliclass: Any = None
        if self.cfg.check_classify_document:
            from sct.utils.gliclass_adapter import GLiClassAdapter
            from sct.config import GLICLASS_DEFAULT_MODEL
            self._gliclass = GLiClassAdapter(
                model_id=self.cfg.gliclass_model or GLICLASS_DEFAULT_MODEL,
                revision=self.cfg.gliclass_revision,
                labels=self.cfg.gliclass_labels,
                threshold=self.cfg.gliclass_threshold,
                classification_type=self.cfg.gliclass_classification_type,
                onnx=self.cfg.gliclass_onnx,
                # Auto stays on CPU (0.7 behaviour); only an explicit 'cuda' moves GLiClass.
                device=f"cuda:{self.cfg.cuda_device_id}" if self.cfg.ner_device == 'cuda' else 'cpu',
            )

        self._pipeline: List[Callable[[str], str]] = []
        self._post_fuzzy_pipeline: List[Callable[[str], str]] = []
        self._init_pipeline()

    def _init_pipeline(self):
        """Build the pipeline steps lists based on config.

        Steps that require language context (fuzzy date replacement) are split
        into pre- and post-fuzzy lists.  _process_single calls them in order:
        self._pipeline → fuzzy (explicit, with ctx) → self._post_fuzzy_pipeline.
        """
        cfg = self.cfg

        if cfg.check_fix_bad_unicode:
            self._pipeline.append(self._fix_bad_unicode)
        if cfg.check_to_ascii_unicode:
            self._pipeline.append(self._to_ascii_unicode)
        if cfg.check_remove_emoji:
            self._pipeline.append(self._remove_emoji)
        # Steps before this index only normalise; markers they surface are source text.
        self._normalise_steps = len(self._pipeline)
        if cfg.check_fix_bad_unicode:  # escaped "<b>" stays text: decoded after the view, hidden from tags
            self._pipeline.append(self._unescape_markup)
        if cfg.check_replace_html:
            self._pipeline.append(self._replace_html)
        if cfg.check_fix_bad_unicode:  # brackets hide from the tag step only; contacts see "john<at>x.com"
            self._pipeline.append(self._restore_brackets)
        if cfg.check_replace_urls:
            self._pipeline.append(self._replace_urls)
        if cfg.check_replace_emails:
            self._pipeline.append(self._replace_emails)
        if cfg.check_replace_ips:
            self._pipeline.append(self._replace_ips)
        if cfg.check_replace_dates:
            self._pipeline.append(self._replace_dates)
        # fuzzy_replace_dates runs here (between replace_dates and replace_years)
        # but is called explicitly in _process_single with language context.
        # Phones before years, so a year step cannot split "+31 6 1234 2019".
        # Checksum-valid IBANs/cards and identifier tokens before phones, so a phone cannot take their digits.
        if cfg.check_replace_iban_cards:
            self._post_fuzzy_pipeline.append(self._replace_iban_cards)
        if cfg.check_replace_identifiers:
            self._post_fuzzy_pipeline.append(self._replace_identifiers)
        if cfg.check_replace_phone_numbers:
            self._post_fuzzy_pipeline.append(self._replace_phone_numbers)
        if cfg.check_replace_years:
            self._post_fuzzy_pipeline.append(self._replace_years)
        if cfg.check_replace_numbers:
            self._post_fuzzy_pipeline.append(self._replace_numbers)
        if cfg.check_replace_currency_symbols:
            self._post_fuzzy_pipeline.append(self._replace_currency_symbols)
        # Emitted tokens and kept IPs are hidden from the steps above; put them back.
        self._post_fuzzy_pipeline.append(self._restore_protected)
        if cfg.check_remove_isolated_letters:
            self._post_fuzzy_pipeline.append(self._remove_isolated_letters)
        if cfg.check_remove_isolated_special_symbols:
            self._post_fuzzy_pipeline.append(self._remove_isolated_special_symbols)
        if cfg.check_normalize_whitespace:
            self._post_fuzzy_pipeline.append(self._normalize_whitespace)
        # Deferred values stay hidden through the cleanup above, so a tag it would remove
        # ("[EMAIL]" under bracket removal) still carries its value to the swap after NER.
        self._post_fuzzy_pipeline.append(self._restore_deferred)
        # User-provided custom steps: after the built-in pre-NER steps, before NER and _after_ner
        for step_fn in cfg.custom_pipeline_steps:
            self._post_fuzzy_pipeline.append(step_fn)

    def _detect_language(self, text: str) -> Tuple[Optional[str], Optional[float], str, Optional[float]]:
        """Detect language as a pure function (no instance mutation).

        Returns ``(name, confidence, source, margin)``; source is ``pinned``, ``detected``,
        ``fallback`` (tuple restriction fell back to its first entry) or ``none``;
        margin is top minus runner-up confidence, only for ``detected``.

        Behavior depends on ``cfg.language``:
          - ``str``: pinned to that language (skip detection)
          - ``tuple``: restrict detection to those languages (detect per text)
          - ``None``: auto-detection (all 75 Lingua languages by default)
        """
        language_config = self.cfg.language

        # Pinned: single string → return immediately (already resolved to uppercase)
        if isinstance(language_config, str):
            return language_config, 1.0, 'pinned', None

        restricted = isinstance(language_config, tuple)
        if not restricted and not self._language_needed():
            return None, None, 'none', None

        values = self._detector.compute_language_confidence_values(text)
        # Same "unknown" rule as Lingua's detect_language_of: nothing scored, or a tie at the top.
        if values and values[0].value > 0 and (len(values) < 2 or values[1].value < values[0].value):
            top = values[0]
            if not restricted or top.language.name in language_config:
                # Lingua's sum order varies per process (last-ulp noise); round so workers match.
                margin = top.value - (values[1].value if len(values) > 1 else 0.0)
                return top.language.name, round(top.value, 6), 'detected', round(margin, 6)
        if restricted:
            first = language_config[0]
            conf = next((v.value for v in values if v.language.name == first), 0.0)
            return first, round(conf, 6), 'fallback', None
        return None, None, 'none', None

    def _language_needed(self) -> bool:
        """Whether auto-detection runs at all (else no language is expected)."""
        return any([self.cfg.check_detect_language, self.cfg.check_ner_process,
                    self.cfg.check_remove_stopwords])

    def _language_low_confidence(self, source, confidence, margin) -> bool:
        """Whether a detected language gets the language_low_confidence warning."""
        return source == 'detected' and (
            confidence < self.cfg.language_confidence_warning
            or margin < self.cfg.language_margin_warning)

    def _process_single(self, text: str) -> ProcessResult:
        try:
            return self._process_single_with_context(text)
        finally:
            self._document_context.anon_map = None
            self._document_context.protected = None
            self._document_context.regex_counts = None
            self._document_context.deferred = None

    def _process_single_with_context(self, text: str) -> ProcessResult:
        """Process a single text through the entire pipeline.

        Returns:
            ProcessResult (unpacks as 3-tuple for backward compat).
        """
        state = self._before_ner(text)
        current_text, language, anon_map = state[0], state[1], state[2]

        # NER processing; always detailed (same detection) so entity counts are known.
        findings = ()
        if self.cfg.check_ner_process and self.GeneralNER is not None:
            detailed = self.GeneralNER.ner_process_detailed(
                current_text,
                positional_tags=self.cfg.positional_tags,
                ner_confidence_threshold=self.cfg.effective_ner_confidence_threshold,
                language=language,
                anon_map=anon_map,
                privacy_policy=self._privacy_policy,
                source_markers=state[9],
            )
            current_text = detailed.text
            findings = detailed.findings
        return self._after_ner(state, current_text, findings)

    def normalise(self, text: str) -> str:
        """The steps before the first regex step (unicode fix, ASCII, emoji), as ``process`` runs them.

        The streaming split guard reads this same view, so the two cannot differ.
        """
        for step in self._pipeline[:self._normalise_steps]:
            text = step(text)
        return text

    def _before_ner_isolated(self, text: str) -> tuple:
        """``_before_ner`` that clears the per-thread document context afterwards."""
        try:
            return self._before_ner(text)
        finally:
            self._document_context.anon_map = None
            self._document_context.protected = None
            self._document_context.regex_counts = None
            self._document_context.deferred = None

    def _before_ner(self, text: str) -> tuple:
        """Every step before NER.

        Returns ``(text, language, anon_map, doc_classes, started_at, regex_counts,
        language_confidence, language_source, source_chars, source_markers, language_margin)``.
        """
        started_at = time.perf_counter()

        # Per-document anonymization map (reversible mode only)
        anon_map = None
        if self.cfg.replacement_mode == 'reversible':
            anon_map = AnonymizationMap()
        self._document_context.anon_map = anon_map
        self._document_context.protected = {}
        # The dict travels in the returned state, so it survives the context reset.
        regex_counts: Dict[str, Dict[str, int]] = {}
        self._document_context.regex_counts = regex_counts
        # (protected key -> tag NER reads, [(tag, value or None) in text order])
        deferred = ({}, []) if self._defer_tags else None
        self._document_context.deferred = deferred

        # Document-level classification (before any text modification)
        doc_classes = None
        if self.cfg.check_classify_document and self._gliclass is not None:
            doc_classes = self._gliclass.classify(text)

        # Detect language (pure function, thread-safe)
        language, language_confidence, language_source, language_margin = self._detect_language(text)

        # Reset per-document synthetic consistency cache (thread-safe via threading.local);
        # the language picks the Faker locale.
        if self._synthetic_replacer:
            self._synthetic_replacer.reset_cache(text, language, self._language_low_confidence(
                language_source, language_confidence, language_margin))

        current_text = text

        # Pre-fuzzy pipeline steps (unicode fix → html → urls → emails → ips → dates)
        current_text = self.normalise(current_text)
        # Placeholder spellings the source holds, raw or once normalised ("＜LONDON＞",
        # "&lt;LONDON&gt;"): NER gives them no protection and reversible mode never
        # allocates them.
        source_markers = frozenset(ner._PLACEHOLDER_TOKEN.findall(text)) | frozenset(
            ner._PLACEHOLDER_TOKEN.findall(normtext.unescape_markup(current_text))) | frozenset(
            tag for tag in self._custom_markers if tag in text or tag in current_text)
        if anon_map is not None:
            anon_map.reserved.update(source_markers)
        # (marker, emitted) for each marker occurrence, recorded by _restore_protected
        self._document_context.marker_order = [] if source_markers else None
        for step in self._pipeline[self._normalise_steps:]:
            current_text = step(current_text)

        # Fuzzy date replacement — requires language context, called explicitly
        # to avoid thread-local; positioned between replace_dates and replace_years.
        if self.cfg.check_fuzzy_replace_dates:
            current_text = self.ProcessDateTime.fuzzy_replace_dates(
                current_text,
                replace_with=lambda orig: self._emit('DATE', orig, self.cfg.replace_with_dates),
                score_cutoff=self.cfg.fuzzy_date_score_cutoff,
                language=language,
            )

        # Post-fuzzy pipeline steps (phones → years → numbers → symbols → whitespace)
        for step in self._post_fuzzy_pipeline:
            current_text = step(current_text)
        source_markers = self._source_marker_spans(current_text, source_markers)

        return (current_text, language, anon_map, doc_classes, started_at, regex_counts,
                language_confidence, language_source, len(text), source_markers, language_margin,
                deferred[1] if deferred else None)

    def _after_ner(self, state: tuple, current_text: str, findings) -> ProcessResult:
        """Every step after NER, for the document *state* from ``_before_ner``."""
        (_, language, anon_map, doc_classes, started_at, regex_counts,
         language_confidence, language_source, source_chars, _, language_margin, deferred) = state
        current_text = self._swap_deferred(current_text, deferred, state[0], findings, anon_map)

        # Statistical model processing (always returns stext, even if None)
        stext = None
        if self.cfg.check_statistical_model_processing:
            stext = self._statistical_model_processing(current_text, language)

        # Value-free counts are always present; entity text only on opt-in below.
        # Overlapping findings count once, as their union labelled by the highest
        # score: the same rule NER uses when it replaces them.
        spans: List[list] = []
        for finding in sorted(findings, key=lambda f: (f.entity.start, -f.entity.end)):
            e = finding.entity
            if spans and e.start < spans[-1][3]:
                last = spans[-1]
                if e.score > last[1]:
                    last[0], last[1] = e.entity_type, e.score
                last[3] = max(last[3], e.end)
            else:
                spans.append([e.entity_type, e.score, e.start, e.end])
        entity_counts: Dict[str, Dict[str, int]] = {}
        for entity_type, _, start, end in spans:
            entry = entity_counts.setdefault(entity_type, {'count': 0, 'chars': 0})
            entry['count'] += 1
            entry['chars'] += end - start
        warnings = set()
        if language is None and language_source == 'none' and self._language_needed():
            warnings.add('language_unknown')
        if language_source == 'fallback':
            warnings.add('language_fallback')
        if self._language_low_confidence(language_source, language_confidence, language_margin):
            warnings.add('language_low_confidence')
        ner_on = self.cfg.check_ner_process and self.GeneralNER is not None
        if ner_on and self.GeneralNER.uses_fallback_models(language):
            warnings.add('ner_fallback_models')
        models = self.GeneralNER.models_for(language) if ner_on else []
        if any(m['provider'] == 'CPUExecutionProvider' for m in models) \
                and self.GeneralNER.device == 'cuda':
            warnings.add('cuda_fallback')
        if any(m['provider'] == 'cpu' for m in models) and self.GeneralNER.device == 'mps':
            warnings.add('mps_fallback')
        metadata: Dict[str, Any] = {
            'source_chars': source_chars,
            'regex_counts': regex_counts,
            'entity_counts': entity_counts,
            'language_confidence': language_confidence,
            'language_source': language_source,
            'warnings': sorted(warnings),
            'provenance': {
                'backend': self.cfg.ner_backend if ner_on else None,
                'models': models,
            },
        }
        if anon_map is not None:
            if self._token_store is not None:
                metadata['anon_map_reference'] = self._token_store.save(anon_map)
            else:
                metadata['anon_map'] = anon_map
        if doc_classes is not None:
            metadata['classes'] = doc_classes
        if self._privacy_policy is not None:
            metadata['policy'] = {
                'name': self._privacy_policy.name,
                'version': self._privacy_policy.version,
            }
        if self._include_entities or self._privacy_policy is not None:
            metadata['findings'] = findings

        if self._metrics_callback is not None:
            self._metrics_callback(ProcessingMetrics(
                duration_ms=(time.perf_counter() - started_at) * 1000,
                language=language,
                backend=self.cfg.ner_backend if self.cfg.check_ner_process else None,
                policy=self._privacy_policy.name if self._privacy_policy else None,
                entity_counts={k: v['count'] for k, v in entity_counts.items()},
            ))

        return ProcessResult(current_text, stext, language, metadata=metadata)

    def process_batch(self, texts: List[str], batch_size: Optional[int] = None) -> List[ProcessResult]:
        """Process multiple texts with bounded document concurrency.

        Returns:
            List of ProcessResult (each unpacks as 3-tuple for backward compat).
        """
        if batch_size is not None and batch_size <= 0:
            raise ValueError("batch_size must be >= 1")
        if not texts:
            return []

        results: List[Optional[ProcessResult]] = [None] * len(texts)
        to_process = []

        for i, text in enumerate(texts):
            if not isinstance(text, str):
                raise ValueError(f"Input must be string, got {type(text)}")
            if not text or text.isspace():
                results[i] = ProcessResult("", "", None, metadata=self._empty_metadata(text))
            else:
                to_process.append((i, text))

        if not to_process:
            return [r for r in results if r is not None]

        # Parallel processing: ONNX Runtime releases the GIL during C++ inference
        # ops, so threads achieve real concurrency for the NER inference bottleneck.
        worker_limit = batch_size if batch_size is not None else (os.cpu_count() or 4)
        max_workers = min(len(to_process), worker_limit)

        def run(fn):
            if max_workers <= 1:
                return [fn(text) for _, text in to_process]
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                return list(executor.map(fn, [text for _, text in to_process]))

        if (
            self.cfg.check_ner_process
            and self.GeneralNER is not None
            and self.cfg.ner_batch_size > 1
        ):
            # Batched NER: pre-NER steps per document in threads, one NER call over
            # all documents (model calls batch chunks across documents), then the rest.
            replacer = self._synthetic_replacer

            def before_ner(text):
                # Keep the regex steps' synthetic cache with the document for its NER step.
                state = self._before_ner_isolated(text)
                return state, replacer.swap_cache() if replacer else None

            staged = run(before_ner)
            states = [state for state, _ in staged]

            def restore_synthetic(index):
                if replacer:
                    replacer.swap_cache(staged[index][1])

            ner_results = self.GeneralNER.ner_process_batch_detailed(
                [state[0] for state in states],
                [state[1] for state in states],
                positional_tags=self.cfg.positional_tags,
                ner_confidence_threshold=self.cfg.effective_ner_confidence_threshold,
                anon_maps=[state[2] for state in states],
                privacy_policy=self._privacy_policy,
                before_finalize=restore_synthetic,
                max_workers=max_workers,
                source_markers=[state[9] for state in states],
            )
            for (i, _), state, ner_result in zip(to_process, states, ner_results):
                results[i] = self._after_ner(state, ner_result.text, ner_result.findings)
        else:
            # Each text goes through the full pipeline independently.
            for (i, _), result in zip(to_process, run(self._process_single)):
                results[i] = result

        return [r for r in results if r is not None]

    def _empty_metadata(self, text: str) -> Dict[str, Any]:
        """Metadata for empty or whitespace input (nothing ran)."""
        ner_on = self.cfg.check_ner_process and self.GeneralNER is not None
        metadata = {
            'source_chars': len(text),
            'regex_counts': {},
            'entity_counts': {},
            'language_confidence': None,
            'language_source': 'none',
            'warnings': ['empty_input'],
            'provenance': {'backend': self.cfg.ner_backend if ner_on else None, 'models': []},
        }
        # Reversible results always carry a (here empty) map, like non-empty input.
        if self.cfg.replacement_mode == 'reversible':
            if self._token_store is not None:
                metadata['anon_map_reference'] = self._token_store.save(AnonymizationMap())
            else:
                metadata['anon_map'] = AnonymizationMap()
        return metadata

    async def aprocess_batch(self, texts: List[str], batch_size: Optional[int] = None) -> List[ProcessResult]:
        """Async version of process_batch for use with asyncio-based frameworks (FastAPI, aiohttp).

        Runs process_batch in a thread-pool executor so it doesn't block the event loop.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.process_batch, texts, batch_size)

    def warmup(self, languages: Optional[List[str]] = None) -> None:
        """Pre-load NER models to avoid first-request latency.

        Args:
            languages: Language names to pre-load (e.g. ``['ENGLISH', 'DUTCH']``).
                       If ``None``, pre-loads models for all supported languages.
        """
        if not self.cfg.check_ner_process or self.GeneralNER is None:
            return
        langs = languages or list(self.cfg.supported_languages)
        for lang in langs:
            try:
                self.GeneralNER.load_language(lang)
            except Exception as e:
                logger.warning("warmup: skipping %s: %s", lang, e)

    def process(self, text: str) -> ProcessResult:
        """Process a single text. Maintains backward compatibility.

        Returns:
            ProcessResult (unpacks as 3-tuple: lm_text, stat_text, language).
        """
        return self.process_batch([text])[0]

    # --- Pipeline step methods (private, used by _init_pipeline) ---

    def _emit(self, entity_type, original, value):
        """Count one regex replacement of *original*, then protect its *value*."""
        counts = getattr(self._document_context, 'regex_counts', None)
        if counts is not None:
            entry = counts.setdefault(entity_type, {'count': 0, 'chars': 0})
            entry['count'] += 1
            entry['chars'] += len(original)
        key = self._protect(value)
        deferred = getattr(self._document_context, 'deferred', None)
        if deferred is not None:
            tag = (constants.CURRENCIES.get(original)
                   if entity_type == 'CURRENCY' and self.cfg.replace_with_currency_symbols is None
                   else self._tags.get(entity_type))
            if tag and tag != value:
                deferred[0][key] = tag
        return key

    def _protect(self, value):
        """Hide an emitted value from later regex steps (``<EMAIL1>``, kept IPs)."""
        protected = getattr(self._document_context, 'protected', None)
        if protected is None:
            return value
        # Private-use planes 15-16 only: outside the BMP letter ranges the email and URL
        # rules accept, and no digit, letter or space for later steps to match.
        return protected.setdefault(value, f"\U00100000{chr(0xF0000 + len(protected))}\U00100001")

    def _restore_protected(self, text):
        protected = getattr(self._document_context, 'protected', None) or {}
        deferred = getattr(self._document_context, 'deferred', None)
        tags = deferred[0] if deferred else {}
        order = getattr(self._document_context, 'marker_order', None)
        if order is not None:
            order += self._marker_occurrences(text, protected, tags)
        for value, key in protected.items():
            if key not in tags:
                text = text.replace(key, value)
        return text

    def _marker_occurrences(self, text, protected, tags):
        """(marker, emitted) for each marker in *text*, whose emitted values are still behind keys."""
        values = {key: tags.get(key, value) for value, key in protected.items()}
        parts, emitted, pos, size = [], [], 0, 0
        for m in _KEY.finditer(text):
            value = values.get(m.group(), m.group())
            size += m.start() - pos
            if value not in ('<', '>'):  # a bracket unescape_markup hid is source text, not an emit
                emitted.append((size, size + len(value)))
            parts += [text[pos:m.start()], value]
            size, pos = size + len(value), m.end()
        resolved = ''.join(parts) + text[pos:]
        return [(m.group(), any(a < m.end() and m.start() < b for a, b in emitted))
                for m in self._marker_regex.finditer(resolved)]

    def _source_marker_spans(self, text, source_markers):
        """The spans of source-text markers in *text*, so NER exempts only those and protects every
        emitted one ("EMAIL" in the source and as a replacement); by text if a step moved markers."""
        order = getattr(self._document_context, 'marker_order', None)
        self._document_context.marker_order = None
        if not source_markers or order is None:
            return source_markers
        found = list(self._marker_regex.finditer(text))
        if [m.group() for m in found] != [marker for marker, _ in order]:
            return source_markers
        return frozenset(m.span() for m, (_, emitted) in zip(found, order) if not emitted)

    def _restore_deferred(self, text):
        protected = getattr(self._document_context, 'protected', None) or {}
        deferred = getattr(self._document_context, 'deferred', None)
        if not deferred or not deferred[0]:
            return text
        # Deferred keys become their tag; every tag occurrence is recorded in order
        # (source text and plain tag emits as None) so _swap_deferred can pair them up.
        tags, order = deferred
        values = {key: value for value, key in protected.items() if key in tags}
        tracked = set(tags.values())
        pattern = re.compile('|'.join(map(re.escape, sorted({*values, *tracked}, key=len, reverse=True))))

        def put(match):
            found = match.group()
            if found in tags:
                order.append((tags[found], values[found]))
                return tags[found]
            value = values.get(found, found)
            if value in tracked:
                order.append((value, None))
            return value
        return pattern.sub(put, text)

    @staticmethod
    def _swap_deferred(text, order, source='', findings=(), anon_map=None):
        """After NER: each tag occurrence NER left gets its recorded value, in order.

        *source* is the text NER read; occurrences inside an NER finding are skipped, and in
        reversible mode their originals go back into that NER map entry.
        """
        if not order:
            return text
        values: Dict[str, list] = {}
        for tag, value in order:
            values.setdefault(tag, []).append(value)
        tags = sorted(values, key=len, reverse=True)
        pattern = re.compile('|'.join(map(re.escape, tags)))
        # An NER placeholder in the text is matched whole, so a marker such as "PERSON" is
        # not counted (or swapped) inside "<PERSON_0>".
        held = {e.placeholder for e in anon_map.entries} - set(values) if anon_map is not None else set()
        in_text = re.compile('|'.join(map(re.escape, sorted([*held, *tags], key=len, reverse=True))))
        found = Counter(match.group() for match in in_text.finditer(text) if match.group() in values)
        where: Dict[str, list] = {}
        for match in pattern.finditer(source):
            where.setdefault(match.group(), []).append(match.span())
        spans = [(f.entity.start, f.entity.end) for f in findings]
        pending, swallowed = {}, []
        for tag, vals in values.items():
            if found[tag] == len(vals):
                pending[tag] = iter(vals)
                continue
            if len(where.get(tag, ())) != len(vals):
                continue
            inside = [any(s < b and a < e for s, e in spans) for a, b in where[tag]]
            if found[tag] == inside.count(False):
                pending[tag] = iter(v for v, hit in zip(vals, inside) if not hit)
                swallowed += [(a, b, v) for (a, b), v, hit in zip(where[tag], vals, inside) if hit and v]
            # shortcut: any other count change (NER added a tag) keeps that tag plain
            # everywhere (no value lost to the wrong place, no PII).
        if swallowed and anon_map is not None:
            originals = {e.placeholder: e.original for e in anon_map.entries}
            for entry in anon_map.entries:
                if source[entry.start:entry.end] != entry.original:
                    continue  # regex entry: its offsets index another text
                # A tag the span only cuts ("of <EMAIL") goes back whole; its cut-off part
                # left beside the placeholder in the text is dropped.
                for a, b, value in sorted(swallowed, reverse=True):
                    if a < entry.end and entry.start < b:
                        entry.original = (entry.original[:max(a, entry.start) - entry.start]
                                          + originals.get(value, value)
                                          + entry.original[min(b, entry.end) - entry.start:])
                        if b > entry.end:
                            text = text.replace(entry.placeholder + source[entry.end:b], entry.placeholder, 1)
                        if a < entry.start:
                            text = text.replace(source[a:entry.start] + entry.placeholder, entry.placeholder, 1)

        def put(match):
            tag = match.group()
            value = next(pending[tag]) if tag in pending else None
            return tag if value is None else value
        return in_text.sub(put, text)

    def _replace_reversible(self, text, pattern, entity_type, keep=None):
        anon_map = getattr(self._document_context, 'anon_map', None)
        if anon_map is None:
            return None

        def replace(match):
            if keep is not None and keep(match.group()):
                return self._protect(match.group())
            placeholder = anon_map.next_placeholder(entity_type)
            anon_map.add(
                placeholder,
                match.group(),
                entity_type,
                match.start(),
                match.end(),
            )
            return self._emit(entity_type, match.group(), placeholder)

        return pattern.sub(replace, text)

    def _fix_bad_unicode(self, text):
        return self.NormaliseText.fix_bad_unicode(
            text, decode_escapes=self.cfg.check_decode_unicode_escapes,
            fix_character_width=self.cfg.fix_character_width,
            fold_styled_characters=self.cfg.fold_styled_characters)

    def _unescape_markup(self, text):
        return normtext.unescape_markup(text, self._protect)

    def _restore_brackets(self, text):
        protected = getattr(self._document_context, 'protected', None) or {}
        for value in ('<', '>'):
            if value in protected:
                text = text.replace(protected[value], value)
        return text

    def _to_ascii_unicode(self, text):
        return self.NormaliseText.to_ascii_unicode(text)

    def _remove_emoji(self, text):
        return self.NormaliseText.remove_emoji(text)

    def _replace_html(self, text):
        reversible = self._replace_reversible(
            text,
            self._html_regex,
            'HTML',
        )
        if reversible is not None:
            return reversible
        return self._html_regex.sub(lambda m: self._emit('HTML', m.group(), self.cfg.replace_with_html), text)

    def _replace_urls(self, text):
        reversible = self._replace_reversible(
            text,
            self._url_regex,
            'URL',
        )
        if reversible is not None:
            return reversible
        if self._synthetic_replacer:
            return self._url_regex.sub(
                lambda m: self._emit('URL', m.group(), self._synthetic_replacer.get_replacement('URL', m.group())), text,
            )
        return self._url_regex.sub(lambda m: self._emit('URL', m.group(), self.cfg.replace_with_url), text)

    def _replace_emails(self, text):
        if not constants.EMAIL_GATE.search(text):  # no separator, no email: skip the slow regex
            return text
        reversible = self._replace_reversible(
            text,
            constants.EMAIL_REGEX,
            'EMAIL',
        )
        if reversible is not None:
            return reversible
        if self._synthetic_replacer:
            return self.ProcessContacts.replace_emails_with_fn(
                text, lambda orig: self._emit('EMAIL', orig, self._synthetic_replacer.get_replacement('EMAIL', orig)),
            )
        return self.ProcessContacts.replace_emails_with_fn(
            text, lambda orig: self._emit('EMAIL', orig, self.cfg.replace_with_email),
        )

    def _replace_ips(self, text):
        public_only = self.cfg.replace_ips_public_only
        reversible = self._replace_reversible(
            text,
            contact.IP_VIEW,
            self._ip_type,
            keep=lambda orig: not contact.is_replaceable_ip(orig, public_only),
        )
        if reversible is not None:
            return reversible
        if self._synthetic_replacer:
            def emit(orig):
                return self._synthetic_replacer.get_replacement(self._ip_type, orig)
        else:
            def emit(orig):
                return self.cfg.replace_with_ips
        return self.ProcessContacts.replace_ips_with_fn(
            text,
            lambda orig: (
                self._emit(self._ip_type, orig, emit(orig))
                if contact.is_replaceable_ip(orig, public_only) else self._protect(orig)
            ),
        )

    def _replace_dates(self, text):
        reversible = self._replace_reversible(
            text,
            self._date_view,
            'DATE',
        )
        if reversible is not None:
            return reversible
        return self._date_view.sub(lambda m: self._emit('DATE', m.group(), self.cfg.replace_with_dates), text)

    def _replace_years(self, text):
        reversible = self._replace_reversible(
            text,
            constants.YEAR_REGEX,
            'YEAR',
        )
        if reversible is not None:
            return reversible
        return self.ProcessDateTime.replace_years(
            text, replace_with=lambda m: self._emit('YEAR', m.group(), self.cfg.replace_with_years),
        )

    def _replace_phone_numbers(self, text):
        phone_view = contact.PHONE_VIEW if self.cfg.fold_styled_characters else contact.PHONE_DIGIT_VIEW
        reversible = self._replace_reversible(
            text,
            phone_view,
            self._phone_type,
        )
        if reversible is not None:
            return reversible
        if self._synthetic_replacer:
            return self.ProcessContacts.replace_phone_numbers_with_fn(
                text, lambda orig: self._emit(
                    self._phone_type, orig, self._synthetic_replacer.get_replacement(self._phone_type, orig),
                ), view=phone_view,
            )
        return self.ProcessContacts.replace_phone_numbers_with_fn(
            text, lambda orig: self._emit(self._phone_type, orig, self.cfg.replace_with_phone_numbers),
            view=phone_view,
        )

    def _replace_typed(self, text, pattern, entity_type):
        """Replace *pattern* matches as *entity_type* in every replacement mode."""
        reversible = self._replace_reversible(text, pattern, entity_type)
        if reversible is not None:
            return reversible
        if self._synthetic_replacer:
            def value(orig):
                return self._synthetic_replacer.get_replacement(entity_type, orig)
        else:
            def value(orig):
                return self._tags[entity_type]
        return pattern.sub(lambda m: self._emit(entity_type, m.group(), value(m.group())), text)

    def _replace_iban_cards(self, text):
        return self._replace_typed(self._replace_typed(text, contact.IBAN_VIEW, 'IBAN'), contact.CARD_VIEW, 'CARD')

    def _replace_identifiers(self, text):
        return self._replace_typed(text, constants.IDENTIFIER_REGEX, 'ID')

    def _replace_numbers(self, text):
        anon_map = getattr(self._document_context, 'anon_map', None)
        if anon_map is not None:
            protected = {}

            def protect(match):
                key = f"\ue000{chr(65 + len(protected))}\ue001"
                protected[key] = match.group()
                return key

            placeholder_pattern = re.compile(r"<[A-Z][A-Z0-9_]*_\d+>")
            text = placeholder_pattern.sub(protect, text)
            reversible = self._replace_reversible(
                text,
                contact.numbers_regex(self.cfg.replace_numbers_in_words),
                'NUMBER',
            )
            for key, placeholder in protected.items():
                reversible = reversible.replace(key, placeholder)
            return reversible
        if self._synthetic_replacer:
            return self.ProcessContacts.replace_numbers_with_fn(
                text, lambda orig: self._emit('NUMBER', orig, self._synthetic_replacer.get_replacement('NUMBER', orig)),
                in_words=self.cfg.replace_numbers_in_words,
            )
        return self.ProcessContacts.replace_numbers_with_fn(
            text, lambda orig: self._emit('NUMBER', orig, self.cfg.replace_with_numbers),
            in_words=self.cfg.replace_numbers_in_words,
        )

    def _replace_currency_symbols(self, text):
        reversible = self._replace_reversible(
            text,
            constants.CURRENCY_REGEX,
            'CURRENCY',
        )
        if reversible is not None:
            return reversible
        replace_with = self.cfg.replace_with_currency_symbols
        if replace_with is None:
            # Symbol -> ISO code, one count per symbol occurrence.
            return self.ProcessSpecialSymbols.replace_currency_symbols_with_fn(
                text, lambda symbol: self._emit('CURRENCY', symbol, constants.CURRENCIES[symbol]),
            )
        return self.ProcessSpecialSymbols.replace_currency_symbols(
            text, replace_with=lambda m: self._emit('CURRENCY', m.group(), replace_with),
        )

    def _remove_isolated_letters(self, text):
        return self.ProcessSpecialSymbols.remove_isolated_letters(text)

    def _remove_isolated_special_symbols(self, text):
        return self.ProcessSpecialSymbols.remove_isolated_special_symbols(
            text,
            remove_brackets=self.cfg.check_remove_bracket_content,
            remove_braces=self.cfg.check_remove_brace_content,
            max_chars=self.cfg.bracket_content_max_chars,
            max_words=self.cfg.bracket_content_max_words,
            keep_json=self.cfg.keep_json_braces,
        )

    def _normalize_whitespace(self, text):
        return self.NormaliseText.normalize_whitespace(text, no_line_breaks=True)

    def _statistical_model_processing(self, text: str, language: Optional[str]) -> str:
        """Generate statistical model text (lowercase, no stopwords, no punctuation)."""
        stext = text
        if self.cfg.check_smart_casefold:
            stop_words = self.ProcessStopwords.get_stop_words(language) if language else set()
            stext = self.NormaliseText.smart_casefold(stext, stop_words=stop_words)
        elif self.cfg.check_casefold:
            stext = stext.casefold()
        if self.cfg.check_remove_stopwords and language:
            stext = self.ProcessStopwords.remove_stopwords(stext, language)
        if self.cfg.check_remove_punctuation:
            stext = self.ProcessSpecialSymbols.remove_punctuation(stext)
        if self.cfg.check_remove_isolated_letters:
            stext = self.ProcessSpecialSymbols.remove_isolated_letters(stext)
        if self.cfg.check_normalize_whitespace:
            stext = self.NormaliseText.normalize_whitespace(stext)
        return stext
