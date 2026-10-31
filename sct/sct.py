"""
Comprehensive text cleaning and preprocessing pipeline.
"""
import asyncio
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple

from sct.config import TextCleanerConfig, GLINER_BACKENDS, _config_from_module_globals
from sct.privacy import PrivacyPolicy, ProcessingMetrics, TokenStore
from sct.utils import constants, contact, datetime, ner, normtext, resources, special, stopwords
from sct.utils.anonymization_map import AnonymizationMap
from sct.utils.process_result import ProcessResult

logger = logging.getLogger(__name__)

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
            custom_stopwords=dict(self.cfg.custom_stopwords) if self.cfg.custom_stopwords else None,
        )

        # Instance-scoped datetime (with custom month names -> per-instance regex)
        if self.cfg.custom_month_names:
            extra = {}
            for lang, names in self.cfg.custom_month_names.items():
                extra[resources.language_to_iso(lang)] = names
            mn_dict = constants.build_month_names_dict(extra)
            date_regex = constants.build_date_regex(mn_dict)
            fuzzy_all, fuzzy_by_lang = constants.build_fuzzy_vocabulary(
                mn_dict,
                {**constants._LANG_TO_VOCAB,
                 **{lang: resources.language_to_iso(lang) for lang in self.cfg.custom_month_names}},
            )
            self.ProcessDateTime = datetime.ProcessDateTime(
                date_regex=date_regex,
                fuzzy_vocabulary=fuzzy_all,
                fuzzy_vocabulary_by_lang=fuzzy_by_lang,
            )
        else:
            self.ProcessDateTime = datetime.ProcessDateTime()

        self.ProcessContacts = contact.ProcessContacts()
        self.ProcessSpecialSymbols = special.ProcessSpecialSymbols()
        self.NormaliseText = normtext.NormaliseText()

        # Synthetic replacer: shared by regex pipeline steps + NER processor
        self._synthetic_replacer = None
        if self.cfg.replacement_mode == 'synthetic':
            from sct.utils.synthetic import SyntheticReplacer
            self._synthetic_replacer = SyntheticReplacer()

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
                    'threshold': self.cfg.gliner_threshold,
                    'label_map': (
                        dict(self.cfg.gliner_label_map)
                        if self.cfg.gliner_label_map else None
                    ),
                    'label_descriptions': (
                        dict(self.cfg.gliner_label_descriptions)
                        if self.cfg.gliner_label_descriptions else None
                    ),
                    'onnx': self.cfg.gliner_onnx,
                }

            # Determine torch model names (if needed)
            torch_model_names = None
            if self.cfg.ner_backend in ('torch', 'ensemble_torch'):
                torch_model_names = dict(self.cfg.torch_ner_models) if self.cfg.torch_ner_models else None

            self.GeneralNER = ner.GeneralNER(
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
                ensemble_models=dict(self.cfg.ner_ensemble) if self.cfg.ner_ensemble is not None else None,
                ensemble_default_keys=(
                    tuple(self.cfg.ner_ensemble_default_keys)
                    if self.cfg.ner_ensemble_default_keys is not None
                    else None
                ),
                replacement_mode=self.cfg.replacement_mode,
                synthetic_replacer=self._synthetic_replacer,
                onnx_intra_op_threads=self.cfg.onnx_intra_op_threads,
                onnx_inter_op_threads=self.cfg.onnx_inter_op_threads,
                ner_chunk_overlap=self.cfg.ner_chunk_overlap,
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
        if cfg.check_replace_html:
            self._pipeline.append(self._replace_html)
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
        # User-provided custom steps — appended after all built-in steps
        for step_fn in cfg.custom_pipeline_steps:
            self._post_fuzzy_pipeline.append(step_fn)

    def _detect_language(self, text: str) -> Tuple[Optional[str], Optional[float], str]:
        """Detect language as a pure function (no instance mutation).

        Returns ``(name, confidence, source)``; source is ``pinned``, ``detected``,
        ``fallback`` (tuple restriction fell back to its first entry) or ``none``.

        Behavior depends on ``cfg.language``:
          - ``str``: pinned to that language (skip detection)
          - ``tuple``: restrict detection to those languages (detect per text)
          - ``None``: auto-detection (all 75 Lingua languages by default)
        """
        language_config = self.cfg.language

        # Pinned: single string → return immediately (already resolved to uppercase)
        if isinstance(language_config, str):
            return language_config, 1.0, 'pinned'

        restricted = isinstance(language_config, tuple)
        if not restricted and not self._language_needed():
            return None, None, 'none'

        values = self._detector.compute_language_confidence_values(text)
        # Same "unknown" rule as Lingua's detect_language_of: nothing scored, or a tie at the top.
        if values and values[0].value > 0 and (len(values) < 2 or values[1].value < values[0].value):
            top = values[0]
            if not restricted or top.language.name in language_config:
                return top.language.name, top.value, 'detected'
        if restricted:
            first = language_config[0]
            conf = next((v.value for v in values if v.language.name == first), 0.0)
            return first, conf, 'fallback'
        return None, None, 'none'

    def _language_needed(self) -> bool:
        """Whether auto-detection runs at all (else no language is expected)."""
        return any([self.cfg.check_detect_language, self.cfg.check_ner_process,
                    self.cfg.check_remove_stopwords])

    def _process_single(self, text: str) -> ProcessResult:
        try:
            return self._process_single_with_context(text)
        finally:
            self._document_context.anon_map = None
            self._document_context.protected = None
            self._document_context.regex_counts = None

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
                ner_confidence_threshold=self.cfg.ner_confidence_threshold,
                language=language,
                anon_map=anon_map,
                privacy_policy=self._privacy_policy,
            )
            current_text = detailed.text
            findings = detailed.findings
        return self._after_ner(state, current_text, findings)

    def _before_ner_isolated(self, text: str) -> tuple:
        """``_before_ner`` that clears the per-thread document context afterwards."""
        try:
            return self._before_ner(text)
        finally:
            self._document_context.anon_map = None
            self._document_context.protected = None
            self._document_context.regex_counts = None

    def _before_ner(self, text: str) -> tuple:
        """Every step before NER.

        Returns ``(text, language, anon_map, doc_classes, started_at, regex_counts,
        language_confidence, language_source)``.
        """
        started_at = time.perf_counter()

        # Reset per-document synthetic consistency cache (thread-safe via threading.local)
        if self._synthetic_replacer:
            self._synthetic_replacer.reset_cache()

        # Per-document anonymization map (reversible mode only)
        anon_map = None
        if self.cfg.replacement_mode == 'reversible':
            anon_map = AnonymizationMap()
        self._document_context.anon_map = anon_map
        self._document_context.protected = {}
        # The dict travels in the returned state, so it survives the context reset.
        regex_counts: Dict[str, Dict[str, int]] = {}
        self._document_context.regex_counts = regex_counts

        # Document-level classification (before any text modification)
        doc_classes = None
        if self.cfg.check_classify_document and self._gliclass is not None:
            doc_classes = self._gliclass.classify(text)

        # Detect language (pure function, thread-safe)
        language, language_confidence, language_source = self._detect_language(text)

        current_text = text

        # Pre-fuzzy pipeline steps (unicode fix → html → urls → emails → ips → dates)
        for step in self._pipeline:
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

        return (current_text, language, anon_map, doc_classes, started_at, regex_counts,
                language_confidence, language_source)

    def _after_ner(self, state: tuple, current_text: str, findings) -> ProcessResult:
        """Every step after NER, for the document *state* from ``_before_ner``."""
        (_, language, anon_map, doc_classes, started_at, regex_counts,
         language_confidence, language_source) = state

        # Statistical model processing (always returns stext, even if None)
        stext = None
        if self.cfg.check_statistical_model_processing:
            stext = self._statistical_model_processing(current_text, language)

        # Value-free counts are always present; entity text only on opt-in below.
        entity_counts: Dict[str, Dict[str, int]] = {}
        for finding in findings:
            entry = entity_counts.setdefault(
                finding.entity.entity_type, {'count': 0, 'chars': 0},
            )
            entry['count'] += 1
            entry['chars'] += finding.entity.end - finding.entity.start
        warnings = set()
        if language is None and language_source == 'none' and self._language_needed():
            warnings.add('language_unknown')
        if language_source == 'fallback':
            warnings.add('language_fallback')
        if (language_source == 'detected'
                and language_confidence < self.cfg.language_confidence_warning):
            warnings.add('language_low_confidence')
        if (self.cfg.check_ner_process and self.GeneralNER is not None
                and self.GeneralNER.uses_fallback_models(language)):
            warnings.add('ner_fallback_models')
        metadata: Dict[str, Any] = {
            'regex_counts': regex_counts,
            'entity_counts': entity_counts,
            'language_confidence': language_confidence,
            'language_source': language_source,
            'warnings': sorted(warnings),
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
            detector_models: Dict[str, Dict[str, Optional[str]]] = {}
            if self.cfg.ner_backend in ('onnx', 'ensemble_onnx'):
                detector_models.update({
                    key: {
                        'id': model_id,
                        'revision': (
                            self.cfg.ner_model_revisions or {}
                        ).get(key),
                    }
                    for key, model_id in (
                        self.cfg.ner_models or {}
                    ).items()
                })
            if self.cfg.ner_backend in ('torch', 'ensemble_torch'):
                detector_models.update({
                    key: {
                        'id': model_id,
                        'revision': (
                            self.cfg.torch_ner_model_revisions or {}
                        ).get(key),
                    }
                    for key, model_id in (
                        self.cfg.torch_ner_models or {}
                    ).items()
                })
            if (
                self.cfg.ner_backend in GLINER_BACKENDS
                or self.cfg.ner_backend == 'presidio_gliner'
            ):
                detector_models['GLINER'] = {
                    'id': self.cfg.gliner_model,
                    'revision': self.cfg.gliner_revision,
                }
            metadata['detector'] = {
                'backend': self.cfg.ner_backend,
                'models': detector_models,
            }

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
                results[i] = ProcessResult("", "", None)
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
                ner_confidence_threshold=self.cfg.ner_confidence_threshold,
                anon_maps=[state[2] for state in states],
                privacy_policy=self._privacy_policy,
                before_finalize=restore_synthetic,
                max_workers=max_workers,
            )
            for (i, _), state, ner_result in zip(to_process, states, ner_results):
                results[i] = self._after_ner(state, ner_result.text, ner_result.findings)
        else:
            # Each text goes through the full pipeline independently.
            for (i, _), result in zip(to_process, run(self._process_single)):
                results[i] = result

        return [r for r in results if r is not None]

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
        return self._protect(value)

    def _protect(self, value):
        """Hide an emitted value from later regex steps (``<EMAIL1>``, kept IPs)."""
        protected = getattr(self._document_context, 'protected', None)
        if protected is None:
            return value
        # Private-use planes 15-16 only: outside the BMP letter ranges the email and URL
        # rules accept, and no digit, letter or space for later steps to match.
        return protected.setdefault(value, f"\U00100000{chr(0xF0000 + len(protected))}\U00100001")

    def _restore_protected(self, text):
        for value, key in (getattr(self._document_context, 'protected', None) or {}).items():
            text = text.replace(key, value)
        return text

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
        return self.NormaliseText.fix_bad_unicode(text)

    def _to_ascii_unicode(self, text):
        return self.NormaliseText.to_ascii_unicode(text)

    def _remove_emoji(self, text):
        return self.NormaliseText.remove_emoji(text)

    def _replace_html(self, text):
        reversible = self._replace_reversible(
            text,
            constants.HTML_REGEX,
            'HTML',
        )
        if reversible is not None:
            return reversible
        return self.ProcessContacts.replace_html(
            text, replace_with=lambda m: self._emit('HTML', m.group(), self.cfg.replace_with_html),
        )

    def _replace_urls(self, text):
        reversible = self._replace_reversible(
            text,
            constants.URL_REGEX,
            'URL',
        )
        if reversible is not None:
            return reversible
        if self._synthetic_replacer:
            return self.ProcessContacts.replace_urls_with_fn(
                text, lambda orig: self._emit('URL', orig, self._synthetic_replacer.get_replacement('URL', orig)),
            )
        return self.ProcessContacts.replace_urls_with_fn(
            text, lambda orig: self._emit('URL', orig, self.cfg.replace_with_url),
        )

    def _replace_emails(self, text):
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
            constants.IP_REGEX,
            'IP_ADDRESS',
            keep=lambda orig: not contact.is_replaceable_ip(orig, public_only),
        )
        if reversible is not None:
            return reversible
        if self._synthetic_replacer:
            def emit(orig):
                return self._synthetic_replacer.get_replacement('IP_ADDRESS', orig)
        else:
            def emit(orig):
                return self.cfg.replace_with_ips
        return self.ProcessContacts.replace_ips_with_fn(
            text,
            lambda orig: (
                self._emit('IP_ADDRESS', orig, emit(orig))
                if contact.is_replaceable_ip(orig, public_only) else self._protect(orig)
            ),
        )

    def _replace_dates(self, text):
        reversible = self._replace_reversible(
            text,
            self.ProcessDateTime._date_regex,
            'DATE',
        )
        if reversible is not None:
            return reversible
        return self.ProcessDateTime.replace_dates(
            text, replace_with=lambda m: self._emit('DATE', m.group(), self.cfg.replace_with_dates),
        )

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
        reversible = self._replace_reversible(
            text,
            constants.PHONE_REGEX,
            'PHONE_NUMBER',
        )
        if reversible is not None:
            return reversible
        if self._synthetic_replacer:
            return self.ProcessContacts.replace_phone_numbers_with_fn(
                text, lambda orig: self._emit(
                    'PHONE_NUMBER', orig, self._synthetic_replacer.get_replacement('PHONE_NUMBER', orig),
                ),
            )
        return self.ProcessContacts.replace_phone_numbers_with_fn(
            text, lambda orig: self._emit('PHONE_NUMBER', orig, self.cfg.replace_with_phone_numbers),
        )

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
