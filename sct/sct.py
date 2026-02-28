"""
Comprehensive text cleaning and preprocessing pipeline.
"""
import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional

from sct.config import TextCleanerConfig, GLINER_BACKENDS, _config_from_module_globals
from sct.utils import constants, contact, datetime, ner, normtext, resources, special, stopwords
from sct.utils.anonymization_map import AnonymizationMap
from sct.utils.process_result import ProcessResult

logger = logging.getLogger(__name__)

class TextCleaner:

    def __init__(self, cfg: Optional[TextCleanerConfig] = None):
        """Initialize the text cleaning pipeline.

        Args:
            cfg: Immutable configuration. If None, reads from module-level
                 config variables (backward compatible).
        """
        self.cfg = cfg or _config_from_module_globals()

        # Instance-scoped language detector — restricted when language is a tuple
        if isinstance(self.cfg.language, tuple):
            self._detector = resources.build_detector(frozenset(self.cfg.language))
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
                ner_backend=self.cfg.ner_backend,
                gliner_config=gliner_config,
                torch_model_names=torch_model_names,
                ner_batch_size=self.cfg.ner_batch_size,
                ensemble_models=dict(self.cfg.ner_ensemble) if self.cfg.ner_ensemble is not None else None,
                ensemble_default_keys=(
                    tuple(self.cfg.ner_ensemble_default_keys)
                    if self.cfg.ner_ensemble_default_keys is not None
                    else None
                ),
                replacement_mode=self.cfg.replacement_mode,
                synthetic_replacer=self._synthetic_replacer,
            )
        else:
            pass  # self.GeneralNER already initialized to None above

        # GLiClass document-level pre-classification (optional, lazy-loaded)
        self._gliclass: Any = None
        if self.cfg.check_classify_document:
            from sct.utils.gliclass_adapter import GLiClassAdapter
            from sct.config import GLICLASS_DEFAULT_MODEL
            self._gliclass = GLiClassAdapter(
                model_id=self.cfg.gliclass_model or GLICLASS_DEFAULT_MODEL,
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
        if cfg.check_replace_dates:
            self._pipeline.append(self._replace_dates)
        # fuzzy_replace_dates runs here (between replace_dates and replace_years)
        # but is called explicitly in _process_single with language context.
        if cfg.check_replace_years:
            self._post_fuzzy_pipeline.append(self._replace_years)
        if cfg.check_replace_phone_numbers:
            self._post_fuzzy_pipeline.append(self._replace_phone_numbers)
        if cfg.check_replace_numbers:
            self._post_fuzzy_pipeline.append(self._replace_numbers)
        if cfg.check_replace_currency_symbols:
            self._post_fuzzy_pipeline.append(self._replace_currency_symbols)
        if cfg.check_remove_isolated_letters:
            self._post_fuzzy_pipeline.append(self._remove_isolated_letters)
        if cfg.check_remove_isolated_special_symbols:
            self._post_fuzzy_pipeline.append(self._remove_isolated_special_symbols)
        if cfg.check_normalize_whitespace:
            self._post_fuzzy_pipeline.append(self._normalize_whitespace)
        # User-provided custom steps — appended after all built-in steps
        for step_fn in cfg.custom_pipeline_steps:
            self._post_fuzzy_pipeline.append(step_fn)

    def _detect_language(self, text: str) -> Optional[str]:
        """Detect language as a pure function (no instance mutation).

        Behavior depends on ``cfg.language``:
          - ``str``: pinned to that language (skip detection)
          - ``tuple``: restrict detection to those languages (detect per text)
          - ``None``: full auto-detection among all supported languages
        """
        language_config = self.cfg.language

        # Pinned: single string → return immediately (already resolved to uppercase)
        if isinstance(language_config, str):
            return language_config

        # Tuple: restrict detection to listed languages
        if isinstance(language_config, tuple):
            detected = self._detector.detect_language_of(text)
            if detected is not None and detected.name in language_config:
                return detected.name
            return language_config[0]  # Fallback to first listed

        # None: full auto-detection
        if any([self.cfg.check_detect_language, self.cfg.check_ner_process,
                self.cfg.check_remove_stopwords]):
            detected = self._detector.detect_language_of(text)
            if detected is not None:
                return detected.name

        return None

    def _process_single(self, text: str) -> ProcessResult:
        """Process a single text through the entire pipeline.

        Returns:
            ProcessResult (unpacks as 3-tuple for backward compat).
        """
        # Reset per-document synthetic consistency cache (thread-safe via threading.local)
        if self._synthetic_replacer:
            self._synthetic_replacer.reset_cache()

        # Per-document anonymization map (reversible mode only)
        anon_map = None
        if self.cfg.replacement_mode == 'reversible':
            anon_map = AnonymizationMap()

        # Document-level classification (before any text modification)
        doc_classes = None
        if self.cfg.check_classify_document and self._gliclass is not None:
            doc_classes = self._gliclass.classify(text)

        # Detect language (pure function, thread-safe)
        language = self._detect_language(text)

        current_text = text

        # Pre-fuzzy pipeline steps (unicode fix → html → urls → emails → dates)
        for step in self._pipeline:
            current_text = step(current_text)

        # Fuzzy date replacement — requires language context, called explicitly
        # to avoid thread-local; positioned between replace_dates and replace_years.
        if self.cfg.check_fuzzy_replace_dates:
            current_text = self.ProcessDateTime.fuzzy_replace_dates(
                current_text,
                replace_with=self.cfg.replace_with_dates,
                score_cutoff=self.cfg.fuzzy_date_score_cutoff,
                language=language,
            )

        # Post-fuzzy pipeline steps (years → phones → numbers → symbols → whitespace)
        for step in self._post_fuzzy_pipeline:
            current_text = step(current_text)

        # NER processing
        if self.cfg.check_ner_process and self.GeneralNER is not None:
            current_text = self.GeneralNER.ner_process(
                current_text,
                positional_tags=self.cfg.positional_tags,
                ner_confidence_threshold=self.cfg.ner_confidence_threshold,
                language=language,
                anon_map=anon_map,
            )

        # Statistical model processing (always returns stext, even if None)
        stext = None
        if self.cfg.check_statistical_model_processing:
            stext = self._statistical_model_processing(current_text, language)

        # Build metadata dict (only when extended features are active)
        metadata: Optional[Dict[str, Any]] = None
        if anon_map is not None:
            metadata = metadata or {}
            metadata['anon_map'] = anon_map
        if doc_classes is not None:
            metadata = metadata or {}
            metadata['classes'] = doc_classes

        return ProcessResult(current_text, stext, language, metadata=metadata)

    def process_batch(self, texts: List[str], batch_size: Optional[int] = None) -> List[ProcessResult]:
        """Process multiple texts.

        Returns:
            List of ProcessResult (each unpacks as 3-tuple for backward compat).
        """
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

        # Parallel processing: each text goes through the full pipeline independently.
        # ONNX Runtime releases the GIL during C++ inference ops, so threads
        # achieve real concurrency for the NER inference bottleneck.
        max_workers = min(len(to_process), os.cpu_count() or 4)
        if max_workers > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self._process_single, text): i
                    for i, text in to_process
                }
                for future in futures:
                    results[futures[future]] = future.result()
        else:
            for i, text in to_process:
                results[i] = self._process_single(text)

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

    def _fix_bad_unicode(self, text):
        return self.NormaliseText.fix_bad_unicode(text)

    def _to_ascii_unicode(self, text):
        return self.NormaliseText.to_ascii_unicode(text)

    def _remove_emoji(self, text):
        return self.NormaliseText.remove_emoji(text)

    def _replace_html(self, text):
        return self.ProcessContacts.replace_html(text, replace_with=self.cfg.replace_with_html)

    def _replace_urls(self, text):
        if self._synthetic_replacer:
            return self.ProcessContacts.replace_urls_with_fn(
                text, lambda orig: self._synthetic_replacer.get_replacement('URL', orig),
            )
        return self.ProcessContacts.replace_urls(text, replace_with=self.cfg.replace_with_url)

    def _replace_emails(self, text):
        if self._synthetic_replacer:
            return self.ProcessContacts.replace_emails_with_fn(
                text, lambda orig: self._synthetic_replacer.get_replacement('EMAIL', orig),
            )
        return self.ProcessContacts.replace_emails(text, replace_with=self.cfg.replace_with_email)

    def _replace_dates(self, text):
        return self.ProcessDateTime.replace_dates(text, replace_with=self.cfg.replace_with_dates)

    def _fuzzy_replace_dates(self, text, language=None):
        return self.ProcessDateTime.fuzzy_replace_dates(
            text,
            replace_with=self.cfg.replace_with_dates,
            score_cutoff=self.cfg.fuzzy_date_score_cutoff,
            language=language,
        )

    def _replace_years(self, text):
        return self.ProcessDateTime.replace_years(text, replace_with=self.cfg.replace_with_years)

    def _replace_phone_numbers(self, text):
        if self._synthetic_replacer:
            return self.ProcessContacts.replace_phone_numbers_with_fn(
                text, lambda orig: self._synthetic_replacer.get_replacement('PHONE_NUMBER', orig),
            )
        return self.ProcessContacts.replace_phone_numbers(text, replace_with=self.cfg.replace_with_phone_numbers)

    def _replace_numbers(self, text):
        if self._synthetic_replacer:
            return self.ProcessContacts.replace_numbers_with_fn(
                text, lambda orig: self._synthetic_replacer.get_replacement('NUMBER', orig),
            )
        return self.ProcessContacts.replace_numbers(text, replace_with=self.cfg.replace_with_numbers)

    def _replace_currency_symbols(self, text):
        return self.ProcessSpecialSymbols.replace_currency_symbols(text, replace_with=self.cfg.replace_with_currency_symbols)

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
