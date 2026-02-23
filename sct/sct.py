"""
Comprehensive text cleaning and preprocessing pipeline.
"""
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Optional

from sct.config import TextCleanerConfig, _config_from_module_globals
from sct.utils import contact, datetime, ner, normtext, resources, special, stopwords

logger = logging.getLogger(__name__)

# Thread-local storage for passing detected language to pipeline steps
# that need it (e.g. fuzzy date replacement).  Each thread running
# _process_single sets its own value, so there is no cross-thread leaking.
_thread_ctx = threading.local()


class TextCleaner:

    def __init__(self, cfg: Optional[TextCleanerConfig] = None):
        """Initialize the text cleaning pipeline.

        Args:
            cfg: Immutable configuration. If None, reads from module-level
                 config variables (backward compatible).
        """
        self.cfg = cfg or _config_from_module_globals()

        self.ProcessContacts = contact.ProcessContacts()
        self.ProcessDateTime = datetime.ProcessDateTime()
        self.ProcessSpecialSymbols = special.ProcessSpecialSymbols()
        self.NormaliseText = normtext.NormaliseText()
        self.ProcessStopwords = stopwords.ProcessStopwords()

        if self.cfg.check_ner_process:
            self.GeneralNER = ner.GeneralNER(
                model_names=dict(self.cfg.ner_models)
            )
        else:
            self.GeneralNER = None

        self.batch_size = 8
        self._pipeline = []
        self._init_pipeline()

    def _init_pipeline(self):
        """Build the pipeline steps list based on config."""
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
        if cfg.check_fuzzy_replace_dates:
            self._pipeline.append(self._fuzzy_replace_dates)
        if cfg.check_replace_years:
            self._pipeline.append(self._replace_years)
        if cfg.check_replace_phone_numbers:
            self._pipeline.append(self._replace_phone_numbers)
        if cfg.check_replace_numbers:
            self._pipeline.append(self._replace_numbers)
        if cfg.check_replace_currency_symbols:
            self._pipeline.append(self._replace_currency_symbols)
        if cfg.check_remove_isolated_letters:
            self._pipeline.append(self._remove_isolated_letters)
        if cfg.check_remove_isolated_special_symbols:
            self._pipeline.append(self._remove_isolated_special_symbols)
        if cfg.check_normalize_whitespace:
            self._pipeline.append(self._normalize_whitespace)

    def _detect_language(self, text: str) -> Optional[str]:
        """Detect language as a pure function (no instance mutation)."""
        language_config = self.cfg.language
        if language_config:
            lc = language_config.lower()
            if lc in resources.LANGUAGE_NAME:
                return language_config.upper()

        if any([self.cfg.check_detect_language, self.cfg.check_ner_process,
                self.cfg.check_remove_stopwords]):
            return str(resources.DETECTOR.detect_language_of(text)).split(".")[-1]

        return None

    def _process_single(self, text: str) -> Tuple[str, Optional[str], Optional[str]]:
        """Process a single text through the entire pipeline.

        Returns:
            Always a 3-tuple: (lm_text, stat_text_or_None, language_or_None)
        """
        # Detect language (pure function, thread-safe)
        language = self._detect_language(text)

        current_text = text

        # Store language in thread-local so pipeline steps can access it
        _thread_ctx.language = language

        # Apply non-NER pipeline steps
        for step in self._pipeline:
            current_text = step(current_text)

        # NER processing
        if self.cfg.check_ner_process and self.GeneralNER is not None:
            current_text = self.GeneralNER.ner_process(
                current_text,
                positional_tags=list(self.cfg.positional_tags),
                ner_confidence_threshold=self.cfg.ner_confidence_threshold,
                language=language,
            )

        # Statistical model processing (always returns stext, even if None)
        stext = None
        if self.cfg.check_statistical_model_processing:
            stext = self._statistical_model_processing(current_text, language)

        return (current_text, stext, language)

    def process_batch(self, texts: List[str], batch_size: int = None) -> List[Tuple[str, Optional[str], Optional[str]]]:
        """Process multiple texts.

        Returns:
            List of 3-tuples: (lm_text, stat_text_or_None, language_or_None)
        """
        if not texts:
            return []

        results = [None] * len(texts)
        to_process = []

        for i, text in enumerate(texts):
            if not isinstance(text, str):
                raise ValueError(f"Input must be string, got {type(text)}")
            if not text or text.isspace():
                results[i] = ("", "", None)
            else:
                to_process.append((i, text))

        if not to_process:
            return results

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

        return results

    def process(self, text: str) -> Tuple[str, Optional[str], Optional[str]]:
        """Process a single text. Maintains backward compatibility.

        Returns:
            3-tuple: (lm_text, stat_text_or_None, language_or_None)
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
        return self.ProcessContacts.replace_urls(text, replace_with=self.cfg.replace_with_url)

    def _replace_emails(self, text):
        return self.ProcessContacts.replace_emails(text, replace_with=self.cfg.replace_with_email)

    def _replace_dates(self, text):
        return self.ProcessDateTime.replace_dates(text, replace_with=self.cfg.replace_with_dates)

    def _fuzzy_replace_dates(self, text):
        return self.ProcessDateTime.fuzzy_replace_dates(
            text,
            replace_with=self.cfg.replace_with_dates,
            score_cutoff=self.cfg.fuzzy_date_score_cutoff,
            language=getattr(_thread_ctx, 'language', 'ENGLISH'),
        )

    def _replace_years(self, text):
        return self.ProcessDateTime.replace_years(text, replace_with=self.cfg.replace_with_years)

    def _replace_phone_numbers(self, text):
        return self.ProcessContacts.replace_phone_numbers(text, replace_with=self.cfg.replace_with_phone_numbers)

    def _replace_numbers(self, text):
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
            stop_words = self.ProcessStopwords.get_stop_words(language)
            stext = self.NormaliseText.smart_casefold(stext, stop_words=stop_words)
        elif self.cfg.check_casefold:
            stext = stext.casefold()
        if self.cfg.check_remove_stopwords:
            stext = self.ProcessStopwords.remove_stopwords(stext, language)
        if self.cfg.check_remove_punctuation:
            stext = self.ProcessSpecialSymbols.remove_punctuation(stext)
        if self.cfg.check_remove_isolated_letters:
            stext = self.ProcessSpecialSymbols.remove_isolated_letters(stext)
        if self.cfg.check_normalize_whitespace:
            stext = self.NormaliseText.normalize_whitespace(stext)
        return stext
