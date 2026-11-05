"""
Comprehensive text cleaning and preprocessing pipeline.
"""
import asyncio
import copy
import logging
import os
import re
import threading
import time
import warnings
from bisect import bisect_right
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Any, NamedTuple, cast
from collections.abc import Callable

import regex

from sct.config import GLICLASS_DEFAULT_MODEL, TextCleanerConfig, _config_from_module_globals
from sct.privacy import EntityFinding, PrivacyPolicy, ProcessingMetrics, TokenStore
from sct.utils import constants, contact, datetime, ner, normtext, resources, special, stopwords
from sct.utils.anonymization_map import AnonymizationMap
from sct.utils.gliclass_adapter import GLiClassAdapter
from sct.utils.synthetic import FAKER_GENERATORS, SyntheticReplacer
from sct.utils.process_result import ProcessMetadata, ProcessResult

logger = logging.getLogger(__name__)


class CustomPatternTimeout(TimeoutError):
    """A custom_patterns regex ran past custom_patterns_timeout on one document. Names the label, never the text."""

    def __init__(self, label: str, timeout: float):
        super().__init__(label, timeout)
        self.label, self.timeout = label, timeout

    def __str__(self):
        return (f"custom_patterns[{self.label!r}] ran longer than {self.timeout}s on one document; "
                "simplify the pattern, raise custom_patterns_timeout or set custom_patterns_on_timeout='warn'")
# A _protect key (an emitted value hidden from later steps).
_KEY = re.compile(f'{constants.KEY_OPEN}[\U000F0000-\U0010FFFF]{constants.KEY_CLOSE}')


# Without a policy a finding's action is the replacement mode, which conceals the value too.
_EXEMPT_ACTIONS = frozenset({"keep", "allow", "ignore"})


class _DocumentState(NamedTuple):
    """One document on its way from ``_before_ner`` to ``_after_ner``."""

    text: str
    language: str | None
    anon_map: AnonymizationMap | None
    doc_classes: Any
    started_at: float
    regex_counts: dict[str, dict[str, int]]
    language_confidence: float | None
    language_source: str
    source_chars: int
    source_markers: frozenset  # marker spans, or marker strings when a step moved markers (_source_marker_spans)
    language_margin: float | None
    deferred: list | None  # (tag, value) in text order, for _swap_deferred
    warnings: set


def _entity_counts(findings) -> dict[str, dict[str, int]]:
    """Value-free counts per type. Overlapping findings count once, as their union labelled by the highest
    score: the same rule NER uses when it replaces them."""
    spans: list[list] = []
    for finding in sorted(findings, key=lambda f: (f.entity.start, -f.entity.end)):
        e = finding.entity
        if spans and e.start < spans[-1][3]:
            last = spans[-1]
            if e.score > last[1]:
                last[0], last[1] = e.entity_type, e.score
            last[3] = max(last[3], e.end)
        else:
            spans.append([e.entity_type, e.score, e.start, e.end])
    counts: dict[str, dict[str, int]] = {}
    for entity_type, _, start, end in spans:
        entry = counts.setdefault(entity_type, {'count': 0, 'chars': 0})
        entry['count'] += 1
        entry['chars'] += end - start
    return counts


class _DocumentContext(threading.local):
    """One document's state on the running thread; every field reads None outside a document."""

    anon_map: Any = None
    protected: Any = None
    regex_counts: Any = None
    deferred_tags: dict[str, str] | None = None  # protected key -> tag NER reads
    deferred_order: list | None = None  # (tag, value or None) in text order
    warnings: Any = None
    language: Any = None
    marker_order: Any = None
    step_text: Any = None
    allow_spans: Any = None

    def clear(self) -> None:
        self.__dict__.clear()


def _renamed_attribute(old: str, new: str) -> property:
    """A deprecated public name that reads and writes the attribute that replaced it."""
    message = f"TextCleaner.{old} is deprecated; it is a private attribute now and this alias will be removed"

    def read(self):
        warnings.warn(message, DeprecationWarning, stacklevel=2)
        return getattr(self, new)

    def write(self, value):
        warnings.warn(message, DeprecationWarning, stacklevel=2)
        setattr(self, new, value)
    return property(read, write)


def _resolve_max_workers(max_workers: int | None, batch_size: int | None) -> int | None:
    """*max_workers*, or the value of its deprecated alias *batch_size*."""
    if batch_size is None:
        return max_workers
    if max_workers is not None:
        raise TypeError("pass max_workers or its deprecated alias batch_size, not both")
    warnings.warn("batch_size is deprecated; use max_workers", DeprecationWarning, stacklevel=3)
    return batch_size


class TextCleaner:
    """Runs one configured cleaning pipeline; ``process`` and ``process_batch`` return ``ProcessResult``s.

    ``ProcessResult.metadata`` always has source_chars, regex_counts, entity_counts, language_confidence,
    language_source, warnings and provenance. These options add keys:

    - ``replacement_mode='reversible'``: anon_map, or anon_map_reference when a *token_store* is given.
    - ``check_classify_document``: classes.
    - *privacy_policy*: policy and findings.
    - *include_entities*: findings.
    """

    NormaliseText = _renamed_attribute('NormaliseText', '_text_normaliser')
    ProcessStopwords = _renamed_attribute('ProcessStopwords', '_stopword_remover')
    ProcessDateTime = _renamed_attribute('ProcessDateTime', '_date_masker')
    ProcessContacts = _renamed_attribute('ProcessContacts', '_contact_masker')
    ProcessSpecialSymbols = _renamed_attribute('ProcessSpecialSymbols', '_symbol_masker')

    def __init__(
        self,
        cfg: TextCleanerConfig | None = None,
        *,
        privacy_policy: PrivacyPolicy | None = None,
        token_store: TokenStore | None = None,
        include_entities: bool = False,
        metrics_callback: Callable[[ProcessingMetrics], None] | None = None,
    ):
        """Initialize the text cleaning pipeline.

        Args:
            cfg: Immutable configuration. If None, reads from module-level
                 config variables (backward compatible).
            privacy_policy: Chooses each NER entity's action in place of the replacement; adds the policy and
                findings metadata keys. Requires ``replacement_mode='placeholder'`` (ValueError otherwise).
            token_store: Reversible mode saves each map here, and metadata has anon_map_reference
                in place of anon_map.
            include_entities: Add findings, which hold entity text, to metadata.
            metrics_callback: Called with value-free ProcessingMetrics after each non-blank document; empty or
                whitespace input runs nothing and does not call it.
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
        self._document_context = _DocumentContext()
        self._tags = {
            'HTML': self.cfg.replace_with_html, 'URL': self.cfg.replace_with_url,
            'EMAIL': self.cfg.replace_with_email, self._ip_type: self.cfg.replace_with_ips,
            'DATE': self.cfg.replace_with_dates, 'YEAR': self.cfg.replace_with_years,
            self._phone_type: self.cfg.replace_with_phone_numbers,
            'NUMBER': self.cfg.replace_with_numbers, 'CURRENCY': self.cfg.replace_with_currency_symbols,
            'IBAN': self.cfg.replace_with_iban, 'CARD': self.cfg.replace_with_cards,
            'ID': self.cfg.replace_with_identifiers,
        }
        self._text_normaliser = normtext.NormaliseText()
        self._init_user_rules()
        self._detector = self._build_detector()

        # Instance-scoped stopwords (with custom override support)
        self._stopword_remover = stopwords.ProcessStopwords(
            supported_languages=self.cfg.supported_languages,
            custom_stopwords=(dict(self.cfg.custom_stopwords)
                              if self.cfg.custom_stopwords and self.cfg.check_remove_stext_custom_stop_words
                              else None),
        )
        self._init_dates()

        self._contact_masker = contact.ProcessContacts()
        self._html_regex = constants.HTML_REGEX if self.cfg.html_tag_strict else constants.HTML_REGEX_ANY
        self._url_regex = constants.URL_REGEX if self.cfg.url_trailing_punct_outside else constants.URL_REGEX_LEGACY
        self._symbol_masker = special.ProcessSpecialSymbols()

        # Synthetic replacer: shared by regex pipeline steps + NER processor
        self._synthetic_replacer = None
        if self.cfg.replacement_mode == 'synthetic':
            self._synthetic_replacer = SyntheticReplacer(
                seed=self.cfg.synthetic_seed, per_document=self.cfg.synthetic_per_document,
                locale=self.cfg.synthetic_locale,
                low_confidence_locale=self.cfg.synthetic_low_confidence_locale,
                collapse_period=self.cfg.synthetic_collapse_period,
                reference_date=self.cfg.synthetic_reference_date,
            )

        self.GeneralNER: ner.GeneralNER | None = self._build_ner() if self.cfg.check_ner_process else None
        # GLiClass document-level pre-classification (optional, lazy-loaded)
        self._gliclass: Any = self._build_gliclass() if self.cfg.check_classify_document else None

        self._pipeline: list[Callable[[str], str]] = []
        self._init_pipeline()

    @property
    def _defer_tags(self) -> bool:
        """NER reads the placeholder-mode tags; fakes / indexed placeholders go in after NER."""
        return (self.cfg.ner_sees_tags and self.cfg.check_ner_process
                and self.cfg.replacement_mode != 'placeholder')

    def _normalised_terms(self, terms):
        """User terms as this config's normalising steps leave them: the text is normalised before the term
        rules run, so the terms are too ("José" in NFD, fullwidth "ＡＣＭＥ")."""
        for step in self._normalising_steps():
            terms = list(map(step, terms))
        return [term for term in terms if term.strip()]

    def _deny_labels(self) -> dict[str, str]:
        """Each normalised deny term -> its label."""
        labels: dict[str, str] = {}
        for label, terms in (self.cfg.custom_deny_terms or {}).items():
            for term in self._normalised_terms(terms):
                labels.setdefault(term, label)
        return labels

    def _build_user_rules(self, deny_labels) -> list:
        """(pattern, label of a match, custom_patterns label for the time limit or None) per rule."""
        rules: list[tuple[Any, Callable[..., Any], str | None]] = []
        if deny_labels:  # each term is a group named by its label: m.lastgroup is the label
            rules.append((constants.terms_regex(deny_labels, self.cfg.custom_terms_case_sensitive, deny_labels),
                          lambda m: m.lastgroup, None))
        for label, pattern in (self.cfg.custom_patterns or {}).items():
            rules.append((regex.compile(pattern), lambda m, label=label: label, label))
        return rules

    def _init_user_rules(self) -> None:
        """User term lists: deny terms, then each user pattern, before the built-in steps; allow terms last."""
        self._user_rules = self._build_user_rules(self._deny_labels())
        for label in [*(self.cfg.custom_deny_terms or {}), *(self.cfg.custom_patterns or {})]:
            self._tags.setdefault(label, f'<{label}>')
        allow_terms = self._normalised_terms(self.cfg.custom_allow_terms)
        self._allow_regex = (constants.terms_regex(allow_terms, self.cfg.custom_terms_case_sensitive)
                             if allow_terms else None)
        # Markers that are not <TAG>-shaped ("EMAIL"): NER protects them like <EMAIL> (ner.regex_markers).
        self._custom_markers = tuple(sorted({
            tag for tag in self._tags.values() if tag and not ner.PLACEHOLDER_TOKEN.fullmatch(tag)}))
        self._marker_regex = re.compile('|'.join([ner.PLACEHOLDER_TOKEN.pattern, *map(
            re.escape, sorted(self._custom_markers, key=len, reverse=True))]))

    def _build_detector(self):
        """Language detector: tuple restricts; else all 75 Lingua languages unless turned off."""
        if isinstance(self.cfg.language, tuple):
            return resources.build_detector(frozenset(self.cfg.language))
        if self.cfg.detect_all_languages:
            return resources.build_all_detector()
        return resources.build_detector(self.cfg.supported_languages)

    def _init_dates(self) -> None:
        """Instance-scoped datetime (with custom month names -> per-instance regex)."""
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
            self._date_masker = datetime.ProcessDateTime(
                date_regex=date_regex,
                fuzzy_vocabulary=fuzzy_all,
                fuzzy_vocabulary_by_lang=fuzzy_by_lang,
            )
        else:
            self._date_masker = datetime.ProcessDateTime()
        # fold_styled_characters: dates, like phones, read any space as " " and any dash as "-" (1987\u201105\u201112),
        # any dot form as "." (1990﹒ 1﹒ 15﹒), and a month-name date's day may be a circled number (⑮ January 1990).
        self._date_view = (
            contact.DigitView(constants.build_date_regex(mn_dict, styled=True, slash_year=slash_year),
                              contact.DATE_SEPARATOR_VIEW)
            if self.cfg.fold_styled_characters else self._date_masker._date_regex)

    def _build_ner(self) -> ner.GeneralNER:
        return ner.GeneralNER(
            self.cfg,
            synthetic_replacer=self._synthetic_replacer,
            regex_markers=self._custom_markers,
            allow_regex=self._allow_regex,
            privacy_policy=self._privacy_policy,
        )

    def _build_gliclass(self):
        return GLiClassAdapter(
            model_id=self.cfg.gliclass_model or GLICLASS_DEFAULT_MODEL,
            revision=self.cfg.gliclass_revision,
            labels=self.cfg.gliclass_labels,
            threshold=self.cfg.gliclass_threshold,
            classification_type=self.cfg.gliclass_classification_type,
            onnx=self.cfg.gliclass_onnx,
            # Auto stays on CPU (0.7 behaviour); only an explicit 'cuda' moves GLiClass.
            device=f"cuda:{self.cfg.cuda_device_id}" if self.cfg.ner_device == 'cuda' else 'cpu',
        )

    def _with_config(self, cfg: TextCleanerConfig) -> 'TextCleaner':
        """A view of this cleaner running *cfg*, which only turns steps off; shares its loaded models."""
        view = copy.copy(self)
        view.cfg = cfg
        if not cfg.check_ner_process:
            view.GeneralNER = None
        if not cfg.check_classify_document:
            view._gliclass = None
        # Deny terms as either config normalises them: *cfg* may keep the "José" this one folds to "Jose".
        labels = self._deny_labels()
        merged = {**view._deny_labels(), **labels}
        if merged != labels:
            view._user_rules = view._build_user_rules(merged)
        view._pipeline = []
        view._init_pipeline()
        return view

    def _normalising_steps(self):
        """The steps before the first regex step (unicode fix, ASCII, emoji), in pipeline order."""
        cfg = self.cfg
        return [step for on, step in ((cfg.check_fix_bad_unicode, self._fix_bad_unicode),
                                      (cfg.check_to_ascii_unicode, self._text_normaliser.to_ascii_unicode),
                                      (cfg.check_remove_emoji, self._text_normaliser.remove_emoji)) if on]

    def _init_pipeline(self):
        """Build the pipeline steps list based on config."""
        cfg = self.cfg

        self._pipeline.extend(self._normalising_steps())
        # Steps before this index only normalise; markers they surface are source text.
        self._normalise_step_count = len(self._pipeline)
        rows = (
            # escaped "<b>" stays text: decoded after the view, hidden from tags
            (cfg.check_fix_bad_unicode, self._unescape_markup),
            (bool(self._user_rules), self._replace_user_terms),
            (cfg.check_replace_html, self._replace_html),
            # brackets hide from the tag step only; contacts see "john<at>x.com"
            (cfg.check_fix_bad_unicode, self._restore_brackets),
            (cfg.check_replace_urls, self._replace_urls),
            (cfg.check_replace_emails, self._replace_emails),
            (cfg.check_replace_ips, self._replace_ips),
            (cfg.check_replace_dates, self._replace_dates),
            (cfg.check_fuzzy_replace_dates, self._replace_fuzzy_dates),
            # Phones before years, so a year step cannot split "+31 6 1234 2019".
            # Checksum-valid IBANs/cards and identifier tokens before phones, so a phone cannot take their digits.
            (cfg.check_replace_iban_cards, self._replace_iban_cards),
            (cfg.check_replace_identifiers, self._replace_identifiers),
            (cfg.check_replace_phone_numbers, self._replace_phone_numbers),
            (cfg.check_replace_years, self._replace_years),
            (cfg.check_replace_numbers, self._replace_numbers),
            (cfg.check_replace_currency_symbols, self._replace_currency_symbols),
            # Emitted tokens and kept IPs are hidden from the steps above; put them back.
            (True, self._restore_protected),
            (cfg.check_remove_isolated_letters, self._symbol_masker.remove_isolated_letters),
            (cfg.check_remove_isolated_special_symbols, self._remove_isolated_special_symbols),
            (cfg.check_normalize_whitespace, self._normalize_whitespace),
            # Deferred values stay hidden through the cleanup above, so a tag it would remove
            # ("[EMAIL]" under bracket removal) still carries its value to the swap after NER.
            (True, self._restore_deferred),
        )
        self._pipeline.extend(step for on, step in rows if on)
        # User-provided custom steps: after the built-in pre-NER steps, before NER and _after_ner
        self._pipeline.extend(cfg.custom_pipeline_steps)

    def _detect_language(self, text: str) -> tuple[str | None, float | None, str, float | None]:
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

        if not isinstance(language_config, tuple) and not self._language_needed():
            return None, None, 'none', None

        values = self._detector.compute_language_confidence_values(text)
        # Same "unknown" rule as Lingua's detect_language_of: nothing scored, or a tie at the top.
        if values and values[0].value > 0 and (len(values) < 2 or values[1].value < values[0].value):
            top = values[0]
            if not isinstance(language_config, tuple) or top.language.name in language_config:
                # Lingua's sum order varies per process (last-ulp noise); round so workers match.
                margin = top.value - (values[1].value if len(values) > 1 else 0.0)
                return top.language.name, round(top.value, 6), 'detected', round(margin, 6)
        if isinstance(language_config, tuple):
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

    def _process_single_isolated(self, text: str) -> ProcessResult:
        """``_run_pipeline`` that clears the per-thread document context afterwards."""
        try:
            return self._run_pipeline(text)
        finally:
            self._document_context.clear()

    def _run_pipeline(self, text: str) -> ProcessResult:
        """Process a single text through the entire pipeline.

        Returns:
            ProcessResult (unpacks as 3-tuple for backward compat).
        """
        state = self._before_ner(text)
        current_text = state.text

        # NER processing; always detailed (same detection) so entity counts are known.
        findings: tuple[EntityFinding, ...] = ()
        if self.GeneralNER is not None:
            detailed = self.GeneralNER.ner_process_detailed(
                current_text,
                language=state.language,
                anon_map=state.anon_map,
                source_markers=state.source_markers,
            )
            current_text = detailed.text
            findings = detailed.findings
        return self._after_ner(state, current_text, findings)

    def normalise(self, text: str) -> str:
        """The steps before the first regex step (unicode fix, ASCII, emoji), as ``process`` runs them.

        The streaming split guard reads this same view, so the two cannot differ.
        """
        for step in self._pipeline[:self._normalise_step_count]:
            text = step(text)
        return text

    def _before_ner_isolated(self, text: str) -> _DocumentState:
        """``_before_ner`` that clears the per-thread document context afterwards."""
        try:
            return self._before_ner(text)
        finally:
            self._document_context.clear()

    def _before_ner(self, text: str) -> _DocumentState:
        """Every step before NER."""
        started_at = time.perf_counter()

        # Per-document anonymization map (reversible mode only)
        anon_map = None
        if self.cfg.replacement_mode == 'reversible':
            anon_map = AnonymizationMap()
        self._document_context.anon_map = anon_map
        self._document_context.protected = {}
        # The dict travels in the returned state, so it survives the context reset.
        regex_counts: dict[str, dict[str, int]] = {}
        self._document_context.regex_counts = regex_counts
        doc_warnings: set = set()  # step warnings ('custom_pattern_timeout'), travel in the state like the counts
        self._document_context.warnings = doc_warnings
        deferred_order: list | None = [] if self._defer_tags else None
        self._document_context.deferred_tags = {} if self._defer_tags else None
        self._document_context.deferred_order = deferred_order

        # Document-level classification (before any text modification)
        doc_classes = None
        if self._gliclass is not None:
            doc_classes = self._gliclass.classify(text)

        # Detect language (pure function, thread-safe)
        language, language_confidence, language_source, language_margin = self._detect_language(text)
        self._document_context.language = language  # _replace_fuzzy_dates reads it

        # Reset per-document synthetic consistency cache (thread-safe via threading.local);
        # the language picks the Faker locale.
        if self._synthetic_replacer:
            self._synthetic_replacer.reset_cache(text, language, self._language_low_confidence(
                language_source, language_confidence, language_margin))

        current_text = text

        current_text = self.normalise(current_text)
        # Placeholder spellings the source holds, raw or once normalised ("＜LONDON＞",
        # "&lt;LONDON&gt;"): NER gives them no protection and reversible mode never
        # allocates them.
        source_markers = frozenset(ner.PLACEHOLDER_TOKEN.findall(text)) | frozenset(
            ner.PLACEHOLDER_TOKEN.findall(normtext.unescape_markup(current_text))) | frozenset(
            tag for tag in self._custom_markers if tag in text or tag in current_text)
        if anon_map is not None:
            anon_map.reserved.update(source_markers)
        # (marker, emitted) for each marker occurrence, recorded by _restore_protected
        self._document_context.marker_order = [] if source_markers else None
        for step in self._pipeline[self._normalise_step_count:]:
            self._document_context.step_text = current_text
            current_text = step(current_text)
        source_markers = self._source_marker_spans(current_text, source_markers)

        return _DocumentState(
            text=current_text, language=language, anon_map=anon_map, doc_classes=doc_classes, started_at=started_at,
            regex_counts=regex_counts, language_confidence=language_confidence, language_source=language_source,
            source_chars=len(text), source_markers=source_markers, language_margin=language_margin,
            deferred=deferred_order, warnings=doc_warnings)

    def _document_warnings(self, state: _DocumentState, models) -> set:
        """The warning codes of a processed document; *models* are the NER models it ran."""
        warnings = set(state.warnings)
        if state.language_source == 'none' and self._language_needed():
            warnings.add('language_unknown')
        if state.language_source == 'fallback':
            warnings.add('language_fallback')
        if self._language_low_confidence(state.language_source, state.language_confidence, state.language_margin):
            warnings.add('language_low_confidence')
        general_ner = self.GeneralNER
        if general_ner is None:
            return warnings
        if general_ner.uses_fallback_models(state.language):
            warnings.add('ner_fallback_models')
        if any(m['provider'] == 'CPUExecutionProvider' for m in models) and general_ner.device == 'cuda':
            warnings.add('cuda_fallback')
        if any(m['provider'] == 'cpu' for m in models) and general_ner.device == 'mps':
            warnings.add('mps_fallback')
        return warnings

    def _policy_metadata(self, findings) -> ProcessMetadata:
        """The metadata a privacy policy or include_entities adds, for empty input too."""
        metadata: ProcessMetadata = {}
        if self._privacy_policy is not None:
            metadata['policy'] = {
                'name': self._privacy_policy.name,
                'version': self._privacy_policy.version,
            }
        if self._include_entities or self._privacy_policy is not None:
            metadata['findings'] = findings
        return metadata

    def _after_ner(self, state: _DocumentState, current_text: str, findings) -> ProcessResult:
        """Every step after NER, for the document *state* from ``_before_ner``."""
        language, anon_map = state.language, state.anon_map
        current_text = self._swap_deferred(current_text, state.deferred, state.text, findings, anon_map)
        if anon_map is not None and self.GeneralNER is not None and self.cfg.check_replace_numbers:
            current_text = self._mask_numbers_glued_to_ner(current_text, state)

        # Statistical model processing (always returns stat_text, even if None)
        stat_text = None
        if self.cfg.check_statistical_model_processing:
            stat_text = self._build_stat_text(current_text, language)

        # Value-free counts of concealed entities are always present; metadata['findings'], on opt-in
        # below, also lists what the policy exempted.
        entity_counts = _entity_counts(f for f in findings if f.action not in _EXEMPT_ACTIONS)
        ner_on = self.GeneralNER is not None
        models = self.GeneralNER.models_for(language) if self.GeneralNER is not None else []
        metadata: ProcessMetadata = {
            'source_chars': state.source_chars,
            'regex_counts': state.regex_counts,
            'entity_counts': entity_counts,
            'language_confidence': state.language_confidence,
            'language_source': state.language_source,
            'warnings': sorted(self._document_warnings(state, models)),
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
        if state.doc_classes is not None:
            metadata['classes'] = state.doc_classes
        metadata.update(self._policy_metadata(findings))

        if self._metrics_callback is not None:
            self._metrics_callback(ProcessingMetrics(
                duration_ms=(time.perf_counter() - state.started_at) * 1000,
                language=language,
                backend=self.cfg.ner_backend if ner_on else None,
                policy=self._privacy_policy.name if self._privacy_policy else None,
                entity_counts={k: v['count'] for k, v in entity_counts.items()},
            ))

        return ProcessResult(current_text, stat_text, language, metadata=metadata)

    def process_batch(
        self, texts: list[str], max_workers: int | None = None, *, batch_size: int | None = None,
    ) -> list[ProcessResult]:
        """Process multiple texts with bounded document concurrency.

        Args:
            max_workers: Maximum number of worker threads (default ``os.cpu_count()``). Documents per NER
                model call is ``cfg.ner_batch_size``.
            batch_size: Deprecated alias of *max_workers*.

        Returns:
            List of ProcessResult (each unpacks as 3-tuple for backward compat).
        """
        max_workers = _resolve_max_workers(max_workers, batch_size)
        if max_workers is not None and max_workers <= 0:
            raise ValueError("max_workers must be >= 1")
        if not texts:
            return []
        if isinstance(texts, str):
            raise ValueError("texts must be a list of strings; use process() for one text")

        results: list[ProcessResult | None] = [None] * len(texts)
        to_process = []

        for i, text in enumerate(texts):
            if not isinstance(text, str):
                raise ValueError(f"Input must be string, got {type(text)}")
            if not text or text.isspace():
                results[i] = ProcessResult("", "", None, metadata=self._empty_metadata(text))
            else:
                to_process.append((i, text))

        if not to_process:
            return cast(list[ProcessResult], results)

        # Parallel processing: ONNX Runtime releases the GIL during C++ inference
        # ops, so threads achieve real concurrency for the NER inference bottleneck.
        worker_limit = max_workers if max_workers is not None else (os.cpu_count() or 4)
        workers = min(len(to_process), worker_limit)

        def run(fn):
            if workers <= 1:
                return [fn(text) for _, text in to_process]
            with ThreadPoolExecutor(max_workers=workers) as executor:
                return list(executor.map(fn, [text for _, text in to_process]))

        if self.GeneralNER is not None and self.cfg.ner_batch_size > 1:
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
                [state.text for state in states],
                [state.language for state in states],
                anon_maps=[state.anon_map for state in states],
                before_finalize=restore_synthetic,
                max_workers=workers,
                source_markers=[state.source_markers for state in states],
            )
            for (i, _), state, ner_result in zip(to_process, states, ner_results, strict=True):
                results[i] = self._after_ner(state, ner_result.text, ner_result.findings)
        else:
            # Each text goes through the full pipeline independently.
            for (i, _), result in zip(to_process, run(self._process_single_isolated), strict=True):
                results[i] = result

        return cast(list[ProcessResult], results)

    def _empty_metadata(self, text: str) -> ProcessMetadata:
        """Metadata for empty or whitespace input (nothing ran)."""
        metadata: ProcessMetadata = {
            'source_chars': len(text),
            'regex_counts': {},
            'entity_counts': {},
            'language_confidence': None,
            'language_source': 'none',
            'warnings': ['empty_input'],
            'provenance': {'backend': self.cfg.ner_backend if self.GeneralNER is not None else None, 'models': []},
            **self._policy_metadata(()),
        }
        # Reversible results always carry a (here empty) map, like non-empty input.
        if self.cfg.replacement_mode == 'reversible':
            if self._token_store is not None:
                metadata['anon_map_reference'] = self._token_store.save(AnonymizationMap())
            else:
                metadata['anon_map'] = AnonymizationMap()
        if self._gliclass is not None:
            metadata['classes'] = []
        return metadata

    async def aprocess_batch(
        self, texts: list[str], max_workers: int | None = None, *, batch_size: int | None = None,
    ) -> list[ProcessResult]:
        """Async version of process_batch for use with asyncio-based frameworks (FastAPI, aiohttp).

        Runs process_batch in a thread-pool executor so it doesn't block the event loop.
        *max_workers* is the maximum number of worker threads, as in process_batch; *batch_size* is its
        deprecated alias.
        """
        max_workers = _resolve_max_workers(max_workers, batch_size)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.process_batch, texts, max_workers)

    def warmup(self, languages: list[str] | None = None) -> None:
        """Pre-load the NER models each language routes to, to avoid first-request latency.

        A language that fails to load is logged as a warning and skipped; it loads (or fails) on first use.

        Args:
            languages: Language names to pre-load (e.g. ``['ENGLISH', 'DUTCH']``).
                       If ``None``, pre-loads models for all supported languages.
        """
        if self.GeneralNER is None:
            return
        langs = languages or list(self.cfg.supported_languages)
        for lang in langs:
            try:
                self.GeneralNER._load_models([lang])
            except ner.ModelLoadError as e:
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
        if self._allowed(original):  # plain text, not protected: a later detection holding it still sees it
            return original
        counts = self._document_context.regex_counts
        if counts is not None:
            entry = counts.setdefault(entity_type, {'count': 0, 'chars': 0})
            entry['count'] += 1
            entry['chars'] += len(original)
        key = self._protect(value)
        tags = self._document_context.deferred_tags
        if tags is not None:
            tag = (constants.CURRENCIES.get(original)
                   if entity_type == 'CURRENCY' and self.cfg.replace_with_currency_symbols is None
                   else self._tags.get(entity_type))
            if tag and tag != value:
                tags[key] = tag
        return key

    def _allowed(self, original):
        """Whether a detection of *original* is kept as text (custom_allow_terms): every occurrence of it in
        the step's text lies inside an allowed term occurrence."""
        text = self._document_context.step_text
        if self._allow_regex is None or not text or not original:
            return False
        cached = self._document_context.allow_spans
        if cached is None or cached[0] is not text:
            spans = [m.span() for m in self._allow_regex.finditer(text)]
            cached = (text, [a for a, _ in spans], [b for _, b in spans], {})
            self._document_context.allow_spans = cached
        _, starts, ends, verdicts = cached
        if original not in verdicts:  # once per value and step text, each occurrence a bisect (no cubic scan)
            def occurrences():
                at = text.find(original)
                while at >= 0:
                    yield at
                    at = text.find(original, at + len(original))
            # shortcut: position-free; a value also found outside an allowed term is masked everywhere (no leak).
            # Allowed matches never overlap, so only the last one starting at or before s can hold s.
            verdicts[original] = bool(starts) and text.find(original) >= 0 and all(
                (i := bisect_right(starts, s) - 1) >= 0 and s + len(original) <= ends[i] for s in occurrences())
        return verdicts[original]

    def _replace_user_terms(self, text):
        """custom_deny_terms, then each custom_patterns regex: <LABEL> in placeholder mode, <LABEL_0> in reversible
        mode, and a fake in synthetic mode when the label has a generator."""
        anon_map = self._document_context.anon_map

        def source(original):
            """*original* with an earlier rule's protected emit inside it put back as that emit's source, so
            the map holds "Ann secret", not a private-use key ("Ann" as <PERSON_0>, then ".+ secret")."""
            if not _KEY.search(original):
                return original
            values = {key: value for value, key in (self._document_context.protected or {}).items()}
            held = {e.placeholder: e.original for e in anon_map.entries} if anon_map is not None else {}
            return _KEY.sub(lambda k: held.get(values.get(k.group()), values.get(k.group(), k.group())), original)

        def emit(match, label) -> str:
            original = match.group()
            if not original or self._allowed(original):
                return original
            if anon_map is not None:
                placeholder = anon_map.next_placeholder(label)
                anon_map.add(placeholder, source(original), label, match.start(), match.end())
                return self._emit(label, original, placeholder)
            if self._synthetic_replacer and label in FAKER_GENERATORS:
                return self._emit(label, original, self._synthetic_replacer.get_replacement(label, original))
            return self._emit(label, original, self._tags[label])
        for pattern, label_of, timed_label in self._user_rules:
            self._document_context.step_text = text
            text = self._sub_user_rule(pattern, timed_label, lambda m, label_of=label_of: emit(m, label_of(m)), text)
        return text

    def _sub_user_rule(self, pattern, timed_label, repl, text):
        """*text* with each match of a user rule replaced. Matches first, then *repl*: a pattern that times out
        (see _user_matches) adds nothing to the map (the streaming guard runs this too)."""
        parts, last = [], 0
        for match in self._user_matches(pattern, timed_label, text) or ():
            parts += [text[last:match.start()], repl(match)]
            last = match.end()
        return ''.join(parts) + text[last:]

    def _user_matches(self, pattern, timed_label, text, **kwargs):
        """Every match of a user rule in *text*; a custom pattern (*timed_label* set) stops at custom_patterns_timeout
        and then raises CustomPatternTimeout, or with custom_patterns_on_timeout='warn' logs it, adds the
        'custom_pattern_timeout' warning and returns None."""
        timeout = self.cfg.custom_patterns_timeout if timed_label is not None else None
        try:
            return list(pattern.finditer(text, timeout=timeout, **kwargs))
        except TimeoutError:
            assert timeout is not None  # no limit, no TimeoutError
            if self.cfg.custom_patterns_on_timeout == 'raise':
                raise CustomPatternTimeout(timed_label, timeout) from None
            logger.warning("custom_patterns[%r] ran longer than %ss; skipped for this document", timed_label, timeout)
            warnings = self._document_context.warnings
            if warnings is not None:
                warnings.add('custom_pattern_timeout')
            return None

    def _protect(self, value):
        """Hide an emitted value from later regex steps (``<EMAIL1>``, kept IPs)."""
        protected = self._document_context.protected
        if protected is None:
            return value
        # Private-use planes 15-16 only: outside the BMP letter ranges the email and URL
        # rules accept, and no digit, letter or space for later steps to match.
        return protected.setdefault(value, f"{constants.KEY_OPEN}{chr(0xF0000 + len(protected))}{constants.KEY_CLOSE}")

    def _restore_protected(self, text):
        protected = self._document_context.protected or {}
        tags = self._document_context.deferred_tags or {}
        order = self._document_context.marker_order
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
        order = self._document_context.marker_order
        self._document_context.marker_order = None
        if order is None:
            return source_markers
        found = list(self._marker_regex.finditer(text))
        if [m.group() for m in found] != [marker for marker, _ in order]:
            return source_markers
        return frozenset(m.span() for m, (_, emitted) in zip(found, order) if not emitted)

    def _restore_deferred(self, text):
        protected = self._document_context.protected or {}
        tags, order = self._document_context.deferred_tags, self._document_context.deferred_order
        if not tags or order is None:
            return text
        # Deferred keys become their tag; every tag occurrence is recorded in order
        # (source text and plain tag emits as None) so _swap_deferred can pair them up.
        values = {key: value for value, key in protected.items() if key in tags}
        tracked = set(tags.values())
        pattern = re.compile('|'.join(map(re.escape, sorted({*values, *tracked}, key=len, reverse=True))))

        def put(match):
            found = match.group()
            if found in tags:
                order.append((tags[found], values[found]))
                return tags[found]
            order.append((found, None))
            return found
        return pattern.sub(put, text)

    @staticmethod
    def _swap_deferred(text, order, source='', findings=(), anon_map=None):
        """After NER: each tag occurrence NER left gets its recorded value, in order.

        *source* is the text NER read; occurrences inside an NER finding are skipped, and in
        reversible mode their originals go back into that NER map entry.
        """
        if not order:
            return text
        values: dict[str, list] = {}
        for tag, value in order:
            values.setdefault(tag, []).append(value)
        tags = sorted(values, key=len, reverse=True)
        pattern = re.compile('|'.join(map(re.escape, tags)))
        # An NER placeholder in the text is matched whole, so a marker such as "PERSON" is
        # not counted (or swapped) inside "<PERSON_0>".
        held = {e.placeholder for e in anon_map.entries} - set(values) if anon_map is not None else set()
        in_text = re.compile('|'.join(map(re.escape, sorted([*held, *tags], key=len, reverse=True))))
        found = Counter(match.group() for match in in_text.finditer(text) if match.group() in values)
        where: dict[str, list] = {}
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
        anon_map = self._document_context.anon_map
        if anon_map is None:
            return None

        def replace(match) -> str:
            if self._allowed(match.group()):
                return match.group()
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
        return self._text_normaliser.fix_bad_unicode(
            text, decode_escapes=self.cfg.check_decode_unicode_escapes,
            fix_character_width=self.cfg.fix_character_width,
            fold_styled_characters=self.cfg.fold_styled_characters)

    def _unescape_markup(self, text):
        return normtext.unescape_markup(text, self._protect)

    def _restore_brackets(self, text):
        protected = self._document_context.protected or {}
        for value in ('<', '>'):
            if value in protected:
                text = text.replace(protected[value], value)
        return text

    def _replace_html(self, text):
        return self._replace_typed(text, self._html_regex, 'HTML', synthetic=False)

    def _replace_urls(self, text):
        return self._replace_typed(text, self._url_regex, 'URL')

    def _replace_emails(self, text):
        if not constants.EMAIL_GATE.search(text):  # no separator, no email: skip the slow regex
            return text
        return self._replace_typed(text, constants.EMAIL_REGEX, 'EMAIL')

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
        replacer = self._synthetic_replacer
        if replacer:
            def emit(orig):
                return replacer.get_replacement(self._ip_type, orig)
        else:
            def emit(orig):
                return self.cfg.replace_with_ips
        return self._contact_masker.replace_ips_with_fn(
            text,
            lambda orig: (
                self._emit(self._ip_type, orig, emit(orig))
                if contact.is_replaceable_ip(orig, public_only) else self._protect(orig)
            ),
        )

    def _replace_dates(self, text):
        return self._replace_typed(text, self._date_view, 'DATE', synthetic=False)

    def _replace_fuzzy_dates(self, text):
        return self._date_masker.fuzzy_replace_dates(
            text,
            replace_with=lambda orig: self._emit('DATE', orig, self.cfg.replace_with_dates),
            score_cutoff=self.cfg.fuzzy_date_score_cutoff,
            language=self._document_context.language,
        )

    def _replace_years(self, text):
        return self._replace_typed(text, self._year_view(text), 'YEAR', synthetic=False)

    def _year_view(self, text):
        """The year pattern the year step applies to *text*."""
        view = constants.YEAR_REGEX
        if self.cfg.check_replace_numbers and self.cfg.replace_numbers_mode == 'identifiers':
            # A year-like group of a multi-group ID ("123-45 2020") is left to the numbers step, which masks the
            # whole run; masking the year alone would leave "123-45" visible. A run of years ("1999 2000 2001")
            # and a date-shaped run ("05 11 2023") stay years.
            runs = [m.span() for m in self._numbers_regex.finditer(text) if not m.group().isdecimal()
                    and not all(map(constants.YEAR_REGEX.fullmatch, re.split(r"\D+", m.group())))
                    and not constants.SPACED_DATE_REGEX.fullmatch(m.group())]
            if runs:
                return contact.RejectView(view, lambda _, start, end: any(a <= start and end <= b for a, b in runs))
        return view

    @property
    def _phone_folded_view(self):
        """The phone pattern before phone_reject_embedded (the streaming guard reads it too)."""
        return contact.PHONE_VIEW if self.cfg.fold_styled_characters else contact.PHONE_DIGIT_VIEW

    @property
    def _phone_view(self):
        """The phone pattern the phone step applies (the streaming guard reads it too)."""
        view = self._phone_folded_view
        return contact.RejectView(view, contact.embedded_phone) if self.cfg.phone_reject_embedded else view

    def _replace_phone_numbers(self, text):
        return self._replace_typed(text, self._phone_view, self._phone_type)

    def _replace_typed(self, text, pattern, entity_type, synthetic=True):
        """Replace *pattern* matches as *entity_type* in each replacement mode (fakes only if *synthetic*)."""
        reversible = self._replace_reversible(text, pattern, entity_type)
        if reversible is not None:
            return reversible
        replacer = self._synthetic_replacer
        if synthetic and replacer:
            def value(orig):
                return replacer.get_replacement(entity_type, orig)
        else:
            def value(orig):
                return self._tags[entity_type]
        return pattern.sub(lambda m: self._emit(entity_type, m.group(), value(m.group())), text)

    def _replace_iban_cards(self, text):
        return self._replace_typed(self._replace_typed(text, contact.IBAN_VIEW, 'IBAN'), contact.CARD_VIEW, 'CARD')

    def _replace_identifiers(self, text):
        return self._replace_typed(text, constants.IDENTIFIER_REGEX, 'ID')

    def _replace_numbers(self, text):
        if self._document_context.anon_map is None:
            return self._replace_typed(text, self._numbers_regex, 'NUMBER')
        return self._mask_numbers(text, self._glued_number_regex, self._numbers_regex)

    def _mask_numbers(self, text, *patterns):
        """Reversible mode: mask *patterns* as NUMBER with every placeholder hidden."""
        # A placeholder-shaped source token keeps its digits, so it cannot collide with an allocated one. It is
        # hidden, not just skipped: the pattern's lookarounds treat '<' and '>' as edges and would spare a joined run.
        hidden: dict[str, str] = {}

        def hide(match):
            key = f'{constants.HIDDEN_OPEN}{chr(65 + len(hidden))}{constants.HIDDEN_CLOSE}'
            hidden[key] = match.group()
            return key

        replaced = ner.PLACEHOLDER_TOKEN.sub(hide, text)
        for pattern in patterns:
            replaced = self._replace_typed(replaced, pattern, 'NUMBER')
        for key, placeholder in hidden.items():
            replaced = replaced.replace(key, placeholder)
        return replaced

    @property
    def _glued_number_regex(self):
        min_digits = 1 if self.cfg.replace_numbers_mode == 'all' else self.cfg.replace_numbers_min_digits
        return constants.placeholder_glued_number_regex(min_digits, self.cfg.replace_numbers_in_words)

    def _mask_numbers_glued_to_ner(self, text: str, state: _DocumentState) -> str:
        """A run glued to a name was a token part in the number step; once NER replaces the name it stands
        beside a placeholder."""
        context = self._document_context
        context.clear()  # no key to hide a value behind any more: no step follows to restore it
        context.anon_map, context.regex_counts, context.step_text = state.anon_map, state.regex_counts, text
        try:
            return self._mask_numbers(text, self._glued_number_regex)
        finally:
            context.clear()

    @property
    def _numbers_regex(self):
        return contact.config_numbers_regex(self.cfg)

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
            return self._symbol_masker.replace_currency_symbols_with_fn(
                text, lambda symbol: self._emit('CURRENCY', symbol, constants.CURRENCIES[symbol]),
            )
        return self._symbol_masker.replace_currency_symbols(
            text, replace_with=lambda m: self._emit('CURRENCY', m.group(), replace_with),
        )

    def _remove_isolated_special_symbols(self, text):
        return self._symbol_masker.remove_isolated_special_symbols(
            text,
            remove_brackets=self.cfg.check_remove_bracket_content,
            remove_braces=self.cfg.check_remove_brace_content,
            max_chars=self.cfg.bracket_content_max_chars,
            max_words=self.cfg.bracket_content_max_words,
            keep_json=self.cfg.keep_json_braces,
            keep_glued_to_placeholder=self._document_context.anon_map is not None,
        )

    def _normalize_whitespace(self, text):
        return self._text_normaliser.normalize_whitespace(text, no_line_breaks=True)

    def _build_stat_text(self, text: str, language: str | None) -> str:
        """Generate statistical model text (lowercase, no stopwords, no punctuation)."""
        stat_text = text
        if self.cfg.check_smart_casefold:
            stop_words = self._stopword_remover.get_stop_words(language) if language else set()
            stat_text = self._text_normaliser.smart_casefold(stat_text, stop_words=stop_words)
        elif self.cfg.check_casefold:
            stat_text = stat_text.casefold()
        if self.cfg.check_remove_stopwords and language:
            stat_text = self._stopword_remover.remove_stopwords(stat_text, language)
        if self.cfg.check_remove_punctuation:
            stat_text = self._symbol_masker.remove_punctuation(stat_text)
        if self.cfg.check_remove_isolated_letters:
            stat_text = self._symbol_masker.remove_isolated_letters(stat_text)
        if self.cfg.check_normalize_whitespace:
            stat_text = self._text_normaliser.normalize_whitespace(stat_text)
        return stat_text
