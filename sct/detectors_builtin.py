"""Native detectors share legacy matching decisions without emitting replacements."""

from dataclasses import dataclass

from sct.config import TextCleanerConfig
from sct.findings import Detection, DetectionContext
from sct.utils import constants, contact, normtext
from sct.utils.ner import GeneralNER


@dataclass(frozen=True)
class _Capabilities:
    id: str
    version: str = '1'
    entity_types: tuple[str, ...] = ()
    languages: tuple[str, ...] = ()
    coordinate_system: str = 'unicode_codepoint'
    supports_context: bool = False
    supports_windowed: bool = False
    network_access: bool = False


class BuiltinRegexDetector:
    """Collect enabled format matches on one input stage using existing rule helpers."""

    def __init__(self, cfg: TextCleanerConfig):
        from sct.sct import TextCleaner, _DocumentContext

        # Reuse setup helpers without initializing language detectors, models or a cleaner pipeline.
        helper = object.__new__(TextCleaner)
        helper.cfg = cfg
        helper._text_normaliser = normtext.NormaliseText()
        helper._tags = {}
        helper._document_context = _DocumentContext()
        helper._init_user_rules()
        helper._init_dates()
        self._helper = helper
        types = [label for enabled, label, _ in self._rules('') if enabled]
        types += ['IP'] if cfg.check_replace_ips else []
        types += ['DATE'] if cfg.check_fuzzy_replace_dates else []
        types += [*(cfg.custom_deny_terms or {}), *(cfg.custom_patterns or {})]
        self._capabilities = _Capabilities('builtin_regex', entity_types=tuple(dict.fromkeys(types)))

    @property
    def capabilities(self):
        return self._capabilities

    def detect(self, text: str, *, language: str | None, context: DetectionContext) -> tuple[Detection, ...]:
        helper, hits = self._helper, []
        cfg = helper.cfg

        def collect(label, pattern, valid=lambda value: True):
            def match(m):
                if m.end() > m.start() and valid(m.group()):
                    hits.append(Detection(label, m.start(), m.end(), 'builtin_regex', None, 'canonical'))
                return m.group()
            pattern.sub(match, text)

        for pattern, label_of, timed_label in helper._user_rules:
            for m in helper._user_matches(pattern, timed_label, text) or ():
                if m.end() > m.start():
                    hits.append(Detection(label_of(m), m.start(), m.end(), 'builtin_regex', None, 'canonical'))
        for enabled, label, pattern in self._rules(text):
            if enabled:
                collect(label, pattern)
        if cfg.check_replace_ips:
            collect('IP', contact.IP_VIEW, lambda value: contact.is_replaceable_ip(value, cfg.replace_ips_public_only))
        if cfg.check_fuzzy_replace_dates:
            for match in helper._date_masker.fuzzy_date_matches(
                    text, score_cutoff=cfg.fuzzy_date_score_cutoff, language=language):
                hits.append(Detection('DATE', match.start(), match.end(), 'builtin_regex', None, 'canonical'))
        from sct.utils.ner_spans import drop_allowed
        return tuple(drop_allowed(helper._allow_regex, text, hits, lambda h: (h.start, h.end)))

    def _rules(self, text):
        helper = self._helper
        cfg = helper.cfg
        return (
            (cfg.check_replace_urls, 'URL',
             constants.URL_REGEX if cfg.url_trailing_punct_outside else constants.URL_REGEX_LEGACY),
            (cfg.check_replace_emails, 'EMAIL', constants.EMAIL_REGEX),
            (cfg.check_replace_phone_numbers, 'PHONE', helper._phone_view),
            (cfg.check_replace_dates, 'DATE', helper._date_view),
            (cfg.check_replace_years, 'YEAR', helper._year_view(text)),
            (cfg.check_replace_iban_cards, 'IBAN', contact.IBAN_VIEW),
            (cfg.check_replace_iban_cards, 'CARD', contact.CARD_VIEW),
            (cfg.check_replace_identifiers, 'ID', constants.IDENTIFIER_REGEX),
            (cfg.check_replace_numbers, 'NUMBER', helper._numbers_regex),
            (cfg.check_replace_currency_symbols, 'CURRENCY', constants.CURRENCY_REGEX),
        )


class BuiltinNERDetector:
    """Expose an existing process-owned NER instance through the detector interface."""

    def __init__(self, ner: GeneralNER):
        self._ner = ner
        from sct.utils.ner_spans import ENTITY_TYPE_MAP
        labels = (*ner.cfg.positional_tags, *(ner.cfg.gliner_label_map or {}).values())
        self._capabilities = _Capabilities(
            'builtin_ner', entity_types=tuple(dict.fromkeys(ENTITY_TYPE_MAP.get(t, t) for t in labels)),
            supports_windowed=True)

    @property
    def capabilities(self):
        return self._capabilities

    def detect(self, text: str, *, language: str | None, context: DetectionContext) -> tuple[Detection, ...]:
        return self._ner.analyze(text, language=language, context=context)
