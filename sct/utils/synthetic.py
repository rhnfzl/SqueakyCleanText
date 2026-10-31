"""Synthetic replacement engine — generates realistic fake values via Faker.

Thread-safe: uses threading.local() for both Faker instances and consistency
caches, so parallel batch processing in TextCleaner has no shared mutable state.
"""
import functools
import hashlib
import locale as _stdlib_locale
import logging
import os
import random
import threading
from datetime import date, datetime, timedelta
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Some locale providers (th_TH, uk_UA ssn; uk_UA plates) draw from the module-level
# random, which seed_instance() leaves alone; seeded calls hold this lock.
_GLOBAL_RANDOM_LOCK = threading.Lock()
if hasattr(os, "register_at_fork"):
    # A child forked while another thread held the lock must not inherit it held.
    os.register_at_fork(after_in_child=lambda: globals().update(_GLOBAL_RANDOM_LOCK=threading.Lock()))

# Maps entity tags to Faker method names
FAKER_GENERATORS: Dict[str, str] = {
    # Standard NER tags
    'PER': 'name', 'PERSON': 'name',
    'LOC': 'city', 'LOCATION': 'city',
    'ORG': 'company', 'ORGANISATION': 'company',
    # PII contact
    'EMAIL': 'email', 'PHONE': 'phone_number', 'PHONE_NUMBER': 'phone_number',
    'URL': 'url', 'IP': 'ipv4', 'IP_ADDRESS': 'ipv4',
    # PII financial
    'CREDIT_CARD_NUMBER': 'credit_card_number',
    'SOCIAL_SECURITY_NUMBER': 'ssn',
    'BANK_ACCOUNT_NUMBER': 'bban', 'TAX_ID': 'ssn',
    # PII personal
    'DATE_OF_BIRTH': 'date_of_birth', 'DATE': 'date',
    'FIRST_NAME': 'first_name', 'LAST_NAME': 'last_name',
    'AGE': 'random_int', 'GENDER': 'prefix',
    'NATIONALITY': 'country',
    # PII address
    'STREET_ADDRESS': 'street_address', 'CITY': 'city',
    'STATE': 'state', 'COUNTRY': 'country', 'ZIP_CODE': 'zipcode',
    # PII employment
    'JOB_TITLE': 'job', 'COMPANY_NAME': 'company',
    'EMPLOYEE_ID': 'bothify', 'SALARY': 'random_int',
    # PII identity
    'USERNAME': 'user_name', 'PASSWORD': 'password',
    'PASSPORT_NUMBER': 'bothify', 'DRIVER_LICENSE_NUMBER': 'bothify',
    # PII digital
    'MAC_ADDRESS': 'mac_address', 'DEVICE_ID': 'uuid4',
    'API_KEY': 'sha256', 'ACCESS_TOKEN': 'sha256',
    # PII misc
    'LICENSE_PLATE': 'license_plate',
    'SERIAL_NUMBER': 'bothify',
    # PII relationship
    'SPOUSE_NAME': 'name', 'CHILD_NAME': 'first_name',
    'EMERGENCY_CONTACT': 'name',
}


@functools.lru_cache(maxsize=None)
def faker_locale_for(language: Optional[str]) -> str:
    """Faker locale for a Lingua language name (``'DUTCH'`` -> ``'nl_NL'``).

    The territory comes from the C library's locale alias table (``locale.normalize``),
    then Faker's bare language locale, then its first locale of that language; Faker's
    default (``en_US``) when Faker has none (Norwegian Bokmal/Nynorsk, Malay, Urdu...).
    """
    from faker.config import AVAILABLE_LOCALES, DEFAULT_LOCALE

    if not language:
        return DEFAULT_LOCALE
    from sct.utils.resources import language_to_iso

    iso = language_to_iso(language)
    normalized = _stdlib_locale.normalize(iso).split('.')[0].split('@')[0]
    if normalized in AVAILABLE_LOCALES:
        return normalized
    if iso in AVAILABLE_LOCALES:
        return iso
    same = sorted(loc for loc in AVAILABLE_LOCALES if loc.split('_')[0] == iso)
    return same[0] if same else DEFAULT_LOCALE


class _DocumentCache(dict):
    """Per-document ``(entity_group, original) -> fake`` map.

    ``seed`` is set in per-document mode; ``language`` is the detected document language.
    """

    seed: Optional[int] = None
    language: Optional[str] = None
    locale: Optional[str] = None  # overrides the language's locale (low-confidence detection)


class SyntheticReplacer:
    """Generate synthetic replacements for detected entities using Faker.

    Thread-safe: each thread gets its own Faker instance and consistency cache
    via ``threading.local()``. Call ``reset_cache(text)`` at the start of each
    document to ensure consistency within (but not across) documents.

    ``per_document=True`` derives each fake from *seed*, the document text and
    the fake's ordinal in the document, so fakes differ across documents yet the
    same document always gets the same fakes, whatever the thread, worker or
    batch order. ``False`` is the 0.7 behaviour: one Faker stream per thread.

    ``locale=None`` picks the Faker locale from the document language passed to
    ``reset_cache``; a string forces that locale (``'en_US'`` = 0.7).
    ``low_confidence_locale`` replaces the language's locale for documents whose
    detection is low confidence (None = follow the language anyway).
    ``collapse_period`` drops a fake's final ``.`` when the text already has one
    right after the entity (``B.Eng..`` -> ``B.Eng.``).
    ``reference_date`` (ISO date) is the "today" that date fakes (``DATE``,
    ``DATE_OF_BIRTH``) end on, so they do not change with the clock; None = today.
    """

    def __init__(self, seed: int = 42, per_document: bool = True, locale: Optional[str] = None,
                 low_confidence_locale: Optional[str] = None, collapse_period: bool = False,
                 reference_date: Optional[str] = None):
        self._seed = seed
        self._per_document = per_document
        self._locale = locale
        self._low_confidence_locale = low_confidence_locale
        self._collapse_period = collapse_period
        self._reference = None if reference_date is None else datetime.combine(
            date.fromisoformat(reference_date), datetime.min.time())
        self._local = threading.local()

    def _get_faker(self, locale: Optional[str] = None):
        """Get or create the thread-local Faker instance for *locale* (default: en_US)."""
        fakers = getattr(self._local, 'fakers', None)
        if fakers is None:
            fakers = self._local.fakers = {}
        if locale not in fakers:
            try:
                from faker import Faker  # noqa: S404
            except ImportError:
                raise ImportError(
                    "faker is required for synthetic replacement mode. "
                    "Install with: pip install squeakycleantext[synthetic]"
                )
            fake = Faker(locale)
            fake.seed_instance(self._seed)
            for provider in fake.providers:
                # shortcut: it_IT builds its city list from a set (hash-seed order); sort any
                # set-built list Faker adds later the same way.
                if isinstance(getattr(provider, 'cities', None), list):
                    provider.cities = sorted(provider.cities)
            fakers[locale] = fake
        return fakers[locale]

    def reset_cache(self, text: Optional[str] = None, language: Optional[str] = None,
                    low_confidence: bool = False) -> None:
        """Start a document: fresh consistency cache, fakes seeded from *text*.

        *language* is the document's Lingua language name; it picks the Faker locale
        unless the replacer was given a fixed one, or *low_confidence* flags the
        detection and a ``low_confidence_locale`` is set.
        """
        self._local.cache = _DocumentCache()
        self._local.cache.language = language
        if low_confidence:
            self._local.cache.locale = self._low_confidence_locale
        if self._per_document and text is not None:
            digest = hashlib.blake2b(
                text.encode('utf-8', 'surrogatepass'), digest_size=8,
                key=str(self._seed).encode(),
            ).digest()
            self._local.cache.seed = int.from_bytes(digest, 'big')

    def swap_cache(self, cache: Optional[Dict[tuple, str]] = None) -> Dict[tuple, str]:
        """Install *cache* (default: a new one) for this thread; return the previous one.

        Lets a document's cache move between threads and processing stages.
        """
        previous = self._get_cache()
        self._local.cache = _DocumentCache() if cache is None else cache
        return previous

    def _get_cache(self) -> Dict[tuple, str]:
        """Get or create the thread-local consistency cache."""
        if not hasattr(self._local, 'cache'):
            self._local.cache = _DocumentCache()
        return self._local.cache

    def get_replacement(self, entity_group: str, original_text: str) -> str:
        """Generate a synthetic replacement for an entity.

        Maintains consistency: same (entity_group, original_text) pair maps to
        the same fake value within a single document.
        """
        cache = self._get_cache()
        cache_key = (entity_group, original_text)
        if cache_key in cache:
            return cache[cache_key]

        fake = self._get_faker(self._locale or getattr(cache, 'locale', None)
                               or faker_locale_for(getattr(cache, 'language', None)))
        doc_seed = getattr(cache, 'seed', None)
        if doc_seed is not None:
            # The ordinal travels with the cache, so a document's fakes do not
            # depend on which thread or stage generates them.
            fake.seed_instance(doc_seed + len(cache))
        method_name = FAKER_GENERATORS.get(entity_group)

        generate = getattr(fake, method_name) if method_name and hasattr(fake, method_name) else fake.name
        if self._reference is not None and method_name == 'date':
            generate = functools.partial(fake.date, end_datetime=self._reference)
        elif self._reference is not None and method_name == 'date_of_birth':
            # Faker's date_of_birth range (ages 0-115) ending on the reference date, not today.
            generate = functools.partial(fake.date_between_dates,
                                         self._reference - timedelta(days=115 * 365 + 28),
                                         self._reference)
        if doc_seed is None:
            replacement = str(generate())
        else:
            with _GLOBAL_RANDOM_LOCK:
                state = random.getstate()
                random.seed(doc_seed + len(cache))
                try:
                    replacement = str(generate())
                finally:
                    random.setstate(state)

        cache[cache_key] = replacement
        return replacement

    def generate_for_entities(
        self, text: str, filtered_data: list,
    ) -> str:
        """Replace entities in text with Faker-generated synthetic values.

        Processes entities right-to-left to preserve character offsets.
        """
        if not filtered_data:
            return text

        sorted_data = sorted(filtered_data, key=lambda x: x['start'], reverse=True)
        for item in sorted_data:
            replacement = self.get_replacement(item['entity_group'], item['word'])
            if self._collapse_period and replacement.endswith('.') and text.startswith('.', item['end']):
                replacement = replacement[:-1]
            text = text[:item['start']] + replacement + text[item['end']:]
        return text
