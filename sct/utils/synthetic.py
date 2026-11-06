"""Synthetic replacement engine — generates realistic fake values via Faker.

Thread-safe: uses threading.local() for both Faker instances and consistency
caches, so parallel batch processing in TextCleaner has no shared mutable state.
"""
import contextlib
import datetime as _datetime_module
import functools
import hashlib
import locale as _stdlib_locale
import logging
import os
import random
import re
import sys
import threading
import types
from datetime import date, datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)

# Some locale providers (th_TH, uk_UA ssn; uk_UA plates) draw from the module-level
# random, which seed_instance() leaves alone; seeded calls swap it for a seeded Random under this lock.
_GLOBAL_RANDOM_LOCK = threading.Lock()
_CLOCK_LOCK = threading.Lock()
# (module, name, original) of every global swap a lock holder has applied, if any.
_SWAPPED: list[tuple[Any, str, Any]] = []


def _after_fork_in_child():
    # A child forked while another thread held a lock must not inherit it held, nor keep the
    # held clock that thread would have restored (it does not exist in the child).
    global _GLOBAL_RANDOM_LOCK, _CLOCK_LOCK
    for mod, name, value in _SWAPPED:
        setattr(mod, name, value)
    _SWAPPED.clear()
    _GLOBAL_RANDOM_LOCK, _CLOCK_LOCK = threading.Lock(), threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork_in_child)


class _HeldClock(type):
    """isinstance against a held-clock class still accepts every real date/datetime."""
    def __instancecheck__(cls, obj):
        return isinstance(obj, cls.__mro__[1])


def _date_part(generate):
    return generate().date()


def _provider_modules(generate):
    """The modules whose clock and random to hold: those of the Faker call inside *generate*."""
    method = getattr(generate, 'func', generate)
    if method is _date_part:
        method = generate.args[0].func
    return {sys.modules.get(getattr(method, '__module__', ''))} - {None}


@contextlib.contextmanager
def _clock_at(reference, modules):
    """Faker code in *modules* reads *reference* as today and now.

    Native SSN/TAX_ID providers (zh_CN, fi_FI, sv_SE, pl_PL ...) derive the birthday inside
    the number from the clock, not from any argument.
    shortcut: swaps module globals process-wide under a lock, so a Faker call outside SCT on
    another thread meanwhile sees the reference date too; upgrade if Faker gains a clock hook.
    """
    class Date(date, metaclass=_HeldClock):
        @classmethod
        def today(cls):
            return reference.date()

    class DateTime(datetime, metaclass=_HeldClock):
        @classmethod
        def now(cls, tz=None):
            return reference.replace(tzinfo=tz)

        @classmethod
        def today(cls):
            return reference

    module = types.SimpleNamespace(**vars(_datetime_module))
    module.date, module.datetime = Date, DateTime
    held = {id(_datetime_module): module, id(datetime): DateTime, id(date): Date}
    with _swap_globals(_CLOCK_LOCK, modules, held):
        yield


@contextlib.contextmanager
def _seeded_random(seed, modules):
    """Faker code in *modules* draws from a Random seeded with *seed*, not the shared global one.

    The global random stays untouched, so another thread drawing from it meanwhile neither
    changes these values nor loses its own draws. shortcut: like _clock_at, a Faker call
    outside SCT on another thread meanwhile draws from this Random too.
    """
    rng = random.Random(seed)  # noqa: S311  # reproducible fakes, not crypto
    held = {id(random): rng}  # "import random" and "from random import randint" alike
    held.update((id(value), getattr(rng, name)) for name, value in vars(random).items()
                if isinstance(getattr(value, '__self__', None), random.Random))
    with _swap_globals(_GLOBAL_RANDOM_LOCK, modules, held):
        yield


@contextlib.contextmanager
def _swap_globals(lock, modules, held):
    """Under *lock*, swap each global of *modules* that *held* maps (by id) for its stand-in."""
    with lock:
        saved = [(mod, name, value) for mod in modules for name, value in vars(mod).items()
                 if id(value) in held]
        _SWAPPED.extend(saved)
        for mod, name, value in saved:
            setattr(mod, name, held[id(value)])
        try:
            yield
        finally:
            for patch in saved:
                setattr(patch[0], patch[1], patch[2])
                _SWAPPED.remove(patch)

_FAKER_INSTALL_HINT = (
    "faker is required for synthetic replacement mode. "
    "Install with: pip install squeakycleantext[synthetic]"
)
# A fake that echoes its input is drawn again this often, then the type's tag takes its place.
_ECHO_RETRIES = 8

# Maps entity tags to Faker method names; a tag not here is written as <TAG> (get_replacement).
FAKER_GENERATORS: dict[str, str] = {
    # Standard NER tags
    'PER': 'name', 'PERSON': 'name',
    'LOC': 'city', 'LOCATION': 'city',
    'ORG': 'company', 'ORGANISATION': 'company', 'ORGANIZATION': 'company',
    'NAME': 'name', 'BENEFICIARY': 'name',
    # PII contact
    'EMAIL': 'email', 'PHONE': 'phone_number', 'PHONE_NUMBER': 'phone_number',
    'EMAIL_ADDRESS': 'email',
    'URL': 'url', 'IP': 'ipv4', 'IP_ADDRESS': 'ipv4',
    # PII financial
    'CREDIT_CARD_NUMBER': 'credit_card_number', 'CREDIT_CARD': 'credit_card_number',
    'CVV': 'credit_card_security_code', 'EXPIRATION_DATE': 'credit_card_expire',
    'ROUTING_NUMBER': 'aba', 'ABA_ROUTING_NUMBER': 'aba',
    'ACCOUNT_NUMBER': 'bban', 'US_BANK_NUMBER': 'bban', 'CRYPTO': 'sha256',
    'SOCIAL_SECURITY_NUMBER': 'ssn',
    'BANK_ACCOUNT_NUMBER': 'bban', 'TAX_ID': 'ssn',
    # PII personal
    'DATE_OF_BIRTH': 'date_of_birth', 'DATE': 'date', 'DATE_TIME': 'date',
    'TIME': 'time', 'TIMESTAMP': 'iso8601', 'NUMBER': 'random_int',
    'FIRST_NAME': 'first_name', 'LAST_NAME': 'last_name',
    'AGE': 'random_int', 'GENDER': 'prefix',
    'NATIONALITY': 'country', 'NRP': 'country',
    # PII address
    'STREET_ADDRESS': 'street_address', 'CITY': 'city',
    # administrative_unit and postcode equal state and zipcode where a locale has those.
    'STATE': 'administrative_unit', 'COUNTRY': 'country', 'ZIP_CODE': 'postcode',
    'LATITUDE': 'latitude', 'LONGITUDE': 'longitude',
    # PII employment
    'JOB_TITLE': 'job', 'COMPANY_NAME': 'company',
    'EMPLOYEE_ID': 'bothify', 'SALARY': 'random_int',
    # PII identity
    'USERNAME': 'user_name', 'PASSWORD': 'password',
    'PASSPORT_NUMBER': 'passport_number', 'DRIVER_LICENSE_NUMBER': 'bothify',
    # PII digital
    'MAC_ADDRESS': 'mac_address', 'DEVICE_ID': 'uuid4', 'UUID': 'uuid4', 'IMEI': 'bothify',
    'API_KEY': 'sha256', 'ACCESS_TOKEN': 'sha256',
    # PII misc
    'LICENSE_PLATE': 'license_plate', 'VEHICLE_IDENTIFICATION_NUMBER': 'vin',
    'IBAN': 'iban', 'IBAN_CODE': 'iban', 'CARD': 'credit_card_number', 'ID': 'bothify',
    'SERIAL_NUMBER': 'bothify', 'MEDICAL_CODE': 'bothify', 'HEALTHCARE_NUMBER': 'bothify',
    'INSURANCE_POLICY_NUMBER': 'bothify',
    # PII relationship
    'SPOUSE_NAME': 'name', 'CHILD_NAME': 'first_name',
    'EMERGENCY_CONTACT': 'name',
    # Presidio's country recognizers
    **dict.fromkeys(('ZA_MOBILE_NUMBER', 'ZA_TELEPHONE_NUMBER'), 'phone_number'),
    **dict.fromkeys(('CA_POSTAL_CODE', 'DE_PLZ', 'UK_POSTCODE'), 'postcode'),
    **dict.fromkeys(('DE_KFZ', 'IN_VEHICLE_REGISTRATION', 'NG_VEHICLE_REGISTRATION', 'TR_LICENSE_PLATE',
                     'UK_VEHICLE_REGISTRATION', 'ZA_LICENSE_PLATE'), 'license_plate'),
    **dict.fromkeys(('DE_PASSPORT', 'ES_PASSPORT', 'IN_PASSPORT', 'IT_PASSPORT', 'KR_PASSPORT', 'PH_PASSPORT',
                     'UK_PASSPORT', 'US_PASSPORT', 'ZA_PASSPORT'), 'passport_number'),
    **dict.fromkeys(('AU_TFN', 'CA_SIN', 'DE_SOCIAL_SECURITY', 'DE_TAX_ID', 'DE_TAX_NUMBER', 'ES_NIE', 'ES_NIF',
                     'FI_PERSONAL_IDENTITY_CODE', 'IN_AADHAAR', 'IT_FISCAL_CODE', 'KR_FRN', 'KR_RRN', 'NG_NIN',
                     'PH_TIN', 'PL_PESEL', 'SE_PERSONNUMMER', 'SG_NRIC_FIN', 'TH_TNIN', 'TR_NATIONAL_ID', 'UK_NINO',
                     'US_ITIN', 'US_PROVIDER_TAX_ID', 'US_SSN', 'ZA_ID_NUMBER', 'ZA_INCOME_TAX_NUMBER'), 'ssn'),
    **dict.fromkeys(('AU_ABN', 'AU_ACN', 'AU_MEDICARE', 'DE_BSNR', 'DE_FUEHRERSCHEIN', 'DE_HANDELSREGISTER',
                     'DE_HEALTH_INSURANCE', 'DE_ID_CARD', 'DE_LANR', 'DE_VAT_ID', 'IN_GSTIN', 'IN_PAN', 'IN_VOTER',
                     'IT_DRIVER_LICENSE', 'IT_IDENTITY_CARD', 'IT_VAT_CODE', 'KR_BRN', 'KR_DRIVER_LICENSE',
                     'MEDICAL_LICENSE', 'PH_UMID', 'SE_ORGANISATIONSNUMMER', 'SG_UEN', 'UK_DRIVING_LICENCE', 'UK_NHS',
                     'US_CLAIM_NUMBER', 'US_DRIVER_LICENSE', 'US_HEALTH_INSURANCE_MEMBER_ID', 'US_MBI', 'US_NPI',
                     'US_PRESCRIPTION_NUMBER', 'US_PRIOR_AUTHORIZATION_NUMBER', 'US_REFERRAL_NUMBER',
                     'ZA_COMPANY_REGISTRATION', 'ZA_DRIVER_LICENSE', 'ZA_TRAFFIC_REGISTER_NUMBER',
                     'ZA_VAT_NUMBER'), 'bothify'),
}


@functools.cache
def faker_locale_for(language: str | None) -> str:
    """Faker locale for a Lingua language name (``'DUTCH'`` -> ``'nl_NL'``).

    The territory comes from the C library's locale alias table (``locale.normalize``),
    then Faker's bare language locale, then its first locale of that language; Faker's
    default (``en_US``) when Faker has none (Norwegian Bokmal/Nynorsk, Malay, Urdu...).
    """
    try:
        from faker.config import AVAILABLE_LOCALES, DEFAULT_LOCALE
    except ImportError as exc:
        raise ImportError(_FAKER_INSTALL_HINT) from exc

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

    seed: int | None = None
    language: str | None = None
    locale: str | None = None  # overrides the language's locale (low-confidence detection)


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
    ``DATE_TIME``, ``DATE_OF_BIRTH``) end on, so they do not change with the clock; None = today.
    """

    def __init__(self, seed: int = 42, per_document: bool = True, locale: str | None = None,
                 low_confidence_locale: str | None = None, collapse_period: bool = False,
                 reference_date: str | None = None):
        self._seed = seed
        self._per_document = per_document
        self._locale = locale
        self._low_confidence_locale = low_confidence_locale
        self._collapse_period = collapse_period
        self._reference = None if reference_date is None else datetime.combine(
            date.fromisoformat(reference_date), datetime.min.time())
        self._local = threading.local()

    def _get_faker(self, locale: str | None = None):
        """Get or create the thread-local Faker instance for *locale* (default: en_US)."""
        fakers = getattr(self._local, 'fakers', None)
        if fakers is None:
            fakers = self._local.fakers = {}
        if locale not in fakers:
            try:
                from faker import Faker  # noqa: S404
            except ImportError as exc:
                raise ImportError(_FAKER_INSTALL_HINT) from exc
            fake = Faker(locale)
            fake.seed_instance(self._seed)
            for provider in fake.providers:
                # shortcut: it_IT builds its city list from a set (hash-seed order); sort any
                # set-built list Faker adds later the same way.
                if isinstance(getattr(provider, 'cities', None), list):
                    provider.cities = sorted(provider.cities)
            fakers[locale] = fake
        return fakers[locale]

    def reset_cache(self, text: str | None = None, language: str | None = None,
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

    def swap_cache(self, cache: dict[tuple, str] | None = None) -> dict[tuple, str]:
        """Install *cache* (default: a new one) for this thread; return the previous one.

        Lets a document's cache move between threads and processing stages.
        """
        previous = self._get_cache()
        self._local.cache = _DocumentCache() if cache is None else cache
        return previous

    def _get_cache(self) -> dict[tuple, str]:
        """Get or create the thread-local consistency cache."""
        if not hasattr(self._local, 'cache'):
            self._local.cache = _DocumentCache()
        return self._local.cache

    def get_replacement(self, entity_group: str, original_text: str) -> str:
        """Generate a synthetic replacement for an entity.

        Maintains consistency: same (entity_group, original_text) pair maps to
        the same fake value within a single document. A type Faker has no generator for,
        in this locale or at all, is written as its tag (``<ANIMAL>``). A fake never holds
        *original_text* as a whole word.
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
        tag = f'<{entity_group}>'
        method_name = FAKER_GENERATORS.get(entity_group)
        if method_name is None or not hasattr(fake, method_name):
            cache[cache_key] = tag
            return tag
        generate = self._faker_call(fake, method_name)
        modules = _provider_modules(generate)
        if self._reference is not None:
            clock_modules = modules | ({sys.modules.get('faker.providers.date_time')} - {None})
            generate = functools.partial(self._at_reference, generate, clock_modules)
        echo = re.compile(rf'(?<!\w){re.escape(original_text)}(?!\w)', re.IGNORECASE)

        def draw():
            for _ in range(_ECHO_RETRIES):
                value = str(generate())
                if not echo.search(value):
                    return value
            return tag

        if doc_seed is None:
            replacement = draw()
        else:
            with _seeded_random(doc_seed + len(cache), modules):
                replacement = draw()

        cache[cache_key] = replacement
        return replacement

    def _faker_call(self, fake, method_name):
        """The Faker call for *method_name*; date fakes end on the reference date, if set."""
        generate = getattr(fake, method_name)
        if self._reference is not None and method_name == 'date':
            return functools.partial(fake.date, end_datetime=self._reference)
        if self._reference is not None and method_name == 'date_of_birth':
            # Faker's date_of_birth range (ages 0-115) ending on the reference date, not today; drawn in
            # UTC, since Faker reads a naive range in the worker's local timezone (fakes differ per machine).
            between = functools.partial(fake.date_time_between_dates,
                                        self._reference - timedelta(days=115 * 365 + 28),
                                        self._reference, _datetime_module.UTC)
            return functools.partial(_date_part, between)
        return generate

    def _at_reference(self, generate, modules):
        with _clock_at(self._reference, modules):
            return generate()

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
