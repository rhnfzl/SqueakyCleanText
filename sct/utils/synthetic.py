"""Synthetic replacement engine — generates realistic fake values via Faker.

Thread-safe: uses threading.local() for both Faker instances and consistency
caches, so parallel batch processing in TextCleaner has no shared mutable state.
"""
import hashlib
import logging
import threading
from typing import Dict, Optional

logger = logging.getLogger(__name__)

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


class _DocumentCache(dict):
    """Per-document ``(entity_group, original) -> fake`` map; ``seed`` set in per-document mode."""

    seed: Optional[int] = None


class SyntheticReplacer:
    """Generate synthetic replacements for detected entities using Faker.

    Thread-safe: each thread gets its own Faker instance and consistency cache
    via ``threading.local()``. Call ``reset_cache(text)`` at the start of each
    document to ensure consistency within (but not across) documents.

    ``per_document=True`` derives each fake from *seed*, the document text and
    the fake's ordinal in the document, so fakes differ across documents yet the
    same document always gets the same fakes, whatever the thread, worker or
    batch order. ``False`` is the 0.7 behaviour: one Faker stream per thread.
    """

    def __init__(self, seed: int = 42, per_document: bool = True):
        self._seed = seed
        self._per_document = per_document
        self._local = threading.local()

    def _get_faker(self):
        """Get or create the thread-local Faker instance."""
        if not hasattr(self._local, 'faker'):
            try:
                from faker import Faker  # noqa: S404
            except ImportError:
                raise ImportError(
                    "faker is required for synthetic replacement mode. "
                    "Install with: pip install squeakycleantext[synthetic]"
                )
            self._local.faker = Faker()
            self._local.faker.seed_instance(self._seed)
        return self._local.faker

    def reset_cache(self, text: Optional[str] = None) -> None:
        """Start a document: fresh consistency cache, fakes seeded from *text*."""
        self._local.cache = _DocumentCache()
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

        fake = self._get_faker()
        doc_seed = getattr(cache, 'seed', None)
        if doc_seed is not None:
            # The ordinal travels with the cache, so a document's fakes do not
            # depend on which thread or stage generates them.
            fake.seed_instance(doc_seed + len(cache))
        method_name = FAKER_GENERATORS.get(entity_group)

        if method_name and hasattr(fake, method_name):
            replacement = str(getattr(fake, method_name)())
        else:
            # Fallback: generate a name-like replacement for unknown types
            replacement = fake.name()

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
            text = text[:item['start']] + replacement + text[item['end']:]
        return text
