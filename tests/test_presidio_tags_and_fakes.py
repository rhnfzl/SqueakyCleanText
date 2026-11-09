"""presidio_gliner keeps the pipeline's tags; each type gets a fake of its kind; a missing Faker gives the install hint."""

import re
import sys
import types

import faker
import pytest
from presidio_anonymizer.entities import RecognizerResult

from sct import TextCleaner, TextCleanerConfig
from sct.config import PII_LABEL_MAP, PII_LABELS
from sct.utils.ner import _presidio_type_map
from sct.utils.ner_spans import ENTITY_TYPE_MAP
from sct.utils.synthetic import FAKER_GENERATORS, SyntheticReplacer, faker_locale_for

from tests.helpers import bare_ner, set_cfg

_INSTALL_HINT = r"squeakycleantext\[synthetic\]"

# Every entity of presidio-analyzer 2.2.366's predefined recognizers, plus what its spaCy
# recognizer maps NER labels to, that has a fake of its own kind.
_PRESIDIO_TYPES = {
    "name": {"PERSON"},
    "city": {"LOCATION"},
    "company": {"ORGANIZATION", "ORGANISATION"},
    "date": {"DATE_TIME"},
    "email": {"EMAIL_ADDRESS"},
    "url": {"URL"},
    "ipv4": {"IP_ADDRESS"},
    "mac_address": {"MAC_ADDRESS"},
    "uuid4": {"UUID"},
    "sha256": {"CRYPTO"},
    "country": {"NRP"},
    "credit_card_number": {"CREDIT_CARD"},
    "iban": {"IBAN_CODE"},
    "bban": {"US_BANK_NUMBER"},
    "aba": {"ABA_ROUTING_NUMBER"},
    "phone_number": {"PHONE_NUMBER", "ZA_MOBILE_NUMBER", "ZA_TELEPHONE_NUMBER"},
    "postcode": {"UK_POSTCODE", "DE_PLZ", "CA_POSTAL_CODE"},
    "license_plate": {"DE_KFZ", "IN_VEHICLE_REGISTRATION", "NG_VEHICLE_REGISTRATION", "TR_LICENSE_PLATE",
                      "UK_VEHICLE_REGISTRATION", "ZA_LICENSE_PLATE"},
    "passport_number": {"DE_PASSPORT", "ES_PASSPORT", "IN_PASSPORT", "IT_PASSPORT", "KR_PASSPORT", "PH_PASSPORT",
                        "UK_PASSPORT", "US_PASSPORT", "ZA_PASSPORT"},
    "ssn": {"AU_TFN", "CA_SIN", "DE_SOCIAL_SECURITY", "DE_TAX_ID", "DE_TAX_NUMBER", "ES_NIE", "ES_NIF",
            "FI_PERSONAL_IDENTITY_CODE", "IN_AADHAAR", "IT_FISCAL_CODE", "KR_FRN", "KR_RRN", "NG_NIN", "PH_TIN",
            "PL_PESEL", "SE_PERSONNUMMER", "SG_NRIC_FIN", "TH_TNIN", "TR_NATIONAL_ID", "UK_NINO", "US_ITIN",
            "US_PROVIDER_TAX_ID", "US_SSN", "ZA_ID_NUMBER", "ZA_INCOME_TAX_NUMBER"},
    "bothify": {"AU_ABN", "AU_ACN", "AU_MEDICARE", "DE_BSNR", "DE_FUEHRERSCHEIN", "DE_HANDELSREGISTER",
                "DE_HEALTH_INSURANCE", "DE_ID_CARD", "DE_LANR", "DE_VAT_ID", "IN_GSTIN", "IN_PAN", "IN_VOTER",
                "IT_DRIVER_LICENSE", "IT_IDENTITY_CARD", "IT_VAT_CODE", "KR_BRN", "KR_DRIVER_LICENSE",
                "MEDICAL_LICENSE", "PH_UMID", "SE_ORGANISATIONSNUMMER", "SG_UEN",
                "UK_DRIVING_LICENCE", "UK_NHS", "US_CLAIM_NUMBER", "US_DRIVER_LICENSE",
                "US_HEALTH_INSURANCE_MEMBER_ID", "US_MBI", "US_NPI", "US_PRESCRIPTION_NUMBER",
                "US_PRIOR_AUTHORIZATION_NUMBER", "US_REFERRAL_NUMBER", "ZA_COMPANY_REGISTRATION",
                "ZA_DRIVER_LICENSE", "ZA_TRAFFIC_REGISTER_NUMBER", "ZA_VAT_NUMBER"},
}
_TYPE_METHODS = sorted((kind, method) for method, kinds in _PRESIDIO_TYPES.items() for kind in kinds)
# Labels Faker has nothing of their kind for.
_NO_GENERATOR = {"MARITAL_STATUS", "MEDICAL_CONDITION", "DRUG_NAME", "DOSAGE", "BLOOD_TYPE", "GPS_COORDINATES", "MISC"}


# --- Fakes of the right kind ---
@pytest.mark.parametrize("entity_type, method", _TYPE_METHODS)
def test_presidio_type_gets_a_fake_of_its_kind(entity_type, method):
    assert FAKER_GENERATORS[entity_type] == method
    assert SyntheticReplacer(seed=42, locale="en_US").get_replacement(entity_type, "value")


@pytest.fixture
def no_person_name(monkeypatch):
    monkeypatch.setattr(faker.Faker, "name", lambda self: pytest.fail("a person name for a non-person type"),
                        raising=False)


@pytest.mark.parametrize("entity_type, original, shape", [
    ("ORGANIZATION", "Siemens", None),
    ("DATE_TIME", "Tuesday", r"\d{4}-\d{2}-\d{2}"),
    ("CREDIT_CARD", "4111 1111 1111 1111", r"\d{12,19}"),
    ("EMAIL_ADDRESS", "a@b.com", r".+@.+"),
    ("IP_ADDRESS", "8.8.8.8", r"(\d+\.){3}\d+"),
    ("IBAN_CODE", "NL91ABNA0417164300", r"[A-Z]{2}\d{2}\w+"),
    ("US_SSN", "123-45-6789", r"\d{3}-\d{2}-\d{4}"),
    ("US_PASSPORT", "912803456", r"\w+"),
    ("US_BANK_NUMBER", "12345678", r"\w+"),
])
def test_fake_reads_as_its_type(no_person_name, entity_type, original, shape):
    replacer = SyntheticReplacer(seed=42, locale="en_US")
    value = replacer.get_replacement(entity_type, original)
    assert value != original
    assert shape is None or re.fullmatch(shape, value), value


@pytest.mark.parametrize("label", [*sorted(PII_LABELS)])
def test_every_pii_label_has_a_fake_or_its_tag(label):
    tag = PII_LABEL_MAP.get(label, label.upper())
    tag = ENTITY_TYPE_MAP.get(tag, tag)
    value = SyntheticReplacer(seed=42, locale="en_US").get_replacement(tag, "some value")
    if tag in _NO_GENERATOR:
        assert value == f"<{tag}>"
    else:
        assert tag in FAKER_GENERATORS and value != f"<{tag}>"


def test_type_without_generator_gets_its_tag_consistently():
    replacer = SyntheticReplacer(seed=42)
    replacer.reset_cache("doc")
    first = replacer.get_replacement("ANIMAL", "Rex")
    assert first == "<ANIMAL>"
    assert replacer.get_replacement("ANIMAL", "Rex") == first
    assert replacer.get_replacement("ANIMAL", "Fido") == "<ANIMAL>"


@pytest.mark.parametrize("entity_type", ["STATE", "ZIP_CODE"])
def test_generator_covers_a_locale_without_the_us_method(no_person_name, entity_type):
    """nl_NL has no state() or zipcode(); these fell back to a person name."""
    assert not SyntheticReplacer(seed=42, locale="nl_NL").get_replacement(entity_type, "some value").startswith("<")


def test_method_missing_from_the_locale_gets_the_tag():
    """en_PH has no phone_number(): the type's tag, never a person name."""
    assert SyntheticReplacer(seed=42, locale="en_PH").get_replacement("PHONE_NUMBER", "0612345678") == "<PHONE_NUMBER>"


def test_new_types_keep_document_determinism():
    def fakes(text):
        replacer = SyntheticReplacer(seed=42, locale="en_US", reference_date="2026-01-01")
        replacer.reset_cache(text)
        return [replacer.get_replacement(t, v) for t, v in
                (("ANIMAL", "Rex"), ("ORGANIZATION", "Siemens"), ("DATE_TIME", "Tuesday"), ("PERSON", "John"))]

    assert fakes("doc one") == fakes("doc one")
    assert fakes("doc one") != fakes("doc two")


def test_tag_takes_an_ordinal_like_any_fake():
    """The fakes after a type without a generator do not move."""
    def person_after(entity_type):
        replacer = SyntheticReplacer(seed=42, locale="en_US")
        replacer.reset_cache("doc")
        replacer.get_replacement(entity_type, "x")
        return replacer.get_replacement("PERSON", "John")

    assert person_after("ANIMAL") == person_after("EMAIL")


class _StubFaker:
    def __init__(self, *values):
        self.values = iter(values)

    def seed_instance(self, seed):
        pass

    def company(self):
        return next(self.values)


@pytest.mark.parametrize("original, values, expected", [
    ("Siemens", ("Siemens", "Acme"), "Acme"),
    ("siemens", ("SIEMENS", "Acme"), "Acme"),
    ("John", ("John Smith and Sons", "Acme"), "Acme"),  # the input inside the fake is an echo too
    ("Ann", ("Annette Ltd", ), "Annette Ltd"),  # part of a longer word is not
    ("Siemens", ("Siemens",) * 100, "<ORGANIZATION>"),  # nothing but echoes: the tag
])
def test_fake_never_echoes_the_input(original, values, expected):
    replacer = SyntheticReplacer(per_document=False)
    stub = _StubFaker(*values)
    replacer._get_faker = lambda locale: stub
    assert replacer.get_replacement("ORGANIZATION", original) == expected


# --- presidio_gliner and the pipeline's tags ---
class _Analyzer:
    """Reports each regex's matches as its entity type, like a detector reading the tag's word."""

    def __init__(self, *rules):
        self.rules = rules
        self.seen = []

    def analyze(self, *, text, language):
        self.seen.append(text)
        return [RecognizerResult(kind, m.start(), m.end(), 0.9)
                for pattern, kind in self.rules for m in re.finditer(pattern, text)]


_TAG_WORDS = ((r"(?<=<)EMAIL(?:_0)?(?=>)", "EMAIL_ADDRESS"), (r"(?<=<)PHONE(?:_0)?(?=>)", "LOCATION"),
              (r"(?<=<)CARD(?:_0)?(?=>)", "CREDIT_CARD"), (r"John", "PERSON"))


def _ner(*rules, **fields):
    ner = bare_ner()
    set_cfg(ner, ner_backend="presidio_gliner", **fields)
    ner._analyzer = _Analyzer(*rules)
    ner._presidio_types = _presidio_type_map(False, None)
    return ner


def _cleaner(monkeypatch, *rules, **fields):
    class Engine:
        registry = types.SimpleNamespace(add_recognizer=lambda r: None)

        def analyze(self, *, text, language):
            return analyzer.analyze(text=text, language=language)

    analyzer = _Analyzer(*rules)
    monkeypatch.setitem(sys.modules, "presidio_analyzer", types.SimpleNamespace(AnalyzerEngine=Engine))
    monkeypatch.setitem(sys.modules, "presidio_analyzer.predefined_recognizers",
                        types.SimpleNamespace(GLiNERRecognizer=lambda **kwargs: None))
    fields.setdefault("gliner_model", "fake/gliner")
    return TextCleaner(TextCleanerConfig(ner_backend="presidio_gliner", **fields)), analyzer


_TEXT = "John: mail john.doe@example.com or call +31 6 12345678"


def test_placeholder_mode_keeps_tags(monkeypatch):
    cleaner, _ = _cleaner(monkeypatch, *_TAG_WORDS)
    result = cleaner.process(_TEXT)
    assert "<<" not in result.lm_text
    assert "<EMAIL>" in result.lm_text and "<PHONE>" in result.lm_text
    assert "<PERSON>" in result.lm_text and "John" not in result.lm_text


@pytest.mark.parametrize("sees_tags", [True, False])
def test_reversible_tags_stay_as_written(monkeypatch, sees_tags):
    cleaner, _ = _cleaner(monkeypatch, *_TAG_WORDS, replacement_mode="reversible", ner_sees_tags=sees_tags)
    result = cleaner.process(_TEXT)
    assert "<EMAIL_0>" in result.lm_text and "<PHONE_0>" in result.lm_text
    assert "<PERSON_0>" in result.lm_text and "<<" not in result.lm_text
    anon_map = result.metadata["anon_map"]
    assert {e.placeholder for e in anon_map.entries} == {"<EMAIL_0>", "<PHONE_0>", "<PERSON_0>"}
    assert anon_map.deanonymize(result.lm_text).startswith("John: mail john.doe@example.com")


def test_synthetic_mode_writes_the_regex_fake_not_a_fake_in_the_tag(monkeypatch):
    cleaner, analyzer = _cleaner(monkeypatch, *_TAG_WORDS, replacement_mode="synthetic")
    text = cleaner.process(_TEXT).lm_text
    assert "<PHONE>" in analyzer.seen[0]  # NER read the tags, the fakes went in after
    assert "<" not in text and "John" not in text and "john.doe" not in text


def test_result_inside_a_tag_is_dropped_and_not_a_finding():
    ner = _ner(*_TAG_WORDS)
    result = ner.ner_process_detailed("Mail <EMAIL> or <PHONE> or <CARD> to John")
    assert result.text == "Mail <EMAIL> or <PHONE> or <CARD> to <PERSON>"
    assert [f.entity.entity_type for f in result.findings] == ["PERSON"]


def test_indexed_tag_is_kept():
    result = _ner(*_TAG_WORDS).ner_process_detailed("Mail <EMAIL_0> now")
    assert result.text == "Mail <EMAIL_0> now" and result.findings == ()


@pytest.mark.parametrize("mode", ["placeholder", "reversible", "synthetic"])
def test_result_cutting_a_tag_still_masks_the_text_outside_it(mode):
    ner = _ner((r"John <EMAIL", "PERSON"), replacement_mode=mode)
    if mode == "synthetic":
        ner._synthetic_replacer = SyntheticReplacer(locale="en_US")
    result = ner.ner_process_detailed("Ask John <EMAIL> now")
    assert "John" not in result.text
    assert "EMAIL>" not in result.text and "<EMAIL" not in result.text  # no fragment beside the replacement
    assert result.findings[0].entity.text == "John <EMAIL>"


def test_result_containing_a_tag_masks_the_whole_span():
    result = _ner((r"John <EMAIL> Doe", "PERSON")).ner_process_detailed("Ask John <EMAIL> Doe now")
    assert result.text == "Ask <PERSON> now"


def test_results_outside_tags_are_kept():
    result = _ner((r"Anna", "PERSON"), (r"Paris", "LOCATION")).ner_process_detailed("<EMAIL> Anna <PHONE> Paris")
    assert result.text == "<EMAIL> <PERSON> <PHONE> <LOCATION>"


def test_literal_angle_text_is_still_masked():
    """<John> is not tag-shaped, so nothing the pipeline wrote protects its contents."""
    result = _ner((r"John", "PERSON")).ner_process_detailed("Hi <John> there")
    assert "John" not in result.text and len(result.findings) == 1


def test_tag_shaped_source_text_is_still_masked(monkeypatch):
    cleaner, _ = _cleaner(monkeypatch, (r"LONDON", "LOCATION"), check_replace_html=False)
    text = cleaner.process("Meet at <LONDON> and mail a@b.com").lm_text
    assert "LONDON" not in text and "<EMAIL>" in text


def test_batch_path_passes_source_markers():
    ner = _ner((r"LONDON", "LOCATION"), (r"(?<=<)EMAIL(?=>)", "EMAIL_ADDRESS"))
    source = ner.ner_process_batch_detailed(["<LONDON> <EMAIL>"], ["ENGLISH"], source_markers=[frozenset({"<LONDON>"})])
    assert "LONDON" not in source[0].text and "<EMAIL>" in source[0].text


def test_batch_synthetic_caches_reach_presidio_replacement():
    ner = _ner((r"Alice", "PERSON"), replacement_mode="synthetic")
    ner._synthetic_replacer = SyntheticReplacer(locale="en_US")
    results = ner.ner_process_batch_detailed(
        ["Alice called Alice."] * 2, ["ENGLISH"] * 2,
        synthetic_caches=[{("PERSON", "Alice"): name} for name in ("Beatrice", "Catherine")],
    )
    assert [r.text for r in results] == ["Beatrice called Beatrice.", "Catherine called Catherine."]


def test_process_batch_keeps_tags(monkeypatch):
    cleaner, _ = _cleaner(monkeypatch, *_TAG_WORDS)
    assert "<<" not in cleaner.process_batch([_TEXT, _TEXT], max_workers=1)[0].lm_text


def test_custom_regex_marker_is_kept(monkeypatch):
    cleaner, _ = _cleaner(monkeypatch, (r"MAILTAG", "ORGANIZATION"), replace_with_email="MAILTAG")
    assert "MAILTAG" in cleaner.process("mail john.doe@example.com now").lm_text


def test_policy_sees_only_results_outside_tags():
    from sct.privacy import EntityRule, PrivacyPolicy

    from tests.helpers import set_policy

    ner = _ner(*_TAG_WORDS)
    set_policy(ner, PrivacyPolicy(name="p", version="1", rules={"PERSON": EntityRule(action="redact")},
                                  default_rule=EntityRule(action="redact")))
    result = ner.ner_process_detailed("Mail <EMAIL> to John")
    assert "<EMAIL>" in result.text and "John" not in result.text
    assert [f.entity.entity_type for f in result.findings] == ["PERSON"]


def test_fake_written_by_a_regex_step_is_not_protected_from_presidio():
    """Fakes are not tags: real text equal to a fake must not hide from Presidio."""
    ner = _ner((r"Paris", "LOCATION"), replacement_mode="synthetic")
    ner._synthetic_replacer = SyntheticReplacer(locale="en_US")
    ner._synthetic_replacer._get_cache()[("CITY", "Rome")] = "Paris"
    assert "Paris" not in ner.ner_process_detailed("She lives in Paris").text


# --- Missing Faker ---
@pytest.fixture
def no_faker(monkeypatch):
    for name in [m for m in sys.modules if m == "faker" or m.startswith("faker.")] + ["faker", "faker.config"]:
        monkeypatch.setitem(sys.modules, name, None)
    faker_locale_for.cache_clear()
    yield
    faker_locale_for.cache_clear()


@pytest.mark.parametrize("locale", [None, "en_US"])
def test_missing_faker_regex_path(no_faker, locale):
    cleaner = TextCleaner(TextCleanerConfig(replacement_mode="synthetic", check_ner_process=False,
                                            synthetic_locale=locale))
    with pytest.raises(ImportError, match=_INSTALL_HINT):
        cleaner.process("mail john.doe@example.com")


@pytest.mark.parametrize("locale", [None, "en_US"])
def test_missing_faker_ner_path(no_faker, locale):
    ner = _ner((r"John", "PERSON"), replacement_mode="synthetic")
    ner._synthetic_replacer = SyntheticReplacer(locale=locale)
    with pytest.raises(ImportError, match=_INSTALL_HINT):
        ner.ner_process_detailed("Hi John")


def test_missing_faker_type_without_generator(no_faker):
    with pytest.raises(ImportError, match=_INSTALL_HINT):
        SyntheticReplacer().get_replacement("ANIMAL", "Rex")


def test_missing_faker_locale_lookup(no_faker):
    with pytest.raises(ImportError, match=_INSTALL_HINT):
        faker_locale_for("DUTCH")


@pytest.mark.parametrize("mode", ["placeholder", "reversible"])
def test_other_modes_need_no_faker(no_faker, mode):
    cleaner = TextCleaner(TextCleanerConfig(replacement_mode=mode, check_ner_process=False))
    assert "john.doe" not in cleaner.process("mail john.doe@example.com").lm_text
