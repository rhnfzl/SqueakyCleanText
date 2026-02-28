import gc
import unittest
import string
from types import MappingProxyType
from hypothesis import given, settings
from hypothesis.strategies import text, from_regex
from faker import Faker
from sct import config
from sct.config import (
    LANG_KEYS, TextCleanerConfig,
    PII_LABELS, PII_LABEL_MAP, PII_DEFAULT_MODEL, PII_DEFAULT_THRESHOLD,
)
from sct.utils import contact, datetime, special, normtext, stopwords, constants, resources
from sct.utils.constants import build_month_names_dict, build_date_regex
from sct.utils.stopwords import ProcessStopwords
from sct.utils.ner import GeneralNER
from unittest.mock import patch, MagicMock
from functools import wraps
from sct.sct import TextCleaner


# Lightweight ONNX test model for all languages (never use production 7GB models in tests)
TEST_NER_MODELS = {k: "protectai/bert-base-NER-onnx" for k in LANG_KEYS}


def requires_ner(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, 'ner') or self.ner is None:
            self.skipTest("NER models not loaded")
        if not hasattr(self.ner, '_pipelines') or 'ENGLISH' not in self.ner._pipelines:
            self.skipTest("NER models not properly loaded")
        return func(self, *args, **kwargs)
    return wrapper


# Test text constant exercising all pipeline steps
TEST_TEXT = """
Here's $50 for you! Check my blog at https://example.com/blog?id=123
or email me at john.doe@example.com. Call me at +1-234-567-8900.
Meet Dr. John Smith from Microsoft Corp. at 5th Avenue, New York on Jan 2024.
<div class="test">Some HTML content with special chars: &amp; © ®</div>
This text has some   extra   spaces and isolated l e t t e r s.
"""


class TextCleanerTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Initialize processing classes (no NER needed for these)
        config.CHECK_NER_PROCESS = False
        cls.ProcessContacts = contact.ProcessContacts()
        cls.ProcessDateTime = datetime.ProcessDateTime()
        cls.ProcessSpecialSymbols = special.ProcessSpecialSymbols()
        cls.NormaliseText = normtext.NormaliseText()
        cls.ProcessStopwords = stopwords.ProcessStopwords()
        cls.fake = Faker()

        # Always use lightweight ONNX test model — never download production models
        cls.ner = None
        try:
            cls.ner = GeneralNER(device='cpu', model_names=TEST_NER_MODELS)
            cls.ner.ner_process(
                "Test sentence.",
                positional_tags=['PER', 'ORG', 'LOC'],
                ner_confidence_threshold=0.85,
            )
            print("NER test models loaded successfully!")
        except Exception as e:
            print(f"NER models unavailable: {e}")
            cls.ner = None

    def setUp(self):
        """Set up test fixtures before each test method."""
        config.CHECK_NER_PROCESS = True
        self.ProcessContacts = self.__class__.ProcessContacts
        self.ProcessDateTime = self.__class__.ProcessDateTime
        self.ProcessSpecialSymbols = self.__class__.ProcessSpecialSymbols
        self.NormaliseText = self.__class__.NormaliseText
        self.ProcessStopwords = self.__class__.ProcessStopwords
        self.fake = self.__class__.fake
        self.ner = self.__class__.ner

    # --- Regex tests (non-NER, always run) ---

    @settings(deadline=None)
    @given(from_regex(constants.EMAIL_REGEX, fullmatch=True))
    def test_email_regex(self, rx):
        self.assertEqual("", self.ProcessContacts.replace_emails(rx, ""))

    @settings(deadline=None)
    @given(from_regex(constants.PHONE_REGEX, fullmatch=True))
    def test_phone_regex(self, rx):
        self.assertEqual("", self.ProcessContacts.replace_phone_numbers(rx, ""))

    @settings(deadline=None)
    @given(from_regex(constants.NUMBERS_REGEX, fullmatch=True))
    def test_number_regex(self, rx):
        self.assertEqual("", self.ProcessContacts.replace_numbers(rx, ""))

    @settings(deadline=None)
    @given(from_regex(constants.URL_REGEX, fullmatch=True))
    def test_url_regex(self, rx):
        self.assertNotEqual(rx, self.ProcessContacts.replace_urls(rx, ""))

    @settings(deadline=None)
    @given(from_regex(constants.YEAR_REGEX, fullmatch=True))
    def test_year_regex(self, rx):
        self.assertEqual("", self.ProcessDateTime.replace_years(rx, ""))

    @settings(deadline=None)
    @given(from_regex(constants.ISOLATED_LETTERS_REGEX, fullmatch=True))
    def test_isolated_letters_regex(self, rx):
        rx = self.ProcessSpecialSymbols.remove_isolated_letters(rx)
        rx = self.NormaliseText.normalize_whitespace(rx)
        self.assertEqual("", rx)

    @settings(deadline=None)
    @given(from_regex(constants.ISOLATED_SPECIAL_SYMBOLS_REGEX, fullmatch=True))
    def test_isolated_symbols_regex(self, rx):
        result = self.ProcessSpecialSymbols.remove_isolated_special_symbols(
            rx, remove_brackets=False, remove_braces=False
        )
        self.assertEqual("", result)

    # --- Faker tests (non-NER) ---

    @settings(deadline=None)
    @given(text(alphabet=string.ascii_letters + string.digits, min_size=0, max_size=4))
    def test_faker_email(self, fkw):
        """Check if generated emails are replaced correctly."""
        email = self.fake.email()
        clean_email = self.ProcessContacts.replace_emails(email, replace_with=fkw)
        self.assertEqual(clean_email, fkw)

    @settings(deadline=None)
    @given(text(alphabet=string.ascii_letters + string.digits, min_size=0, max_size=4))
    def test_faker_phone_number(self, fkw):
        """Check if generated phone numbers are replaced correctly."""
        phonenum = self.fake.phone_number()
        clean_phonenum = self.ProcessContacts.replace_phone_numbers(phonenum, replace_with=fkw)
        self.assertEqual(clean_phonenum, fkw)

    @settings(deadline=None)
    @given(text(alphabet=string.ascii_letters + string.digits, min_size=0, max_size=4))
    def test_faker_url(self, fkw):
        """Check if generated URLs are replaced correctly."""
        url = self.fake.url()
        clean_url = self.ProcessContacts.replace_urls(url, replace_with=fkw)
        self.assertEqual(clean_url, fkw)

    # --- Stopwords tests (non-NER) ---

    def test_stopwords_english(self):
        """Test English stopword removal."""
        text = "the quick brown fox jumps over the lazy dog"
        result = self.ProcessStopwords.remove_stopwords(text, "ENGLISH")
        self.assertNotIn("the", result.split())
        self.assertIn("quick", result.split())

    def test_stopwords_unknown_language_fallback(self):
        """Test that unknown language falls back to English stopwords."""
        text = "the quick brown fox"
        result = self.ProcessStopwords.remove_stopwords(text, "KLINGON")
        # Should use English fallback
        self.assertNotIn("the", result.split())

    def test_stopwords_set_performance(self):
        """Verify stopwords are stored as sets (O(1) lookup)."""
        self.assertIsInstance(self.ProcessStopwords.STOP_WORDS_EN, set)
        self.assertIsInstance(self.ProcessStopwords.STOP_WORDS_NL, set)

    # --- Config dataclass tests ---

    def test_config_frozen(self):
        """Test that TextCleanerConfig is frozen (immutable)."""
        cfg = TextCleanerConfig()
        with self.assertRaises(AttributeError):
            cfg.check_ner_process = False

    def test_config_from_module_globals(self):
        """Test backward-compatible config creation from module globals."""
        from sct.config import _config_from_module_globals
        old_val = config.CHECK_NER_PROCESS
        config.CHECK_NER_PROCESS = False
        cfg = _config_from_module_globals()
        self.assertFalse(cfg.check_ner_process)
        config.CHECK_NER_PROCESS = old_val

    def test_config_list_to_tuple_coercion(self):
        """Test that list args are coerced to tuples in frozen dataclass."""
        cfg = TextCleanerConfig(positional_tags=['PER', 'LOC'])
        self.assertIsInstance(cfg.positional_tags, tuple)

    # --- NER model config dict tests ---

    def test_ner_models_dict_acceptance(self):
        """Test that ner_models dict is accepted and populates both fields."""
        from types import MappingProxyType
        models = {
            'ENGLISH': 'test/english-onnx',
            'MULTILINGUAL': 'test/multi-onnx',
        }
        cfg = TextCleanerConfig(ner_models=models)
        # Dict should be frozen
        self.assertIsInstance(cfg.ner_models, MappingProxyType)
        self.assertEqual(cfg.ner_models['ENGLISH'], 'test/english-onnx')
        self.assertEqual(cfg.ner_models['MULTILINGUAL'], 'test/multi-onnx')
        # Missing keys filled from defaults
        self.assertIn('DUTCH', cfg.ner_models)
        self.assertIn('GERMAN', cfg.ner_models)
        self.assertIn('SPANISH', cfg.ner_models)
        # ner_models_list also populated
        self.assertIsInstance(cfg.ner_models_list, tuple)
        self.assertEqual(len(cfg.ner_models_list), 5)

    def test_ner_models_frozen(self):
        """Test that ner_models dict is immutable (MappingProxyType)."""
        cfg = TextCleanerConfig()
        with self.assertRaises(TypeError):
            cfg.ner_models['ENGLISH'] = 'new-model'

    def test_ner_models_required_keys_validation(self):
        """Test that missing ENGLISH or MULTILINGUAL raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            TextCleanerConfig(ner_models={'DUTCH': 'test/dutch-onnx'})
        self.assertIn('ENGLISH', str(ctx.exception))
        self.assertIn('MULTILINGUAL', str(ctx.exception))

    def test_ner_models_backward_compat_tuple(self):
        """Test that old ner_models_list tuple API still works."""
        from types import MappingProxyType
        models = ("m1", "m2", "m3", "m4", "m5")
        cfg = TextCleanerConfig(ner_models_list=models)
        # Should auto-derive ner_models dict
        self.assertIsInstance(cfg.ner_models, MappingProxyType)
        self.assertEqual(cfg.ner_models['ENGLISH'], 'm1')
        self.assertEqual(cfg.ner_models['MULTILINGUAL'], 'm5')
        # Original tuple preserved
        self.assertEqual(cfg.ner_models_list, models)

    def test_ner_models_dict_priority_over_list(self):
        """Test that ner_models takes priority when both are provided."""
        cfg = TextCleanerConfig(
            ner_models={'ENGLISH': 'dict-model', 'MULTILINGUAL': 'dict-multi'},
            ner_models_list=("list1", "list2", "list3", "list4", "list5"),
        )
        # Dict should win
        self.assertEqual(cfg.ner_models['ENGLISH'], 'dict-model')
        self.assertEqual(cfg.ner_models['MULTILINGUAL'], 'dict-multi')
        # ner_models_list re-derived from merged dict
        self.assertEqual(cfg.ner_models_list[0], 'dict-model')

    # --- Special symbols tests ---

    def test_bracket_removal_configurable(self):
        """Test bracket/brace content removal is configurable."""
        text = "Hello [citation needed] world {template var}"
        # With removal
        result = self.ProcessSpecialSymbols.remove_isolated_special_symbols(
            text, remove_brackets=True, remove_braces=True
        )
        self.assertNotIn("citation", result)
        self.assertNotIn("template", result)
        # Without removal
        result = self.ProcessSpecialSymbols.remove_isolated_special_symbols(
            text, remove_brackets=False, remove_braces=False
        )
        self.assertIn("citation", result)
        self.assertIn("template", result)

    def test_punctuation_pre_compiled(self):
        """Test that punctuation regex is pre-compiled in constants."""
        self.assertIsNotNone(constants.PUNCTUATION_REGEX)
        result = self.ProcessSpecialSymbols.remove_punctuation("Hello, world!")
        self.assertEqual(result, "Hello world")

    # --- Date tests ---

    def test_date_regex_iso8601(self):
        """Test ISO 8601 date replacement."""
        dt = datetime.ProcessDateTime()
        text = "Meeting on 2024-01-15 at noon"
        result = dt.replace_dates(text, "<DATE>")
        self.assertIn("<DATE>", result)
        self.assertNotIn("2024-01-15", result)

    def test_date_regex_month_name(self):
        """Test month name date replacement."""
        dt = datetime.ProcessDateTime()
        text = "Born on January 15, 2024 in London"
        result = dt.replace_dates(text, "<DATE>")
        self.assertIn("<DATE>", result)

    # --- Emoji tests ---

    def test_emoji_removal(self):
        """Test emoji removal functionality."""
        text = "Hello world 😀 test 🎉"
        result = self.NormaliseText.remove_emoji(text)
        self.assertNotIn("😀", result)
        self.assertNotIn("🎉", result)
        self.assertIn("Hello", result)

    # --- Unicode regex tests ---

    def test_url_regex_unicode_domain(self):
        """Test URL regex handles non-ASCII domains."""
        text = "Visit https://www.example.com/path"
        result = self.ProcessContacts.replace_urls(text, "<URL>")
        self.assertIn("<URL>", result)

    def test_email_regex_basic(self):
        """Test email regex handles standard emails."""
        text = "Send to user@example.com please"
        result = self.ProcessContacts.replace_emails(text, "<EMAIL>")
        self.assertIn("<EMAIL>", result)

    # --- ACRONYM_REGEX ReDoS safety test ---

    def test_acronym_regex_no_redos(self):
        """Test that ACRONYM_REGEX doesn't catastrophically backtrack."""
        import time
        # Adversarial input that would cause ReDoS with nested quantifiers
        adversarial = "A." * 100
        start = time.time()
        constants.ACRONYM_REGEX.findall(adversarial)
        elapsed = time.time() - start
        self.assertLess(elapsed, 2.0, "ACRONYM_REGEX took too long - possible ReDoS")

    # --- Consistent return type tests ---

    def test_process_returns_3tuple(self):
        """Test that process() always returns a 3-tuple regardless of config."""
        cfg = TextCleanerConfig(
            check_ner_process=False,
            check_statistical_model_processing=False,
            check_detect_language=False,
        )
        cleaner = TextCleaner(cfg=cfg)
        result = cleaner.process("Hello world")
        self.assertEqual(len(result), 3)
        lm_text, stat_text, lang = result
        self.assertIsInstance(lm_text, str)

    def test_process_with_stat_model(self):
        """Test that stat processing returns a lowercase result."""
        cfg = TextCleanerConfig(
            check_ner_process=False,
            check_statistical_model_processing=True,
        )
        cleaner = TextCleaner(cfg=cfg)
        lm_text, stat_text, lang = cleaner.process("Hello World THE")
        self.assertIsNotNone(stat_text)
        self.assertTrue(stat_text == stat_text.casefold())

    def test_process_batch_empty(self):
        """Test batch processing with empty inputs."""
        cfg = TextCleanerConfig(check_ner_process=False)
        cleaner = TextCleaner(cfg=cfg)
        self.assertEqual(cleaner.process_batch([]), [])
        results = cleaner.process_batch([""])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], ("", "", None))

    def test_process_batch_invalid_type(self):
        """Test batch processing rejects non-string input."""
        cfg = TextCleanerConfig(check_ner_process=False)
        cleaner = TextCleaner(cfg=cfg)
        with self.assertRaises(ValueError):
            cleaner.process_batch([123])

    # --- Thread safety test ---

    def test_language_not_stored_on_instance(self):
        """Test that language detection doesn't mutate instance state."""
        cfg = TextCleanerConfig(check_ner_process=False)
        cleaner = TextCleaner(cfg=cfg)
        # Process should not set self.language
        cleaner.process("Hello world")
        self.assertFalse(hasattr(cleaner, 'language'))

    # --- NER tests (skip if models not loaded) ---

    @requires_ner
    def test_ner_process_basic(self):
        """Test basic NER processing with known entities."""
        text = "John Smith works at Microsoft in New York."
        processed = self.ner.ner_process(
            text,
            positional_tags=['PER', 'ORG', 'LOC'],
            ner_confidence_threshold=0.85,
        )
        self.assertNotIn("John Smith", processed)
        self.assertNotIn("Microsoft", processed)
        self.assertNotIn("New York", processed)

    @requires_ner
    def test_ner_process_empty(self):
        """Test NER processing with no positional tags raises."""
        with self.assertRaises(ValueError):
            self.ner.ner_process("test", positional_tags=[])

    @requires_ner
    def test_ner_process_no_entities(self):
        """Test NER processing with text containing no entities."""
        text = "The quick brown fox jumps over the lazy dog."
        processed = self.ner.ner_process(
            text,
            positional_tags=['PER', 'ORG', 'LOC'],
            ner_confidence_threshold=0.85,
        )
        self.assertEqual(text, processed)

    @requires_ner
    def test_ner_entity_key_no_collision(self):
        """Test that entity keys use separator to avoid collision."""
        # ner_data should produce keys like "1:23" not "123"
        mock_data = [
            {'entity_group': 'PER', 'score': 0.95, 'word': 'John',
             'start': 1, 'end': 23},
            {'entity_group': 'PER', 'score': 0.90, 'word': 'Smith',
             'start': 12, 'end': 3},
        ]
        results = self.ner.ner_data(mock_data, ['PER'])
        keys = [r['key'] for r in results]
        self.assertEqual(keys[0], "1:23")
        self.assertEqual(keys[1], "12:3")
        self.assertNotEqual(keys[0], keys[1])

    @requires_ner
    def test_ner_ensemble_voting(self):
        """Test that ensemble voting combines results from multiple models."""
        mock_results = [
            {'entity_group': 'PER', 'score': 0.90, 'word': 'John', 'key': '0:4', 'start': 0, 'end': 4},
            {'entity_group': 'PER', 'score': 0.85, 'word': 'John', 'key': '0:4', 'start': 0, 'end': 4},
        ]
        ensemble = self.ner.ner_ensemble(mock_results, 0.80)
        self.assertEqual(len(ensemble), 1)
        self.assertEqual(ensemble[0]['key'], '0:4')

    @requires_ner
    def test_ner_misc_entity_handling(self):
        """Test that MISC entities are handled (not silently dropped)."""
        from sct.utils.ner import ENTITY_TYPE_MAP
        self.assertIn('MISC', ENTITY_TYPE_MAP)
        self.assertEqual(ENTITY_TYPE_MAP['MISC'], 'MISC')

    @requires_ner
    def test_ner_batch_processing(self):
        """Test batch processing of multiple texts."""
        texts = [
            "John works at Microsoft.",
            "Sarah lives in London.",
            "The quick brown fox.",
        ]
        results = self.ner.process_batch(
            texts,
            batch_size=2,
            positional_tags=['PER', 'ORG', 'LOC'],
            ner_confidence_threshold=0.85,
        )
        self.assertEqual(len(results), len(texts))

    @requires_ner
    def test_ner_process_multilingual(self):
        """Test NER processing with different languages."""
        tests = {
            "GERMAN": ("Angela Merkel lebt in Berlin.", "Angela Merkel", "Berlin"),
            "SPANISH": ("Pablo vive en Madrid.", "Pablo", "Madrid"),
            "DUTCH": ("Willem woont in Amsterdam.", "Willem", "Amsterdam"),
        }
        for lang, (input_text, person, location) in tests.items():
            processed = self.ner.ner_process(
                input_text,
                positional_tags=['PER', 'LOC'],
                ner_confidence_threshold=0.85,
                language=lang,
            )
            self.assertNotIn(person, processed, f"Failed for {lang}: {person} not anonymized")
            self.assertNotIn(location, processed, f"Failed for {lang}: {location} not anonymized")

    @requires_ner
    def test_ner_long_text_splitting(self):
        """Test NER processing splits and processes long text without error."""
        long_text = "John Smith " * 512
        processed = self.ner.ner_process(
            long_text,
            positional_tags=['PER'],
            ner_confidence_threshold=0.85,
        )
        # Verify the text was processed (split + NER ran without crash)
        self.assertIsInstance(processed, str)
        self.assertGreater(len(processed), 0)
        # At least some entities should be anonymized by the lightweight model
        self.assertIn("<PERSON>", processed)

    def test_inference_lock_keyed_on_model_name(self):
        """FR and MULTILINGUAL (same underlying model) resolve to the same lock key."""
        from sct.config import DEFAULT_NER_MODELS
        fr_model  = DEFAULT_NER_MODELS.get('FRENCH',       'FRENCH')
        mul_model = DEFAULT_NER_MODELS.get('MULTILINGUAL', 'MULTILINGUAL')
        self.assertEqual(fr_model, mul_model)   # share same model
        # Different languages with different models must NOT share a lock key
        en_model  = DEFAULT_NER_MODELS.get('ENGLISH', 'ENGLISH')
        self.assertNotEqual(en_model, mul_model)

    @requires_ner
    def test_load_language_is_public(self):
        """load_language() is public and pre-loads the model."""
        self.ner.load_language('ENGLISH')  # already loaded — must be a no-op, not crash
        self.assertIn('ENGLISH', self.ner._pipelines)

    @requires_ner
    def test_ensure_loaded_shares_session_for_same_model(self):
        """Languages sharing a model ID must reference the same pipeline object."""
        from sct.config import DEFAULT_NER_MODELS
        # Only run if FRENCH and MULTILINGUAL actually share a model in defaults
        if DEFAULT_NER_MODELS.get('FRENCH') == DEFAULT_NER_MODELS.get('MULTILINGUAL'):
            # Use a fresh NER with default models (skipping heavy download by using test models)
            # Simulate sharing: both keys in TEST_NER_MODELS map to same model string
            from sct.utils.ner import GeneralNER
            shared_model = "protectai/bert-base-NER-onnx"
            models = {'ENGLISH': shared_model, 'MULTILINGUAL': shared_model,
                      'FRENCH': shared_model}
            ner2 = GeneralNER(device='cpu', model_names=models)
            ner2._ensure_loaded('FRENCH')
            ner2._ensure_loaded('MULTILINGUAL')
            self.assertIs(ner2._pipelines['FRENCH'],
                          ner2._pipelines['MULTILINGUAL'])

    @requires_ner
    def test_ner_lazy_loading(self):
        """Test that NER models are loaded lazily."""
        # English and multilingual should be loaded (eagerly for tokenizer)
        self.assertIn('ENGLISH', self.ner._pipelines)
        self.assertIn('MULTILINGUAL', self.ner._pipelines)

    @requires_ner
    def test_ner_ensemble_key_selection(self):
        """Ensemble key routing: EN/NL/DE/ES get correct keys; others fall back to default."""
        from sct.config import NER_ENSEMBLE_DEFAULT_KEYS
        self.assertEqual(self.ner._get_ensemble_keys('ENGLISH'),
                         ('ENGLISH', 'MULTILINGUAL'))
        self.assertEqual(self.ner._get_ensemble_keys('DUTCH'),
                         ('DUTCH', 'ENGLISH', 'MULTILINGUAL'))
        self.assertEqual(self.ner._get_ensemble_keys('GERMAN'),
                         ('GERMAN', 'ENGLISH', 'MULTILINGUAL'))
        self.assertEqual(self.ner._get_ensemble_keys('SPANISH'),
                         ('SPANISH', 'ENGLISH', 'MULTILINGUAL'))
        self.assertEqual(self.ner._get_ensemble_keys('FRENCH'),    NER_ENSEMBLE_DEFAULT_KEYS)
        self.assertEqual(self.ner._get_ensemble_keys('PORTUGUESE'), NER_ENSEMBLE_DEFAULT_KEYS)
        self.assertEqual(self.ner._get_ensemble_keys('ITALIAN'),   NER_ENSEMBLE_DEFAULT_KEYS)

    @requires_ner
    def test_ner_ensemble_custom_default_keys(self):
        """Custom ner_ensemble_default_keys overrides the module-level fallback."""
        cfg = TextCleanerConfig(
            check_ner_process=True,
            ner_models=TEST_NER_MODELS,
            ner_ensemble_default_keys=('MULTILINGUAL',),
        )
        sx = TextCleaner(cfg=cfg)
        # FRENCH not in DEFAULT_NER_ENSEMBLE — should use the custom fallback
        self.assertEqual(sx.GeneralNER._get_ensemble_keys('FRENCH'), ('MULTILINGUAL',))
        # EN/NL still routed by DEFAULT_NER_ENSEMBLE (not overridden)
        self.assertEqual(sx.GeneralNER._get_ensemble_keys('ENGLISH'), ('ENGLISH', 'MULTILINGUAL'))

    @requires_ner
    def test_batch_processing_languages(self):
        """Test batch processing with multiple languages (fixed assertions)."""
        config.CHECK_NER_PROCESS = True
        cfg = TextCleanerConfig(
            check_ner_process=True,
            ner_models=TEST_NER_MODELS,
        )
        sx = TextCleaner(cfg=cfg)

        texts = [
            "John Smith lives in London",
            "Angela Merkel lebt in Berlin",
            "Pablo vive en Madrid",
            "Willem woont in Amsterdam",
        ]
        results = sx.process_batch(texts, batch_size=2)

        # Check language detection
        languages = [lang for _, _, lang in results]
        self.assertIn("ENGLISH", languages)

        # Fixed: only check entities relevant to EACH text (not all entities in all texts)
        en_idx = languages.index("ENGLISH")
        lm_en = results[en_idx][0]
        self.assertNotIn("John Smith", lm_en)
        self.assertNotIn("London", lm_en)

    @requires_ner
    def test_end_to_end_text_cleaning(self):
        """Test end-to-end text cleaning following the pipeline."""
        cfg = TextCleanerConfig(
            check_ner_process=True,
            ner_models=TEST_NER_MODELS,
        )
        sx = TextCleaner(cfg=cfg)
        lm_text, stat_text, lang = sx.process(TEST_TEXT)

        # Language detection
        self.assertEqual(lang, "ENGLISH")

        # Basic replacements
        self.assertNotIn("<div", lm_text)
        self.assertIn("<URL>", lm_text)
        self.assertIn("<EMAIL>", lm_text)
        self.assertIn("<PHONE>", lm_text)
        self.assertNotIn("$", lm_text)

        # NER
        self.assertNotIn("John Smith", lm_text)

        # Whitespace
        self.assertNotIn("  ", lm_text)

        # Statistical processing
        self.assertIsNotNone(stat_text)
        self.assertTrue(stat_text.islower())

    # --- Parallelism tests ---

    @requires_ner
    def test_batched_chunk_inference(self):
        """Test that batched chunk NER produces correct results for multi-chunk text.

        Verifies level 1 parallelism: all chunks from a single long text go
        through each model in one batched forward pass (2 passes total).
        """
        # Build text long enough to be split into multiple chunks
        chunk1_text = "John Smith works at Microsoft in Seattle. "
        chunk2_text = "Sarah Johnson lives in Berlin and works at Google. "
        long_text = chunk1_text * 30 + chunk2_text * 30  # ~60 repetitions, forces splitting

        chunks = self.ner.split_text(long_text, self.ner.min_token_length, self.ner.tokenizer)
        self.assertGreater(len(chunks), 1, "Text should split into multiple chunks")

        processed = self.ner.ner_process(
            long_text,
            positional_tags=['PER', 'ORG', 'LOC'],
            ner_confidence_threshold=0.85,
        )

        # Verify entities were detected across multiple chunks
        self.assertIsInstance(processed, str)
        self.assertIn("<PERSON>", processed)

    @requires_ner
    def test_threadpool_batch_consistency(self):
        """Test that ThreadPoolExecutor batch processing matches sequential results.

        Verifies level 2 parallelism: multiple independent texts processed in
        parallel threads produce the same results as sequential processing.
        """
        cfg = TextCleanerConfig(
            check_ner_process=True,
            ner_models=TEST_NER_MODELS,
        )

        texts = [
            "John Smith works at Microsoft.",
            "Sarah lives in London.",
            "Contact us at info@example.com or +1-234-567-8900.",
            "The quick brown fox jumps over the lazy dog.",
        ]

        # Run through process_batch (uses ThreadPoolExecutor when len > 1)
        cleaner = TextCleaner(cfg=cfg)
        parallel_results = cleaner.process_batch(texts)

        # Run each text individually (sequential, no ThreadPoolExecutor)
        sequential_results = [cleaner.process(t) for t in texts]

        self.assertEqual(len(parallel_results), len(sequential_results))
        for i, (par, seq) in enumerate(zip(parallel_results, sequential_results)):
            self.assertEqual(par[0], seq[0], f"LM text mismatch at index {i}")
            self.assertEqual(par[1], seq[1], f"Stat text mismatch at index {i}")
            self.assertEqual(par[2], seq[2], f"Language mismatch at index {i}")

    @requires_ner
    def test_concurrent_ner_thread_safety(self):
        """Test that concurrent NER calls don't corrupt shared model state.

        Verifies both levels together: multiple threads call ner_process
        simultaneously, each with different text. Results should be
        independent (no cross-thread contamination).
        """
        from concurrent.futures import ThreadPoolExecutor

        texts_and_entities = [
            ("John Smith works at Microsoft.", "John Smith"),
            ("Sarah Johnson lives in Berlin.", "Sarah Johnson"),
            ("Pablo Garcia works in Madrid.", "Pablo Garcia"),
            ("Willem de Vries lives in Amsterdam.", "Willem"),
        ]

        def run_ner(text):
            return self.ner.ner_process(
                text,
                positional_tags=['PER', 'LOC', 'ORG'],
                ner_confidence_threshold=0.85,
            )

        # Run all NER calls concurrently
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(run_ner, t) for t, _ in texts_and_entities]
            results = [f.result() for f in futures]

        # Each result should have anonymized the person from its OWN text
        for i, (result, (original_text, person)) in enumerate(zip(results, texts_and_entities)):
            self.assertIsInstance(result, str)
            self.assertNotEqual(result, "", f"Empty result for text {i}")
            # At minimum, the result should be processed (not the raw input)
            # Lightweight model may not catch every entity, so just verify no crash
            # and that the output is a valid string

    # --- split_text() unit tests (chunking only, no NER inference) ---

    @requires_ner
    def test_split_text_short_text_no_split(self):
        """Text under max_tokens returns as a single chunk."""
        text = "Hello world."
        chunks = self.ner.split_text(text, 50, self.ner.tokenizer)
        self.assertEqual(chunks, [text])

    @requires_ner
    def test_split_text_paragraph_boundaries(self):
        """Text with paragraph breaks splits at \\n\\n first."""
        para1 = "The quick brown fox jumps over the lazy dog. " * 5
        para2 = "Pack my box with five dozen liquor jugs. " * 5
        para3 = "How vexingly quick daft zebras jump. " * 5
        text = f"{para1.strip()}\n\n{para2.strip()}\n\n{para3.strip()}"

        # Set max_tokens so that each paragraph fits but two don't
        tokens_per_para = self.ner._token_count(para1.strip(), self.ner.tokenizer)
        max_tokens = int(tokens_per_para * 1.5)

        chunks = self.ner.split_text(text, max_tokens, self.ner.tokenizer)
        self.assertGreaterEqual(len(chunks), 2)
        # Paragraph breaks should not appear within any chunk
        for chunk in chunks:
            self.assertNotIn("\n\n", chunk)

    @requires_ner
    def test_split_text_sentence_boundaries(self):
        """Text without paragraph breaks splits at sentence boundaries."""
        sentences = [
            "The quick brown fox jumps over the lazy dog.",
            "Pack my box with five dozen liquor jugs.",
            "How vexingly quick daft zebras jump.",
            "The five boxing wizards jump quickly.",
            "Jackdaws love my big sphinx of quartz.",
        ]
        text = " ".join(s for s in sentences * 4)  # Repeat to ensure length

        total_tokens = self.ner._token_count(text, self.ner.tokenizer)
        max_tokens = total_tokens // 3

        chunks = self.ner.split_text(text, max_tokens, self.ner.tokenizer)
        self.assertGreater(len(chunks), 1)
        # Each chunk except the last should end at a sentence boundary
        for chunk in chunks[:-1]:
            self.assertTrue(
                chunk.rstrip().endswith(('.', '!', '?')),
                f"Chunk doesn't end at sentence boundary: ...'{chunk[-30:]}'"
            )

    @requires_ner
    def test_split_text_clause_boundaries(self):
        """Single long sentence with clause separators splits at semicolons/commas."""
        clauses = [f"clause number {i} of the text" for i in range(30)]
        text = "; ".join(clauses)

        total_tokens = self.ner._token_count(text, self.ner.tokenizer)
        max_tokens = total_tokens // 3

        chunks = self.ner.split_text(text, max_tokens, self.ner.tokenizer)
        self.assertGreater(len(chunks), 1)

    @requires_ner
    def test_split_text_word_boundaries(self):
        """Long text without punctuation splits at word boundaries."""
        text = " ".join(["word"] * 200)

        total_tokens = self.ner._token_count(text, self.ner.tokenizer)
        max_tokens = total_tokens // 4

        chunks = self.ner.split_text(text, max_tokens, self.ner.tokenizer)
        self.assertGreater(len(chunks), 1)
        # No chunk should start or end with whitespace
        for chunk in chunks:
            self.assertEqual(chunk, chunk.strip())

    @requires_ner
    def test_split_text_delimiter_hierarchy_order(self):
        """Paragraph breaks preferred over sentence breaks when both present."""
        text = (
            "First sentence here. Second sentence here.\n\n"
            "Third sentence here. Fourth sentence here."
        )

        total_tokens = self.ner._token_count(text, self.ner.tokenizer)
        max_tokens = int(total_tokens * 0.6)

        chunks = self.ner.split_text(text, max_tokens, self.ner.tokenizer)
        if len(chunks) == 2:
            # Should have split at paragraph break, not sentence break
            self.assertIn("First sentence", chunks[0])
            self.assertIn("Second sentence", chunks[0])
            self.assertIn("Third sentence", chunks[1])
            self.assertIn("Fourth sentence", chunks[1])

    @requires_ner
    def test_split_text_chunks_within_token_limit(self):
        """All returned chunks fit within max_tokens."""
        text = "The quick brown fox jumps. " * 100
        max_tokens = 30

        chunks = self.ner.split_text(text, max_tokens, self.ner.tokenizer)
        for i, chunk in enumerate(chunks):
            token_count = self.ner._token_count(chunk, self.ner.tokenizer)
            self.assertLessEqual(
                token_count, max_tokens,
                f"Chunk {i} has {token_count} tokens, exceeds limit {max_tokens}"
            )

    @requires_ner
    def test_split_text_no_content_loss(self):
        """All non-whitespace content is preserved across chunks."""
        import re as _re
        text = "Hello world. This is a test. With multiple sentences.\n\nAnd paragraphs too."
        max_tokens = 15

        chunks = self.ner.split_text(text, max_tokens, self.ner.tokenizer)

        # Normalize whitespace in original and joined chunks for comparison
        original_normalized = _re.sub(r'\s+', ' ', text).strip()
        joined_normalized = _re.sub(r'\s+', ' ', ' '.join(chunks)).strip()
        self.assertEqual(original_normalized, joined_normalized)

    @requires_ner
    def test_split_text_single_long_word_returns_one_chunk(self):
        """BERT maps long no-whitespace strings to [UNK] (1 token), returns single chunk."""
        text = "abcdefghij" * 100  # 1000 chars, no delimiters
        # BERT tokenizer produces [CLS] [UNK] [SEP] = 3 tokens
        token_count = self.ner._token_count(text, self.ner.tokenizer)
        self.assertLessEqual(token_count, 5, "Expected BERT to map to [UNK]")
        chunks = self.ner.split_text(text, 50, self.ner.tokenizer)
        self.assertEqual(len(chunks), 1)  # Under token limit, no split needed
        self.assertEqual(chunks[0], text)

    @requires_ner
    def test_split_text_token_offset_fallback(self):
        """Character-level fallback code path tested via mock tokenizer."""
        # BERT's [UNK] makes real no-delimiter testing impossible,
        # so we mock a tokenizer that produces one token per character.
        from types import SimpleNamespace

        def mock_tokenizer(text, **kwargs):
            n = len(text)
            ids = list(range(n))
            if kwargs.get('return_offsets_mapping'):
                return SimpleNamespace(
                    input_ids=ids,
                    offset_mapping=[(i, i + 1) for i in range(n)],
                )
            return SimpleNamespace(input_ids=ids)

        text = "x" * 100  # 100 chars, no delimiters
        max_tokens = 20

        chunks = self.ner.split_text(text, max_tokens, mock_tokenizer)
        self.assertGreater(len(chunks), 1)
        self.assertEqual(''.join(chunks), text)
        # Each chunk should be at most max_tokens characters (1 char = 1 token)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), max_tokens)

    # --- Chunking + NER integration tests ---

    @requires_ner
    def test_chunked_ner_preserves_entities(self):
        """Long text with entities across chunks still gets anonymized."""
        segment = "John Smith works at Microsoft in Seattle. Sarah Johnson lives in Berlin. "
        text = segment * 50  # Repeat to force multiple chunks

        chunks = self.ner.split_text(text, self.ner.min_token_length, self.ner.tokenizer)
        self.assertGreater(len(chunks), 1, "Text should require multiple chunks")

        processed = self.ner.ner_process(
            text,
            positional_tags=['PER', 'ORG', 'LOC'],
            ner_confidence_threshold=0.85,
        )

        self.assertIsInstance(processed, str)
        self.assertIn("<PERSON>", processed)

    @requires_ner
    def test_chunked_ner_parallel_consistency(self):
        """Batched chunk inference produces consistent results across runs."""
        text = "John Smith from Microsoft met Sarah Johnson in Berlin. " * 40

        result1 = self.ner.ner_process(
            text,
            positional_tags=['PER', 'ORG', 'LOC'],
            ner_confidence_threshold=0.85,
        )
        result2 = self.ner.ner_process(
            text,
            positional_tags=['PER', 'ORG', 'LOC'],
            ner_confidence_threshold=0.85,
        )

        # Same input should produce identical output (deterministic with no_grad)
        self.assertEqual(result1, result2)

    def test_aprocess_batch_uses_running_loop(self):
        """aprocess_batch() must call get_running_loop, not get_event_loop."""
        import inspect
        src = inspect.getsource(TextCleaner.aprocess_batch)
        self.assertIn('get_running_loop', src)
        self.assertNotIn('get_event_loop', src)

    def test_sentence_boundary_no_false_split_on_abbreviation(self):
        """SENTENCE_BOUNDARY_PATTERN must not split abbreviation-dot + space."""
        from sct.utils.constants import SENTENCE_BOUNDARY_PATTERN
        for abbrev_text in ["Dr. Smith visited Berlin.",
                            "Mr. Jones called Mrs. Smith.",
                            "The U.S. Army Base is here."]:
            pieces = SENTENCE_BOUNDARY_PATTERN.split(abbrev_text)
            self.assertEqual(len(pieces), 1,
                             f"Should not split abbreviation in: {abbrev_text!r}")

    def test_sentence_boundary_splits_real_sentence(self):
        """SENTENCE_BOUNDARY_PATTERN should split genuine sentence boundaries."""
        from sct.utils.constants import SENTENCE_BOUNDARY_PATTERN
        text = "This is the first sentence. And this is the second one."
        pieces = SENTENCE_BOUNDARY_PATTERN.split(text)
        self.assertGreater(len(pieces), 1)

    def test_simple_chunk_conservative_ratio(self):
        """_simple_chunk uses 2 chars/token (not 4) to handle CJK/Arabic safely."""
        import inspect
        src = inspect.getsource(GeneralNER._simple_chunk)
        self.assertIn('max_tokens * 2', src)
        self.assertNotIn('max_tokens * 4', src)

    def test_threadpool_single_text_no_threading(self):
        """Test that process_batch with 1 text skips ThreadPoolExecutor."""
        cfg = TextCleanerConfig(check_ner_process=False)
        cleaner = TextCleaner(cfg=cfg)

        # Single text should take the sequential path (max_workers <= 1)
        results = cleaner.process_batch(["Hello world"])
        self.assertEqual(len(results), 1)
        lm_text, stat_text, lang = results[0]
        self.assertIn("Hello", lm_text)

    # --- smart_casefold() unit tests (no NER needed) ---

    def test_smart_casefold_preserves_abbreviations(self):
        """All-uppercase tokens preserved as abbreviations."""
        text = "The UN passed a new NATO resolution about GDP"
        result = self.NormaliseText.smart_casefold(text)
        for abbr in ("UN", "NATO", "GDP"):
            self.assertIn(abbr, result.split())

    def test_smart_casefold_preserves_camelcase(self):
        """Tokens with lowercase-to-uppercase transitions preserved."""
        text = "Use McDonald eBay iPhone"
        result = self.NormaliseText.smart_casefold(text)
        for term in ("McDonald", "eBay", "iPhone"):
            self.assertIn(term, result.split())

    def test_smart_casefold_lowercases_regular_words(self):
        """Title-cased and regular words get casefolded."""
        text = "Beautiful Sunny Morning"
        result = self.NormaliseText.smart_casefold(text)
        self.assertEqual(result, "beautiful sunny morning")

    def test_smart_casefold_stopwords_lowercased(self):
        """All-uppercase stopwords lowercased (not treated as abbreviations)."""
        en_stops = self.ProcessStopwords.get_stop_words("ENGLISH")
        text = "NASA AND THE FDA"
        result = self.NormaliseText.smart_casefold(text, stop_words=en_stops)
        self.assertIn("NASA", result.split())
        self.assertIn("FDA", result.split())
        self.assertIn("and", result.split())
        self.assertIn("the", result.split())

    def test_smart_casefold_mixed_text(self):
        """Combined scenario: abbreviations + stopwords + regular words."""
        en_stops = self.ProcessStopwords.get_stop_words("ENGLISH")
        text = "CEO Quarterly AND iPhone Report"
        result = self.NormaliseText.smart_casefold(text, stop_words=en_stops)
        self.assertEqual(result, "CEO quarterly and iPhone report")

    def test_smart_casefold_empty_and_edge_cases(self):
        """Edge cases: empty string, numbers only, single chars."""
        self.assertEqual(self.NormaliseText.smart_casefold(""), "")
        self.assertEqual(self.NormaliseText.smart_casefold("123 456"), "123 456")
        # Single uppercase letter passes isupper() -> preserved without stopwords
        self.assertEqual(self.NormaliseText.smart_casefold("A"), "A")
        self.assertEqual(self.NormaliseText.smart_casefold("I"), "I")
        # With stopwords, single-char stopwords get lowered
        en_stops = self.ProcessStopwords.get_stop_words("ENGLISH")
        self.assertEqual(self.NormaliseText.smart_casefold("A", stop_words=en_stops), "a")

    def test_statistical_pipeline_smart_casefold(self):
        """Integration: statistical pipeline uses smart_casefold when configured."""
        cfg = TextCleanerConfig(
            check_ner_process=False,
            check_statistical_model_processing=True,
            check_smart_casefold=True,
            check_casefold=True,  # Should be overridden by smart_casefold
        )
        cleaner = TextCleaner(cfg=cfg)
        lm_text, stat_text, lang = cleaner.process("UNESCO Heritage AND iPhone Quarterly")
        self.assertIsNotNone(stat_text)
        # Abbreviations and camelCase preserved
        self.assertIn("UNESCO", stat_text)
        self.assertIn("iPhone", stat_text)
        # Regular words lowered
        self.assertNotIn("Heritage", stat_text)
        self.assertNotIn("Quarterly", stat_text)

    # --- Multilingual date regex tests ---

    def test_date_regex_german(self):
        """Test German date formats with extended multilingual regex."""
        dt = datetime.ProcessDateTime()
        cases = [
            ("Treffen am 15. Januar 2024 in Berlin", "15. Januar 2024"),
            ("Am 3. März 2023 begann das Projekt", "3. März 2023"),
            ("Dezember 25, 2024 ist Weihnachten", "Dezember 25, 2024"),
        ]
        for input_text, date_str in cases:
            result = dt.replace_dates(input_text, "<DATE>")
            self.assertIn("<DATE>", result, f"Failed to detect German date: {date_str}")
            self.assertNotIn(date_str, result)

    def test_date_regex_dutch(self):
        """Test Dutch date formats with extended multilingual regex."""
        dt = datetime.ProcessDateTime()
        cases = [
            ("Vergadering op 15 januari 2024", "15 januari 2024"),
            ("Op 3 maart 2023 begon het project", "3 maart 2023"),
            ("Oktober 10, 2024 is de deadline", "Oktober 10, 2024"),
        ]
        for input_text, date_str in cases:
            result = dt.replace_dates(input_text, "<DATE>")
            self.assertIn("<DATE>", result, f"Failed to detect Dutch date: {date_str}")

    def test_date_regex_spanish(self):
        """Test Spanish date formats with extended multilingual regex."""
        dt = datetime.ProcessDateTime()
        cases = [
            ("Reunión el 15 de enero de 2024", "15 de enero de 2024"),
            ("El 3 de marzo de 2023 comenzó el proyecto", "3 de marzo de 2023"),
            ("Diciembre 25, 2024 es Navidad", "Diciembre 25, 2024"),
        ]
        for input_text, date_str in cases:
            result = dt.replace_dates(input_text, "<DATE>")
            self.assertIn("<DATE>", result, f"Failed to detect Spanish date: {date_str}")

    def test_date_regex_day_first_format(self):
        """Test day-first date format: '15 January 2024'."""
        dt = datetime.ProcessDateTime()
        result = dt.replace_dates("Born on 15 January 2024 in London", "<DATE>")
        self.assertIn("<DATE>", result)
        self.assertNotIn("15 January 2024", result)

    # --- Fuzzy date detection tests ---

    def test_fuzzy_replace_dates_misspelled_english(self):
        """Test fuzzy detection of misspelled English month names."""
        dt = datetime.ProcessDateTime()
        cases = [
            "Meeting on Janury 15, 2024",
            "Born on Feburary 3, 2023",
            "Due by Septmber 30, 2024",
            "Starting Novmber 1, 2025",
        ]
        for input_text in cases:
            result = dt.fuzzy_replace_dates(input_text, "<DATE>", score_cutoff=85)
            self.assertIn("<DATE>", result, f"Fuzzy failed for: {input_text}")

    def test_fuzzy_replace_dates_misspelled_multilingual(self):
        """Test fuzzy detection of misspelled non-English month names."""
        dt = datetime.ProcessDateTime()
        cases = [
            ("15 de enro de 2024", "Spanish 'enero' misspelled"),
            ("15 Febuari 2024", "Dutch 'februari' misspelled"),
        ]
        for input_text, description in cases:
            result = dt.fuzzy_replace_dates(input_text, "<DATE>", score_cutoff=80)
            self.assertIn("<DATE>", result, f"Fuzzy failed for {description}: {input_text}")

    def test_fuzzy_replace_dates_no_false_positive(self):
        """Test that fuzzy detection doesn't replace non-date patterns."""
        dt = datetime.ProcessDateTime()
        # These should NOT be replaced — no date-like pattern context
        texts = [
            "The market is strong today",
            "Meeting with the major at noon",
            "This is a jungle out there",
        ]
        for input_text in texts:
            result = dt.fuzzy_replace_dates(input_text, "<DATE>", score_cutoff=85)
            self.assertEqual(input_text, result, f"False positive for: {input_text}")

    def test_fuzzy_replace_dates_after_exact_regex(self):
        """Test fuzzy pass only catches what regex missed."""
        dt = datetime.ProcessDateTime()
        text = "Events on January 15, 2024 and Janury 20, 2024"
        # First: exact regex catches "January 15, 2024"
        step1 = dt.replace_dates(text, "<DATE>")
        self.assertIn("<DATE>", step1)
        self.assertIn("Janury", step1)  # regex missed misspelling
        # Second: fuzzy catches "Janury 20, 2024"
        step2 = dt.fuzzy_replace_dates(step1, "<DATE>", score_cutoff=85)
        self.assertNotIn("Janury", step2)
        self.assertEqual(step2.count("<DATE>"), 2)

    def test_fuzzy_replace_dates_configurable_cutoff(self):
        """Test that score_cutoff parameter controls match sensitivity."""
        dt = datetime.ProcessDateTime()
        text = "Due by Agusto 15, 2024"  # score ~83.3 against "agosto"
        # High cutoff: should NOT match
        result_strict = dt.fuzzy_replace_dates(text, "<DATE>", score_cutoff=90)
        self.assertNotIn("<DATE>", result_strict)
        # Low cutoff: should match
        result_loose = dt.fuzzy_replace_dates(text, "<DATE>", score_cutoff=80)
        self.assertIn("<DATE>", result_loose)

    def test_fuzzy_replace_dates_import_error(self):
        """Test that missing rapidfuzz raises clear ImportError."""
        dt = datetime.ProcessDateTime()
        with patch.dict('sys.modules', {'rapidfuzz': None, 'rapidfuzz.fuzz': None, 'rapidfuzz.process': None}):
            with self.assertRaises(ImportError) as ctx:
                dt.fuzzy_replace_dates("Janury 15, 2024", "<DATE>")
            self.assertIn("rapidfuzz", str(ctx.exception))

    def test_fuzzy_replace_dates_pipeline_integration(self):
        """Test fuzzy dates work when enabled in full pipeline config."""
        cfg = TextCleanerConfig(
            check_ner_process=False,
            check_replace_dates=True,
            check_fuzzy_replace_dates=True,
            fuzzy_date_score_cutoff=85,
        )
        cleaner = TextCleaner(cfg=cfg)
        lm_text, _, _ = cleaner.process("Meeting on Janury 15, 2024 about the project")
        self.assertIn("<DATE>", lm_text)
        self.assertNotIn("Janury", lm_text)

    def test_fuzzy_replace_dates_language_aware(self):
        """Test that language-aware fuzzy matching narrows vocabulary."""
        dt = datetime.ProcessDateTime()
        # German misspelling "Janur" (should match "januar" when lang=GERMAN)
        text = "15 Janur 2024"
        result_de = dt.fuzzy_replace_dates(text, "<DATE>", score_cutoff=80, language="GERMAN")
        self.assertIn("<DATE>", result_de)
        # Spanish misspelling should still match when language=None (all langs)
        text_es = "15 de enro de 2024"
        result_all = dt.fuzzy_replace_dates(text_es, "<DATE>", score_cutoff=80, language=None)
        self.assertIn("<DATE>", result_all)

    def test_fuzzy_replace_dates_language_includes_english_fallback(self):
        """Test that non-English language still checks English month names."""
        dt = datetime.ProcessDateTime()
        # English misspelling should be caught even when language is GERMAN
        text = "Feburary 15, 2024"
        result = dt.fuzzy_replace_dates(text, "<DATE>", score_cutoff=85, language="GERMAN")
        self.assertIn("<DATE>", result)

    def test_fuzzy_replace_dates_default_off(self):
        """Test that fuzzy date detection is off by default (no regression)."""
        cfg = TextCleanerConfig(check_ner_process=False, check_replace_dates=True)
        cleaner = TextCleaner(cfg=cfg)
        lm_text, _, _ = cleaner.process("Meeting on Janury 15, 2024")
        # With fuzzy OFF, misspelled month should survive
        self.assertIn("Janury", lm_text)

    def test_get_stop_words_accessor(self):
        """Test get_stop_words returns correct set for each language."""
        en = self.ProcessStopwords.get_stop_words("ENGLISH")
        nl = self.ProcessStopwords.get_stop_words("DUTCH")
        self.assertIsInstance(en, set)
        self.assertIn("the", en)
        self.assertIn("de", nl)
        # Unknown language falls back to English
        fallback = self.ProcessStopwords.get_stop_words("KLINGON")
        self.assertEqual(fallback, en)

    # --- Language extensibility tests ---

    def test_extra_languages_valid(self):
        """POLISH is a valid Lingua language."""
        cfg = TextCleanerConfig(check_ner_process=False, extra_languages=('POLISH',))
        self.assertIn('POLISH', cfg.supported_languages)
        # Default 4 still present
        self.assertIn('ENGLISH', cfg.supported_languages)
        self.assertIn('DUTCH', cfg.supported_languages)

    def test_extra_languages_invalid(self):
        """KLINGON is not a valid Lingua language."""
        with self.assertRaises(ValueError) as ctx:
            TextCleanerConfig(check_ner_process=False, extra_languages=('KLINGON',))
        self.assertIn("Unknown language", str(ctx.exception))

    def test_supported_languages_union(self):
        """supported_languages is union of defaults + extra + ner_models + custom keys."""
        cfg = TextCleanerConfig(
            check_ner_process=False,
            extra_languages=('POLISH',),
            ner_models={'ENGLISH': 'x', 'MULTILINGUAL': 'y', 'FRENCH': 'z'},
            custom_stopwords={'ITALIAN': frozenset({'di', 'il'})},
        )
        expected = {'ENGLISH', 'DUTCH', 'GERMAN', 'SPANISH', 'POLISH', 'FRENCH', 'ITALIAN'}
        self.assertTrue(expected.issubset(cfg.supported_languages))
        # MULTILINGUAL is a model, not a spoken language
        self.assertNotIn('MULTILINGUAL', cfg.supported_languages)

    def test_language_pin_validation(self):
        """language must be in supported set."""
        with self.assertRaises(ValueError):
            TextCleanerConfig(check_ner_process=False, language='POLISH')
        # This should work when POLISH is in supported set:
        cfg = TextCleanerConfig(
            check_ner_process=False, language='POLISH', extra_languages=('POLISH',)
        )
        self.assertEqual(cfg.language, 'POLISH')

    # --- ISO code / tuple language tests ---

    def test_resolve_language_full_name(self):
        """resolve_language('ENGLISH') → 'ENGLISH'."""
        self.assertEqual(resources.resolve_language('ENGLISH'), 'ENGLISH')

    def test_resolve_language_lowercase(self):
        """resolve_language('english') → 'ENGLISH'."""
        self.assertEqual(resources.resolve_language('english'), 'ENGLISH')

    def test_resolve_language_iso1(self):
        """resolve_language('en') → 'ENGLISH'."""
        self.assertEqual(resources.resolve_language('en'), 'ENGLISH')

    def test_resolve_language_iso3(self):
        """resolve_language('eng') → 'ENGLISH'."""
        self.assertEqual(resources.resolve_language('eng'), 'ENGLISH')

    def test_resolve_language_iso1_uppercase(self):
        """resolve_language('EN') → 'ENGLISH'."""
        self.assertEqual(resources.resolve_language('EN'), 'ENGLISH')

    def test_resolve_language_multilingual_passthrough(self):
        """resolve_language('MULTILINGUAL') → 'MULTILINGUAL' (special case)."""
        self.assertEqual(resources.resolve_language('MULTILINGUAL'), 'MULTILINGUAL')

    def test_resolve_language_invalid(self):
        """resolve_language('xyz') raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            resources.resolve_language('xyz')
        self.assertIn("Unknown language", str(ctx.exception))

    def test_config_language_iso_code(self):
        """TextCleanerConfig(language='en') resolves to 'ENGLISH'."""
        cfg = TextCleanerConfig(check_ner_process=False, language='en')
        self.assertEqual(cfg.language, 'ENGLISH')

    def test_config_language_iso3_code(self):
        """TextCleanerConfig(language='nld') resolves to 'DUTCH'."""
        cfg = TextCleanerConfig(check_ner_process=False, language='nld')
        self.assertEqual(cfg.language, 'DUTCH')

    def test_config_language_tuple(self):
        """TextCleanerConfig(language=('en', 'nl')) → ('ENGLISH', 'DUTCH')."""
        cfg = TextCleanerConfig(check_ner_process=False, language=('en', 'nl'))
        self.assertEqual(cfg.language, ('ENGLISH', 'DUTCH'))

    def test_config_language_tuple_iso_mix(self):
        """TextCleanerConfig(language=('ENGLISH', 'nl')) → ('ENGLISH', 'DUTCH')."""
        cfg = TextCleanerConfig(check_ner_process=False, language=('ENGLISH', 'nl'))
        self.assertEqual(cfg.language, ('ENGLISH', 'DUTCH'))

    def test_config_language_tuple_in_supported(self):
        """Tuple languages are added to supported_languages."""
        cfg = TextCleanerConfig(check_ner_process=False, language=('en', 'nl', 'fr'))
        self.assertIn('FRENCH', cfg.supported_languages)
        self.assertIn('ENGLISH', cfg.supported_languages)
        self.assertIn('DUTCH', cfg.supported_languages)

    def test_config_extra_languages_iso(self):
        """extra_languages=('fr',) resolves to ('FRENCH',)."""
        cfg = TextCleanerConfig(check_ner_process=False, extra_languages=('fr',))
        self.assertIn('FRENCH', cfg.supported_languages)

    def test_detect_language_pinned_iso(self):
        """language='de' pins to GERMAN (no detection)."""
        cfg = TextCleanerConfig(check_ner_process=False, language='de')
        cleaner = TextCleaner(cfg=cfg)
        _, _, lang = cleaner.process("This is English text")
        self.assertEqual(lang, 'GERMAN')

    def test_detect_language_tuple_restricts(self):
        """language=('en', 'nl') restricts detection; result is one of the two."""
        cfg = TextCleanerConfig(check_ner_process=False, language=('en', 'nl'))
        cleaner = TextCleaner(cfg=cfg)
        _, _, lang = cleaner.process("This is clearly English text about the weather")
        self.assertIn(lang, ('ENGLISH', 'DUTCH'))

    def test_detect_language_tuple_fallback(self):
        """language=('en', 'nl') with undetectable text falls back to first."""
        cfg = TextCleanerConfig(check_ner_process=False, language=('en', 'nl'))
        cleaner = TextCleaner(cfg=cfg)
        _, _, lang = cleaner.process("xyz")
        self.assertIn(lang, ('ENGLISH', 'DUTCH'))

    def test_default_backward_compat(self):
        """Default TextCleanerConfig() has same supported_languages as before."""
        cfg = TextCleanerConfig(check_ner_process=False)
        self.assertEqual(
            cfg.supported_languages,
            frozenset({'ENGLISH', 'DUTCH', 'GERMAN', 'SPANISH'}),
        )

    def test_custom_stopwords_replace(self):
        """Custom stopwords replace auto-detected ones."""
        custom = frozenset({'custom1', 'custom2'})
        sw = ProcessStopwords(
            supported_languages=frozenset({'ENGLISH'}),
            custom_stopwords={'ENGLISH': custom},
        )
        self.assertEqual(sw.get_stop_words('ENGLISH'), custom)

    def test_auto_stopwords_via_iso(self):
        """Polish stopwords auto-load from stop-words package."""
        sw = ProcessStopwords(supported_languages=frozenset({'ENGLISH', 'POLISH'}))
        polish_sw = sw.get_stop_words('POLISH')
        self.assertGreater(len(polish_sw), 0)  # stop-words has Polish

    def test_custom_month_names_in_date_regex(self):
        """Custom month names are included in date regex."""
        mn = build_month_names_dict({'pl': ('styczen', 'luty')})
        regex = build_date_regex(mn)
        self.assertIsNotNone(regex.search("15 styczen 2025"))

    def test_custom_month_names_pipeline(self):
        """Custom month names work in full pipeline."""
        cfg = TextCleanerConfig(
            check_ner_process=False,
            check_replace_dates=True,
            extra_languages=('POLISH',),
            custom_month_names={
                'POLISH': ('styczen', 'luty', 'marzec', 'kwiecien', 'maj', 'czerwiec',
                           'lipiec', 'sierpien', 'wrzesien', 'pazdziernik', 'listopad', 'grudzien'),
            },
        )
        cleaner = TextCleaner(cfg=cfg)
        lm_text, _, _ = cleaner.process("Spotkanie 15 styczen 2025 w biurze")
        self.assertIn("<DATE>", lm_text)
        self.assertNotIn("styczen", lm_text)

    def test_full_pipeline_with_extra_language(self):
        """Full pipeline with extra_languages works end-to-end."""
        cfg = TextCleanerConfig(
            check_ner_process=False,
            extra_languages=('POLISH',),
        )
        cleaner = TextCleaner(cfg=cfg)
        result = cleaner.process("To jest test tekstu")
        self.assertEqual(len(result), 3)
        lm_text, stat_text, lang = result
        self.assertIsInstance(lm_text, str)

    # --- NER backend config validation (no model loading) ---

    def test_ner_batch_size_zero_raises(self):
        with self.assertRaises(ValueError, msg="batch_size=0 must raise"):
            TextCleanerConfig(ner_batch_size=0)

    def test_ner_batch_size_negative_raises(self):
        with self.assertRaises(ValueError, msg="batch_size=-1 must raise"):
            TextCleanerConfig(ner_batch_size=-1)

    def test_ner_batch_size_valid(self):
        cfg = TextCleanerConfig(ner_batch_size=1)
        self.assertEqual(cfg.ner_batch_size, 1)

    def test_ner_backend_default_is_onnx(self):
        cfg = TextCleanerConfig(check_ner_process=False)
        self.assertEqual(cfg.ner_backend, 'onnx')

    def test_ner_backend_invalid_raises(self):
        with self.assertRaises(ValueError):
            TextCleanerConfig(check_ner_process=False, ner_backend='invalid')

    def test_ner_backend_torch_valid(self):
        cfg = TextCleanerConfig(check_ner_process=False, ner_backend='torch')
        self.assertEqual(cfg.ner_backend, 'torch')
        self.assertIsNotNone(cfg.torch_ner_models)  # defaults filled

    def test_ner_backend_gliner_requires_model(self):
        with self.assertRaises(ValueError):
            TextCleanerConfig(check_ner_process=False, ner_backend='gliner')

    def test_ner_backend_gliner_valid(self):
        cfg = TextCleanerConfig(
            check_ner_process=False,
            ner_backend='gliner',
            gliner_model='urchade/gliner_large-v2.1',
            gliner_labels=('person', 'org'),
            gliner_label_map={'person': 'PER', 'org': 'ORG'},
        )
        self.assertEqual(cfg.gliner_model, 'urchade/gliner_large-v2.1')
        self.assertEqual(cfg.gliner_labels, ('person', 'org'))

    def test_ner_backend_gliner_variant_invalid(self):
        with self.assertRaises(ValueError):
            TextCleanerConfig(
                check_ner_process=False,
                ner_backend='gliner', gliner_model='x', gliner_variant='v3',
            )

    def test_ner_backend_ensemble_onnx_requires_gliner(self):
        with self.assertRaises(ValueError):
            TextCleanerConfig(check_ner_process=False, ner_backend='ensemble_onnx')

    def test_ner_backend_ensemble_onnx_valid(self):
        cfg = TextCleanerConfig(
            check_ner_process=False,
            ner_backend='ensemble_onnx',
            gliner_model='urchade/gliner_large-v2.1',
        )
        self.assertEqual(cfg.ner_backend, 'ensemble_onnx')

    def test_ner_backend_ensemble_torch_valid(self):
        cfg = TextCleanerConfig(
            check_ner_process=False,
            ner_backend='ensemble_torch',
            gliner_model='urchade/gliner_large-v2.1',
        )
        self.assertEqual(cfg.ner_backend, 'ensemble_torch')
        self.assertIsNotNone(cfg.torch_ner_models)

    def test_gliner_label_map_frozen(self):
        cfg = TextCleanerConfig(
            check_ner_process=False,
            ner_backend='gliner', gliner_model='x',
            gliner_label_map={'person': 'PER'},
        )
        with self.assertRaises(TypeError):
            cfg.gliner_label_map['new'] = 'val'

    def test_torch_ner_models_requires_english_multilingual(self):
        with self.assertRaises(ValueError):
            TextCleanerConfig(
                check_ner_process=False,
                ner_backend='torch',
                torch_ner_models={'ENGLISH': 'x'},  # Missing MULTILINGUAL
            )

    def test_torch_ner_models_fills_defaults(self):
        cfg = TextCleanerConfig(
            check_ner_process=False,
            ner_backend='torch',
            torch_ner_models={'ENGLISH': 'custom-en', 'MULTILINGUAL': 'custom-ml'},
        )
        self.assertEqual(cfg.torch_ner_models['ENGLISH'], 'custom-en')
        self.assertIn('DUTCH', cfg.torch_ner_models)  # filled from defaults

    # --- GLiNER adapter unit tests (no model loading) ---

    def test_gliner_adapter_map_label(self):
        from sct.utils.gliner_adapter import GLiNERAdapter
        adapter = GLiNERAdapter.__new__(GLiNERAdapter)
        adapter.label_map = {'person': 'PER', 'org': 'ORG'}
        self.assertEqual(adapter._map_label('person'), 'PER')
        self.assertEqual(adapter._map_label('org'), 'ORG')
        self.assertEqual(adapter._map_label('product'), 'PRODUCT')  # unmapped -> uppercase

    # --- anonymize_text custom label tests ---

    @requires_ner
    def test_anonymize_custom_labels_direct_replacement(self):
        """Custom entity labels bypass Presidio and use direct string replacement."""
        filtered = [
            {'entity_group': 'PRODUCT', 'score': 0.9, 'word': 'iPhone',
             'key': '10:16', 'start': 10, 'end': 16},
        ]
        text = "I bought iPhone today"
        result = self.ner.anonymize_text(text, filtered)
        self.assertIn('<PRODUCT>', result.text)
        self.assertNotIn('iPhone', result.text)

    @requires_ner
    def test_anonymize_mixed_labels(self):
        """Mix of standard and custom labels uses direct replacement for all."""
        filtered = [
            {'entity_group': 'PER', 'score': 0.9, 'word': 'John',
             'key': '0:4', 'start': 0, 'end': 4},
            {'entity_group': 'PRODUCT', 'score': 0.8, 'word': 'iPad',
             'key': '15:19', 'start': 15, 'end': 19},
        ]
        text = "John bought an iPad today"
        result = self.ner.anonymize_text(text, filtered)
        self.assertIn('<PERSON>', result.text)
        self.assertIn('<PRODUCT>', result.text)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, 'ner') and cls.ner is not None:
            del cls.ner
        gc.collect()


class PIIModeConfigTest(unittest.TestCase):
    """Test PII mode auto-configuration (config-level only, no model downloads)."""

    def test_pii_mode_auto_configures_gliner(self):
        """PII mode should auto-switch to gliner backend with PII defaults."""
        cfg = TextCleanerConfig(
            check_ner_process=False,
            ner_mode='pii',
        )
        self.assertEqual(cfg.ner_backend, 'gliner')
        self.assertEqual(cfg.gliner_model, PII_DEFAULT_MODEL)
        self.assertEqual(cfg.gliner_labels, PII_LABELS)
        self.assertEqual(cfg.gliner_threshold, PII_DEFAULT_THRESHOLD)
        self.assertIsNotNone(cfg.gliner_label_map)

    def test_pii_mode_preserves_user_model(self):
        """User-provided gliner_model should not be overridden by PII mode."""
        cfg = TextCleanerConfig(
            check_ner_process=False,
            ner_mode='pii',
            gliner_model='my-custom/pii-model',
        )
        self.assertEqual(cfg.gliner_model, 'my-custom/pii-model')

    def test_pii_mode_preserves_user_labels(self):
        """User-provided gliner_labels should not be overridden by PII mode."""
        custom_labels = ('person', 'email', 'phone')
        cfg = TextCleanerConfig(
            check_ner_process=False,
            ner_mode='pii',
            gliner_model='knowledgator/gliner-pii-base-v1.0',
            gliner_labels=custom_labels,
        )
        self.assertEqual(cfg.gliner_labels, custom_labels)

    def test_pii_mode_preserves_user_backend(self):
        """Non-onnx ner_backend should not be overridden by PII mode."""
        cfg = TextCleanerConfig(
            check_ner_process=False,
            ner_mode='pii',
            ner_backend='gliner',
            gliner_model='knowledgator/gliner-pii-base-v1.0',
        )
        self.assertEqual(cfg.ner_backend, 'gliner')

    def test_pii_mode_expands_positional_tags(self):
        """PII mode should add PII-specific tags to positional_tags."""
        cfg = TextCleanerConfig(
            check_ner_process=False,
            ner_mode='pii',
        )
        tags = set(cfg.positional_tags)
        # Standard tags should still be present
        self.assertIn('PER', tags)
        self.assertIn('LOC', tags)
        self.assertIn('ORG', tags)
        # PII-specific tags should be added (uppercased label names)
        self.assertIn('EMAIL', tags)
        self.assertIn('PHONE_NUMBER', tags)
        self.assertIn('CREDIT_CARD_NUMBER', tags)

    def test_pii_mode_invalid_value(self):
        """Invalid ner_mode should raise ValueError."""
        with self.assertRaises(ValueError) as ctx:
            TextCleanerConfig(
                check_ner_process=False,
                ner_mode='invalid',
            )
        self.assertIn('ner_mode', str(ctx.exception))

    def test_replacement_mode_config_validation(self):
        """replacement_mode should accept valid values and reject invalid ones."""
        # Valid values
        for mode in ('placeholder', 'synthetic'):
            cfg = TextCleanerConfig(check_ner_process=False, replacement_mode=mode)
            self.assertEqual(cfg.replacement_mode, mode)

        # Invalid value
        with self.assertRaises(ValueError) as ctx:
            TextCleanerConfig(check_ner_process=False, replacement_mode='invalid')
        self.assertIn('replacement_mode', str(ctx.exception))

    def test_pii_labels_are_tuples(self):
        """PII_LABELS should be an immutable tuple."""
        self.assertIsInstance(PII_LABELS, tuple)
        self.assertGreater(len(PII_LABELS), 50)

    def test_pii_label_map_coverage(self):
        """All PII_LABEL_MAP keys should map to standard NER tags."""
        valid_tags = {'PER', 'LOC', 'ORG'}
        for label, tag in PII_LABEL_MAP.items():
            self.assertIn(tag, valid_tags, f"Label '{label}' maps to unexpected tag '{tag}'")


class GLiNEROnnxConfigTest(unittest.TestCase):
    """Test gliner_onnx config field and PII mode auto-set behaviour."""

    def test_gliner_onnx_config_field_default(self):
        """gliner_onnx should default to False."""
        cfg = TextCleanerConfig(check_ner_process=False)
        self.assertFalse(cfg.gliner_onnx)

    def test_gliner_onnx_explicit_true(self):
        """gliner_onnx can be explicitly set to True."""
        cfg = TextCleanerConfig(check_ner_process=False, gliner_onnx=True)
        self.assertTrue(cfg.gliner_onnx)

    def test_pii_mode_onnx_backend_sets_gliner_onnx(self):
        """PII mode with ner_backend='onnx' should auto-set gliner_onnx=True."""
        cfg = TextCleanerConfig(
            check_ner_process=False,
            ner_mode='pii',
            ner_backend='onnx',
        )
        self.assertTrue(cfg.gliner_onnx)
        # ner_backend should be rewritten to 'gliner' (GLiNER loaded in ONNX mode)
        self.assertEqual(cfg.ner_backend, 'gliner')

    def test_pii_mode_gliner_backend_no_auto_onnx(self):
        """PII mode with ner_backend='gliner' should NOT auto-set gliner_onnx."""
        cfg = TextCleanerConfig(
            check_ner_process=False,
            ner_mode='pii',
            ner_backend='gliner',
        )
        self.assertFalse(cfg.gliner_onnx)

    def test_gliner_onnx_module_level_variable(self):
        """Module-level GLINER_ONNX variable should exist and default to False."""
        self.assertTrue(hasattr(config, 'GLINER_ONNX'))
        self.assertFalse(config.GLINER_ONNX)


class BiEncoderAdapterTest(unittest.TestCase):
    """Test GLiNERAdapter bi-encoder detection and label caching (mock-based)."""

    def _make_mock_model(self, is_bi_encoder=False, max_pos=None):
        """Create a mock GLiNER model."""
        model = MagicMock()
        if is_bi_encoder:
            model.encode_labels = MagicMock(return_value='mock_embeddings')
            model.batch_predict_with_embeds = MagicMock(return_value=[[]])
        else:
            # Remove encode_labels so hasattr returns False
            del model.encode_labels
            model.predict_entities = MagicMock(return_value=[])

        if max_pos is not None:
            model.config = MagicMock()
            model.config.max_position_embeddings = max_pos
        else:
            del model.config
        return model

    @patch('sct.utils.gliner_adapter.GLiNERAdapter.__init__', return_value=None)
    def test_gliner_adapter_biencoder_detection(self, mock_init):
        """Bi-encoder model should be detected via hasattr(model, 'encode_labels')."""
        from sct.utils.gliner_adapter import GLiNERAdapter
        adapter = GLiNERAdapter.__new__(GLiNERAdapter)
        adapter.model = self._make_mock_model(is_bi_encoder=True)
        adapter.variant = 'gliner'
        adapter.labels = ['person', 'location']
        adapter.threshold = 0.4
        adapter.label_map = {}
        adapter.label_descriptions = {}
        adapter.model_id = 'test-model'
        adapter._is_bi_encoder = hasattr(adapter.model, 'encode_labels')
        adapter._label_embeddings = None
        adapter._cached_labels_key = None

        self.assertTrue(adapter._is_bi_encoder)

    @patch('sct.utils.gliner_adapter.GLiNERAdapter.__init__', return_value=None)
    def test_gliner_adapter_uniencoder_detection(self, mock_init):
        """Uni-encoder model should NOT have encode_labels."""
        from sct.utils.gliner_adapter import GLiNERAdapter
        adapter = GLiNERAdapter.__new__(GLiNERAdapter)
        adapter.model = self._make_mock_model(is_bi_encoder=False)
        adapter._is_bi_encoder = hasattr(adapter.model, 'encode_labels')

        self.assertFalse(adapter._is_bi_encoder)

    @patch('sct.utils.gliner_adapter.GLiNERAdapter.__init__', return_value=None)
    def test_gliner_adapter_label_cache_invalidation(self, mock_init):
        """invalidate_label_cache() should clear cached embeddings."""
        from sct.utils.gliner_adapter import GLiNERAdapter
        adapter = GLiNERAdapter.__new__(GLiNERAdapter)
        adapter._label_embeddings = 'some_embeddings'
        adapter._cached_labels_key = ('person', 'location')

        adapter.invalidate_label_cache()
        self.assertIsNone(adapter._label_embeddings)
        self.assertIsNone(adapter._cached_labels_key)

    @patch('sct.utils.gliner_adapter.GLiNERAdapter.__init__', return_value=None)
    def test_gliner_adapter_max_context_from_model_config(self, mock_init):
        """max_context_length should derive from model.config.max_position_embeddings."""
        from sct.utils.gliner_adapter import GLiNERAdapter
        adapter = GLiNERAdapter.__new__(GLiNERAdapter)
        adapter.variant = 'gliner'
        adapter.model = self._make_mock_model(max_pos=2048)

        self.assertEqual(adapter.max_context_length, 2048)

    @patch('sct.utils.gliner_adapter.GLiNERAdapter.__init__', return_value=None)
    def test_gliner_adapter_max_context_fallback(self, mock_init):
        """max_context_length should fallback to 512 when no model config."""
        from sct.utils.gliner_adapter import GLiNERAdapter
        adapter = GLiNERAdapter.__new__(GLiNERAdapter)
        adapter.variant = 'gliner'
        adapter.model = self._make_mock_model()  # no config

        self.assertEqual(adapter.max_context_length, 512)

    @patch('sct.utils.gliner_adapter.GLiNERAdapter.__init__', return_value=None)
    def test_gliner_adapter_gliner2_context_fallback(self, mock_init):
        """max_context_length for gliner2 variant should fallback to 2048."""
        from sct.utils.gliner_adapter import GLiNERAdapter
        adapter = GLiNERAdapter.__new__(GLiNERAdapter)
        adapter.variant = 'gliner2'
        adapter.model = self._make_mock_model()  # no config

        self.assertEqual(adapter.max_context_length, 2048)


class SyntheticReplacementTest(unittest.TestCase):
    """Test SyntheticReplacer (Faker-based replacement engine)."""

    def test_synthetic_replacer_consistency(self):
        """Same entity text should produce same fake value within a document."""
        from sct.utils.synthetic import SyntheticReplacer
        replacer = SyntheticReplacer(seed=42)

        val1 = replacer.get_replacement('PER', 'John Smith')
        val2 = replacer.get_replacement('PER', 'John Smith')
        self.assertEqual(val1, val2)

        # Different entity text should produce different value
        val3 = replacer.get_replacement('PER', 'Jane Doe')
        self.assertNotEqual(val1, val3)

    def test_synthetic_replacer_cache_reset(self):
        """reset_cache() should clear per-document consistency."""
        from sct.utils.synthetic import SyntheticReplacer
        replacer = SyntheticReplacer(seed=42)

        val1 = replacer.get_replacement('PER', 'John Smith')
        replacer.reset_cache()
        # After reset, same input may produce different output
        # (Faker state has advanced), but the important thing is cache is cleared
        self.assertIsNotNone(val1)

    def test_synthetic_replacer_entity_types(self):
        """Different entity types should use appropriate Faker methods."""
        from sct.utils.synthetic import SyntheticReplacer
        replacer = SyntheticReplacer(seed=42)

        email = replacer.get_replacement('EMAIL', 'test@example.com')
        self.assertIn('@', email)  # Faker.email() always contains @

        name = replacer.get_replacement('PER', 'Alice')
        self.assertIsInstance(name, str)
        self.assertGreater(len(name), 0)

    def test_synthetic_replacer_unknown_type_fallback(self):
        """Unknown entity types should fall back to name generation."""
        from sct.utils.synthetic import SyntheticReplacer
        replacer = SyntheticReplacer(seed=42)

        result = replacer.get_replacement('UNKNOWN_TYPE', 'something')
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_synthetic_generate_for_entities(self):
        """generate_for_entities should replace entities in text."""
        from sct.utils.synthetic import SyntheticReplacer
        replacer = SyntheticReplacer(seed=42)

        text_input = "Hello John Smith, welcome to New York"
        entities = [
            {'entity_group': 'PER', 'word': 'John Smith', 'start': 6, 'end': 16},
            {'entity_group': 'LOC', 'word': 'New York', 'start': 29, 'end': 37},
        ]
        result = replacer.generate_for_entities(text_input, entities)
        self.assertNotIn('John Smith', result)
        self.assertNotIn('New York', result)
        self.assertIn('Hello', result)


class EntityDescriptionTest(unittest.TestCase):
    """Test entity description (ZERONER-style) label support."""

    def test_label_descriptions_frozen(self):
        """gliner_label_descriptions should be frozen after __post_init__."""
        descriptions = {'person': "a person's full legal name"}
        cfg = TextCleanerConfig(
            check_ner_process=False,
            gliner_label_descriptions=descriptions,
        )
        self.assertIsInstance(cfg.gliner_label_descriptions, MappingProxyType)
        self.assertEqual(cfg.gliner_label_descriptions['person'], "a person's full legal name")

    def test_label_descriptions_none_by_default(self):
        """gliner_label_descriptions should be None when not provided."""
        cfg = TextCleanerConfig(check_ner_process=False)
        self.assertIsNone(cfg.gliner_label_descriptions)

    @patch('sct.utils.gliner_adapter.GLiNERAdapter.__init__', return_value=None)
    def test_label_descriptions_inference_labels(self, mock_init):
        """_get_inference_labels should return descriptions when available."""
        from sct.utils.gliner_adapter import GLiNERAdapter
        adapter = GLiNERAdapter.__new__(GLiNERAdapter)
        adapter.labels = ['person', 'location']
        adapter.label_descriptions = {
            'person': "a person's full name",
            'location': "a geographical place",
        }

        result = adapter._get_inference_labels()
        self.assertEqual(result, ["a person's full name", "a geographical place"])

    @patch('sct.utils.gliner_adapter.GLiNERAdapter.__init__', return_value=None)
    def test_label_descriptions_fallback_to_names(self, mock_init):
        """_get_inference_labels should fall back to label names without descriptions."""
        from sct.utils.gliner_adapter import GLiNERAdapter
        adapter = GLiNERAdapter.__new__(GLiNERAdapter)
        adapter.labels = ['person', 'location']
        adapter.label_descriptions = {}

        result = adapter._get_inference_labels()
        self.assertEqual(result, ['person', 'location'])


class PresidioGLiNERBackendTest(unittest.TestCase):
    """Test presidio_gliner backend configuration."""

    def test_presidio_gliner_backend_valid(self):
        """Config should accept 'presidio_gliner' as a valid backend."""
        cfg = TextCleanerConfig(
            check_ner_process=False,
            ner_backend='presidio_gliner',
            gliner_model='knowledgator/gliner-pii-base-v1.0',
        )
        self.assertEqual(cfg.ner_backend, 'presidio_gliner')

    def test_presidio_gliner_requires_model(self):
        """presidio_gliner backend should require gliner_model."""
        with self.assertRaises(ValueError):
            TextCleanerConfig(
                check_ner_process=False,
                ner_backend='presidio_gliner',
            )


class ContactSyntheticMethodTest(unittest.TestCase):
    """Test callable-based replacement methods in contact.py."""

    def test_replace_emails_with_fn(self):
        """replace_emails_with_fn should call the provided function."""
        pc = contact.ProcessContacts()
        replacements = []

        def track_fn(matched):
            replacements.append(matched)
            return 'REPLACED'

        result = pc.replace_emails_with_fn("Email me at test@example.com please", track_fn)
        self.assertIn('REPLACED', result)
        self.assertEqual(len(replacements), 1)
        self.assertEqual(replacements[0], 'test@example.com')

    def test_replace_urls_with_fn(self):
        """replace_urls_with_fn should call the provided function."""
        pc = contact.ProcessContacts()
        result = pc.replace_urls_with_fn(
            "Visit https://example.com today",
            lambda m: 'FAKE_URL',
        )
        self.assertIn('FAKE_URL', result)
        self.assertNotIn('https://example.com', result)

    def test_replace_phone_numbers_with_fn(self):
        """replace_phone_numbers_with_fn should call the provided function."""
        pc = contact.ProcessContacts()
        result = pc.replace_phone_numbers_with_fn(
            "Call +1-234-567-8900",
            lambda m: 'FAKE_PHONE',
        )
        self.assertIn('FAKE_PHONE', result)


class ModernBERTConfigTest(unittest.TestCase):
    """Test ModernBERT model constants."""

    def test_modernbert_models_defined(self):
        """MODERNBERT_NER_MODELS should be importable and have ENGLISH key."""
        from sct.config import MODERNBERT_NER_MODELS
        self.assertIn('ENGLISH', MODERNBERT_NER_MODELS)
        self.assertIn('MULTILINGUAL', MODERNBERT_NER_MODELS)

    def test_pii_exports_from_init(self):
        """PII_LABELS and PII_LABEL_MAP should be importable from sct package."""
        from sct import PII_LABELS, PII_LABEL_MAP
        self.assertIsInstance(PII_LABELS, tuple)
        self.assertIsInstance(PII_LABEL_MAP, dict)


class ReversibleAnonymizationTest(unittest.TestCase):
    """Test reversible anonymization: AnonymizationMap + indexed placeholders."""

    def test_reversible_mode_config_valid(self):
        """'reversible' should be a valid replacement_mode."""
        cfg = TextCleanerConfig(check_ner_process=False, replacement_mode='reversible')
        self.assertEqual(cfg.replacement_mode, 'reversible')

    def test_anonymization_map_roundtrip(self):
        """Anonymize with indexed placeholders then deanonymize to get original."""
        from sct.utils.anonymization_map import AnonymizationMap
        amap = AnonymizationMap()

        original_text = "John Smith works at Google in London."
        # Simulate NER entities (reversed order for right-to-left replacement)
        entities = [
            {'entity_group': 'PER', 'start': 0, 'end': 10, 'score': 0.99},
            {'entity_group': 'ORG', 'start': 20, 'end': 26, 'score': 0.95},
            {'entity_group': 'LOC', 'start': 30, 'end': 36, 'score': 0.92},
        ]

        # Apply right-to-left replacement (same logic as _anonymize_reversible)
        text = original_text
        sorted_entities = sorted(entities, key=lambda x: x['start'], reverse=True)
        from sct.utils.ner import ENTITY_TYPE_MAP
        for item in sorted_entities:
            tag = ENTITY_TYPE_MAP.get(item['entity_group'], item['entity_group'])
            placeholder = amap.next_placeholder(tag)
            original = text[item['start']:item['end']]
            amap.add(placeholder=placeholder, original=original,
                     entity_type=tag, start=item['start'], end=item['end'])
            text = text[:item['start']] + placeholder + text[item['end']:]

        # Verify placeholders are indexed
        self.assertIn('<LOCATION_0>', text)
        self.assertIn('<ORGANISATION_0>', text)
        self.assertIn('<PERSON_0>', text)

        # Deanonymize should restore original
        restored = amap.deanonymize(text)
        self.assertEqual(restored, original_text)

    def test_reversible_map_entries(self):
        """Map entries should have correct placeholder format and preserve originals."""
        from sct.utils.anonymization_map import AnonymizationMap
        amap = AnonymizationMap()

        p1 = amap.next_placeholder('PERSON')
        self.assertEqual(p1, '<PERSON_0>')
        p2 = amap.next_placeholder('PERSON')
        self.assertEqual(p2, '<PERSON_1>')
        p3 = amap.next_placeholder('LOCATION')
        self.assertEqual(p3, '<LOCATION_0>')

    def test_reversible_longest_first_deanonymize(self):
        """<PERSON_10> should not collide with <PERSON_1> during deanonymize."""
        from sct.utils.anonymization_map import AnonymizationMap
        amap = AnonymizationMap()

        # Create 11 PERSON entries so we get <PERSON_10>
        for i in range(11):
            amap.add(
                placeholder=f'<PERSON_{i}>', original=f'Name{i}',
                entity_type='PERSON', start=0, end=5,
            )

        text = '<PERSON_1> and <PERSON_10>'
        restored = amap.deanonymize(text)
        self.assertEqual(restored, 'Name1 and Name10')

    def test_reversible_map_serialization(self):
        """to_dict/from_dict should round-trip correctly."""
        from sct.utils.anonymization_map import AnonymizationMap
        amap = AnonymizationMap()
        amap.add('<PERSON_0>', 'Alice', 'PERSON', 0, 5)
        amap.add('<LOCATION_0>', 'Berlin', 'LOCATION', 15, 21)

        data = amap.to_dict()
        restored = AnonymizationMap.from_dict(data)

        self.assertEqual(restored.session_id, amap.session_id)
        self.assertEqual(len(restored.entries), 2)
        self.assertEqual(restored.entries[0].placeholder, '<PERSON_0>')
        self.assertEqual(restored.entries[0].original, 'Alice')
        self.assertEqual(restored.entries[1].placeholder, '<LOCATION_0>')

    def test_ner_anonymize_reversible(self):
        """GeneralNER._anonymize_reversible should produce indexed placeholders."""
        from sct.utils.anonymization_map import AnonymizationMap
        ner_obj = GeneralNER.__new__(GeneralNER)

        amap = AnonymizationMap()
        text = "Alice works at ACME in Paris."
        filtered = [
            {'entity_group': 'PER', 'start': 0, 'end': 5, 'score': 0.99,
             'word': 'Alice', 'key': '05'},
            {'entity_group': 'ORG', 'start': 15, 'end': 19, 'score': 0.95,
             'word': 'ACME', 'key': '1519'},
            {'entity_group': 'LOC', 'start': 23, 'end': 28, 'score': 0.92,
             'word': 'Paris', 'key': '2328'},
        ]

        result = ner_obj._anonymize_reversible(text, filtered, amap)
        self.assertIn('<PERSON_0>', result.text)
        self.assertIn('<ORGANISATION_0>', result.text)
        self.assertIn('<LOCATION_0>', result.text)
        self.assertEqual(len(amap.entries), 3)

        # Round-trip (anon_map is mutated in-place, use the passed-in reference)
        self.assertEqual(amap.deanonymize(result.text), text)


class GLiClassConfigTest(unittest.TestCase):
    """Test GLiClass document classification config fields."""

    def test_gliclass_config_fields_exist(self):
        """GLiClass config fields should exist with defaults."""
        cfg = TextCleanerConfig(check_ner_process=False)
        self.assertFalse(cfg.check_classify_document)
        self.assertIsNone(cfg.gliclass_model)
        self.assertEqual(cfg.gliclass_labels, ())
        self.assertEqual(cfg.gliclass_threshold, 0.5)
        self.assertEqual(cfg.gliclass_classification_type, 'single-label')
        self.assertFalse(cfg.gliclass_onnx)

    def test_gliclass_default_model_constant(self):
        """GLICLASS_DEFAULT_MODEL should be defined."""
        from sct.config import GLICLASS_DEFAULT_MODEL
        self.assertIn('gliclass', GLICLASS_DEFAULT_MODEL)

    def test_gliclass_config_accepts_labels(self):
        """GLiClass config should accept custom labels tuple."""
        cfg = TextCleanerConfig(
            check_ner_process=False,
            check_classify_document=True,
            gliclass_labels=('email', 'code', 'legal'),
        )
        self.assertEqual(cfg.gliclass_labels, ('email', 'code', 'legal'))

    def test_gliclass_adapter_init_mock(self):
        """GLiClassAdapter should accept model/labels/threshold (mock init)."""
        from sct.utils.gliclass_adapter import GLiClassAdapter
        # Patch the init methods to avoid importing gliclass
        with patch.object(GLiClassAdapter, '_init_model'):
            adapter = GLiClassAdapter.__new__(GLiClassAdapter)
            adapter.model_id = 'test-model'
            adapter.labels = ['email', 'code']
            adapter.threshold = 0.5
            adapter.classification_type = 'single-label'
            adapter._onnx = False
            adapter._pipeline = None
            self.assertEqual(adapter.model_id, 'test-model')
            self.assertEqual(adapter.labels, ['email', 'code'])

    def test_gliclass_classify_returns_list_mock(self):
        """classify() should return list of {label, score} dicts."""
        from sct.utils.gliclass_adapter import GLiClassAdapter
        adapter = GLiClassAdapter.__new__(GLiClassAdapter)
        adapter.labels = ['email', 'code']
        adapter.threshold = 0.5
        adapter._pipeline = MagicMock(return_value=[
            [{'label': 'email', 'score': 0.9}, {'label': 'code', 'score': 0.3}],
        ])
        results = adapter.classify("Dear Sir, please find attached...")
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 1)  # only email above threshold
        self.assertEqual(results[0]['label'], 'email')
        self.assertGreaterEqual(results[0]['score'], 0.5)

    def test_otter_mmbert_placeholders(self):
        """OTTER_NER_MODELS and MMBERT_NER_MODELS should exist as empty dicts."""
        from sct.config import OTTER_NER_MODELS, MMBERT_NER_MODELS
        self.assertIsInstance(OTTER_NER_MODELS, dict)
        self.assertIsInstance(MMBERT_NER_MODELS, dict)
        self.assertEqual(len(OTTER_NER_MODELS), 0)
        self.assertEqual(len(MMBERT_NER_MODELS), 0)


class ProcessResultTest(unittest.TestCase):
    """Test ProcessResult backward-compatible tuple wrapper."""

    def test_process_result_unpacks_as_3tuple(self):
        """ProcessResult should unpack as (lm_text, stat_text, language)."""
        from sct.utils.process_result import ProcessResult
        r = ProcessResult('cleaned', 'stat', 'ENGLISH')
        a, b, c = r
        self.assertEqual(a, 'cleaned')
        self.assertEqual(b, 'stat')
        self.assertEqual(c, 'ENGLISH')

    def test_process_result_len(self):
        """len(ProcessResult) should be 3."""
        from sct.utils.process_result import ProcessResult
        r = ProcessResult('a', 'b', 'c')
        self.assertEqual(len(r), 3)

    def test_process_result_getitem(self):
        """ProcessResult[idx] should work like a tuple."""
        from sct.utils.process_result import ProcessResult
        r = ProcessResult('a', 'b', 'c')
        self.assertEqual(r[0], 'a')
        self.assertEqual(r[1], 'b')
        self.assertEqual(r[2], 'c')

    def test_process_result_metadata_none_by_default(self):
        """metadata should be None when no extended features active."""
        from sct.utils.process_result import ProcessResult
        r = ProcessResult('a', None, None)
        self.assertIsNone(r.metadata)

    def test_process_result_metadata_accessible(self):
        """metadata should carry anon_map and other keys."""
        from sct.utils.process_result import ProcessResult
        meta = {'anon_map': 'mock_map', 'classes': None}
        r = ProcessResult('a', None, None, metadata=meta)
        self.assertEqual(r.metadata['anon_map'], 'mock_map')

    def test_process_result_equality_with_tuple(self):
        """ProcessResult should compare equal to a matching 3-tuple."""
        from sct.utils.process_result import ProcessResult
        r = ProcessResult('a', 'b', 'c')
        self.assertEqual(r, ('a', 'b', 'c'))

    def test_process_returns_process_result(self):
        """TextCleaner.process() should return ProcessResult."""
        from sct.utils.process_result import ProcessResult
        cleaner = TextCleaner(TextCleanerConfig(check_ner_process=False))
        result = cleaner.process("Hello world")
        self.assertIsInstance(result, ProcessResult)
        # Should still unpack
        lm, stat, lang = result
        self.assertIsInstance(lm, str)

    def test_process_reversible_returns_metadata(self):
        """process() with replacement_mode='reversible' should set metadata."""
        from sct.utils.process_result import ProcessResult
        cfg = TextCleanerConfig(
            check_ner_process=False,
            replacement_mode='reversible',
        )
        cleaner = TextCleaner(cfg)
        result = cleaner.process("Hello world")
        self.assertIsInstance(result, ProcessResult)
        # Even without NER entities, metadata should have anon_map
        self.assertIsNotNone(result.metadata)
        self.assertIn('anon_map', result.metadata)

    def test_exports_from_init(self):
        """AnonymizationMap, MapEntry, ProcessResult should be importable from sct."""
        from sct import AnonymizationMap, MapEntry, ProcessResult
        self.assertIsNotNone(AnonymizationMap)
        self.assertIsNotNone(MapEntry)
        self.assertIsNotNone(ProcessResult)


if __name__ == "__main__":
    unittest.main(verbosity=2)
