import unittest
import random
import string
from hypothesis import given, settings
from hypothesis.strategies import text, from_regex
from faker import Faker
from sct import config
from sct.utils import contact, datetime, special, normtext, stopwords, constants
from sct.utils.ner import GeneralNER
import torch
from unittest.mock import patch
from functools import wraps
import timeout_decorator
import signal
from contextlib import contextmanager
from sct.sct import TextCleaner
from sct.utils import ner
import os

def requires_ner(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if os.getenv('GITHUB_ACTIONS'):
            self.skipTest("Skipping NER tests in GitHub Actions")
        if not hasattr(self.ner, 'en_model'):
            self.skipTest("NER models not properly loaded")
        return func(self, *args, **kwargs)
    return wrapper

class TimeoutException(Exception):
    pass

@contextmanager
def timeout(seconds):
    def timeout_handler(signum, frame):
        raise TimeoutException("Test timed out")
    
    # Set the timeout handler
    original_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, original_handler)

# Add this test text constant near the top of the file
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
        if os.getenv('GITHUB_ACTIONS'):
            cls.ner = None
            return
            
        try:
            with timeout(1200):  # 20 minute timeout
                config.CHECK_NER_PROCESS = False
                # Initialize all the processing classes
                cls.ProcessContacts = contact.ProcessContacts()
                cls.ProcessDateTime = datetime.ProcessDateTime()
                cls.ProcessSpecialSymbols = special.ProcessSpecialSymbols()
                cls.NormaliseText = normtext.NormaliseText()
                cls.ProcessStopwords = stopwords.ProcessStopwords()
                cls.fake = Faker()  # Initialize Faker
                
                # Override default models with smaller model for testing
                test_models = ["dslim/bert-base-NER"] * 5  # Same small model for all languages
                
                print("Downloading and loading NER models... This may take a few minutes...")
                cls.ner = GeneralNER(device='cpu')  # Force CPU usage for tests
                cls.ner.DEFAULT_MODELS = test_models  # Override default models
                
                # Test model loading with a simple inference
                try:
                    cls.ner.ner_process(
                        "Test sentence.", 
                        positional_tags=['PER', 'ORG', 'LOC'],
                        ner_confidence_threshold=0.85
                    )
                    print("NER models loaded successfully!")
                except Exception as e:
                    print(f"Failed to validate NER models: {e}")
                    cls.ner = None
                    raise
        except TimeoutException:
            print("Setup timed out after 20 minutes")
            raise
        except Exception as e:
            print(f"Setup failed: {e}")
            raise

    def setUp(self):
        """Set up test fixtures before each test method."""
        config.CHECK_NER_PROCESS = True
        # Copy class-level attributes to instance level
        self.ProcessContacts = self.__class__.ProcessContacts
        self.ProcessDateTime = self.__class__.ProcessDateTime
        self.ProcessSpecialSymbols = self.__class__.ProcessSpecialSymbols
        self.NormaliseText = self.__class__.NormaliseText
        self.ProcessStopwords = self.__class__.ProcessStopwords
        self.fake = self.__class__.fake
        self.ner = self.__class__.ner

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
        self.assertEqual("", self.ProcessSpecialSymbols.remove_isolated_special_symbols(rx))

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
        
    @requires_ner
    def test_ner_process_basic(self):
        """Test basic NER processing with known entities."""
        text = "John Smith works at Microsoft in New York."
        processed = self.ner.ner_process(
            text,
            positional_tags=['PER', 'ORG', 'LOC'],
            ner_confidence_threshold=0.85
        )
        
        # Check that entities are replaced
        self.assertNotIn("John Smith", processed)
        self.assertNotIn("Microsoft", processed)
        self.assertNotIn("New York", processed)
        
    @requires_ner
    def test_ner_process_empty(self):
        """Test NER processing with empty input."""
        with self.assertRaises(ValueError):
            self.ner.ner_process("")
            
    @requires_ner
    def test_ner_process_no_entities(self):
        """Test NER processing with text containing no entities."""
        text = "The quick brown fox jumps over the lazy dog."
        processed = self.ner.ner_process(
            text,
            positional_tags=['PER', 'ORG', 'LOC'],
            ner_confidence_threshold=0.85
        )
        self.assertEqual(text, processed)
        
    @requires_ner
    def test_ner_batch_processing(self):
        """Test batch processing of multiple texts."""
        texts = [
            "John works at Microsoft.",
            "Sarah lives in London.",
            "The quick brown fox."
        ]
        
        results = self.ner.process_batch(
            texts, 
            batch_size=2,
            positional_tags=['PER', 'ORG', 'LOC'],
            ner_confidence_threshold=0.85
        )
        
        self.assertEqual(len(results), len(texts))
        # Check first text has entities replaced
        self.assertNotIn("John", results[0])
        self.assertNotIn("Microsoft", results[0])
        # Check second text has entities replaced
        self.assertNotIn("Sarah", results[1])
        self.assertNotIn("London", results[1])
        # Check third text remains unchanged
        self.assertEqual(results[2], texts[2])
        
    @requires_ner
    def test_ner_process_multilingual(self):
        """Test NER processing with different languages."""
        tests = {
            "GERMAN": "Angela Merkel lebt in Berlin.",
            "SPANISH": "Pablo vive en Madrid.",
            "DUTCH": "Willem woont in Amsterdam."
        }
        
        for lang, text in tests.items():
            processed = self.ner.ner_process(
                text,
                positional_tags=['PER', 'LOC'],
                ner_confidence_threshold=0.85,
                language=lang
            )
            # Verify that known entities are replaced
            if lang == "GERMAN":
                self.assertNotIn("Angela Merkel", processed)
                self.assertNotIn("Berlin", processed)
            elif lang == "SPANISH":
                self.assertNotIn("Pablo", processed)
                self.assertNotIn("Madrid", processed)
            elif lang == "DUTCH":
                self.assertNotIn("Willem", processed)
                self.assertNotIn("Amsterdam", processed)
                
    @requires_ner
    def test_ner_long_text_splitting(self):
        """Test NER processing with text that needs to be split."""
        # Create a long text that exceeds token limit
        long_text = "John Smith " * 512  # Should exceed token limit
        
        processed = self.ner.ner_process(
            long_text,
            positional_tags=['PER'],
            ner_confidence_threshold=0.85
        )
        
        self.assertNotIn("John Smith", processed)
        
    @requires_ner
    def test_ner_process_threshold(self):
        """Test NER processing with different confidence thresholds."""
        text = "Dr. White from TechCorp visited Washington."
        
        # Test that both high and low thresholds work
        high_conf = self.ner.ner_process(
            text,
            positional_tags=['PER', 'ORG', 'LOC'],
            ner_confidence_threshold=0.95
        )
        
        low_conf = self.ner.ner_process(
            text,
            positional_tags=['PER', 'ORG', 'LOC'],
            ner_confidence_threshold=0.3
        )
        
        # Verify that entities are detected regardless of threshold
        self.assertNotIn("White", high_conf)
        self.assertNotIn("TechCorp", high_conf)
        self.assertNotIn("Washington", high_conf)
        
        # Log for debugging
        print(f"\nHigh confidence output: {high_conf}")
        print(f"Low confidence output: {low_conf}")

    @requires_ner
    def test_ner_batch_processing_empty(self):
        """Test batch processing with empty input."""
        self.assertEqual(self.ner.process_batch([]), [])

    @requires_ner
    def test_ner_batch_processing_invalid(self):
        """Test batch processing with invalid inputs."""
        with self.assertRaises(ValueError):
            self.ner.process_batch([None, "", "valid text"])

    @requires_ner
    def test_ner_batch_processing_large(self):
        """Test batch processing with larger batches."""
        texts = [
            f"John Smith works at Microsoft {i}" for i in range(20)
        ]
        
        # Test with different batch sizes
        for batch_size in [1, 5, 10, 20]:
            results = self.ner.process_batch(
                texts,
                batch_size=batch_size,
                positional_tags=['PER', 'ORG'],
                ner_confidence_threshold=0.85
            )
            
            self.assertEqual(len(results), len(texts))
            for result in results:
                self.assertNotIn("John Smith", result)
                self.assertNotIn("Microsoft", result)

    @requires_ner
    def test_ner_memory_management(self):
        """Test memory management during NER processing."""
        # Process a large batch of text
        texts = ["John Smith works at Microsoft"] * 100
        _ = self.ner.process_batch(
            texts, 
            batch_size=10,
            positional_tags=['PER', 'ORG', 'LOC'],  # Add required positional tags
            ner_confidence_threshold=0.85
        )
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # If CUDA is available, test CUDA memory management
        if torch.cuda.is_available():
            initial_memory = torch.cuda.memory_allocated()
            torch.cuda.empty_cache()
            final_memory = torch.cuda.memory_allocated()
            self.assertLess(final_memory - initial_memory, 1024 * 1024 * 100)  # 100MB difference max

    @classmethod
    def tearDownClass(cls):
        # Clean up memory
        if hasattr(cls, 'ner'):
            del cls.ner
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()

    @requires_ner
    def test_end_to_end_text_cleaning(self):
        """Test end-to-end text cleaning following the exact pipeline sequence in sct.py."""
        # Override NER models with smaller model for testing
        test_models = ["dslim/bert-base-NER"] * 5
        
        # Initialize cleaner with all features enabled
        sx = TextCleaner()
        sx.GeneralNER = ner.GeneralNER(device='cpu')
        sx.GeneralNER.DEFAULT_MODELS = test_models
        sx.GeneralNER._load_models(test_models, {})
        
        # Process text and get all outputs
        lm_text, stat_text, lang = sx.process(TEST_TEXT)
        
        print("\nProcessing Results:")
        print(f"Language Model Text: {lm_text}")
        print(f"Statistical Text: {stat_text}")
        print(f"Detected Language: {lang}")
        
        # Test pipeline steps in sequence as defined in sct.py
        
        # 1. Language Detection
        self.assertEqual(lang, "ENGLISH")
        
        # 2. Unicode Handling
        self.assertNotIn("&amp;", lm_text)  # fix_bad_unicode
        self.assertNotIn("©", lm_text)      # to_ascii_unicode
        
        # 3. Basic Replacements
        self.assertNotIn("<div", lm_text)        # replace_html
        self.assertIn("<URL>", lm_text)          # replace_urls
        self.assertIn("<EMAIL>", lm_text)        # replace_emails
        self.assertIn("<YEAR>", lm_text)         # replace_years
        self.assertIn("<PHONE>", lm_text)        # replace_phone_numbers
        self.assertIn("<NUMBER>", lm_text)       # replace_numbers
        self.assertNotIn("$", lm_text)           # replace_currency_symbols
        
        # 4. NER Processing
        self.assertIn("<PERSON>", lm_text)       # Dr. John Smith
        self.assertIn("<ORGANISATION>", lm_text)  # Microsoft
        self.assertIn("<LOCATION>", lm_text)     # New York
        
        # 5. Cleanup
        self.assertNotIn("  ", lm_text)          # normalize_whitespace
        self.assertNotIn("l e t t e r s", lm_text)  # remove_isolated_letters
        
        # 6. Statistical Processing
        self.assertTrue(stat_text.islower())      # casefold
        self.assertNotIn("!", stat_text)         # remove_punctuation
        self.assertNotIn("the", stat_text)       # remove_stopwords
        self.assertNotIn("  ", stat_text)        # normalize_whitespace
        
        # Test that original entities are properly replaced
        self.assertNotIn("John Smith", lm_text)
        self.assertNotIn("Microsoft", lm_text)
        self.assertNotIn("New York", lm_text)
        self.assertNotIn("john.doe@example.com", lm_text)
        self.assertNotIn("https://example.com", lm_text)
        self.assertNotIn("+1-234-567-8900", lm_text)

    @requires_ner
    def test_batch_processing_basic(self):
        """Test basic batch processing functionality."""
        # Override NER models with smaller model for testing
        test_models = ["dslim/bert-base-NER"] * 5
        
        # Initialize cleaner with test model
        sx = TextCleaner()
        sx.GeneralNER = ner.GeneralNER(device='cpu')
        sx.GeneralNER.DEFAULT_MODELS = test_models
        sx.GeneralNER._load_models(test_models, {})
        
        texts = [
            "John Doe works at Apple Inc.",
            "Visit Microsoft in Seattle",
            "The quick brown fox"  # No entities
        ]
        
        results = sx.process_batch(texts, batch_size=2)
        
        # Check we get all results
        self.assertEqual(len(results), len(texts))
        
        # Check each result is a tuple of (lm_text, stat_text, language)
        for result in results:
            self.assertEqual(len(result), 3)
            lm_text, stat_text, lang = result
            self.assertIsInstance(lm_text, str)
            self.assertIsInstance(stat_text, str)
            self.assertIsInstance(lang, str)

    @requires_ner
    def test_batch_processing_empty(self):
        """Test batch processing with empty inputs."""
        sx = TextCleaner()
        
        # Empty list
        results = sx.process_batch([])
        self.assertEqual(results, [])
        
        # List with empty string
        results = sx.process_batch([""])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], ("", "", None))

    @requires_ner
    def test_batch_processing_large(self):
        """Test batch processing with different batch sizes."""
        # Override NER models with smaller model for testing
        test_models = ["dslim/bert-base-NER"] * 5
        
        # Initialize cleaner with test model
        sx = TextCleaner()
        sx.GeneralNER = ner.GeneralNER(device='cpu')
        sx.GeneralNER.DEFAULT_MODELS = test_models
        sx.GeneralNER._load_models(test_models, {})
        
        texts = ["John Smith lives in New York"] * 10
        
        # Process with different batch sizes
        results1 = sx.process_batch(texts, batch_size=2)
        results2 = sx.process_batch(texts, batch_size=5)
        
        # Results should be identical regardless of batch size
        self.assertEqual(len(results1), len(results2))
        self.assertEqual(results1, results2)

    @requires_ner
    def test_batch_processing_memory(self):
        """Test memory usage during batch processing."""
        sx = TextCleaner()
        texts = ["John Smith works at Microsoft"] * 100
        
        # Record initial memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()
        
        # Process large batch
        _ = sx.process_batch(texts, batch_size=10)
        
        # Check memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            final_memory = torch.cuda.memory_allocated()
            # Memory difference should be minimal after cleanup
            self.assertLess(final_memory - initial_memory, 1024 * 1024 * 100)  # 100MB threshold

    @requires_ner
    def test_batch_processing_languages(self):
        """Test batch processing with multiple languages."""
        # Override NER models with smaller model for testing
        test_models = ["dslim/bert-base-NER"] * 5
        
        # Initialize cleaner with test model
        sx = TextCleaner()
        sx.GeneralNER = ner.GeneralNER(device='cpu')
        sx.GeneralNER.DEFAULT_MODELS = test_models
        sx.GeneralNER._load_models(test_models, {})
        
        texts = [
            "John Smith lives in London",  # English
            "Angela Merkel lebt in Berlin",  # German
            "Pablo vive en Madrid",  # Spanish
            "Willem woont in Amsterdam"  # Dutch
        ]
        
        results = sx.process_batch(texts, batch_size=2)
        
        # Check language detection
        languages = [lang for _, _, lang in results]
        self.assertIn("ENGLISH", languages)
        self.assertIn("GERMAN", languages)
        self.assertIn("SPANISH", languages)
        self.assertIn("DUTCH", languages)
        
        # Check entity replacement in each language
        for lm_text, _, _ in results:
            self.assertNotIn("John Smith", lm_text)
            self.assertNotIn("Angela Merkel", lm_text)
            self.assertNotIn("Pablo", lm_text)
            self.assertNotIn("Willem", lm_text)
            self.assertNotIn("London", lm_text)
            self.assertNotIn("Berlin", lm_text)
            self.assertNotIn("Madrid", lm_text)
            self.assertNotIn("Amsterdam", lm_text)

if __name__ == "__main__":
    unittest.main(verbosity=2)