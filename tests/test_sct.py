import unittest
import random
import string
from hypothesis import given, settings
from hypothesis.strategies import text, from_regex
from faker import Faker
from sct import config
from sct.utils import contact, datetime, special, normtext, stopwords, constants

class TextCleanerTest(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        config.CHECK_NER_PROCESS = False
        cls.ProcessContacts = contact.ProcessContacts()
        cls.ProcessDateTime = datetime.ProcessDateTime()
        cls.ProcessSpecialSymbols = special.ProcessSpecialSymbols()
        cls.NormaliseText = normtext.NormaliseText()
        cls.ProcessStopwords = stopwords.ProcessStopwords()
        cls.fake = Faker()

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
        
if __name__ == "__main__":
    unittest.main(verbosity=2)