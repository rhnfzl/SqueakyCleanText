import logging
from sct.utils import constants
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

class ProcessContacts:

    def __init__(self):
        pass

    def replace_urls(self, text, replace_with="<URL>"):
        """Replace all URLs in ``text`` str with ``replace_with`` str."""
        return constants.URL_REGEX.sub(replace_with, text)

    def replace_html(self, text, replace_with="<HTML>"):
        """Replace all html tags in ``text`` str with ``replace_with`` str."""
        try:
            soup = BeautifulSoup(text, 'html.parser')
            text = soup.get_text()
        except Exception:
            logger.debug("HTML parsing failed, falling back to regex", exc_info=True)
            text = constants.HTML_REGEX.sub(replace_with, text)
        return text

    def replace_emails(self, text, replace_with="<EMAIL>"):
        """Replace all emails in ``text`` str with ``replace_with`` str."""
        return constants.EMAIL_REGEX.sub(replace_with, text)

    def replace_phone_numbers(self, text, replace_with="<PHONE>"):
        """Replace all phone numbers in ``text`` str with ``replace_with`` str."""
        return constants.PHONE_REGEX.sub(replace_with, text)

    def replace_numbers(self, text, replace_with="<NUMBER>"):
        """Replace all numbers in ``text`` str with ``replace_with`` str."""
        return constants.NUMBERS_REGEX.sub(replace_with, text)

    # --- Callable-based replacements (for synthetic mode) ---

    def replace_emails_with_fn(self, text, replace_fn):
        """Replace emails using a callable: replace_fn(matched_text) -> replacement."""
        return constants.EMAIL_REGEX.sub(lambda m: replace_fn(m.group()), text)

    def replace_urls_with_fn(self, text, replace_fn):
        """Replace URLs using a callable: replace_fn(matched_text) -> replacement."""
        return constants.URL_REGEX.sub(lambda m: replace_fn(m.group()), text)

    def replace_phone_numbers_with_fn(self, text, replace_fn):
        """Replace phone numbers using a callable: replace_fn(matched_text) -> replacement."""
        return constants.PHONE_REGEX.sub(lambda m: replace_fn(m.group()), text)

    def replace_numbers_with_fn(self, text, replace_fn):
        """Replace numbers using a callable: replace_fn(matched_text) -> replacement."""
        return constants.NUMBERS_REGEX.sub(lambda m: replace_fn(m.group()), text)
