from sct.utils import constants
from bs4 import BeautifulSoup

class ProcessContacts:
    
    def __init__(self):
        pass
    
    def replace_urls(self, text, replace_with="<URL>"):
        """
        Replace all URLs in ``text`` str with ``replace_with`` str.
        """
        # matches = constants.URL_REGEX.finditer(text)
        # result = text
        # # Iterate through matches in reverse order (to avoid index issues)
        # for match in reversed(list(matches)):
        # # Check if the matched substring contains non-ASCII characters
        #     if not any(ord(char) > 127 for char in match.group()):
        #         result = text[:match.start()] + replace_with + text[match.end():]
        return constants.URL_REGEX.sub(replace_with, text)

    def replace_html(self, text, replace_with="<HTML>"):
        """
        Replace all html tags in ``text`` str with ``replace_with`` str.
        """
        try:
            soup = BeautifulSoup(text, 'html.parser')
            text = soup.get_text()
        except:
            text = constants.HTML_REGEX.sub(replace_with, text)
        
        return text
    
    def replace_emails(self, text, replace_with="<EMAIL>"):
        """
        Replace all emails in ``text`` str with ``replace_with`` str.
        """
        return constants.EMAIL_REGEX.sub(replace_with, text)

    def replace_phone_numbers(self, text, replace_with="<PHONE>"):
        """
        Replace all phone numbers in ``text`` str with ``replace_with`` str.
        """
        return constants.PHONE_REGEX.sub(replace_with, text)

    def replace_numbers(self, text, replace_with="<NUMBER>"):
        """
        Replace all numbers in ``text`` str with ``replace_with`` str.
        """
        return constants.NUMBERS_REGEX.sub(replace_with, text)