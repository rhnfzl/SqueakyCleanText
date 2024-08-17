from sct.utils import constants
from bs4 import BeautifulSoup

class ProcessContacts:
    
    def __init__(self):
        pass
    
    def replace_urls(self, input_text: str, replacement: str = "<URL>") -> str:
        """
        Replace all URLs in the given input text with the specified replacement string.

        Args:
            input_text (str): The input text to be processed.
            replacement (str, optional): The string to replace URLs with. Defaults to "<URL>".

        Returns:
            str: The processed text with all URLs replaced.
        """
        return constants.URL_REGEX.sub(replacement, input_text)

    def replace_html(self, text: str, replacement: str = "<HTML>") -> str:
        """
        Replaces all HTML tags in the given text with the specified replacement string.

        Args:
            text (str): The input text to be processed.
            replacement (str, optional): The string to replace HTML tags with. Defaults to "<HTML>".

        Returns:
            str: The processed text with all HTML tags replaced.
        """
        try:
            # Use BeautifulSoup to parse the HTML and extract the text
            cleaned_text = BeautifulSoup(text, 'html.parser').get_text()
        except:
            # If BeautifulSoup fails, use the HTML_REGEX to remove HTML tags
            cleaned_text = constants.HTML_REGEX.sub(replacement, text)
        
        return cleaned_text
    
    def replace_emails(self, input_text: str, replacement: str = "<EMAIL>") -> str:
        """
        Replace all email addresses in the input text with the specified replacement string.

        Args:
            input_text (str): The text to be processed.
            replacement (str, optional): The string to replace email addresses with. Defaults to "<EMAIL>".

        Returns:
            str: The processed text with all email addresses replaced.
        """
        # Use the pre-compiled regular expression to find all email patterns in the text
        # and replace them with the specified replacement string
        return constants.EMAIL_REGEX.sub(replacement, input_text)

    def replace_phone_numbers(self, input_text: str, replacement: str = "<PHONE>") -> str:
        """
        Replaces all occurrences of phone numbers in the input text with a
        specified replacement string.

        Args:
            input_text (str): The text to be processed.
            replacement (str, optional): The string to replace the phone numbers with.
                Defaults to "<PHONE>".

        Returns:
            str: The modified text with phone numbers replaced.
        """
        # Use the pre-compiled regular expression to find all phone number patterns
        # in the text and replace them with the specified replacement string
        return constants.PHONE_REGEX.sub(replacement, input_text)

    def replace_numbers(self, input_text: str, replacement: str = "<NUMBER>") -> str:
        """
        Replace all numbers in the input text with a specified replacement string.

        Args:
            input_text (str): The input text.
            replacement (str, optional): The string to replace the numbers with.
                Defaults to "<NUMBER>".

        Returns:
            str: The modified text with numbers replaced.
        """
        # Use the pre-compiled regular expression to find all number patterns
        # in the text and replace them with the specified replacement string
        return constants.NUMBERS_REGEX.sub(replacement, input_text)
