import re
import string
#---
from sct.utils import constants

class ProcessSpecialSymbols:
    
    def __init__(self):
        pass
    
    def replace_currency_symbols(
        self,
        text: str,
        replace_with: str = "<CUR>"
    ) -> str:
        """
        Replaces currency symbols in the given text with the specified replacement string.

        Args:
            text (str): The raw text.
            replace_with (str, optional): If None (default), replaces symbols with their
                standard 3-letter abbreviations (e.g. '$' with 'USD', '£' with 'GBP').
                Otherwise, passes in a string with which to replace all symbols
                (e.g. "*CURRENCY*").

        Returns:
            str: The modified text with currency symbols replaced.
        """
        # If `replace_with` is None, replace symbols with their standard 3-letter abbreviations
        if replace_with is None:
            # Iterate over the keys and values in CURRENCIES
            for symbol, abbreviation in constants.CURRENCIES.items():
                # Replace the symbol in the text with its standard abbreviation
                text = text.replace(symbol, abbreviation)
            return text
        # If `replace_with` is not None, use regular expressions to replace all currency symbols
        else:
            # Use the pre-compiled regular expression CURRENCY_REGEX to find all currency symbols
            # in the text and replace them with the specified replacement string
            return constants.CURRENCY_REGEX.sub(replace_with, text)
        
        
    def remove_isolated_letters(
        self,
        text: str
    ) -> str:
        """
        Removes any isolated letters which doesn't add any value to the text.

        Args:
            text (str): The raw text to be processed.

        Returns:
            str: The modified text with isolated letters removed.
        """
        # Use the pre-compiled regular expression ISOLATED_LETTERS_REGEX to find all isolated
        # letters in the text and replace them with an empty string
        cleaned_text: str = constants.ISOLATED_LETTERS_REGEX.sub('', text)
        
        return cleaned_text

    def remove_isolated_special_symbols(
        self,
        text: str
    ) -> str:
        """
        Removes any isolated symbols which shouldn't be present in the text.

        This includes:
        - [] content, usually images or file locations
        - {} content, usually HTML links
        - Any isolated special symbols which don't add any value to the text

        Args:
            text (str): The raw text.

        Returns:
            str: The modified text with isolated special symbols removed.
        """
        # Remove [] content, usually images or file locations
        cleaned_text: str = re.sub(r'\[[^\]]+\]', '', text)
        # Remove {} content, usually HTML links
        cleaned_text: str = re.sub(r'\{[^}]+\}', '', cleaned_text)
        # Remove any isolated special symbols which don't add any value to the text
        cleaned_text: str = constants.ISOLATED_SPECIAL_SYMBOLS_REGEX.sub('', cleaned_text)
        # Remove any remaining isolated special symbols which may have been missed
        cleaned_text: str = re.sub(r"(?<![a-zA-Z0-9])['\"\-*%](?![a-zA-Z0-9])", '', cleaned_text, flags=re.UNICODE | re.IGNORECASE)
        
        return cleaned_text
    
    def remove_punctuation(
        self,
        text: str
    ) -> str:
        """
        Removes all punctuation characters from the given text.

        Args:
            text (str): The input text.

        Returns:
            str: The modified text with punctuation characters removed.
        """
        # Create a regular expression pattern of all punctuation characters
        # using the string.punctuation string.
        chars = re.escape(string.punctuation)

        # Use re.sub() to replace all occurrences of punctuation characters with an empty string.
        # The pattern is created by escaping all punctuation characters using re.escape().
        # The '[' and ']' are added to the pattern to match any character in the set.
        # The '+' after the '[' and before the ']' is a quantifier that matches one or more occurrences.
        # The '' (empty string) is the replacement string, which replaces all matches with an empty string.
        return re.sub('['+chars+']', '', text)
