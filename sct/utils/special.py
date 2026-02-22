import re
import string
#---
from sct.utils import constants

# Pre-compile punctuation regex at module level (avoids recompilation per call)
_PUNCTUATION_CHARS = re.escape(string.punctuation)
_PUNCTUATION_REGEX = re.compile('[' + _PUNCTUATION_CHARS + ']')

# Pre-compile bracket/brace regexes
_BRACKET_CONTENT_REGEX = re.compile(r'\[[^\]]+\]')
_BRACE_CONTENT_REGEX = re.compile(r'\{[^}]+\}')
_ISOLATED_QUOTES_REGEX = re.compile(
    r"(?<![a-zA-Z0-9])['\"\-*%](?![a-zA-Z0-9])",
    flags=re.UNICODE | re.IGNORECASE,
)


class ProcessSpecialSymbols:
    
    def __init__(self):
        pass
    
    def replace_currency_symbols(self, text, replace_with="<CUR>"):
        """
        Replace currency symbols in ``text`` str with string specified by ``replace_with`` str.
        Args:
            text (str): raw text
            replace_with (str): if None (default), replace symbols with
                their standard 3-letter abbreviations (e.g. '$' with 'USD', '£' with 'GBP');
                otherwise, pass in a string with which to replace all symbols
                (e.g. "*CURRENCY*")
        """
        if replace_with is None:
            for k, v in constants.CURRENCIES.items():
                text = text.replace(k, v)
            return text
        else:
            return constants.CURRENCY_REGEX.sub(replace_with, text)
        
        
    def remove_isolated_letters(self, text):
        """
        Removes any isolated letters which doesn't add any value to the text.
        """
        cleaned_text = constants.ISOLATED_LETTERS_REGEX.sub('', text)
        
        return cleaned_text

    def remove_isolated_special_symbols(self, text, remove_brackets=True, remove_braces=True):
        """
        Removes any isolated symbols which shouldn't be present in the text.
        
        Args:
            text: Input text.
            remove_brackets: If True, remove [...] content (image/file references).
            remove_braces: If True, remove {...} content (HTML links/templates).
        """
        cleaned_text = text
        if remove_brackets:
            cleaned_text = _BRACKET_CONTENT_REGEX.sub('', cleaned_text)
        if remove_braces:
            cleaned_text = _BRACE_CONTENT_REGEX.sub('', cleaned_text)
        cleaned_text = constants.ISOLATED_SPECIAL_SYMBOLS_REGEX.sub('', cleaned_text)
        cleaned_text = _ISOLATED_QUOTES_REGEX.sub('', cleaned_text)
        
        return cleaned_text
    
    def remove_punctuation(self, text):
        return _PUNCTUATION_REGEX.sub('', text)
