import re
import string
#---
from sct.utils import constants

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

    def remove_isolated_special_symbols(self, text):
        """
        Removes any isolated symbols which shouldn't be present in the text.
        """
        cleaned_text = re.sub(r'\[[^\]]+\]', '', text) # to remove [] content, usianlly they are image or file loc text
        cleaned_text = re.sub(r'\{[^}]+\}', '', cleaned_text) # to remove {}} content, usianlly they are html links
        cleaned_text = constants.ISOLATED_SPECIAL_SYMBOLS_REGEX.sub('', cleaned_text)
        cleaned_text = re.sub(r"(?<![a-zA-Z0-9])['\"\-*%](?![a-zA-Z0-9])", '', cleaned_text, flags=re.UNICODE | re.IGNORECASE)
        
        return cleaned_text
    
    def remove_punctuation(self, text):
        chars = re.escape(string.punctuation)
        return re.sub('['+chars+']', '',text)