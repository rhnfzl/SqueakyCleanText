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
            return self.replace_currency_symbols_with_fn(text, constants.CURRENCIES.__getitem__)
        return constants.CURRENCY_REGEX.sub(replace_with, text)

    def replace_currency_symbols_with_fn(self, text, replace_fn):
        """Replace each single currency symbol: replace_fn(symbol) -> replacement."""
        return constants.CURRENCY_SYMBOL_REGEX.sub(lambda m: replace_fn(m.group()), text)


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
            cleaned_text = constants.BRACKET_CONTENT_REGEX.sub('', cleaned_text)
        if remove_braces:
            cleaned_text = constants.BRACE_CONTENT_REGEX.sub('', cleaned_text)
        cleaned_text = constants.ISOLATED_SPECIAL_SYMBOLS_REGEX.sub('', cleaned_text)
        cleaned_text = constants.ISOLATED_QUOTES_REGEX.sub('', cleaned_text)

        return cleaned_text

    def remove_punctuation(self, text):
        return constants.PUNCTUATION_REGEX.sub('', text)
