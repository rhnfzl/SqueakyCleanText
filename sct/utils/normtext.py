from ftfy import fix_text
from unidecode import unidecode
from emoji import demojize, emojize
#---
from sct.utils import constants


class NormaliseText:
    
    def __init__(self):
        
        pass
    
    def fix_bad_unicode(
        self,
        text: str,
        normalization: str = "NFC",
    ) -> str:
        """
        Fixes broken unicode text using the `ftfy` library.

        This function fixes various issues with unicode text, including mojibake, HTML entities, and other code cruft.
        It also normalizes the text to a specified format, which can improve the readability and consistency of the text.

        Args:
            text (str): The raw text to be fixed.
            normalization (str, optional): The normalization form to use.
                Defaults to "NFC".
                Possible values are:
                    - "NFC" (Canonical Decomposition, followed by Canonical Composition):
                      Combines characters and diacritics written using separate code points,
                      e.g. converting "e" plus an acute accent modifier into "é".
                      Unicode can be converted to NFC form without any change in its meaning.
                    - "NFKC" (Compatibility Decomposition, followed by Canonical Composition):
                      Applies additional normalizations that can change the meanings of characters,
                      e.g. replacing ellipsis characters with three periods.
                    - "NFD" (Canonical Decomposition):
                      Decomposes characters and diacritics written using separate code points.
                    - "NFKD" (Compatibility Decomposition):
                      Decomposes characters and diacritics written using separate code points,
                      and applies additional normalizations.

        Returns:
            str: The fixed and normalized text.
        """
        # Trying to fix backslash-replaced strings (via https://stackoverflow.com/a/57192592/4028896)
        try:
            # Decode text using the "latin" encoding and "backslashreplace" error handler
            text = text.encode("latin", "backslashreplace").decode("unicode-escape")
        except:
            pass

        # Use the fix_text function from the ftfy library to fix the text
        return fix_text(text, normalization=normalization)

    def fix_strange_quotes(self, text: str) -> str:
        """
        Replace strange quotes, i.e., with a single quote ' or a double quote " if it fits better.

        Args:
            text (str): The text to be processed.

        Returns:
            str: The processed text with strange quotes replaced.
        """
        # Replace single quotes
        text = constants.SINGLE_QUOTE_REGEX.sub("'", text)
        # Replace double quotes
        text = constants.DOUBLE_QUOTE_REGEX.sub('"', text)
        return text

    def to_ascii_unicode(
        self,
        text: str,
        no_emoji: bool = True,
    ) -> str:
        """
        Transliterate unicode text into ascii characters.

        This function tries to represent unicode data in ascii characters similar to
        what a human with a US keyboard would choose. It works great for languages of
        Western origin, and gets worse the farther the language gets from Latin-based
        alphabets. It's based on hand-tuned character mappings that also contain ascii
        approximations for symbols and non-Latin alphabets.

        Args:
            text (str): The text to be transliterated.
            no_emoji (bool, optional): Flag to indicate whether to remove emojis or not. Defaults to True.

        Returns:
            str: The transliterated text.

        """
        # Normalize quotes before since this improves transliteration quality.
        text = self.fix_strange_quotes(text)

        # If no_emoji flag is set to False, remove emojis from the text.
        if not no_emoji:
            text = demojize(text, use_aliases=True)

        # Convert unicode characters to their ascii approximations.
        text = unidecode(text)

        return text

    def normalize_whitespace(
        self,
        text: str,
        strip_lines: bool = True,
        no_line_breaks: bool = False,
        keep_two_line_breaks: bool = False,
    ) -> str:
        """
        Normalize whitespace in the given text.

        This function replaces one or more spacings with a single space, and one
        or more line breaks with a single newline. Also, it strips leading/trailing whitespace.

        Args:
            text (str): The input text to be processed.
            strip_lines (bool, optional): Flag to indicate whether to strip leading/trailing whitespace from each line. Defaults to True.
            no_line_breaks (bool, optional): Flag to indicate whether to remove all line breaks or not. Defaults to False.
            keep_two_line_breaks (bool, optional): Flag to indicate whether to keep two consecutive line breaks or not. Defaults to False.

        Returns:
            str: The processed text with normalized whitespace.
        """

        # Strip leading/trailing whitespace from each line if strip_lines flag is set to True.
        if strip_lines:
            text = "\n".join([x.strip() for x in text.splitlines()])

        # If no_line_breaks flag is set to True, remove all line breaks.
        # Otherwise, replace consecutive line breaks with one newline.
        if no_line_breaks:
            text = constants.MULTI_WHITESPACE_TO_ONE_REGEX.sub(" ", text)
        else:
            if keep_two_line_breaks:
                text = constants.NONBREAKING_SPACE_REGEX.sub(
                    " ", constants.TWO_LINEBREAK_REGEX.sub(r"\n\n", text)
                )
            else:
                text = constants.NONBREAKING_SPACE_REGEX.sub(
                    " ", constants.LINEBREAK_REGEX.sub(r"\n", text)
                )

        # Strip leading/trailing whitespace from the final processed text.
        return text.strip()
