from ftfy import fix_text
from unidecode import unidecode
from emoji import demojize, emojize
#---
from sct.utils import constants


class NormaliseText:
    
    def __init__(self):
        
        pass
    
    def fix_bad_unicode(self, text, normalization="NFC"):
        """
        Fix unicode text that's "broken" using `ftfy <http://ftfy.readthedocs.org/>`_;
        this includes mojibake, HTML entities and other code cruft,
        and non-standard forms for display purposes.
        Args:
            text (str): raw text
            normalization ({'NFC', 'NFKC', 'NFD', 'NFKD'}): if 'NFC',
                combines characters and diacritics written using separate code points,
                e.g. converting "e" plus an acute accent modifier into "é"; unicode
                can be converted to NFC form without any change in its meaning!
                if 'NFKC', additional normalizations are applied that can change
                the meanings of characters, e.g. ellipsis characters will be replaced
                with three periods
        """
        # trying to fix backslash-replaced strings (via https://stackoverflow.com/a/57192592/4028896)
        try:
            text = text.encode("latin", "backslashreplace").decode("unicode-escape")
        except:
            pass

        return fix_text(text, normalization=normalization)

    def fix_strange_quotes(self, text):
        """
        Replace strange quotes, i.e., 〞with a single quote ' or a double quote " if it fits better.
        """
        text = constants.SINGLE_QUOTE_REGEX.sub("'", text)
        text = constants.DOUBLE_QUOTE_REGEX.sub('"', text)
        return text

    def to_ascii_unicode(self, text, no_emoji=True):
        """
        Try to represent unicode data in ascii characters similar to what a human
        with a US keyboard would choose.
        Works great for languages of Western origin, worse the farther the language
        gets from Latin-based alphabets. It's based on hand-tuned character mappings
        that also contain ascii approximations for symbols and non-Latin alphabets.
        """
        # normalize quotes before since this improves transliteration quality
        text = self.fix_strange_quotes(text)

        if not no_emoji:
            text = demojize(text, use_aliases=True)

        text = unidecode(text)

        return text

    def normalize_whitespace(self, text, strip_lines=True, no_line_breaks=False, keep_two_line_breaks=False):
        """
        Given ``text`` str, replace one or more spacings with a single space, and one
        or more line breaks with a single newline. Also strip leading/trailing whitespace.
        """
        if strip_lines:
            text = "\n".join([x.strip() for x in text.splitlines()])

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

        return text.strip()