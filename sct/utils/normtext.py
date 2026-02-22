import logging
import re
from ftfy import fix_text
from unidecode import unidecode
from emoji import demojize, emojize, replace_emoji
from sct.utils import constants

logger = logging.getLogger(__name__)

class NormaliseText:
    
    def __init__(self):
        pass
    
    def fix_bad_unicode(self, text, normalization="NFC"):
        """
        Fix unicode text that's "broken" using `ftfy <http://ftfy.readthedocs.org/>`_;
        this includes mojibake, HTML entities and other code cruft,
        and non-standard forms for display purposes.
        """
        try:
            text = text.encode("latin", "backslashreplace").decode("unicode-escape")
        except (UnicodeDecodeError, UnicodeEncodeError):
            logger.debug("Unicode escape decoding failed, using original text", exc_info=True)

        return fix_text(text, normalization=normalization)

    def fix_strange_quotes(self, text):
        """Replace strange quotes with standard single/double quotes."""
        text = constants.SINGLE_QUOTE_REGEX.sub("'", text)
        text = constants.DOUBLE_QUOTE_REGEX.sub('"', text)
        return text

    def to_ascii_unicode(self, text, no_emoji=True):
        """
        Convert non-ASCII characters into their closest ASCII equivalents.
        """
        text = self.fix_strange_quotes(text)

        if not no_emoji:
            text = demojize(text, use_aliases=True)

        text = unidecode(text)
        return text

    def remove_emoji(self, text):
        """Remove all emoji characters from text."""
        return replace_emoji(text, replace='')

    @staticmethod
    def _has_camelcase(text: str) -> bool:
        """Detect lowercase-to-uppercase transition (e.g., GraphQL, DevOps)."""
        return bool(re.search(r'[a-z][A-Z]', text))

    def smart_casefold(self, text: str, stop_words: set = None) -> str:
        """Case-fold text while preserving abbreviations and camelCase.

        Per-token rules (applied in order):
        1. If token is a stopword (case-insensitive) -> casefold
        2. If token is all-uppercase -> preserve (abbreviation/acronym)
        3. If token has a lowercase-to-uppercase transition -> preserve (compound term)
        4. Otherwise -> casefold
        """
        tokens = text.split()
        stop_words = stop_words or set()
        result = []
        for token in tokens:
            alpha = ''.join(c for c in token if c.isalpha())
            if not alpha:
                result.append(token.casefold())
            elif alpha.casefold() in stop_words:
                result.append(token.casefold())
            elif alpha.isupper():
                result.append(token)
            elif self._has_camelcase(token):
                result.append(token)
            else:
                result.append(token.casefold())
        return ' '.join(result)

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
