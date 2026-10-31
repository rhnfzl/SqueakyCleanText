import logging
import re
import unicodedata
from typing import Optional, Set
from ftfy import fix_text
from unidecode import unidecode
from emoji import demojize, replace_emoji
from sct.utils import constants

logger = logging.getLogger(__name__)

# Full-width ASCII letters/digits (U+FF10..FF5A) map to ASCII by a fixed offset.
_FULLWIDTH_ALNUM = {c: c - 0xFEE0 for r in ((0xFF10, 0xFF19), (0xFF21, 0xFF3A), (0xFF41, 0xFF5A))
                    for c in range(r[0], r[1] + 1)}
# Full-width @ . - _ + and ideographic space become ASCII only inside alphanumeric runs (emails,
# phones; ＠ and ． also beside letters of any script), so CJK punctuation such as ， ： ！ keeps its width.
# A word of any script may end in a combining mark (प्रिया), so marks count as its letters.
_WORD_END = rf'(?<=[^\W_{constants.MARKS}])'
_FULLWIDTH_JOINER_REGEX = re.compile(r'(?<=[0-9A-Za-z])[＠．－＿＋\u3000](?=[0-9A-Za-z])|＋(?=[0-9])'
                                     rf'|{_WORD_END}＠(?=[^\W_])|{_WORD_END}．(?=[0-9A-Za-z])|(?<=[0-9A-Za-z])．(?=[^\W_])')
_FULLWIDTH_JOINER = {'＠': '@', '．': '.', '－': '-', '＿': '_', '＋': '+', '\u3000': ' '}
# Inside an address-shaped run (word characters around an @), full-width . - _ + fold whatever
# script sits beside them (john@例え．テスト, josé＿smith@...); prose without an @ keeps them.
_EMAIL_RUN_REGEX = re.compile(rf'\w[\w．－＿＋{constants.MARKS}]*[@＠][\w．－＿{constants.MARKS}]*\w')
_EMAIL_RUN_JOINER = str.maketrans('＠．－＿＋', '@.-_+')


def _fold_email_runs(text):
    return _EMAIL_RUN_REGEX.sub(lambda m: m.group().translate(_EMAIL_RUN_JOINER), text)


def _nfkc_ascii_alnum(tags):
    """Characters with one of the Unicode decomposition ``tags`` whose NFKC form is ASCII alnum."""
    table = {}
    for cp in range(0x20000):  # BMP + SMP hold every such character (Unicode 15)
        ch = chr(cp)
        if unicodedata.decomposition(ch).split(' ', 1)[0] in tags:
            folded = unicodedata.normalize('NFKC', ch)
            if folded.isascii() and folded.isalnum():
                table[cp] = folded
    return table


# fold_styled_characters: font variants (math bold/italic/script/double-struck 𝐞 ℝ 𝟎) and
# full-width letters/digits always fold; circled ones (ⓙ ①) only next to ASCII alnum or URL/email
# punctuation, so a lone ① list marker in CJK prose stays. Super/subscripts (m², H₂O, nº) and
# squared CJK units (㎏) are prose, not PII, and stay.
_STYLED_ALNUM = _nfkc_ascii_alnum({'<font>', '<wide>'})
_CIRCLED_ALNUM = _nfkc_ascii_alnum({'<circle>'})
_CIRCLED_RUN_REGEX = re.compile('[%s]+' % ''.join(map(chr, _CIRCLED_ALNUM)))
_RUN_NEIGHBOUR = re.compile(r'[0-9A-Za-z@＠.:/_+\-]')
# URL/email/phone/date punctuation in compatibility form folds between ASCII alnum; contact
# syntax also beside letters of any script: ＠ between letters/digits, ． with an ASCII side
# (例え．jp) and a ：／／ scheme after ASCII letters (https：／／例え.jp).
_STYLED_JOINER_REGEX = re.compile(r'(?<=[0-9A-Za-z])[＠．－＿＋：／？＝＆＃％～\u3000]+(?=[0-9A-Za-z])|＋(?=[0-9])'
                                  rf'|{_WORD_END}＠(?=[^\W_])|{_WORD_END}．(?=[0-9A-Za-z])|(?<=[0-9A-Za-z])．(?=[^\W_])|(?<=[A-Za-z])：／／')


# After a scheme, full-width query/fragment syntax folds whatever script the host is in
# (https://例え.テスト？email=...), so the whole URL, query values included, is one match.
_URL_RUN_REGEX = re.compile(r'[A-Za-z][A-Za-z0-9+.\-]*://\S+')
_URL_RUN_JOINER = str.maketrans('：／？＝＆＃％～', ':/?=&#%~')


def _fold_circled_run(m):
    s, a, b = m.string, m.start(), m.end()
    if b - a > 1 or (a and _RUN_NEIGHBOUR.match(s, a - 1)) or (b < len(s) and _RUN_NEIGHBOUR.match(s, b)):
        return m.group().translate(_CIRCLED_ALNUM)
    return m.group()


def fold_styled(text):
    """Fold styled/compatibility letters, digits and in-run URL punctuation to ASCII."""
    text = _CIRCLED_RUN_REGEX.sub(_fold_circled_run, text.translate(_STYLED_ALNUM))
    text = _STYLED_JOINER_REGEX.sub(lambda m: unicodedata.normalize('NFKC', m.group()), text)
    return _fold_email_runs(_URL_RUN_REGEX.sub(lambda m: m.group().translate(_URL_RUN_JOINER), text))


class NormaliseText:

    def __init__(self):
        pass

    def fix_bad_unicode(self, text, normalization="NFC", decode_escapes=True, fix_character_width=False,
                        fold_styled_characters=True):
        """
        Fix unicode text that's "broken" using `ftfy <http://ftfy.readthedocs.org/>`_;
        this includes mojibake, HTML entities and other code cruft,
        and non-standard forms for display purposes.

        ``decode_escapes=False`` keeps literal backslash escapes (``\\n``, ``\\u00e9``).
        ``fix_character_width=False`` keeps full-width CJK punctuation and half-width katakana and
        only turns full-width ASCII letters/digits (and @ . - _ + inside them) into ASCII, so
        full-width emails and phones are still found; ``True`` is ftfy's full width fix (0.7).
        ``fold_styled_characters=True`` also folds math-alphanumeric (𝐞), circled (ⓙ) and full-width
        URL punctuation (``：／？``) inside ASCII runs, so styled emails, URLs and phones are found.
        """
        if decode_escapes:
            try:
                text = text.encode("latin", "backslashreplace").decode("unicode-escape")
            except (UnicodeDecodeError, UnicodeEncodeError):
                logger.debug("Unicode escape decoding failed, using original text", exc_info=True)

        text = fix_text(text, normalization=normalization, fix_character_width=fix_character_width)
        if fold_styled_characters:
            return fold_styled(text)
        if fix_character_width:
            return text
        text = text.translate(_FULLWIDTH_ALNUM)
        return _fold_email_runs(_FULLWIDTH_JOINER_REGEX.sub(lambda m: _FULLWIDTH_JOINER[m.group()], text))

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

    def smart_casefold(self, text: str, stop_words: Optional[Set[str]] = None) -> str:
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
