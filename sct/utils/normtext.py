import logging
import re
import unicodedata
from encodings.idna import dots as _idna_dots
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
_WORD_END = rf'(?<=[^\W_]|[{constants.MARKS}])'
_FULLWIDTH_JOINER_REGEX = re.compile(r'(?<=[0-9A-Za-z])[＠．－＿＋\u3000](?=[0-9A-Za-z])|＋(?=[0-9])'
                                     rf'|{_WORD_END}＠(?=[^\W_])|{_WORD_END}．(?=[0-9A-Za-z])|(?<=[0-9A-Za-z])．(?=[^\W_])')
_FULLWIDTH_JOINER = {'＠': '@', '．': '.', '－': '-', '＿': '_', '＋': '+', '\u3000': ' '}
# Invisible format characters (category Cf: ZWNJ/ZWJ, ZWSP, soft hyphen, bidi marks).
_FORMAT_CHARS = re.escape(''.join(chr(cp) for cp in range(0x110000) if unicodedata.category(chr(cp)) == 'Cf'))
# Inside an address-shaped run (word characters around an @), full-width . - _ + fold whatever
# script sits beside them (john@例え．テスト, josé＿smith@...); prose without an @ keeps them.
# Format characters inside a word (ZWNJ in نامه\u200cای, ZWJ in क्\u200dष) continue the run and stay.
# The local part takes every ASCII character EMAIL_REGEX takes there (josé＿smith.work@, anna!＠), so a
# fold run and a detected address start at the same place. A run starts only where no local-part
# character precedes it, so each run is scanned once (linear time).
_LOCAL_ATEXT = re.escape("!#$%&'*+-.")
_EMAIL_RUN_LOCAL = rf'\w．－＿＋{constants.MARKS}{_FORMAT_CHARS}{_LOCAL_ATEXT}'
_EMAIL_RUN_REGEX = re.compile(
    rf'(?<![{_EMAIL_RUN_LOCAL}])[{_EMAIL_RUN_LOCAL}]*+[@＠][\w．－＿{constants.MARKS}{_FORMAT_CHARS}]*\w')
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
# squared CJK units (㎏) are prose, not PII, and stay; squared single letters (🄹) fold like circled ones.
_STYLED_ALNUM = _nfkc_ascii_alnum({'<font>', '<wide>'})
_CIRCLED_ALNUM = {**_nfkc_ascii_alnum({'<circle>'}),
                  **{cp: ch for cp, ch in _nfkc_ascii_alnum({'<square>'}).items() if len(ch) == 1}}
_CIRCLED_RUN_REGEX = re.compile('[%s]+' % ''.join(map(chr, _CIRCLED_ALNUM)))
# Contact punctuation in every compatibility form (full-width ＠, small ﹫, font ﬩ for +), picked by
# decomposition tag and NFKC value so every rule below folds the same set; ideographic space too.
_COMPAT_PUNCT = {ch: unicodedata.normalize('NFKC', ch) for ch in map(chr, range(0x20000))
                 if unicodedata.decomposition(ch).split(' ', 1)[0] in ('<wide>', '<small>', '<font>')
                 and unicodedata.normalize('NFKC', ch) in "@.-_+:/?=&#%~ !$'*"}
_STYLED_SEP = {ch: plain for ch, plain in _COMPAT_PUNCT.items() if plain in '@.-_+:/?=&#%~ '}


def _forms(ascii_chars, table=_STYLED_SEP):
    """Character-class body of the styled forms of ``ascii_chars``."""
    return re.escape(''.join(ch for ch, plain in table.items() if plain in ascii_chars))


_STYLED_PUNCT = _forms('@.-_+:/?=&#%~ ')
# IDNA label separators (。 ｡ ．, as Python's idna codec reads them) are a dot in a mail domain or URL
# host when an ASCII letter or digit follows (example。com); before other text 。 ends a CJK sentence.
_IDNA_DOTS = re.escape(''.join(ch for ch in map(chr, range(0x80, 0x10000)) if _idna_dots.fullmatch(ch)))
_IDNA_DOT = f'[{_IDNA_DOTS}](?=[0-9A-Za-z])'
_IDNA_DOT_REGEX = re.compile(_IDNA_DOT)
_PUNCT_ANY = _STYLED_PUNCT + re.escape('@.-_+:/?=&#%~')
# Dashes of any form (Pd: hyphen, non-breaking hyphen, en dash) join phone groups like "-" (06\u2011 12).
_DASHES = re.escape(''.join(chr(cp) for cp in range(0x110000) if unicodedata.category(chr(cp)) == 'Pd'))
# In a URL host or mail domain any dash is a label hyphen and any character whose NFKC form is "." a label
# dot (my\u2010site, a\u2024b.com, as copied from typeset text).
_HOST_SEP = {cp: '-' if unicodedata.category(chr(cp)) == 'Pd' else '.' for cp in range(0x80, 0x110000)
             if unicodedata.category(chr(cp)) == 'Pd' or unicodedata.normalize('NFKC', chr(cp)) == '.'}
_HOST_SEP_CHARS = re.escape(''.join(map(chr, _HOST_SEP)))
_HOST_DOTS = re.escape(''.join(chr(cp) for cp, plain in _HOST_SEP.items() if plain == '.'))
# A circled run folds beside ASCII alnum, a letter of the Latin script it folds to (ⓓé: décembre), any decimal
# digit (⑩꤈), any dash (2026\u2013⑩) or contact punctuation in any form.
_RUN_NEIGHBOUR = re.compile(rf'[0-9{constants.LATIN_LETTERS}{_PUNCT_ANY}{_DASHES}]|\d')
# Phone parentheses in any form (+49（30）123456).
_PARENS = re.escape('()' + ''.join(constants.PHONE_PARENS))
# URL/email/phone/date punctuation in compatibility form folds between ASCII letters or digits of
# any script (a run may mix forms: http：/／x); contact syntax also beside letters of any script:
# ＠ between letters/digits, ． with an ASCII side (例え．jp) and a ：／／ scheme after ASCII letters (https：／／例え.jp).
# The run is taken whole (possessive) once a styled form is seen in it, so a failing right side is linear.
_STYLED_JOINER_REGEX = re.compile(
    rf'(?<=[A-Za-z\d])(?=[{_PUNCT_ANY}]*?[{_STYLED_PUNCT}])[{_PUNCT_ANY}]++(?=[A-Za-z\d])|[{_forms("+")}](?=\d)'
    rf'|{_WORD_END}[{_forms("@")}](?=[^\W_])|{_WORD_END}[{_forms(".")}](?=[0-9A-Za-z])'
    rf'|(?<=[0-9A-Za-z])[{_forms(".")}](?=[^\W_])|(?<=[A-Za-z])[:{_forms(":")}][/{_forms("/")}]{{2}}')


# After a scheme or a www introducer (as URL_REGEX accepts), full-width query/fragment syntax folds
# whatever script the host is in (https://例え.テスト？email=...), and so do dots in the host
# (https://例え．テスト/), so the whole URL, query values included, is one match.
# CJK and full-width prose punctuation (，、「) ends the run, as it ends URL_REGEX, so prose glued after keeps
# its ：; an IDNA dot before an ASCII letter or digit (例え。jp) and punctuation before an @ (user info
# https://alice:pass！＠example.com) and a katakana middle dot beside kana or Han (例・え.jp, as in URL_REGEX) stay in it.
# A scheme from the start of its [A-Za-z0-9+.-] run (leading digits and +.- are left as they are), so the
# scan runs once per run, not once per letter (linear time).
_SCHEME = r'(?<![A-Za-z0-9+.\-])[0-9+.\-]*+[A-Za-z][A-Za-z0-9+.\-]*+://'
_URL_RUN_REGEX = re.compile(
    rf'(?:{_SCHEME}|(?<![A-Za-z0-9_/.{_HOST_DOTS}])www\d{{0,3}}(?:[.{_HOST_DOTS}]|{_IDNA_DOT}))(?:[^\s{constants._WIDE_PUNCT}]|{_IDNA_DOT}|{constants._KATAKANA_MIDDLE_DOT}'
    rf'|[{constants._WIDE_PUNCT}]+(?=[@{_forms("@")}]))+',
    re.IGNORECASE)
_URL_RUN_JOINER = str.maketrans({ch: plain for ch, plain in _STYLED_SEP.items() if plain in ':/?=&#%~'})
# Dots fold only in a host (up to the path), so prose glued after a URL keeps its ．; a run may hold
# several URLs glued by CJK prose, each with its own host (a host ends where the next scheme starts, so
# 。 before a glued https:// stays a sentence end). ＠ folds there too, after any userinfo character
# (https://alice:pass!＠example.com).
# Scheme-shaped runs are taken whole (possessive) and the lookahead runs once per run, so both stay linear.
_URL_AUTHORITY = re.compile(rf'(?:{_SCHEME}|www\d{{0,3}}(?:[.{_HOST_DOTS}]|{_IDNA_DOT}))'
                            r'(?:[0-9+.\-]*+[A-Za-z][A-Za-z0-9+.\-]*+(?!://)|[0-9+.\-]++(?![A-Za-z])'
                            r'|[^/?#A-Za-z0-9+.\-])*', re.IGNORECASE)
_AUTHORITY_JOINER = {**str.maketrans({ch: plain for ch, plain in _STYLED_SEP.items() if plain == '@'}), **_HOST_SEP}


def _fold_url_run(m):
    # layout hints drop in the authority as in a mail domain (https://例\u200bえ.jp)
    return _URL_AUTHORITY.sub(
        lambda a: _IDNA_DOT_REGEX.sub('.', a.group().translate(_AUTHORITY_JOINER)).translate(_DOMAIN_HINTS),
        m.group().translate(_URL_RUN_JOINER).translate(_CIRCLED_ALNUM))
# Inside an address-shaped run every styled form of @ . - _ + folds (the full-width-only rule above, widened),
# and in the local part, as in the domain, any dash is a hyphen and any NFKC dot a dot (john.smith\u2011billing@).
# Styled forms of the other local-part punctuation (anna＆co＠, ＊＠) fold there too.
_LOCAL_FORMS = _forms("!#$%&'*+-._", _COMPAT_PUNCT)
# Circled and squared letters and digits (ⓣ, 🅂) are address characters there too and fold by NFKC whatever sits
# beside them (ab@éⓣ.net, a@Ŗ🅂Ń.io); outside an address run they keep the neighbour rule below (list markers).
_CIRCLED = re.escape(''.join(map(chr, _CIRCLED_ALNUM)))
_STYLED_EMAIL_RUN_LOCAL = rf'\w{_CIRCLED}{_LOCAL_FORMS}{constants.MARKS}{_FORMAT_CHARS}{_LOCAL_ATEXT}{_HOST_SEP_CHARS}'
_STYLED_EMAIL_RUN_REGEX = re.compile(
    rf'(?<![{_STYLED_EMAIL_RUN_LOCAL}])[{_STYLED_EMAIL_RUN_LOCAL}]*+[@{_forms("@")}]'
    rf'(?:[\w{_CIRCLED}.\-{_forms(".-_")}{_HOST_SEP_CHARS}{constants.MARKS}{_FORMAT_CHARS}]|{_IDNA_DOT}'
    rf'|{constants._KATAKANA_MIDDLE_DOT}|{constants._GERESH})*[\w{_CIRCLED}]')
_STYLED_EMAIL_RUN_JOINER = {**str.maketrans({ch: plain for ch, plain in _COMPAT_PUNCT.items() if plain in "@!#$%&'*+-._"}),
                            **_CIRCLED_ALNUM}
# In the domain every format character but the join controls (ZWNJ/ZWJ, which spell words) is a
# copied layout hint (例\u200bえ.jp), whatever script sits beside it.
_DOMAIN_HINTS = {cp: None for cp in range(0x110000)
                 if unicodedata.category(chr(cp)) == 'Cf' and chr(cp) not in '\u200c\u200d'}


def _fold_styled_email_run(m):
    run = m.group().translate(_STYLED_EMAIL_RUN_JOINER).translate(_HOST_SEP)
    at = run.index('@')
    return run[:at] + _IDNA_DOT_REGEX.sub('.', run[at:]).translate(_DOMAIN_HINTS)


# Invisible format characters (category Cf: ZWSP, word joiner, soft hyphen, bidi marks) beside a
# contact-run character (ASCII alnum, any decimal digit, circled alnum, contact or mail local-part punctuation or a phone
# parenthesis in any form) are line-break or layout hints copied with the text ("06\u00ad12" reads "0612"): a run of
# them drops when its other side is a run character, a letter of any script (例え\u200b．jp), a space
# ("06\u200b 12") or the text edge; between a letter of any script and a space or the text edge it joins nothing
# and drops too (1990년 6월\u2060 1일). Between two other letters they stay (ZWNJ in Persian, ZWSP
# between Thai words).
_INVISIBLE_REGEX = re.compile(f'[{_FORMAT_CHARS}]+')
_RUN_CHAR = re.compile('[A-Za-z\\d%s%s%s%s%s%s%s%s]' % (''.join(map(chr, _CIRCLED_ALNUM)), _PUNCT_ANY, _DASHES, _IDNA_DOTS,
                                                         _HOST_DOTS, _LOCAL_FORMS, _LOCAL_ATEXT, _PARENS))
_LETTER = re.compile(rf'[^\W_]|[{constants.MARKS}]')


def _drop_invisible(m):
    s, a, b = m.string, m.start(), m.end()
    # a text edge counts as a space, so a window of a stream that ends there folds as the whole text does
    left, right = s[a - 1] if a else ' ', s[b] if b < len(s) else ' '
    word_left, word_right = _RUN_CHAR.match(left) or _LETTER.match(left), _RUN_CHAR.match(right) or _LETTER.match(right)
    if (_RUN_CHAR.match(left) and word_right) or (_RUN_CHAR.match(right) and word_left) or (
            word_left and right.isspace()) or (word_right and left.isspace()):
        return ''
    return m.group()


# A lone circled number between space-separated digit groups (06 ⑬ 01 68) or after two or more of
# them (06 73 86 88 ㊺) is one group of them; a lone one before a number (① 2024年) is a list marker.
_DIGIT_GROUP = '(?:\\d|[%s])' % ''.join(chr(cp) for cp, plain in _CIRCLED_ALNUM.items() if plain.isdigit())
# Groups may be set off by any space (Zs: no-break, thin, narrow no-break, ideographic), as the phone view reads them.
_SPACE = '[%s]' % ''.join(chr(cp) for cp in range(0x110000) if unicodedata.category(chr(cp)) == 'Zs')
_DIGIT_BEFORE, _DIGIT_AFTER = re.compile(_DIGIT_GROUP + _SPACE), re.compile(_SPACE + _DIGIT_GROUP)
_GROUPS_BEFORE = re.compile(rf'(?:{_DIGIT_GROUP}+{_SPACE}){{2,}}\Z')
# An IDNA dot after a decimal digit is a host label dot (𖭗𖭗｡⑫｡15.example), so a circled run after it folds.
_IDNA_DOT_AFTER_DIGIT = re.compile(rf'\d[{_IDNA_DOTS}]')
# A circled run in parentheses glued to a digit group is an area code (+80（⑫）808).
_OPEN_AFTER_DIGIT = re.compile(rf'{_DIGIT_GROUP}[{_PARENS}]')
_CLOSE_BEFORE_DIGIT = re.compile(rf'[{_PARENS}]{_DIGIT_GROUP}')


def _fold_circled_run(m):
    s, a, b = m.string, m.start(), m.end()
    if b - a > 1 or (a and _RUN_NEIGHBOUR.match(s, a - 1)) or (b < len(s) and _RUN_NEIGHBOUR.match(s, b)) or (
            a > 1 and _IDNA_DOT_AFTER_DIGIT.match(s, a - 2)) or (
            a > 1 and _OPEN_AFTER_DIGIT.match(s, a - 2)) or _CLOSE_BEFORE_DIGIT.match(s, b) or (
            a > 1 and _DIGIT_BEFORE.match(s, a - 2) and _DIGIT_AFTER.match(s, b)) or (
            _GROUPS_BEFORE.search(s, max(0, a - 64), a)):
        return m.group().translate(_CIRCLED_ALNUM)
    return m.group()


def fold_styled(text):
    """Fold styled/compatibility letters, digits and in-run URL punctuation to ASCII."""
    text = _INVISIBLE_REGEX.sub(_drop_invisible, text.translate(_STYLED_ALNUM))
    text = _CIRCLED_RUN_REGEX.sub(_fold_circled_run, text)
    text = _STYLED_JOINER_REGEX.sub(lambda m: unicodedata.normalize('NFKC', m.group()), text)
    text = _URL_RUN_REGEX.sub(_fold_url_run, text)
    return _STYLED_EMAIL_RUN_REGEX.sub(_fold_styled_email_run, text)


class NormaliseText:

    def __init__(self):
        pass

    def fix_bad_unicode(self, text, normalization="NFC", decode_escapes=True, fix_character_width=False,
                        fold_styled_characters=True, keep_html_entities_with_markup=False):
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
        ``keep_html_entities_with_markup=False`` decodes HTML entities everywhere (each entity on its own,
        so a text split at whitespace decodes as the whole); ``True`` is ftfy's ``"auto"``: from the first
        line holding a ``<`` on, entities stay (0.7).
        """
        if decode_escapes:
            try:
                text = text.encode("latin", "backslashreplace").decode("unicode-escape")
            except (UnicodeDecodeError, UnicodeEncodeError):
                logger.debug("Unicode escape decoding failed, using original text", exc_info=True)

        text = fix_text(text, normalization=normalization, fix_character_width=fix_character_width,
                        unescape_html="auto" if keep_html_entities_with_markup else True)
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
