"""
Constant symbols and compiled RegExs used for cleaning.
"""

import functools
import re
import string
import unicodedata
import regex
from collections.abc import Iterable, Mapping

CURRENCIES = {
    "$": "USD",
    "zł": "PLN",
    "£": "GBP",
    "¥": "JPY",
    "฿": "THB",
    "₡": "CRC",
    "₦": "NGN",
    "₩": "KRW",
    "₪": "ILS",
    "₫": "VND",
    "€": "EUR",
    "₱": "PHP",
    "₲": "PYG",
    "₴": "UAH",
    "₹": "INR",
}
CURRENCY_REGEX = re.compile(
    "({})+".format("|".join(re.escape(c) for c in CURRENCIES.keys()))
)
CURRENCY_SYMBOL_REGEX = re.compile("|".join(re.escape(c) for c in CURRENCIES))

# Fixed: bounded quantifiers to prevent ReDoS (was nested + quantifiers)
ACRONYM_REGEX = re.compile(
    r"(?:^|(?<=\W))(?:(?:[A-Z]\.?){1,15}[a-z0-9&/-]?(?:[A-Z][s.]?|[0-9]s?)|[0-9](?:-?[A-Z]){1,15})(?:$|(?=\W))",
    flags=re.UNICODE,
)

# Canonical PII type name -> its pre-0.8.0 name in reversible placeholders and metadata
# (legacy_type_names=True). Every other type already had one name on every surface.
LEGACY_TYPE_NAMES = {'PHONE': 'PHONE_NUMBER', 'IP': 'IP_ADDRESS'}

# Fixed: Unicode ranges use actual Unicode escapes (not literal \\u in raw strings)
# Local part takes RFC 5322 punctuation (o'connor, jo&co, alice!) after a word start, at most 64 characters
# (RFC 5321) so a long run with no "@" costs a bounded scan from each start (linear time),
# so no prefix of the address stays visible; it never ends in a dot. ZWNJ/ZWJ join letters
# in it as in a domain label (Devanagari half forms, Persian).
# Domain labels take any script but not general, CJK or full-width punctuation, so
# "a@b.com（担当：鈴木）" or "a@b.com、" ends at ".com". The letters and numbers of the
# CJK punctuation block (々 〆 〇, by Unicode category) stay, so "info@佐々木.jp" is whole.
_CJK_SYMBOL_LETTERS = ''.join(chr(c) for c in range(0x3000, 0x3040) if unicodedata.category(chr(c))[0] in 'LN')


def _scan_unicode():
    """One pass over Unicode: {general category: its characters in code point order} and {character: decomposition}.
    Unassigned, private-use and surrogate code points are left out: no class is built from them, none decomposes."""
    by_category, decompositions = {}, {}
    for ch in map(chr, range(0x110000)):
        category = unicodedata.category(ch)
        if category not in ('Cn', 'Co', 'Cs'):
            by_category.setdefault(category, []).append(ch)
            if decomposition := unicodedata.decomposition(ch):
                decompositions[ch] = decomposition
    return {category: ''.join(chars) for category, chars in by_category.items()}, decompositions


# Read by every character class here and in normtext.py and contact.py, so none rescans Unicode at import.
CATEGORY_CHARS, DECOMPOSITIONS = _scan_unicode()
# Characters whose NFKC form is "." (U+2024, small and full-width dots).
NFKC_DOTS = ''.join(ch for ch in DECOMPOSITIONS if unicodedata.normalize('NFKC', ch) == '.')


def category_chars(*prefixes: str) -> list[str]:
    """Characters, in code point order, of the general categories starting with one of *prefixes*."""
    return sorted(''.join(chars for category, chars in CATEGORY_CHARS.items() if category.startswith(prefixes)))


def _class_ranges(categories, *spans):
    """Character-class body (literal ranges) of the code points in *spans* whose category starts with one of *categories*."""
    out, start = [], None
    for lo, hi in spans:
        for cp in range(lo, hi + 1):
            hit = unicodedata.category(chr(cp))[0] in categories
            if hit and start is None:
                start = cp
            elif not hit and start is not None:
                out.append(chr(start) if start == cp - 1 else f"{chr(start)}-{chr(cp - 1)}")
                start = None
        if start is not None:
            out.append(chr(start) if start == hi else f"{chr(start)}-{chr(hi)}")
            start = None
    return ''.join(out)


def _ranges(chars):
    """Character-class body (literal ranges) of the ascending iterable *chars*."""
    spans = []
    for cp in map(ord, chars):
        if spans and spans[-1][1] == cp - 1:
            spans[-1][1] = cp
        else:
            spans.append([cp, cp])
    return ''.join(f"{chr(lo)}-{chr(hi)}" if hi > lo else chr(lo) for lo, hi in spans)


# Combining marks of every script (Mn/Mc/Me: the vowel signs of প্রিয়া, संपर्क), which \w leaves out.
MARKS = _ranges(category_chars('M'))
# Letters, marks and digits beyond the BMP (𠮷野.jp): the BMP ranges below stop at U+FFFF.
_SUPPLEMENTARY_ALNUM = _ranges(ch for ch in category_chars('L', 'M', 'N') if '\U00010000' <= ch <= '\U0003FFFF')
# Latin, Greek and Cyrillic letters, which sit between spaces in prose.
_SPACED_LETTERS = "A-Za-z\u00C0-\u024F\u0370-\u03FF\u0400-\u04FF"
# Letters of the Latin script (Script=Latin, any block: é, ŀ, ḁ), which styled letters fold to.
LATIN_LETTERS = _ranges(regex.findall(r'\p{Script=Latin}', ''.join(category_chars('L'))))
# Letters and marks of scripts written without spaces between words (Thai, Lao, Khmer, Myanmar, CJK, kana:
# Unicode line-break class SA, ID or CJ; not full-width Latin). Where one touches a Latin/Greek/Cyrillic letter or a digit, a
# word ends, so prose glued to an address ("ดูa@b.jpกขค", "https://x.com/aปถว06 10") stays outside it.
_SPACELESS_CHAR = regex.compile(r'(?V1)[[\p{Line_Break=SA}\p{Line_Break=ID}\p{Line_Break=CJ}]--\p{Script=Latin}]')
_SPACELESS = _ranges(_SPACELESS_CHAR.findall(''.join(category_chars('L', 'M'))))
# A percent escape is one character (?email=a%40ตัวอย่าง.ไทย, its digits in any script), so it glues to nothing.
_DECIMAL = _ranges(CATEGORY_CHARS['Nd'])
_GLUE = (rf"(?<=[0-9{_SPACED_LETTERS}])(?<!%[{_DECIMAL}A-Fa-f]{{2}})[{_SPACELESS}]"
         rf"|(?<=[{_SPACELESS}])[0-9{_SPACED_LETTERS}]")
# Before each character of a local part, TLD or URL path: it does not start a new word.
_NO_GLUE = rf"(?!{_GLUE})"
# Label characters leave out punctuation of every script (category P), so a label ends at the Arabic
# comma or the Devanagari danda after an address ("ahmed@example.com، أو"), as at ", ".
_IDN_LETTER = ("a-z" + _class_ranges('CLMNSZ', (0xA1, 0x1FFF), (0x2070, 0x2FFF), (0x3040, 0xFEFF), (0xFF21, 0xFF3A),
                                     (0xFF41, 0xFF5A), (0xFF66, 0xFFFF))
               + _CJK_SYMBOL_LETTERS + _SUPPLEMENTARY_ALNUM)
# ZWNJ/ZWJ (U+200C/D) join letters inside IDNA labels (Persian, Devanagari conjuncts).
_IDN_CHAR = _IDN_LETTER + r"0-9\uff10-\uff19\u200c\u200d"
# The one punctuation IDNA2008 allows in a label in a Latin context (RFC 5892 A.3): the middle dot between
# two l's (col·legi.cat, Catalan); elsewhere it stays punctuation that ends a label.
_MIDDLE_DOT = "(?<=l)\u00b7(?=l)"


def _script_class(*scripts):
    """Character-class body of the code points of *scripts* (Unicode Script property)."""
    script = regex.compile('[{}]'.format(''.join(rf'\p{{Script={name}}}' for name in scripts)))
    # Every code point: regex has its own Unicode tables, which may assign what unicodedata does not.
    return _ranges(script.findall(''.join(map(chr, range(0x110000)))))


# The other CONTEXTO punctuation of RFC 5892 (A.4, A.5, A.7): Hebrew geresh and gershayim after a Hebrew character
# (צ׳.co.il) and the katakana middle dot in a label with Hiragana, Katakana or Han (例・え.jp).
# shortcut: the middle dot needs such a character right beside it, not anywhere in the label; widen if one shows up.
_KANA_HAN = _script_class('Hiragana', 'Katakana', 'Han')
# The two branches never match the same dot (the second needs a non-kana/Han left neighbour), so a run of
# dots between Han has one parse and a host with no TLD fails in linear time.
_KATAKANA_MIDDLE_DOT = f"(?:(?<=[{_KANA_HAN}])\u30fb|(?<![{_KANA_HAN}])\u30fb(?=[{_KANA_HAN}]))"
_GERESH = f"(?<=[{_script_class('Hebrew')}])[\u05f3\u05f4]"
# One run of label characters (X+ with those marks inside, a geresh also at its end), so a label stays X+(-+X+)*.
_IDN_RUN = rf"[{_IDN_CHAR}]+(?:(?:{_MIDDLE_DOT}|{_KATAKANA_MIDDLE_DOT})[{_IDN_CHAR}]+|{_GERESH}[{_IDN_CHAR}]*)*"
# A punycode TLD (xn--p1ai) is tried first, so ".xn" alone never ends the match; ZWNJ/ZWJ join
# letters inside it as in any other label.
_TLD = (rf"(?:xn--[a-z0-9-]*[a-z0-9]"
        rf"|{_NO_GLUE}[{_IDN_LETTER}](?:[\u200c\u200d]*{_NO_GLUE}[{_IDN_LETTER}])+)")
_LOCAL_CHAR = rf"{_NO_GLUE}[\w!#$%&'*+{MARKS}\u200c\u200d-]|\.(?!\.)|{_MIDDLE_DOT}"
EMAIL_REGEX = re.compile(
    rf"(?:(?:^|(?<=[^\w@.){MARKS}])|(?={_GLUE}))[\w+-](?:{_LOCAL_CHAR}){{0,63}}?"
    # A longer local part is taken whole from the start of its run (or a script change), so it is masked, not
    # left as text; one scan per run, so this stays linear too.
    rf"|(?:(?<![\w!#$%&'*+.@)\u00b7{MARKS}\u200c\u200d-])|(?={_GLUE}))[\w!#$%&'*+-](?:{_LOCAL_CHAR}){{64,}})"
    rf"(?<!\.)(@|[(<{{\[]at[)>}}\]])"
    # Host labels as X+(-+X+)*: one way to split a label, so a host with no TLD fails in linear time.
    rf"{_IDN_RUN}(?:-+{_IDN_RUN})*(?:\.{_IDN_RUN}(?:-+{_IDN_RUN})*)*(?:\.{_TLD})",
    flags=re.IGNORECASE | re.UNICODE,
)
# Cheap pre-check: every EMAIL_REGEX match holds its separator group, so text with no hit here has no
# email and the regex (13x slower than 0.7.1 on email-free text) is skipped. Keep in step with that group.
EMAIL_GATE = re.compile(r"@|[(<{\[]at[)>}\]]", re.IGNORECASE)

# Area-code parentheses in compatibility form (full-width （ ）, small ﹙ ﹚): the phone view reads them as
# "(" ")" (+49（30）123456) and the styled fold treats them as phone punctuation.
PHONE_PARENS = {ch: unicodedata.normalize('NFKC', ch) for ch, decomposition in DECOMPOSITIONS.items()
                if decomposition.split(' ', 1)[0] in ('<wide>', '<small>')
                and unicodedata.normalize('NFKC', ch) in '()'}

# Digit boundaries are "not next to a digit", so numbers glued to letters of
# any script (``电话9876543210``, ``फोन9876543210पर``) are still found.
PHONE_REGEX = re.compile(
    # National trunk-0 numbers in groups, 9+ digits: 020 7946 0958, 010-6552 1234, 06 12 34 56 78.
    # First, so the general pattern below cannot take only their leading groups.
    # 00 is the international prefix, left to the general pattern.
    r"(?<![\d_)])0[1-9](?=(?:[ .-]?\d){7})\d{0,3}(?:[ .-]\d{2,4}){2,4}"
    r"(?:\s?(?:ext\.?|[#x])\s?\d{2,6})?(?![\d_])"
    r"|"
    # A leading year group ("1999 2000", "2019 555-1234") stays for the year step.
    r"((?<![\d_)])(?!(?:19|20)\d\d[ .-])((\+?\d+|0{1,2}\d*?)[ .-]?)?(\(?\d{3,4}\)?/?[ .-]?)?"
    r"(\d{3}[ .-]?\d{4})(\s?(?:ext\.?|[#x-])\s?\d{2,6})?(?![\d_]))|\+?\d{4,5}[ .-/]\d{6,9}"
    # Indian mobile: 98765 43210, 098765-43210
    r"|(?<![\d_)])0?[6-9]\d{4}[ -]?\d{5}(?![\d_])"
    # 3-3-3 national numbers (699 123 456, 912 345 678); a space-grouped 9-digit amount reads as one too
    r"|(?<![\d_)])(?<!\d[ .,])\d{3}[ .-]\d{3}[ .-]\d{3}(?![\d_]|[ .,]\d)"
    # International: "+" then at least 8 digits in groups (+34 612 345 678, +91 98765 43210)
    # Area-code parentheses must be balanced, so a closing ")" of the sentence stays;
    # a bracket glued after the number ends it unless digits follow (+49 30 1234567（電話）).
    r"|(?<![\d+])\+(?=(?:[ .()-]{0,2}\d){8})\d{1,3}(?:[ .-]?(?:\(\d{1,5}\)|\d{1,5})){1,5}"
    r"(?:\s?(?:ext\.?|[#x])\s?\d{2,6})?(?![\d_]|\(\d)"
)

# IBAN: country code, check digits, then 11-30 letters/digits, in groups of 4 or unspaced.
# A match is only an IBAN when its ISO 7064 mod-97 checksum holds (contact.IBAN_VIEW).
IBAN_REGEX = re.compile(r"(?<![^\W_])[A-Z]{2}[0-9]{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,3})?(?![^\W_])")
# Payment card: 13-19 digits starting 2-6 (the issuer ranges), optional space/hyphen groups;
# only a card when the Luhn checksum holds (contact.CARD_VIEW).
CARD_REGEX = re.compile(r"(?<![\w+.,])[2-6](?:[ -]?[0-9]){12,18}(?![\w])")
# Identifier token (licence/ID codes like RBCON68T08): 6+ uppercase letters and digits of any
# script, at least 2 digits and 1 letter, not glued to another letter, digit or mark.
IDENTIFIER_REGEX = regex.compile(
    r"(?<![\p{L}\p{N}\p{M}_])(?=(?:\p{Lu}*\p{Nd}){2})(?=\p{Nd}*\p{Lu})[\p{Lu}\p{Nd}]{6,}(?![\p{L}\p{N}\p{M}_])"
)

# IPv4 and IPv6 (incl. "::" forms and IPv4-mapped tails). Runs before the
# date/year/phone steps so "2001:4860:4860::8888" is not read as a year.
_IPV4 = r"(?:(?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])\.){3}(?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])"
_H16 = r"[0-9A-Fa-f]{1,4}"
_LS32 = rf"(?:{_H16}:{_H16}|{_IPV4})"
# Every valid layout (RFC 3986 IPv6address), so a match is always the whole
# address. Bare "::" is left alone: it is not personal data.
_IPV6 = "|".join([
    rf"(?:{_H16}:){{6}}{_LS32}",
    rf"::(?:{_H16}:){{5}}{_LS32}",
    rf"(?:{_H16})?::(?:{_H16}:){{4}}{_LS32}",
    rf"(?:(?:{_H16}:){{0,1}}{_H16})?::(?:{_H16}:){{3}}{_LS32}",
    rf"(?:(?:{_H16}:){{0,2}}{_H16})?::(?:{_H16}:){{2}}{_LS32}",
    rf"(?:(?:{_H16}:){{0,3}}{_H16})?::{_H16}:{_LS32}",
    rf"(?:(?:{_H16}:){{0,4}}{_H16})?::{_LS32}",
    rf"(?:(?:{_H16}:){{0,5}}{_H16})?::{_H16}",
    rf"(?:{_H16}:){{0,6}}{_H16}::",
])
# Only ASCII letters/digits glue to an address, so CJK or Hindi text around it
# ("地址fe80::1") does not hide it. A "label:" prefix is allowed when the label
# cannot itself be the start of the address ("client_ip:fd12::1", "src:fe80::1").
_IPV6_START = (
    r"(?:(?<![A-Za-z0-9_:.])"
    r"|(?<=[G-Zg-z_]:)|(?<=[G-Zg-z_][0-9A-Fa-f]:)"
    r"|(?<=[G-Zg-z_][0-9A-Fa-f]{2}:)|(?<=[G-Zg-z_][0-9A-Fa-f]{3}:)"
    r"|(?<=[0-9A-Fa-f]{5}:))"
)
IP_REGEX = re.compile(
    rf"{_IPV6_START}(?:{_IPV6})(?![A-Za-z0-9_:]|\.[0-9])"
    # A dot before it blocks only after a digit (1.2.3.4.5); after a letter it ends a word or sentence
    # (服务器.8.8.8.8, where a CJK full stop may fold to ".").
    rf"|(?<![0-9])(?<![0-9]\.)(?:{_IPV4})(?![0-9]|\.[0-9])"
    # IPv4 in other digit scripts (١٩٢.١٦٨.١.١): needs one non-ASCII digit ([^\x00-\x7f\D]),
    # so ASCII matching (and the browser export, where \d is ASCII) is unchanged.
    r"|(?<!\d)(?<!\d\.)(?=[\d.]*[^\x00-\x7f\D])\d{1,3}(?:\.\d{1,3}){3}(?!\d|\.\d)"
)

_NUMBER_BODY = (
    r"[+–-]?(([1-9]\d{0,2}(,\d{3})+(\.\d*)?)"
    r"|([1-9]\d{0,2}([ .]\d{3})+(,\d*)?)|(\d*?[.,]\d+)|\d+)"
)
# Latin, Greek and Cyrillic put spaces between words, so a digit glued to their
# letters is part of a word (GPT4, Qwen3-8B, H2O) and is kept by default.
# Scripts without spaces (``金额5000元``) still get the number replaced.
# shortcut: common ISO 4217 codes only (EUR4500 is an amount, GPT4 is a name); add codes as needed.
_CURRENCY_CODES = (
    "EUR|USD|GBP|JPY|CNY|INR|CHF|CAD|AUD|NZD|SEK|NOK|DKK|PLN|CZK|HUF|RUB|UAH|TRY|"
    "BRL|MXN|ARS|ZAR|KRW|SGD|HKD|THB|IDR|MYR|PHP|VND|PKR|BDT|LKR|NPR|AED|SAR|ILS|EGP|NGN|KES"
)
NUMBERS_REGEX = re.compile(
    rf"(?:(?<![\d_,.{_SPACED_LETTERS}])|(?<=\b(?:{_CURRENCY_CODES})))"
    rf"{_NUMBER_BODY}(?![\d_{_SPACED_LETTERS}])"
)
# Strict mode (replace_numbers_in_words=True): only an adjacent digit blocks a match.
NUMBERS_IN_WORDS_REGEX = re.compile(rf"(?<![\d_,.]){_NUMBER_BODY}(?![\d_])")

# replace_numbers_mode='identifiers' and phone_reject_embedded: what makes a digit run part of a longer token.
# Joiners build one token from parts (INV-2023-107683, 2023/11, 1.5, 1,250, a_1, 100–200). ":", "=" and "#"
# are left out: they introduce a value ("ID:12345678", "id=12345678", "#12345"), so that value stays maskable.
PART_JOINERS = r"\-\u2010-\u2015/+*._|,"
# A placeholder edge joined to the digits counts as a token part too ("<YEAR>", or the cleaner's protected
# \U00100000..\U00100001 key for an emitted value), so "INV-<YEAR>-107683" keeps its tail instead of a
# second, partial mask.
KEY_OPEN, KEY_CLOSE = "\U00100000", "\U00100001"
_LEFT_EDGE, _RIGHT_EDGE = ">" + KEY_CLOSE, "<" + KEY_OPEN
CURRENCY_SIGNS = CATEGORY_CHARS['Sc']


def token_glue(in_words: bool = False) -> str:
    """Characters that glue to a digit run: digits, "_", and (unless in_words) Latin/Greek/Cyrillic letters.
    Letters of unspaced scripts (CJK, Thai) do not glue, as for NUMBERS_REGEX: 金额50000元 is a whole number."""
    return r"\d_" if in_words else rf"\d_{_SPACED_LETTERS}"


# A date written as a spaced/hyphen-joined digit run (05 11 2023, 2023 11-05): the year step keeps masking its
# year rather than leaving it to the run's <NUMBER>, as before (the date step has no spaced date).
SPACED_DATE_REGEX = re.compile(
    r"\d{1,2}[ \-\u2010\u2011]\d{1,2}[ \-\u2010\u2011](?:\d\d){1,2}|\d{4}[ \-\u2010\u2011]\d{1,2}[ \-\u2010\u2011]\d{1,2}")
# Most groups a thousands-grouped amount may have (1 234 567 890 123 456); a longer spaced run is masked whole
# even when grouped by thousands, so the streaming guard may split a long run (a number table) where both halves
# keep more groups than this and each half is masked whole too.
THOUSANDS_MAX_GROUPS = 6


@functools.cache
def identifier_numbers_regex(min_digits: int, in_words: bool = False, *, letter_prefix: bool = False,
                             two_part: bool = False, spaced_pair: bool = False,
                             keep_currency_code: bool = False) -> re.Pattern[str]:
    """A whole digit run of at least *min_digits* digits that is not part of a longer token (glued to letters,
    digits or a joiner and another token part) and not an amount, decimal, percent, time or range: those
    shapes either join their parts (1,250.00, 100-200; 14:30 has 2-digit parts) or touch a % / currency sign.
    A whole digit-only token of three or more hyphen-joined groups with *min_digits* digits in all (an SSN-like
    123-45-6789) is one identifier too, unless it is date-shaped (2023-11-05, 05-11-23).
    Opt-in whole-token shapes (format rules, independent of *min_digits*):
    letter_prefix: 1-3 uppercase Latin letters, 6+ digits, 0-2 uppercase letters (AB1234567, X12345678Z),
    not a currency code before digits (USD1000000).
    two_part: two digit groups joined by "/" or a hyphen, the first 4+ digits, 7+ digits in all, an optional
    uppercase check letter (123456/7890, 12345-678901, 123456-789K), or by "." with 4+ digits on each side
    (12345.67890); not a year pair (2023-2024), a round range (1000-3000) or a time (1234-5678:30).
    spaced_pair: "DDD DD" (a 5-digit postal code written 123 45), and a run of 3+ digit groups joined by spaces
    and/or hyphens (at least one space) with 7+ digits (123 45 6789, 123 45-6789, 1234 56-78-90) as one
    identifier (a date-shaped run, 05 11 2023, too: the date step has no spaced date), unless
    thousands-grouped (1 234 567, up to THOUSANDS_MAX_GROUPS groups) or beside a decimal, % or currency sign.
    keep_currency_code: a whole digit run right before or after an uppercase ISO 4217 code (_CURRENCY_CODES), with
    at most one space between ("1250 EUR", "EUR 1250"), is an amount. Only that amount shape: an SSN-, slash-,
    dot- or space-joined ID beside a code ("123-45-6789 USD", "EUR 123 45 6789") is still masked whole."""
    glue, j, unit = token_glue(in_words), PART_JOINERS, CURRENCY_SIGNS
    hyphen, end = r"[\-\u2010\u2011]", rf"(?![\d{j}])"
    date = rf"(?:\d{{1,2}}{hyphen}\d{{1,2}}{hyphen}(?:\d\d){{1,2}}|\d{{4}}{hyphen}\d{{1,2}}{hyphen}\d{{1,2}}){end}"
    grouped = rf"(?!{date})(?=\d(?:{hyphen}?\d){{{min_digits - 1}}})\d+(?:{hyphen}\d+){{2,}}"
    no_unit_before, no_unit_after = rf"(?<![{unit}])(?<![{unit}] )", rf"(?! ?[%‰{unit}])"
    no_code_before = no_code_after = ""
    if keep_currency_code:  # uppercase codes only, not inside a Latin/Greek/Cyrillic word ("PEUR", "EURO")
        code = rf"(?:{_CURRENCY_CODES})"
        no_code_before = rf"(?<!(?<![{_SPACED_LETTERS}]){code})(?<!(?<![{_SPACED_LETTERS}]){code} )"
        no_code_after = rf"(?! ?{code}(?![{_SPACED_LETTERS}]))"
    shapes = [
        rf"(?<![{glue}])(?<![{glue}{_LEFT_EDGE}][{j}]){no_unit_before}"
        rf"(?:{grouped}|{no_code_before}\d{{{min_digits},}}{no_code_after})"
        rf"(?![{glue}])(?![{j}][{glue}{_RIGHT_EDGE}]){no_unit_after}"
    ]
    if letter_prefix:
        shapes.append(rf"(?<![\w\-/.])(?!(?:{_CURRENCY_CODES})\d)[A-Z]{{1,3}}\d{{6,}}[A-Z]{{0,2}}(?!\w)")
    if two_part:
        slash = r"[/\-\u2010\u2011]"
        year = r"(?:1[89]|20)\d\d"
        shapes.append(
            rf"(?<![\w\-/.,+]){no_unit_before}"
            rf"(?!{year}{slash}{year}(?!\w))(?!\d*00{slash}\d*00(?!\w))"
            rf"(?=(?:\d{slash}?){{7}})\d{{4,}}{slash}\d{{2,6}}[A-Z]?(?!\w|[\-/.,:]\d){no_unit_after}"
        )
        shapes.append(rf"(?<![\w\-/.,+]){no_unit_before}\d{{4,}}\.\d{{4,}}(?!\w|[\-/.,]\d){no_unit_after}")
    if spaced_pair:
        # A run of 3+ digit groups joined by spaces and/or hyphens with at least one space and 7+ digits
        # (an ID or SSN written 123 45 6789, 123 45-6789, 1234 56-78-90) is one identifier, unless
        # thousands-grouped (1 234 567, an amount, up to THOUSANDS_MAX_GROUPS groups); an all-hyphen run is
        # the grouped rule's. It goes first, so no other shape takes part of the run.
        sep = r"[ \-\u2010\u2011]"
        tail = rf"(?!\w|[.,:]\d|{sep}\d){no_unit_after}"
        shapes.insert(0, rf"(?<![\w.,:/\-\u2010\u2011])(?<!\d ){no_unit_before}"
                         rf"(?!\d{{1,3}}(?: \d{{3}}){{1,{THOUSANDS_MAX_GROUPS - 1}}}{tail})(?=\d+(?:{sep}\d+)*? \d)"
                         rf"(?=(?:\d{sep}?){{7}})\d+(?:{sep}\d+){{2,}}{tail}")
        # "DDD DD", a 5-digit postal code
        shapes.append(rf"(?<![\w.,:/\-])(?<!\d ){no_unit_before}\d{{3}} \d{{2}}{tail}")
    return re.compile("|".join(shapes))


# phone_reject_embedded: a digit-only phone match with a letter or "_" on either side, directly or across a
# joiner, or one that is a yyyymmdd date or a yyyymmddhhmm[ss] timestamp (20231105123456), not a phone. A
# date with other digits after it (2023010123) can be a real phone, so only these exact shapes are rejected.
# Digits across a joiner do not count: in "06-12345678" the phone regex takes only "12345678", and rejecting
# it would leave the whole phone visible.
_PHONE_GLUE = rf"_{_SPACED_LETTERS}"
PHONE_EMBEDDED_LEFT = re.compile(rf"(?:[{_PHONE_GLUE}]|[{_PHONE_GLUE}{_LEFT_EDGE}][{PART_JOINERS}])\Z")
PHONE_EMBEDDED_RIGHT = re.compile(rf"[{_PHONE_GLUE}]|[{PART_JOINERS}][{_PHONE_GLUE}{_RIGHT_EDGE}]")
YYYYMMDD_STAMP = re.compile(
    r"(?:19|20)\d\d(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])(?:(?:[01]\d|2[0-3])[0-5]\d(?:[0-5]\d)?)?")

LINEBREAK_REGEX = re.compile(r"((\r\n)|[\n\v])+")
TWO_LINEBREAK_REGEX = re.compile(r"((\r\n)|[\n\v])+((\r\n)|[\n\v])+")
MULTI_WHITESPACE_TO_ONE_REGEX = re.compile(r"\s+")
NONBREAKING_SPACE_REGEX = re.compile(r"(?!\n)\s+")

_HTML_ENTITY = r'&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});'
# Tag-shaped only (html_tag_strict): a name after "<" ("<b>", "</b>", "<br/>", "<!DOCTYPE", "<?xml") or a
# comment, so "Name <a@b.com>", "<https://...>" and a stray "a < b ... >" are not markup. A comment ends at
# "-->" (its body may hold "<" and ">"); an unclosed "<!--" falls back to the next ">". Linear: a comment body
# stops at the next "<!--" and [^<>] at the next bracket, and the pattern is JS-portable (browser export).
HTML_REGEX = re.compile(
    r'<!--(?:(?!<!--)[\s\S])*?-->|<!--[^<>]*>|<[!?/]?[A-Za-z][A-Za-z0-9:-]*(?:\s[^<>]*)?/?>|' + _HTML_ENTITY,
    flags=re.UNICODE | re.IGNORECASE,
)
# 0.8.0 rule (html_tag_strict=False): any "<...>" on one line.
HTML_REGEX_ANY = re.compile('<.*?>|' + _HTML_ENTITY, flags=re.UNICODE | re.IGNORECASE)

# Punctuation (category P) of the CJK symbol, katakana middle dot and full-width blocks, except the
# full-width forms of URL syntax (／ ？ ＃ ％ ＆ ＠ ： ． － ＿), which the styled fold turns into ASCII.
_WIDE_PUNCT = re.escape(''.join(
    chr(cp) for cp in [*range(0x3000, 0x3040), 0x30FB, *range(0xFF00, 0xFF66)]
    if unicodedata.category(chr(cp))[0] == 'P' and unicodedata.normalize('NFKC', chr(cp)) not in '/?#%&@:.-_'))

# Fixed: Unicode ranges use actual Unicode escapes (not literal \\u in raw strings)
def _url_regex(trailing_punct_outside):
    # url_trailing_punct_outside: . , ; : ! ? ' belong to a path or query only when more URL text
    # follows ("/a.html", "?a=1,2&b=3"), so "see https://x.com/a/b." keeps its full stop as prose.
    tail = r".,;:!?'" if trailing_punct_outside else ""
    url_char = rf"{_NO_GLUE}[^{tail}\)\]\}}\s\"<>{_WIDE_PUNCT}]"
    punct = rf"|[{tail}]+(?=[\)\]\}}]*{url_char})" if trailing_punct_outside else ""
    path_char = rf"(?:{url_char}{punct}|[\)\]\}}]+(?=[.,;:!$*@]*[\w&=/?#%+~-]))"
    return re.compile(
        # Only an ASCII letter or digit glues to the scheme, so a URL right after CJK/Thai text
        # (請見https://..., ดูhttps://...) is still found.
        r"(?:^|(?<![A-Za-z0-9_\/\.]))(?:"
        # protocol identifier
        r"(?:(?:https?:\/\/|ftp:\/\/|www\d{0,3}\.))"
        # user:pass authentication; never across / ? # " < > so a later "@" in the
        # query or the markup is not taken as user info
        # (user up to the first ":", one way to split it, so a long run with no "@" fails in linear time)
        r"(?:(?=[^\s/?#\"<>@])[^\s/?#\"<>@:]*(?::[^\s/?#\"<>@]*)?@)?" r"(?:"
        # bracketed IPv6 host; private and loopback IPv4 hosts match too, so the whole
        # URL (query values included) is replaced, not just the address
        r"\[[0-9a-f:.]*:[0-9a-f:.]*\]" r"|"
        # IPv4 host, whole: a longer octet (any decimal digit, "1௨3") or another label after it (any
        # label start the host grammar below takes, "école" too) is a host name
        rf"{_IPV4}(?![0-9{_DECIMAL}]|\.[{_IDN_CHAR}])"
        r"|"
        # host name (with proper Unicode support, beyond the BMP too); the label characters of an email
        # domain, so CJK and full-width prose punctuation (，、「) glued after a host ends it
        # (labels as X+(-+X+)*, as in EMAIL_REGEX, so a host with no TLD fails in linear time)
        rf"{_IDN_RUN}(?:-+{_IDN_RUN})*"
        # domain name
        rf"(?:\.{_IDN_RUN}(?:-+{_IDN_RUN})*)*"
        # TLD identifier (punycode first, so ".xn" alone never ends the host)
        rf"(?:\.(?:xn--[a-z0-9-]*[a-z0-9]|(?:{_NO_GLUE}[{_IDN_LETTER}]){{2,}}))"
        r"|" r"(?:(localhost))" r")"
        # fully qualified host's final dot ("example.com./?q="), only when a path, port or query follows,
        # so a sentence-ending period stays outside the URL
        r"(?:\.(?=/|:\d|[?#][^\s\"<>]))?"
        # port number
        r"(?::\d{2,5}\b)?"
        # resource path, or a query/fragment right after the host ("?id=", "#token=");
        # " < > are never raw in a URL, so markup around it survives. A bare trailing "?"
        # stays punctuation. CJK and full-width punctuation (，。！（ ・) ends it, as in an email
        # domain, so CJK prose glued after a URL stays. ) ] } belong to the URL only when more URL text follows
        # ("?f=(a)&email=...", "report(1).pdf", "a(b);jsessionid=", a run as in "/(S(id))/a.aspx"); RFC 3986
        # sub-delimiters (but a possessive ') between bracket and text count as URL text, so a closing bracket
        # around a URL stays outside it.
        rf"(?:\/{path_char}*|[?#]{path_char}+)?"
        # Safety net: a scheme-led run whose host fails the grammar above (a look-alike character in it) is
        # still a URL up to a space or prose punctuation, so its query values never stay visible.
        rf"|(?:https?|ftp):\/\/(?:{_NO_GLUE}[^\s\"<>{_WIDE_PUNCT}])+(?<=[^\W_]|[{MARKS}]))",
        flags=re.UNICODE | re.IGNORECASE,
    )


URL_REGEX = _url_regex(True)
# 0.8.0 rule (url_trailing_punct_outside=False): a path or query takes trailing . , ; : ! ? '
URL_REGEX_LEGACY = _url_regex(False)


strange_double_quotes = [
    "«", "‹", "»", "›", "„", "\u201c", "‟", "\u201d",
    "❝", "❞", "❮", "❯", "〝", "〞", "〟", "＂",
]
strange_single_quotes = ["'", "‛", "\u2019", "❛", "❜", "`", "´", "\u2018", "\u2019"]

DOUBLE_QUOTE_REGEX = re.compile("|".join(strange_double_quotes))
SINGLE_QUOTE_REGEX = re.compile("|".join(strange_single_quotes))

YEAR_REGEX = re.compile(r"(?<![\d_])(19|20)\d{2}(?![\d_])")

# ---------------------------------------------------------------------------
# Multilingual month name vocabulary (EN, NL, DE, ES, FR, PT, IT, TH)
# Used by DATE_REGEX (exact match) and fuzzy_replace_dates (RapidFuzz).
# Keys are canonical lowercase forms; values are frozensets of language codes
# (some month names are shared across languages, e.g. "juni" is NL and DE,
# "agosto" is ES/PT/IT, "marzo" is ES/IT, "novembre" is FR/IT).
# ---------------------------------------------------------------------------
MONTH_NAMES_MULTILINGUAL = {
    # English
    "january": frozenset({"en"}), "february": frozenset({"en"}),
    "march": frozenset({"en"}), "april": frozenset({"en", "nl", "de"}),
    "may": frozenset({"en"}), "june": frozenset({"en"}),
    "july": frozenset({"en"}), "august": frozenset({"en", "de"}),
    "september": frozenset({"en", "nl", "de"}), "october": frozenset({"en"}),
    "november": frozenset({"en", "nl", "de"}), "december": frozenset({"en", "nl"}),
    "jan": frozenset({"en"}), "feb": frozenset({"en"}), "mar": frozenset({"en"}),
    "apr": frozenset({"en"}), "jun": frozenset({"en"}), "jul": frozenset({"en"}),
    "aug": frozenset({"en"}), "sep": frozenset({"en"}), "sept": frozenset({"en"}),
    "oct": frozenset({"en"}), "nov": frozenset({"en"}), "dec": frozenset({"en"}),
    # Dutch (unique to NL; shared forms tagged above)
    "januari": frozenset({"nl"}), "februari": frozenset({"nl"}),
    "maart": frozenset({"nl"}), "mei": frozenset({"nl"}),
    "juni": frozenset({"nl", "de"}), "juli": frozenset({"nl", "de"}),
    "augustus": frozenset({"nl"}), "oktober": frozenset({"nl", "de"}),
    "mrt": frozenset({"nl"}), "okt": frozenset({"nl", "de"}),
    # German (unique to DE; shared forms tagged above)
    "januar": frozenset({"de"}), "februar": frozenset({"de"}),
    "märz": frozenset({"de"}), "mai": frozenset({"de"}),
    "dezember": frozenset({"de"}),
    "mär": frozenset({"de"}), "mrz": frozenset({"de"}), "dez": frozenset({"de"}),
    # Spanish
    "enero": frozenset({"es"}), "febrero": frozenset({"es"}),
    "marzo": frozenset({"es", "it"}), "abril": frozenset({"es", "pt"}),
    "mayo": frozenset({"es"}), "junio": frozenset({"es"}),
    "julio": frozenset({"es"}), "agosto": frozenset({"es", "pt", "it"}),
    "septiembre": frozenset({"es"}), "octubre": frozenset({"es"}),
    "noviembre": frozenset({"es"}), "diciembre": frozenset({"es"}),
    "ene": frozenset({"es"}), "abr": frozenset({"es"}),
    "ago": frozenset({"es"}), "dic": frozenset({"es"}),
    # French
    "janvier": frozenset({"fr"}), "février": frozenset({"fr"}),
    "mars": frozenset({"fr"}), "avril": frozenset({"fr"}),
    "juin": frozenset({"fr"}), "juillet": frozenset({"fr"}),
    "août": frozenset({"fr"}), "septembre": frozenset({"fr"}),
    "octobre": frozenset({"fr"}), "novembre": frozenset({"fr", "it"}),
    "décembre": frozenset({"fr"}),
    "fév": frozenset({"fr"}), "avr": frozenset({"fr"}),
    "aoû": frozenset({"fr"}), "déc": frozenset({"fr"}),
    # Portuguese
    "janeiro": frozenset({"pt"}), "fevereiro": frozenset({"pt"}),
    "março": frozenset({"pt"}), "maio": frozenset({"pt"}),
    "junho": frozenset({"pt"}), "julho": frozenset({"pt"}),
    "setembro": frozenset({"pt"}), "outubro": frozenset({"pt"}),
    "novembro": frozenset({"pt"}), "dezembro": frozenset({"pt"}),
    "set": frozenset({"pt"}), "out": frozenset({"pt"}),
    # Italian
    "gennaio": frozenset({"it"}), "febbraio": frozenset({"it"}),
    "aprile": frozenset({"it"}), "maggio": frozenset({"it"}),
    "giugno": frozenset({"it"}), "luglio": frozenset({"it"}),
    "settembre": frozenset({"it"}), "ottobre": frozenset({"it"}),
    "dicembre": frozenset({"it"}),
    "gen": frozenset({"it"}), "mag": frozenset({"it"}),
    "giu": frozenset({"it"}), "lug": frozenset({"it"}),
    "ott": frozenset({"it"}),
}

# Flat set for quick membership check (lowercase)
_MONTH_NAMES_SET = frozenset(MONTH_NAMES_MULTILINGUAL.keys())

# Full month names only (no abbreviations) — used by fuzzy matching.
# Abbreviations are already caught by exact DATE_REGEX; including them in fuzzy
# matching causes false positives (e.g. "many" -> "may" at 85.7%, "jungle" -> "june"
# at 80%). Full names have much better separation against common English words.
# Empirically calibrated: min TP score 83.3, max non-month FP score 85.7 (contextual
# pre-filter via FUZZY_DATE_CANDIDATE_REGEX eliminates the remaining overlap).
FUZZY_MONTH_VOCABULARY = tuple(
    name for name in _MONTH_NAMES_SET if len(name) > 4
)

# Language code mapping (Lingua detection names -> vocabulary language codes)
_LANG_TO_VOCAB = {
    "ENGLISH": "en",
    "DUTCH": "nl",
    "GERMAN": "de",
    "SPANISH": "es",
    "FRENCH": "fr",
    "PORTUGUESE": "pt",
    "ITALIAN": "it",
}

# Per-language fuzzy vocabulary (full names only, len > 4)
FUZZY_MONTH_VOCABULARY_BY_LANG = {}
for _lang_name, _lang_code in _LANG_TO_VOCAB.items():
    FUZZY_MONTH_VOCABULARY_BY_LANG[_lang_name] = tuple(
        name for name, langs in MONTH_NAMES_MULTILINGUAL.items()
        if _lang_code in langs and len(name) > 4
    )

# Build month name alternation for regex (longest first to avoid prefix issues)
_MONTH_ALTERNATION = '|'.join(
    map(re.escape, sorted(_MONTH_NAMES_SET, key=len, reverse=True))
)


# ---------------------------------------------------------------------------
# Factory functions for building per-instance month/date/fuzzy data.
# Used by TextCleaner to support custom languages via TextCleanerConfig.
# Module-level constants above remain the defaults.
# ---------------------------------------------------------------------------

def build_month_names_dict(extra_month_names: Mapping[str, Iterable[str]] | None = None,
                           builtin_languages: Iterable[str] | None = None) -> dict[str, frozenset[str]]:
    """Build month-name lookup dict, merging defaults + custom names.

    Args:
        extra_month_names: dict mapping language ISO code -> tuple of month names
                          e.g. {'pl': ('styczen', 'luty', ...)}
        builtin_languages: ISO codes of the built-in month data to keep (None = all).
    Returns:
        dict mapping lowercase month name -> frozenset of ISO codes
    """
    keep = None if builtin_languages is None else frozenset(builtin_languages)
    result = {name: langs if keep is None else langs & keep
              for name, langs in MONTH_NAMES_MULTILINGUAL.items() if keep is None or langs & keep}
    if extra_month_names:
        for iso_code, names in extra_month_names.items():
            for name in names:
                key = name.lower()
                if key in result:
                    result[key] = result[key] | frozenset({iso_code})
                else:
                    result[key] = frozenset({iso_code})
    return result


_DAY = r'(?:0?[1-9]|[12]\d|3[01])'
_MONTH = r'(?:0?[1-9]|1[0-2])'
# Numeric dates with a 4-digit year and one separator (-, / or .), optional time:
# 2024-01-15, 2024/1/15, 2024.01.15, 15.01.2024, 01/15/2024, 2024-01-15T10:30:00Z, 15-01-2024 10:30.
# Year-last takes day-month or month-day, so one part must be a month (no 31/31/2024).
# Year-first with dots needs 2-digit month/day so CalVer "2024.1.5" stays; a digit
# joined by . or , on either side means a version, IP or decimal ("10.1.12.2024").
# CJK and Korean write the year, month and day with unit characters (1990年1月15日, 1990년 1월 15일), and Korean
# also as dots each followed by a space (1990. 1. 15.); the unit characters are separators like "/".
_NUMERIC_DATE = (
    r'(?<!\d[.,])(?:'
    r'\d{4}(?:([-/])' + _MONTH + r'\1' + _DAY + r'|\.(?:0[1-9]|1[0-2])\.(?:0[1-9]|[12]\d|3[01])'
    r'|\s?年\s?' + _MONTH + r'\s?月\s?' + _DAY + r'\s?日|년\s?' + _MONTH + r'월\s?' + _DAY + r'일'
    r'|\.\s' + _MONTH + r'\.\s' + _DAY + r'\.)'
    r'|' + _DAY + r'([-/.])' + _MONTH + r'\2\d{4}|' + _MONTH + r'([-/.])' + _DAY + r'\3\d{4}'
    r')'
    r'(?:[T ]\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d+)?(?:\s?Z|\s?[+-]\d{2}:?\d{2})?)?'
    r'(?![.,]\d)'
)


# A date is a word of its own where no digit, "_" or letter of a spaced script (Latin, Greek, Cyrillic) touches it;
# letters of an unspaced script (Thai, CJK) end a word, as they do for numbers (生日是15 January 1990谢谢).
_DATE_START = r'(?<![\d_' + _SPACED_LETTERS + r'])'
_DATE_END = r'(?![\d_' + _SPACED_LETTERS + r'])'


_HAN_NUMERIC = {ch: value for ch in map(chr, range(0x3000, 0x2FA20))
                if (value := unicodedata.numeric(ch, None)) is not None and regex.match(r'\p{Script=Han}', ch)}


def _han_numerals(*values):
    """Han-script numerals (〇 一 十 廿, financial forms) whose Unicode numeric value is in *values*."""
    return ''.join(ch for ch, value in _HAN_NUMERIC.items() if value in values)


_HAN_DIGIT, _HAN_1_9 = f'[{_han_numerals(*range(10))}]', f'[{_han_numerals(*range(1, 10))}]'
_HAN_TEN = f'[{_han_numerals(10)}]'
# A date in Han numerals (一九九〇年一月十五日): four digits of value 0-9, a month 1-12 and a day 1-31 by numeric value.
_HAN_DATE = (
    rf'{_HAN_DIGIT}{{4}}年(?:{_HAN_TEN}[{_han_numerals(1, 2)}]?|{_HAN_1_9})月'
    rf'(?:[{_han_numerals(2)}]{_HAN_TEN}{_HAN_1_9}?|[{_han_numerals(3)}]{_HAN_TEN}[{_han_numerals(1)}]?'
    rf'|[{_han_numerals(20, 30)}]{_HAN_1_9}?|{_HAN_TEN}{_HAN_1_9}?|{_HAN_1_9})日'
)
# fold_styled_characters: a circled day number (⑮ January 1990), which the fold keeps where no digit sits beside it.
_CIRCLED_DAYS = ''.join(chr(cp) for cp in range(0x2460, 0x3300)
                        if unicodedata.decomposition(chr(cp)).startswith('<circle>')
                        and unicodedata.normalize('NFKC', chr(cp)).isdigit()
                        and 1 <= int(unicodedata.normalize('NFKC', chr(cp))) <= 31)


def build_date_regex(month_names_dict: Mapping[str, frozenset[str]] | None = None, styled: bool = False,
                     slash_year: bool = True) -> re.Pattern[str]:
    """Build DATE_REGEX from a month-names dict.

    Args:
        month_names_dict: output of build_month_names_dict().
                         Uses module-level MONTH_NAMES_MULTILINGUAL if None.
        styled: a month-name date's day may be a circled number (fold_styled_characters).
        slash_year: also match a month name, "/" and a 2- or 4-digit year (October/55, Oct/2019).
    Returns:
        Compiled regex pattern
    """
    names = MONTH_NAMES_MULTILINGUAL if month_names_dict is None else month_names_dict
    # Escaped: a dotted abbreviation (ก.พ.) is literal, not "any character".
    alt = '|'.join(map(re.escape, sorted(names.keys(), key=len, reverse=True)))
    day = rf'(?:\d{{1,2}}|[{_CIRCLED_DAYS}])' if styled else r'\d{1,2}'
    month_dates = (
        r'(?:' + alt + r')'
        r'\.?\s+' + day + r'(?:st|nd|rd|th)?,?\s+\d{4}'
        r'|'
        + day + r'\.?\s+(?:de\s+)?(?:' + alt + r')'
        r'(?:\s+de)?\s+\d{4}'
        + (r'|(?:' + alt + r')\.?/(?:\d{4}|\d{2})' if slash_year else '')
    )
    # A numeric date glued to letters of an unspaced script (Thai, CJK) is a date, as numbers are.
    return re.compile(
        _DATE_START + _NUMERIC_DATE + _DATE_END + '|' + _HAN_DATE
        + ('|' + _DATE_START + r'(?:' + month_dates + r')' + _DATE_END if names else ''),
        flags=re.IGNORECASE,
    )


def build_fuzzy_vocabulary(
    month_names_dict: Mapping[str, frozenset[str]] | None = None,
    lang_to_vocab: Mapping[str, str] | None = None,
) -> tuple[tuple[str, ...], dict[str, tuple[str, ...]]]:
    """Build fuzzy month vocabulary (full names only, len > 4).

    Returns:
        (all_vocab_tuple, by_lang_dict)
    """
    names = MONTH_NAMES_MULTILINGUAL if month_names_dict is None else month_names_dict  # {} = no months
    lvocab = lang_to_vocab or _LANG_TO_VOCAB

    all_vocab = tuple(name for name in names if len(name) > 4)

    by_lang = {}
    for lang_name, lang_code in lvocab.items():
        by_lang[lang_name] = tuple(
            name for name, langs in names.items()
            if lang_code in langs and len(name) > 4
        )
    return all_vocab, by_lang

# Date patterns: numeric dates and date-times, multilingual month names
DATE_REGEX = build_date_regex()

# Candidate pattern for fuzzy date matching: a word near a number that could be a date.
# Matches patterns like "Janury 15, 2024" or "15 Feburary 2024" where the month is misspelled.
FUZZY_DATE_CANDIDATE_REGEX = re.compile(
    _DATE_START + r'(?:'
    # Word then day then year: "Janury 15, 2024"
    r'([A-Za-z\u00e0-\u00fc]{3,12})\.?\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}'
    r'|'
    # Day then word then year: "15 Feburary 2024" or "15. Janur 2024" or "15 de enro de 2024"
    r'\d{1,2}\.?\s+(?:de\s+)?([A-Za-z\u00e0-\u00fc]{3,12})(?:\s+de)?\s+\d{4}'
    r')' + _DATE_END,
    flags=re.IGNORECASE,
)

# Fixed: removed commas and spaces from character class (were matching literally)
ISOLATED_LETTERS_REGEX = re.compile(
    r"(?:^|\s)[BCDEFGHIJKLMNOPQRSTVWXYZ](?=\s|$)",
    flags=re.UNICODE | re.IGNORECASE,
)

# Isolated = whitespace (or text edge) on both sides, so "(born <YEAR>)." keeps its ")." in any script.
ISOLATED_SPECIAL_SYMBOLS_REGEX = re.compile(
    r"(?<!\S)[:_.|><;·}@~!?+#)({,/\\\\^]+(?!\S)",
    flags=re.UNICODE | re.IGNORECASE,
)

# Pre-compiled bracket/brace removal patterns (moved from special.py runtime)
BRACKET_CONTENT_REGEX = re.compile(r'\[[^\]]+\]')
BRACE_CONTENT_REGEX = re.compile(r'\{[^}]+\}')
# innermost spans only, used when a length limit is set; [...] never spans a brace, so it cannot cut into JSON
BRACKET_INNER_REGEX = re.compile(r'\[[^\[\]{}]+\]')
BRACE_INNER_REGEX = re.compile(r'\{[^{}]+\}')
# A quote, dash, * or % is isolated unless a letter, digit or mark of any script touches it
# (special.py checks the neighbours), so "ইসলাম-এর" and "東京-大阪" keep their hyphen.
ISOLATED_QUOTES_REGEX = re.compile(r"['\"\-*%]")

# Pre-compiled punctuation pattern (was re-compiled on every call in special.py)
PUNCTUATION_REGEX = re.compile('[' + re.escape(string.punctuation) + ']')

# Sentence boundary pattern for NER text splitting
# Used by ner.py split_text() as preferred split point before CHUNK_DELIMITERS.
# Uses the `regex` library (drop-in for `re`) to support variable-length lookbehind.
# (?<![A-Z][a-z]{0,3}\.) — do NOT split after abbreviation (Dr., Mr., Ltd., Corp.)
#                           [A-Z][a-z]{0,3}\. covers: "Dr.", "Mr.", "Ltd.", "U." etc.
# (?<=[.!?])              — must follow sentence-ending punctuation
# \s+                     — one or more whitespace
# (?=[A-Z])               — must be followed by uppercase (sentence start)
SENTENCE_BOUNDARY_PATTERN = regex.compile(
    r"(?<![A-Z][a-z]{0,3}\.)(?<=[.!?])\s+(?=[A-Z])"
)

# Ordered delimiter hierarchy for text chunking (inspired by semchunk).
# Tried from coarsest to finest; first delimiter that produces a split wins.
# Lookbehind patterns keep the delimiter attached to the preceding chunk.
# A match may be empty (CJK); the matched text is kept as the rejoin separator.
CHUNK_DELIMITERS = (
    re.compile(r'\n\n+'),                     # Paragraph breaks
    re.compile(r'\n'),                         # Single newlines
    # Sentence terminators: . ! ? need whitespace; CJK 。！？ and danda । ॥ do not
    re.compile(r'(?<=[.!?])\s+|(?<=[\u3002\uff01\uff1f\u0964\u0965])\s*'),
    re.compile(r'(?<=[;:,\u2014\u2026])\s+'), # Clause separators (; : , em-dash, ellipsis)
    re.compile(r'\s+'),                        # Any whitespace (word boundaries)
)


# User term lists (custom_deny_terms, custom_allow_terms): a term is a whole word where no letter, mark or
# digit of a spaced script touches it. A term edge in a script written without spaces (CJK, Thai) is a word
# edge already, and letters of those scripts never glue (金额ACME元 holds the term ACME).
_TERM_GLUE = rf'[[\p{{L}}\p{{M}}\p{{N}}]--[{_SPACELESS}]]'


def terms_regex(terms: Iterable[str], case_sensitive: bool = False,
                labels: Mapping[str, str] | None = None) -> regex.Pattern:
    """One regex for *terms* as whole words (longest first, so an overlap keeps the longer term). With
    *labels* ({term: LABEL}) each term is a group named by its label, so ``match.lastgroup`` is the label
    (a case variant such as "İ" for "i" need not equal the term after casefold)."""
    parts = []
    for term in sorted(set(terms), key=len, reverse=True):
        left = '' if _SPACELESS_CHAR.match(term[0]) else rf'(?<!{_TERM_GLUE})'
        right = '' if _SPACELESS_CHAR.match(term[-1]) else rf'(?!{_TERM_GLUE})'
        body = regex.escape(term) if labels is None else f'(?P<{labels[term]}>{regex.escape(term)})'
        parts.append(left + body + right)
    return regex.compile('|'.join(parts), regex.V1 | (0 if case_sensitive else regex.IGNORECASE))
