"""
Constant symbols and compiled RegExs used for cleaning.
"""

import re
import string
import unicodedata
import regex

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
# Local part takes RFC 5322 punctuation (o'connor, jo&co, alice!) after a word start,
# so no prefix of the address stays visible; it never ends in a dot. ZWNJ/ZWJ join letters
# in it as in a domain label (Devanagari half forms, Persian).
# Domain labels take any script but not general, CJK or full-width punctuation, so
# "a@b.com（担当：鈴木）" or "a@b.com、" ends at ".com". The letters and numbers of the
# CJK punctuation block (々 〆 〇, by Unicode category) stay, so "info@佐々木.jp" is whole.
_CJK_SYMBOL_LETTERS = ''.join(chr(c) for c in range(0x3000, 0x3040) if unicodedata.category(chr(c))[0] in 'LN')


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


# Combining marks of every script (Mn/Mc/Me: the vowel signs of প্রিয়া, संपर्क), which \w leaves out.
MARKS = _class_ranges('M', (0x300, 0x3FFFF), (0xE0000, 0xE0FFF))
# Letters, marks and digits beyond the BMP (𠮷野.jp): the BMP ranges below stop at U+FFFF.
_SUPPLEMENTARY_ALNUM = _class_ranges('LMN', (0x10000, 0x3FFFF))
_IDN_LETTER = (r"a-z\u00a1-\u1fff\u2070-\u2fff\u3040-\ufeff\uff21-\uff3a\uff41-\uff5a\uff66-\uffff"
               + _CJK_SYMBOL_LETTERS + _SUPPLEMENTARY_ALNUM)
# ZWNJ/ZWJ (U+200C/D) join letters inside IDNA labels (Persian, Devanagari conjuncts).
_IDN_CHAR = _IDN_LETTER + r"0-9\uff10-\uff19\u200c\u200d"
# A punycode TLD (xn--p1ai) is tried first, so ".xn" alone never ends the match.
_TLD = rf"(?:xn--[a-z0-9-]*[a-z0-9]|[{_IDN_LETTER}]{{2,}})"
EMAIL_REGEX = re.compile(
    rf"(?:^|(?<=[^\w@.){MARKS}]))[\w+-](?:[\w!#$%&'*+{MARKS}\u200c\u200d-]|\.(?!\.))*?(?<!\.)(@|[(<{{\[]at[)>}}\]])"
    rf"(?:(?:[{_IDN_CHAR}]-*)*[{_IDN_CHAR}]+)(?:\.(?:[{_IDN_CHAR}]-*)*[{_IDN_CHAR}]+)*(?:\.{_TLD})",
    flags=re.IGNORECASE | re.UNICODE,
)

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
    # Area-code parentheses must be balanced, so a closing ")" of the sentence stays.
    r"|(?<![\d+])\+(?=(?:[ .()-]{0,2}\d){8})\d{1,3}(?:[ .-]?(?:\(\d{1,5}\)|\d{1,5})){1,5}"
    r"(?:\s?(?:ext\.?|[#x])\s?\d{2,6})?(?![\d_(])"
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
    rf"|(?<![0-9.])(?:{_IPV4})(?![0-9]|\.[0-9])"
    # IPv4 in other digit scripts (١٩٢.١٦٨.١.١): needs one non-ASCII digit ([^\x00-\x7f\D]),
    # so ASCII matching (and the browser export, where \d is ASCII) is unchanged.
    r"|(?<![\d.])(?=[\d.]*[^\x00-\x7f\D])\d{1,3}(?:\.\d{1,3}){3}(?!\d|\.\d)"
)

_NUMBER_BODY = (
    r"[+–-]?(([1-9]\d{0,2}(,\d{3})+(\.\d*)?)"
    r"|([1-9]\d{0,2}([ .]\d{3})+(,\d*)?)|(\d*?[.,]\d+)|\d+)"
)
# Latin, Greek and Cyrillic put spaces between words, so a digit glued to their
# letters is part of a word (GPT4, Qwen3-8B, H2O) and is kept by default.
# Scripts without spaces (``金额5000元``) still get the number replaced.
_SPACED_LETTERS = "A-Za-z\u00C0-\u024F\u0370-\u03FF\u0400-\u04FF"
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

LINEBREAK_REGEX = re.compile(r"((\r\n)|[\n\v])+")
TWO_LINEBREAK_REGEX = re.compile(r"((\r\n)|[\n\v])+((\r\n)|[\n\v])+")
MULTI_WHITESPACE_TO_ONE_REGEX = re.compile(r"\s+")
NONBREAKING_SPACE_REGEX = re.compile(r"(?!\n)\s+")

HTML_REGEX = re.compile(
    '<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});',
    flags=re.UNICODE | re.IGNORECASE,
)

# Punctuation (category P) of the CJK symbol, katakana middle dot and full-width blocks, except the
# full-width forms of URL syntax (／ ？ ＃ ％ ＆ ＠ ： ． － ＿), which the styled fold turns into ASCII.
_WIDE_PUNCT = re.escape(''.join(
    chr(cp) for cp in [*range(0x3000, 0x3040), 0x30FB, *range(0xFF00, 0xFF66)]
    if unicodedata.category(chr(cp))[0] == 'P' and unicodedata.normalize('NFKC', chr(cp)) not in '/?#%&@:.-_'))

# Fixed: Unicode ranges use actual Unicode escapes (not literal \\u in raw strings)
URL_REGEX = re.compile(
    # Only an ASCII letter or digit glues to the scheme, so a URL right after CJK/Thai text
    # (請見https://..., ดูhttps://...) is still found.
    r"(?:^|(?<![A-Za-z0-9_\/\.]))"
    # protocol identifier
    r"(?:(?:https?:\/\/|ftp:\/\/|www\d{0,3}\.))"
    # user:pass authentication; never across / ? # " < > so a later "@" in the
    # query or the markup is not taken as user info
    r"(?:[^\s/?#\"<>@]+(?::[^\s/?#\"<>@]*)?@)?" r"(?:"
    # bracketed IPv6 host; private and loopback IPv4 hosts match too, so the whole
    # URL (query values included) is replaced, not just the address
    r"\[[0-9a-f:.]*:[0-9a-f:.]*\]" r"|"
    # IPv4 host, whole: a longer octet or another label after it is a host name
    rf"{_IPV4}(?![0-9]|\.[0-9a-z])"
    r"|"
    # host name (with proper Unicode support, beyond the BMP too)
    rf"(?:(?:[a-z\u00a1-\uffff0-9{_SUPPLEMENTARY_ALNUM}]-*)*[a-z\u00a1-\uffff0-9{_SUPPLEMENTARY_ALNUM}]+)"
    # domain name
    rf"(?:\.(?:[a-z\u00a1-\uffff0-9{_SUPPLEMENTARY_ALNUM}]-*)*[a-z\u00a1-\uffff0-9{_SUPPLEMENTARY_ALNUM}]+)*"
    # TLD identifier (punycode first, so ".xn" alone never ends the host)
    rf"(?:\.(?:xn--[a-z0-9-]*[a-z0-9]|[a-z\u00a1-\uffff{_SUPPLEMENTARY_ALNUM}]{{2,}}))" r"|" r"(?:(localhost))" r")"
    # port number
    r"(?::\d{2,5}\b)?"
    # resource path, or a query/fragment right after the host ("?id=", "#token=");
    # " < > are never raw in a URL, so markup around it survives. A bare trailing "?"
    # stays punctuation. CJK and full-width punctuation (，。！（ ・) ends it, as in an email
    # domain, so CJK prose glued after a URL stays. ) ] } belong to the URL only when more URL text follows
    # ("?f=(a)&email=..."), so a closing bracket around a URL stays outside it.
    rf"(?:\/(?:[^\)\]\}}\s\"<>{_WIDE_PUNCT}]|[\)\]\}}](?=[\w&=/?#%+~-]))*"
    rf"|[?#](?:[^\)\]\}}\s\"<>{_WIDE_PUNCT}]|[\)\]\}}](?=[\w&=/?#%+~-]))+)?",
    flags=re.UNICODE | re.IGNORECASE,
)


strange_double_quotes = [
    "«", "‹", "»", "›", "„", "\u201c", "‟", "\u201d",
    "❝", "❞", "❮", "❯", "〝", "〞", "〟", "＂",
]
strange_single_quotes = ["'", "‛", "\u2019", "❛", "❜", "`", "´", "\u2018", "\u2019"]

DOUBLE_QUOTE_REGEX = re.compile("|".join(strange_double_quotes))
SINGLE_QUOTE_REGEX = re.compile("|".join(strange_single_quotes))

YEAR_REGEX = re.compile(r"(?<![\d_])(19|20)\d{2}(?![\d_])")

# ---------------------------------------------------------------------------
# Multilingual month name vocabulary (EN, NL, DE, ES, FR, PT, IT)
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
    sorted(_MONTH_NAMES_SET, key=len, reverse=True)
)


# ---------------------------------------------------------------------------
# Factory functions for building per-instance month/date/fuzzy data.
# Used by TextCleaner to support custom languages via TextCleanerConfig.
# Module-level constants above remain the defaults.
# ---------------------------------------------------------------------------

def build_month_names_dict(extra_month_names=None):
    """Build month-name lookup dict, merging defaults + custom names.

    Args:
        extra_month_names: dict mapping language ISO code -> tuple of month names
                          e.g. {'pl': ('styczen', 'luty', ...)}
    Returns:
        dict mapping lowercase month name -> frozenset of ISO codes
    """
    result = dict(MONTH_NAMES_MULTILINGUAL)  # copy defaults
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
_NUMERIC_DATE = (
    r'(?<!\d[.,])(?:'
    r'\d{4}(?:([-/])' + _MONTH + r'\1' + _DAY + r'|\.(?:0[1-9]|1[0-2])\.(?:0[1-9]|[12]\d|3[01]))'
    r'|' + _DAY + r'([-/.])' + _MONTH + r'\2\d{4}|' + _MONTH + r'([-/.])' + _DAY + r'\3\d{4}'
    r')'
    r'(?:[T ]\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d+)?(?:\s?Z|\s?[+-]\d{2}:?\d{2})?)?'
    r'(?![.,]\d)'
)


def build_date_regex(month_names_dict=None):
    """Build DATE_REGEX from a month-names dict.

    Args:
        month_names_dict: output of build_month_names_dict().
                         Uses module-level MONTH_NAMES_MULTILINGUAL if None.
    Returns:
        Compiled regex pattern
    """
    names = month_names_dict or MONTH_NAMES_MULTILINGUAL
    alt = '|'.join(sorted(names.keys(), key=len, reverse=True))
    return re.compile(
        r'\b(?:'
        + _NUMERIC_DATE +
        r'|'
        r'(?:' + alt + r')'
        r'\.?\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}'
        r'|'
        r'\d{1,2}\.?\s+(?:de\s+)?(?:' + alt + r')'
        r'(?:\s+de)?\s+\d{4}'
        r')\b',
        flags=re.IGNORECASE,
    )


def build_fuzzy_vocabulary(month_names_dict=None, lang_to_vocab=None):
    """Build fuzzy month vocabulary (full names only, len > 4).

    Returns:
        (all_vocab_tuple, by_lang_dict)
    """
    names = month_names_dict or MONTH_NAMES_MULTILINGUAL
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
    r'\b(?:'
    # Word then day then year: "Janury 15, 2024"
    r'([A-Za-z\u00e0-\u00fc]{3,12})\.?\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}'
    r'|'
    # Day then word then year: "15 Feburary 2024" or "15. Janur 2024" or "15 de enro de 2024"
    r'\d{1,2}\.?\s+(?:de\s+)?([A-Za-z\u00e0-\u00fc]{3,12})(?:\s+de)?\s+\d{4}'
    r')\b',
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
