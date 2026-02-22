"""
Constant symbols and compiled RegExs used for cleaning.
"""

import re
import string

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

# Fixed: bounded quantifiers to prevent ReDoS (was nested + quantifiers)
ACRONYM_REGEX = re.compile(
    r"(?:^|(?<=\W))(?:(?:[A-Z]\.?){1,15}[a-z0-9&/-]?(?:[A-Z][s.]?|[0-9]s?)|[0-9](?:-?[A-Z]){1,15})(?:$|(?=\W))",
    flags=re.UNICODE,
)

# Fixed: Unicode ranges use actual Unicode escapes (not literal \\u in raw strings)
EMAIL_REGEX = re.compile(
    r"(?:^|(?<=[^\w@.)]))([\w+-](\.(?!\.))?)*?[\w+-](@|[(<{\[]at[)>}\]])(?:(?:[a-z\u00a1-\uffff0-9]-?)*[a-z\u00a1-\uffff0-9]+)(?:\.(?:[a-z\u00a1-\uffff0-9]-?)*[a-z\u00a1-\uffff0-9]+)*(?:\.(?:[a-z\u00a1-\uffff]{2,}))",
    flags=re.IGNORECASE | re.UNICODE,
)

PHONE_REGEX = re.compile(
    r"((?:^|(?<=[^\w)]))((\+?\d+|0{1,2}\d*?)[ .-]?)?(\(?\d{3,4}\)?/?[ .-]?)?(\d{3}[ .-]?\d{4})(\s?(?:ext\.?|[#x-])\s?\d{2,6})?(?:$|(?=\W)))|\+?\d{4,5}[ .-/]\d{6,9}"
)

NUMBERS_REGEX = re.compile(
    r"(?:^|(?<=[^\w,.]))[+–-]?(([1-9]\d{0,2}(,\d{3})+(\.\d*)?)|([1-9]\d{0,2}([ .]\d{3})+(,\d*)?)|(\d*?[.,]\d+)|\d+)(?:$|(?=\b))"
)

LINEBREAK_REGEX = re.compile(r"((\r\n)|[\n\v])+")
TWO_LINEBREAK_REGEX = re.compile(r"((\r\n)|[\n\v])+((\r\n)|[\n\v])+")
MULTI_WHITESPACE_TO_ONE_REGEX = re.compile(r"\s+")
NONBREAKING_SPACE_REGEX = re.compile(r"(?!\n)\s+")

HTML_REGEX = re.compile(
    '<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});',
    flags=re.UNICODE | re.IGNORECASE,
)

# Fixed: Unicode ranges use actual Unicode escapes (not literal \\u in raw strings)
URL_REGEX = re.compile(
    r"(?:^|(?<![\w\/\.]))"
    # protocol identifier
    r"(?:(?:https?:\/\/|ftp:\/\/|www\d{0,3}\.))"
    # user:pass authentication
    r"(?:\S+(?::\S*)?@)?" r"(?:"
    # IP address exclusion - private & local networks
    r"(?!(?:10|127)(?:\.\d{1,3}){3})"
    r"(?!(?:169\.254|192\.168)(?:\.\d{1,3}){2})"
    r"(?!172\.(?:1[6-9]|2\d|3[0-1])(?:\.\d{1,3}){2})"
    # IP address dotted notation octets
    r"(?:[1-9]\d?|1\d\d|2[01]\d|22[0-3])"
    r"(?:\.(?:1?\d{1,2}|2[0-4]\d|25[0-5])){2}"
    r"(?:\.(?:[1-9]\d?|1\d\d|2[0-4]\d|25[0-4]))"
    r"|"
    # host name (with proper Unicode support)
    r"(?:(?:[a-z\u00a1-\uffff0-9]-?)*[a-z\u00a1-\uffff0-9]+)"
    # domain name
    r"(?:\.(?:[a-z\u00a1-\uffff0-9]-?)*[a-z\u00a1-\uffff0-9]+)*"
    # TLD identifier
    r"(?:\.(?:[a-z\u00a1-\uffff]{2,}))" r"|" r"(?:(localhost))" r")"
    # port number
    r"(?::\d{2,5}\b)?"
    # resource path
    r"(?:\/[^\)\]\}\s]*)?",
    flags=re.UNICODE | re.IGNORECASE,
)


strange_double_quotes = [
    "«", "‹", "»", "›", "„", "\u201c", "‟", "\u201d",
    "❝", "❞", "❮", "❯", "〝", "〞", "〟", "＂",
]
strange_single_quotes = ["'", "‛", "\u2019", "❛", "❜", "`", "´", "\u2018", "\u2019"]

DOUBLE_QUOTE_REGEX = re.compile("|".join(strange_double_quotes))
SINGLE_QUOTE_REGEX = re.compile("|".join(strange_single_quotes))

YEAR_REGEX = re.compile(r"\b(19|20)\d{2}\b")

# ---------------------------------------------------------------------------
# Multilingual month name vocabulary (EN, NL, DE, ES)
# Used by DATE_REGEX (exact match) and fuzzy_replace_dates (RapidFuzz).
# Keys are canonical lowercase forms; values are frozensets of language codes
# (some month names are shared across languages, e.g. "juni" is NL and DE).
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
    "marzo": frozenset({"es"}), "abril": frozenset({"es"}),
    "mayo": frozenset({"es"}), "junio": frozenset({"es"}),
    "julio": frozenset({"es"}), "agosto": frozenset({"es"}),
    "septiembre": frozenset({"es"}), "octubre": frozenset({"es"}),
    "noviembre": frozenset({"es"}), "diciembre": frozenset({"es"}),
    "ene": frozenset({"es"}), "abr": frozenset({"es"}),
    "ago": frozenset({"es"}), "dic": frozenset({"es"}),
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

# Date patterns: ISO 8601, common formats, multilingual month names
DATE_REGEX = re.compile(
    r'\b(?:'
    # ISO 8601: 2024-01-15 or 2024-01-15T10:30:00Z
    r'\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)?'
    r'|'
    # Common formats: 01/15/2024, 15.01.2024, 15-01-2024
    r'\d{1,2}[/.-]\d{1,2}[/.-]\d{4}'
    r'|'
    # Month name first: January 15, 2024 or Jan 15 2024 (all 4 langs)
    r'(?:' + _MONTH_ALTERNATION + r')'
    r'\.?\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}'
    r'|'
    # Day first: 15 January 2024, 15. Januar 2024, 15 de enero de 2024
    r'\d{1,2}\.?\s+(?:de\s+)?(?:' + _MONTH_ALTERNATION + r')'
    r'(?:\s+de)?\s+\d{4}'
    r')\b',
    flags=re.IGNORECASE,
)

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

ISOLATED_SPECIAL_SYMBOLS_REGEX = re.compile(
    r"(?<![a-zA-Z0-9])[:_.|><;·}@~!?+#)({,/\\\\^]+(?![a-zA-Z0-9])",
    flags=re.UNICODE | re.IGNORECASE,
)

# Pre-compiled bracket/brace removal patterns (moved from special.py runtime)
BRACKET_CONTENT_REGEX = re.compile(r'\[[^\]]+\]')
BRACE_CONTENT_REGEX = re.compile(r'\{[^}]+\}')
ISOLATED_QUOTES_REGEX = re.compile(
    r"(?<![a-zA-Z0-9])['\"\-*%](?![a-zA-Z0-9])",
    flags=re.UNICODE | re.IGNORECASE,
)

# Pre-compiled punctuation pattern (was re-compiled on every call in special.py)
PUNCTUATION_REGEX = re.compile('[' + re.escape(string.punctuation) + ']')

# Sentence boundary pattern for NER text splitting (legacy, kept for backward compat)
SENTENCE_BOUNDARY_PATTERN = re.compile(r'(?<=[.!?])\s+(?=[^\d])')

# Ordered delimiter hierarchy for text chunking (inspired by semchunk).
# Tried from coarsest to finest; first delimiter that produces a split wins.
# Lookbehind patterns keep the delimiter attached to the preceding chunk.
CHUNK_DELIMITERS = (
    re.compile(r'\n\n+'),                     # Paragraph breaks
    re.compile(r'\n'),                         # Single newlines
    re.compile(r'(?<=[.!?\u3002])\s+'),       # Sentence terminators (. ! ? and CJK period)
    re.compile(r'(?<=[;:,\u2014\u2026])\s+'), # Clause separators (; : , em-dash, ellipsis)
    re.compile(r'\s+'),                        # Any whitespace (word boundaries)
)
