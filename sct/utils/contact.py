import ipaddress
import re
import unicodedata
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from sct.utils import constants

if TYPE_CHECKING:
    from sct.config import TextCleanerConfig

# Phones and IPs are matched on a view of the text where every decimal digit is ASCII and
# full-width . - and the ideographic space are ASCII (٠٦ ١٢ ٣٤, ١٩٢．١٦٨．١．١): one code point
# for one, so a match in the view has the same offsets in the source.
_DIGIT_VIEW = {ord(ch): str(unicodedata.decimal(ch)) for ch in constants.CATEGORY_CHARS['Nd'] if not ch.isascii()}
_DIGIT_VIEW.update({0xFF0E: '.', 0xFF0D: '-', 0x3000: ' '})
# fold_styled_characters: phones also see every other space (Zs: no-break, thin, narrow no-break) as
# " " and every dash (Pd: hyphen, non-breaking hyphen, en dash) as "-", as word processors and PDFs
# set digit groups (06\u00a012\u00a034, 06\u201112345678), and full-width or small area-code
# parentheses as "(" ")" (+49（30）123456); the source text keeps them.
SEPARATOR_VIEW = {**_DIGIT_VIEW, **{ord(ch): ' ' if unicodedata.category(ch) == 'Zs' else '-'
                                    for ch in constants.category_chars('Zs', 'Pd') if not ch.isascii()},
                  **str.maketrans(constants.PHONE_PARENS)}

# Dates (fold_styled_characters) also read every character whose NFKC form is "." as "." (1990﹒ 1﹒ 15﹒).
DATE_SEPARATOR_VIEW = {**SEPARATOR_VIEW, **dict.fromkeys(map(ord, constants.NFKC_DOTS), '.')}


class _Span:
    """*source*[start:end] read as a match."""

    def __init__(self, start, end, source):
        self._start, self._end, self._source = start, end, source

    def group(self, *groups):
        if groups not in ((), (0,)):
            raise IndexError("only the whole match is read")
        return self._source[self._start:self._end]

    def start(self, *groups):
        return self._start

    def end(self, *groups):
        return self._end

    def span(self, *groups):
        return self._start, self._end


class DigitView:
    """*pattern* matched on the digit view of a text; replacements land in the source text."""

    def __init__(self, pattern: re.Pattern[str], table: Mapping[int, str] = _DIGIT_VIEW):
        self.pattern, self.table = pattern, table

    def sub(self, repl: str | Callable[[Any], str], text: str) -> str:
        view = text if text.isascii() else text.translate(self.table)
        if view == text:
            return self.pattern.sub(repl, text)
        out, last = [], 0
        for match in self.pattern.finditer(view):
            out.append(text[last:match.start()])
            out.append(repl(_Span(*match.span(), text)) if callable(repl) else match.expand(repl))
            last = match.end()
        out.append(text[last:])
        return ''.join(out)


PHONE_VIEW = DigitView(constants.PHONE_REGEX, SEPARATOR_VIEW)
PHONE_DIGIT_VIEW = DigitView(constants.PHONE_REGEX)  # fold_styled_characters=False
IP_VIEW = DigitView(constants.IP_REGEX)


class ChecksumView:
    """*pattern* matches that pass *check*. A failing match is cut back at its last space or
    hyphen until a prefix passes ("BE68 5390 0754 7034 EUR1"); with none, the text stays."""

    def __init__(self, pattern: re.Pattern[str], check: Callable[[str], bool]):
        self.pattern, self.check = pattern, check

    def sub(self, repl: Callable[[Any], str], text: str) -> str:
        out, last = [], 0
        for m in self.pattern.finditer(text):
            ends = [m.end(), *(i for i in range(m.end() - 1, m.start(), -1) if text[i] in ' -')]
            end = next((e for e in ends if self.check(text[m.start():e])), None)
            if end is not None:
                out += [text[last:m.start()], repl(_Span(m.start(), end, text))]
                last = end
        out.append(text[last:])
        return ''.join(out)


def iban_ok(value: str) -> bool:
    """ISO 13616 IBAN length and ISO 7064 mod-97 checksum."""
    s = value.replace(' ', '')
    return 15 <= len(s) <= 34 and int(''.join(str(int(c, 36)) for c in s[4:] + s[:4])) % 97 == 1


def luhn_ok(value: str) -> bool:
    """13-19 digits whose Luhn checksum holds."""
    digits = [int(c) for c in value if c.isdigit()]
    total = sum(digits[-1::-2]) + sum(sum(divmod(2 * d, 10)) for d in digits[-2::-2])
    return 13 <= len(digits) <= 19 and total % 10 == 0


IBAN_VIEW = ChecksumView(constants.IBAN_REGEX, iban_ok)
CARD_VIEW = ChecksumView(constants.CARD_REGEX, luhn_ok)

def numbers_regex(in_words: bool = False, mode: str = 'all', min_digits: int = 4,
                  **id_shapes: bool) -> re.Pattern[str]:
    """Number pattern: 'all' (every number; strict with in_words, else Latin/Greek/Cyrillic words kept) or
    'identifiers' (whole identifier-shaped digit runs of at least *min_digits* digits, plus the opt-in
    *id_shapes* of constants.identifier_numbers_regex)."""
    if mode == 'identifiers':
        return constants.identifier_numbers_regex(min_digits, in_words, **id_shapes)
    return constants.NUMBERS_IN_WORDS_REGEX if in_words else constants.NUMBERS_REGEX


def config_numbers_regex(cfg: "TextCleanerConfig") -> re.Pattern[str]:
    """The number pattern a TextCleanerConfig selects."""
    return numbers_regex(cfg.replace_numbers_in_words, cfg.replace_numbers_mode, cfg.replace_numbers_min_digits,
                         letter_prefix=cfg.numbers_letter_prefix_ids, two_part=cfg.numbers_two_part_ids,
                         spaced_pair=cfg.numbers_spaced_pair_ids,
                         keep_currency_code=cfg.numbers_keep_currency_code_amounts)


def embedded_phone(text: str, start: int, end: int) -> bool:
    """True when text[start:end] is a bare digit-only phone match inside a longer token, or one that starts
    is a yyyymmdd date or yyyymmddhhmm[ss] timestamp (phone_reject_embedded)."""
    value = text[start:end]
    if not value.isdecimal():
        return False
    return bool(constants.YYYYMMDD_STAMP.fullmatch(value.translate(_DIGIT_VIEW))
                or constants.PHONE_EMBEDDED_LEFT.search(text[max(0, start - 2):start])
                or constants.PHONE_EMBEDDED_RIGHT.match(text, end))


class RejectView:
    """*view* matches, except those *reject(text, start, end)* holds for: they stay as they are."""

    def __init__(self, view: DigitView | re.Pattern[str], reject: Callable[[str, int, int], bool]):
        self.view, self.reject = view, reject

    def sub(self, repl: Callable[[Any], str], text: str) -> str:
        return self.view.sub(
            lambda m: m.group() if self.reject(text, m.start(), m.end()) else repl(m), text)


def is_replaceable_ip(candidate: str, public_only: bool = False) -> bool:
    """True if ``candidate`` is a valid IP (and public, when ``public_only``)."""
    try:
        # Digits of any script count (١٩٢.١٦٨.١.١ is 192.168.1.1).
        ip = ipaddress.ip_address(candidate.translate(_DIGIT_VIEW))
    except ValueError:
        return False
    return ip.is_global or not public_only


class ProcessContacts:

    def replace_emails_with_fn(self, text: str, replace_fn: Callable[[str], str]) -> str:
        """Replace emails using a callable: replace_fn(matched_text) -> replacement."""
        if not constants.EMAIL_GATE.search(text):
            return text
        return constants.EMAIL_REGEX.sub(lambda m: replace_fn(m.group()), text)

    def replace_ips_with_fn(self, text: str, replace_fn: Callable[[str], str], public_only: bool = False) -> str:
        """Replace IPs using a callable: replace_fn(matched_text) -> replacement."""
        return IP_VIEW.sub(
            lambda m: replace_fn(m.group()) if is_replaceable_ip(m.group(), public_only) else m.group(),
            text,
        )

    def replace_phone_numbers_with_fn(self, text: str, replace_fn: Callable[[str], str],
                                  view: DigitView | RejectView = PHONE_VIEW) -> str:
        """Replace phone numbers using a callable: replace_fn(matched_text) -> replacement."""
        return view.sub(lambda m: replace_fn(m.group()), text)
