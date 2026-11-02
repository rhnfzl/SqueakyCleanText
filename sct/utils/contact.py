import ipaddress
import unicodedata

from sct.utils import constants

# Phones and IPs are matched on a view of the text where every decimal digit is ASCII and
# full-width . - and the ideographic space are ASCII (٠٦ ١٢ ٣٤, ١٩٢．١٦٨．١．١): one code point
# for one, so a match in the view has the same offsets in the source.
_DIGIT_VIEW = {cp: str(unicodedata.decimal(chr(cp))) for cp in range(0x80, 0x20000)
               if unicodedata.category(chr(cp)) == 'Nd'}
_DIGIT_VIEW.update({0xFF0E: '.', 0xFF0D: '-', 0x3000: ' '})
# fold_styled_characters: phones also see every other space (Zs: no-break, thin, narrow no-break) as
# " " and every dash (Pd: hyphen, non-breaking hyphen, en dash) as "-", as word processors and PDFs
# set digit groups (06\u00a012\u00a034, 06\u201112345678), and full-width or small area-code
# parentheses as "(" ")" (+49（30）123456); the source text keeps them.
SEPARATOR_VIEW = {**_DIGIT_VIEW, **{cp: ' ' if unicodedata.category(chr(cp)) == 'Zs' else '-'
                                    for cp in range(0x80, 0x20000) if unicodedata.category(chr(cp)) in ('Zs', 'Pd')},
                  **str.maketrans(constants.PHONE_PARENS)}

# Dates (fold_styled_characters) also read every character whose NFKC form is "." as "." (1990﹒ 1﹒ 15﹒).
DATE_SEPARATOR_VIEW = {**SEPARATOR_VIEW, **{cp: '.' for cp in range(0x80, 0x20000)
                                             if unicodedata.normalize('NFKC', chr(cp)) == '.'}}


class _SourceMatch:
    """A match found in the view, read from the source text."""

    def __init__(self, match, source):
        self._match, self._source = match, source

    def group(self, *groups):
        if groups not in ((), (0,)):
            raise IndexError("only the whole match is read from the source")
        return self._source[self._match.start():self._match.end()]

    def start(self, *groups):
        return self._match.start(*groups)

    def end(self, *groups):
        return self._match.end(*groups)

    def span(self, *groups):
        return self._match.span(*groups)


class DigitView:
    """*pattern* matched on the digit view of a text; replacements land in the source text."""

    def __init__(self, pattern, table=_DIGIT_VIEW):
        self.pattern, self.table = pattern, table

    def sub(self, repl, text):
        view = text if text.isascii() else text.translate(self.table)
        if view == text:
            return self.pattern.sub(repl, text)
        out, last = [], 0
        for match in self.pattern.finditer(view):
            out.append(text[last:match.start()])
            out.append(repl(_SourceMatch(match, text)) if callable(repl) else match.expand(repl))
            last = match.end()
        out.append(text[last:])
        return ''.join(out)


PHONE_VIEW = DigitView(constants.PHONE_REGEX, SEPARATOR_VIEW)
PHONE_DIGIT_VIEW = DigitView(constants.PHONE_REGEX)  # fold_styled_characters=False
IP_VIEW = DigitView(constants.IP_REGEX)



class _Span(_SourceMatch):
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


class ChecksumView:
    """*pattern* matches that pass *check*. A failing match is cut back at its last space or
    hyphen until a prefix passes ("BE68 5390 0754 7034 EUR1"); with none, the text stays."""

    def __init__(self, pattern, check):
        self.pattern, self.check = pattern, check

    def sub(self, repl, text):
        out, last = [], 0
        for m in self.pattern.finditer(text):
            ends = [m.end(), *(i for i in range(m.end() - 1, m.start(), -1) if text[i] in ' -')]
            end = next((e for e in ends if self.check(text[m.start():e])), None)
            if end is not None:
                out += [text[last:m.start()], repl(_Span(m.start(), end, text))]
                last = end
        out.append(text[last:])
        return ''.join(out)


def iban_ok(value):
    """ISO 13616 IBAN length and ISO 7064 mod-97 checksum."""
    s = value.replace(' ', '')
    return 15 <= len(s) <= 34 and int(''.join(str(int(c, 36)) for c in s[4:] + s[:4])) % 97 == 1


def luhn_ok(value):
    """13-19 digits whose Luhn checksum holds."""
    digits = [int(c) for c in value if c.isdigit()]
    total = sum(digits[-1::-2]) + sum(sum(divmod(2 * d, 10)) for d in digits[-2::-2])
    return 13 <= len(digits) <= 19 and total % 10 == 0


IBAN_VIEW = ChecksumView(constants.IBAN_REGEX, iban_ok)
CARD_VIEW = ChecksumView(constants.CARD_REGEX, luhn_ok)

def numbers_regex(in_words=False):
    """Number pattern: strict (any glued letter) or default (Latin/Greek/Cyrillic words kept)."""
    return constants.NUMBERS_IN_WORDS_REGEX if in_words else constants.NUMBERS_REGEX


def is_replaceable_ip(candidate, public_only=False):
    """True if ``candidate`` is a valid IP (and public, when ``public_only``)."""
    try:
        # Digits of any script count (١٩٢.١٦٨.١.١ is 192.168.1.1).
        ip = ipaddress.ip_address(candidate.translate(_DIGIT_VIEW))
    except ValueError:
        return False
    return ip.is_global or not public_only


class ProcessContacts:

    def __init__(self):
        pass

    def replace_urls(self, text, replace_with="<URL>"):
        """Replace all URLs in ``text`` str with ``replace_with`` str."""
        return constants.URL_REGEX.sub(replace_with, text)

    def replace_html(self, text, replace_with="<HTML>"):
        """Replace all html tags in ``text`` str with ``replace_with`` str."""
        return constants.HTML_REGEX.sub(replace_with, text)

    def replace_emails(self, text, replace_with="<EMAIL>"):
        """Replace all emails in ``text`` str with ``replace_with`` str."""
        if not constants.EMAIL_GATE.search(text):
            return text
        return constants.EMAIL_REGEX.sub(replace_with, text)

    def replace_ips(self, text, replace_with="<IP>", public_only=False):
        """Replace IPv4/IPv6 addresses in ``text`` (only public ones if ``public_only``)."""
        return self.replace_ips_with_fn(text, lambda orig: replace_with, public_only)

    def replace_phone_numbers(self, text, replace_with="<PHONE>"):
        """Replace all phone numbers in ``text`` str with ``replace_with`` str."""
        return PHONE_VIEW.sub(replace_with, text)

    def replace_numbers(self, text, replace_with="<NUMBER>", in_words=False):
        """Replace all numbers in ``text`` str with ``replace_with`` str.

        ``in_words=True`` also replaces digits glued to Latin/Greek/Cyrillic letters (``GPT4``).
        """
        return numbers_regex(in_words).sub(replace_with, text)

    # --- Callable-based replacements (for synthetic mode) ---

    def replace_emails_with_fn(self, text, replace_fn):
        """Replace emails using a callable: replace_fn(matched_text) -> replacement."""
        if not constants.EMAIL_GATE.search(text):
            return text
        return constants.EMAIL_REGEX.sub(lambda m: replace_fn(m.group()), text)

    def replace_urls_with_fn(self, text, replace_fn):
        """Replace URLs using a callable: replace_fn(matched_text) -> replacement."""
        return constants.URL_REGEX.sub(lambda m: replace_fn(m.group()), text)

    def replace_ips_with_fn(self, text, replace_fn, public_only=False):
        """Replace IPs using a callable: replace_fn(matched_text) -> replacement."""
        return IP_VIEW.sub(
            lambda m: replace_fn(m.group()) if is_replaceable_ip(m.group(), public_only) else m.group(),
            text,
        )

    def replace_phone_numbers_with_fn(self, text, replace_fn, view=PHONE_VIEW):
        """Replace phone numbers using a callable: replace_fn(matched_text) -> replacement."""
        return view.sub(lambda m: replace_fn(m.group()), text)

    def replace_numbers_with_fn(self, text, replace_fn, in_words=False):
        """Replace numbers using a callable: replace_fn(matched_text) -> replacement."""
        return numbers_regex(in_words).sub(lambda m: replace_fn(m.group()), text)
