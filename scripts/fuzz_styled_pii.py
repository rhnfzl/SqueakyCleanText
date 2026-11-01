#!/usr/bin/env python3
"""Fuzz the styled / full-width PII leak class.

Each seed builds one document: random prose in Latin, CJK, Arabic, Devanagari,
Thai or Cyrillic, plus 1-3 PII values (email, phone, IPv4, URL with a query,
also www-introduced, a bare host or on a host whose first labels look like an IPv4; email and URL
also on a host of non-Latin labels, date, number). Each value
is then rendered in forms a reader still reads as the same value: full-width and
small-form characters, mathematical styled letters and digits, circled/squared
letters and digits, native decimal digits of any script, decomposed letters
(base + combining mark), zero-width characters where they are legal, invisible
break hints (ZWSP, WJ, soft hyphen) inside the value,
typeset phone and date separators (no-break or thin space, non-breaking hyphen, en dash),
IDNA dot separators (。 ｡) in a mail domain or URL host, host hyphens and dots typed as
another dash or dot form (my\u2010site, example\u2024com), upper-case addresses (so squared
letters 🄹 occur), break hints between non-Latin letters of a mail domain or URL host,
area-code parentheses glued to the digits (+49(30)123456), URL user info whose password ends
in punctuation (https://alice:p4ss!@host/), mail local parts mixing separators and other punctuation
(josé_smith.work@, anna.smith!@) with their hyphens and dots typed as another form (john\u2011smith@),
and glued to CJK/Thai prose without a space. For a config that decodes backslash escapes
(``check_decode_unicode_escapes``, the default) some characters are also written as literal
``\\uXXXX`` / ``\\U........`` / ``\\xNN`` escapes, as JSON-escaped text carries them, and some as HTML
entities, also escaped twice (``&#51;``, ``&amp;#51;``), as scraped text carries them. Every variant comes from Unicode
data (decomposition tag, NFKC, decimal value, category), and every
rendered value folds back to the plain value (NFKC, Cf removed, Nd -> ASCII).

The cleaner runs with NER off on the ``training`` preset (also with
``check_replace_dates`` on, so a date has no year step behind it) and the default
config, with ``fold_styled_characters`` on and off, in placeholder, synthetic,
reversible (restore must give every value back) and streaming
(``WindowedStreamingRedactor``, compared with the whole-document output) modes.

A leak is a plain identifying core of a value (email local part / host label,
>= 7 phone or date or number digits, the IPv4, URL query values) found in the
output after the same fold. Each failure is shrunk to the smallest set of
transformations that still fails; the cluster key is that set plus type and
mode. Failures with ``fold_styled_characters=False`` are recorded, not counted.

    python -m scripts.fuzz_styled_pii --seeds 3000 --workers 8 --out fuzz.jsonl --report fuzz.md
"""

import argparse
from collections import Counter, defaultdict
from encodings.idna import dots as IDNA_DOTS
from functools import lru_cache, partial
import json
import multiprocessing
from pathlib import Path
import random
import re
import sys
import unicodedata

from ftfy.fixes import unescape_html

from sct import TextCleaner, TextCleanerConfig
from sct.streaming import WindowedStreamingRedactor
from sct.utils import constants

PRESETS = ("training", "sct", "training+dates")
PRESET_ARGS = {"training": ("training", {}), "sct": ("sct", {}),
               "training+dates": ("training", {"check_replace_dates": True})}
MODES = ("placeholder", "synthetic", "reversible", "stream")
CHECKS = ("leak", "restore", "restore_exact", "false_positive", "stream_mismatch")
STREAM_PARAMS = ((48, 12), (80, 24), (160, 48))
TYPE_FLAG = {
    "EMAIL": "check_replace_emails", "PHONE": "check_replace_phone_numbers", "IP": "check_replace_ips",
    "URL": "check_replace_urls", "DATE": "check_replace_dates", "NUMBER": "check_replace_numbers",
}
# Zero-width characters (all category Cf, removed by the oracle fold).
ZWSP, ZWNJ, ZWJ, WJ, SHY = "​", "‌", "‍", "⁠", "\u00ad"
# Typeset digit-group separators: no-break, figure, thin and narrow no-break space (Zs, NFKC is a
# space) and hyphen, non-breaking hyphen, figure dash, en dash (Pd), as word processors and PDFs write them.
TYPO_SPACES, TYPO_DASHES = "\u00a0\u2007\u2009\u202f", "\u2010\u2011\u2012\u2013"
# Decomposition tags whose NFKC fold is the styled character's plain reading.
STYLE_TAGS = {"wide": "wide", "small": "small", "font": "font", "circle": "enclosed", "square": "enclosed"}
# Script letter ranges for context prose and non-ASCII PII (letters picked by category).
SCRIPT_RANGES = {
    "latin": ((0x61, 0x7A),), "cyrillic": ((0x430, 0x44F),), "arabic": ((0x628, 0x64A),),
    "devanagari": ((0x915, 0x939),), "thai": ((0xE01, 0xE2E),), "cjk": ((0x4E00, 0x9FFF), (0x3041, 0x3096)),
}
UNSPACED = ("cjk", "thai")
PUNCT = {
    "latin": (", ", ". ", "! ", "? "), "cyrillic": (", ", ". ", "! "), "arabic": ("، ", "؟ ", ". "),
    "devanagari": (" । ", ", "), "thai": (" ",), "cjk": ("，", "。", "、", "！", "？", "：", "・"),
}
TLDS = ("com", "org", "net", "nl", "de", "jp", "io", "in")
# Label separators IDNA reads as "." (。 ｡ ．), from Python's idna codec.
IDNA_DOT_CHARS = "".join(ch for ch in map(chr, range(0x80, 0x10000)) if IDNA_DOTS.fullmatch(ch))
# Host separators as typeset text writes them: any dash (Pd) but the wide CJK wave dashes, which are prose
# punctuation, for a label hyphen, and any character whose NFKC form is "." (․ ﹒ ．) for a label dot.
HOST_DASHES = "".join(ch for ch in map(chr, range(0x80, 0x110000)) if unicodedata.category(ch) == "Pd"
                      and unicodedata.east_asian_width(ch) != "W")
HOST_DOTS = "".join(ch for ch in map(chr, range(0x80, 0x110000)) if unicodedata.normalize("NFKC", ch) == ".")


def _unicode_tables():
    """Styled variants per plain ASCII target, digit sets per script, letters per script."""
    variants = defaultdict(lambda: defaultdict(list))  # target -> kind -> [chars]
    zeros = []
    for cp in range(0x110000):
        ch = chr(cp)
        category = unicodedata.category(ch)
        if category in ("Cs", "Cn", "Co"):
            continue
        if category == "Nd" and unicodedata.decimal(ch) == 0 and cp > 0x7F and all(
                unicodedata.decimal(chr(cp + k), -1) == k for k in range(10)):
            zeros.append(cp)
        decomposition = unicodedata.decomposition(ch)
        if not decomposition.startswith("<"):
            continue
        kind = STYLE_TAGS.get(decomposition[1:decomposition.index(">")])
        target = unicodedata.normalize("NFKC", ch)
        if kind and target.isascii() and target.strip() and (len(target) == 1 or target.isdigit()):
            variants[target][kind].append(ch)
    letters = {
        script: [chr(cp) for lo, hi in ranges for cp in range(lo, hi + 1) if unicodedata.category(chr(cp)).startswith("L")]
        for script, ranges in SCRIPT_RANGES.items()
    }
    accented = [chr(cp) for cp in range(0xE0, 0x17F) if unicodedata.category(chr(cp)) == "Ll"
                and unicodedata.normalize("NFD", chr(cp)) != chr(cp)]
    # Devanagari vowel signs (category M*) follow a consonant.
    signs = [chr(cp) for cp in range(0x93E, 0x94D) if unicodedata.category(chr(cp)).startswith("M")]
    return variants, zeros, letters, accented, signs


VARIANTS, DIGIT_ZEROS, LETTERS, ACCENTED, DEVA_SIGNS = _unicode_tables()
_DIGIT_FOLD = {cp: str(unicodedata.decimal(chr(cp))) for zero in DIGIT_ZEROS for cp in range(zero, zero + 10)}


def fold(text: str) -> str:
    """The oracle fold: drop Cf, NFKC, any decimal digit -> ASCII, any dash (Pd) -> "-", IDNA dot -> ".", casefold."""
    text = "".join("." if ch in IDNA_DOT_CHARS else ch for ch in text if unicodedata.category(ch) != "Cf")
    text = unicodedata.normalize("NFKC", text).translate(_DIGIT_FOLD)
    return "".join("-" if unicodedata.category(ch) == "Pd" else ch for ch in text).casefold()


# --- document generation ---


def _word(rng, script, low=2, high=6):
    if script == "latin" and rng.random() < 0.3:
        pool = LETTERS["latin"] + ACCENTED
    else:
        pool = LETTERS[script]
    word = "".join(rng.choice(pool) for _ in range(rng.randint(low, high)))
    if script == "devanagari":
        word = "".join(ch + (rng.choice(DEVA_SIGNS) if rng.random() < 0.4 else "") for ch in word)
    return word


def _sentence(rng, script):
    joiner = "" if script in UNSPACED else " "
    words = [_word(rng, script, 1 if script in UNSPACED else 2, 4) for _ in range(rng.randint(3, 8))]
    text = ""
    for index, word in enumerate(words):
        text += word
        if index < len(words) - 1:
            text += rng.choice(PUNCT[script]) if rng.random() < 0.2 else joiner
    end = rng.choice(PUNCT[script]).strip() or ("" if script == "thai" else ".")  # Thai ends a sentence with a space
    if script == "cjk" and rng.random() < 0.3:
        text = f"（{text}）"
    return text + end


def _ascii_token(rng, low, high):
    """Letters and digits, at least one of each, so a core is unlikely in prose or fakes."""
    size = rng.randint(low, high)
    chars = [rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(size - 1)] + [rng.choice("0123456789")]
    rng.shuffle(chars)
    return "".join(chars)


def _script_token(rng, script):
    token = _word(rng, script, 3, 5)
    if script == "latin":  # an accented (precomposed) letter, so decomposition applies
        token = token[:-1] + rng.choice(ACCENTED)
    if script == "devanagari" and rng.random() < 0.5:  # a conjunct (virama), where ZWJ/ZWNJ are legal
        token = token[0] + "्" + token[1:]
    return token


def _host(rng):
    if rng.random() < 0.5:
        label = _ascii_token(rng, 4, 8)
    else:
        label = _script_token(rng, rng.choice(("latin", "cyrillic", "arabic", "devanagari", "thai", "cjk")))
    sub = _ascii_token(rng, 4, 6) + "." if rng.random() < 0.2 else ""
    return sub + label + "." + rng.choice(TLDS), label


def _email(rng):
    script = rng.choice(("ascii", "ascii", "latin", "cyrillic", "arabic", "devanagari", "thai", "cjk"))
    parts = [_ascii_token(rng, 4, 7) if script == "ascii" else _script_token(rng, script)]
    if rng.random() < 0.4:
        parts.append(_ascii_token(rng, 4, 6))
    local = rng.choice(".-_+").join(parts)
    host, label = _host(rng)
    return f"{local}@{host}", [("str", part) for part in parts] + [("str", label)]


def _digits(rng, size):
    return "".join(rng.choice("0123456789") for _ in range(size))


def _phone(rng):
    shape = rng.choice((
        "+31 6 {4} {4}", "06-{8}", "({3}) {3}-{4}", "+44 20 {4} {4}", "0{z}{1} {3} {2} {2}", "+91 {5} {5}",
        "06 {2} {2} {2} {2}", "+49 30 {8}",
    ))
    # {z} is one non-zero digit: a national area code never starts with 00 (the international prefix)
    value = re.sub(r"\{(\d|z)\}", lambda m: rng.choice("123456789") if m.group(1) == "z"
                   else _digits(rng, int(m.group(1))), shape)
    return value, [("digits", re.sub(r"\D", "", value))]


def _ip(rng):
    octets = [str(rng.randint(1, 254)) for _ in range(4)]
    return ".".join(octets), [("ip", tuple(octets))]


def _url(rng):
    host, _ = _host(rng)
    if rng.random() < 0.15:  # an IPv4-shaped first label ("10.0.0.1.école.fr") is still a host name
        host = ".".join(str(rng.randint(1, 254)) for _ in range(4)) + "." + host
    token = _ascii_token(rng, 10, 14)
    cores = [("str", token)]
    if rng.random() < 0.5:
        local = _ascii_token(rng, 5, 8)
        mail_host, _ = _host(rng)
        value = f"{local}{rng.choice(('%40', '@'))}{mail_host}"
        cores.append(("str", local))
    else:
        value = token
    query = f"?{rng.choice(('email', 'token', 'id', 'user', 'q'))}={value}"
    if value != token:
        query += f"&sid={token}"
    tail = rng.choice(("", f"#{_ascii_token(rng, 10, 12)}"))
    if tail:
        cores.append(("str", tail[1:]))
    return f"{rng.choice(('https', 'http'))}://{host}/{_ascii_token(rng, 3, 6)}{query}{tail}", cores


def _date(rng):
    year, month, day = rng.randint(1950, 2030), rng.randint(1, 12), rng.randint(1, 28)
    value = rng.choice((f"{day:02d}/{month:02d}/{year}", f"{year}-{month:02d}-{day:02d}",
                        f"{day:02d}.{month:02d}.{year}", f"{month:02d}/{day:02d}/{year}"))
    return value, [("digits", re.sub(r"\D", "", value))]


def _number(rng):
    value = str(rng.randint(1, 9)) + _digits(rng, rng.randint(5, 8))
    return value, [("digits", value)]


GENERATORS = {"EMAIL": _email, "PHONE": _phone, "IP": _ip, "URL": _url, "DATE": _date, "NUMBER": _number}


def _plan(rng, plain, legal_join):
    """Per-character styled replacements and zero-width insertions; each tagged by its kind."""
    slots = [None] * len(plain)  # index -> (kind, text)
    inserts = {}  # index -> (kind, text), inserted before plain[index]
    eligible = defaultdict(list)
    for index, ch in enumerate(plain):
        for kind, chars in VARIANTS.get(ch, {}).items():
            eligible[f"{kind}_{'alnum' if ch.isalnum() else 'sep'}"].append(index)
        if ch.isdigit() and ch.isascii():
            eligible["native_digit"].append(index)
            if len(plain[index:index + 2]) == 2 and "enclosed" in VARIANTS.get(plain[index:index + 2], {}):
                eligible["enclosed_pair"].append(index)
        if unicodedata.normalize("NFD", ch) != ch:
            eligible["decomposed"].append(index)
        if index in legal_join:
            eligible["zw_join"].append(index)
    eligible["zw_boundary"] = [0, len(plain)]
    kinds = sorted(k for k in eligible if rng.random() < 0.4)
    if not kinds:
        kinds = [rng.choice(sorted(eligible))]
    rng.shuffle(kinds)
    style = rng.randrange(13)  # one font style per value (bold, italic, ...)
    digit_zero = rng.choice(DIGIT_ZEROS)
    for kind in kinds:
        density = rng.choice((1.0, 1.0, 0.6, 0.3))
        for index in eligible[kind]:
            if rng.random() >= density:
                continue
            if kind == "zw_boundary":
                inserts[index] = (kind, rng.choice((ZWSP, WJ)))
            elif kind == "zw_join":
                inserts[index] = (kind, rng.choice((ZWNJ, ZWJ)))
            elif slots[index] is not None:
                continue
            elif kind == "native_digit":
                slots[index] = (kind, chr(digit_zero + int(plain[index])))
            elif kind == "decomposed":
                slots[index] = (kind, unicodedata.normalize("NFD", plain[index]))
            elif kind == "enclosed_pair":
                if slots[index + 1] is None:
                    slots[index] = (kind, rng.choice(VARIANTS[plain[index:index + 2]]["enclosed"]))
                    slots[index + 1] = (kind, "")
            else:
                chars = VARIANTS[plain[index]][kind.rsplit("_", 1)[0]]
                slots[index] = (kind, chars[min(style, len(chars) - 1)] if kind.startswith("font") else rng.choice(chars))
    used = {s[0] for s in slots if s} | {i[0] for i in inserts.values()}
    return slots, inserts, sorted(used)


def _legal_join(plain, typ):
    """Positions where ZWNJ/ZWJ are legal inside a label: after a virama or between Arabic letters."""
    if typ != "EMAIL":
        return set()
    legal = set()
    for index in range(1, len(plain)):
        before, after = plain[index - 1], plain[index]
        if unicodedata.combining(before) == 9 or (
                "ARABIC" in unicodedata.name(before, "") and "ARABIC" in unicodedata.name(after, "")):
            legal.add(index)
    return legal


def _zw_inside(rng, plain, inserts, kinds):
    """Invisible line-break hints (ZWSP, WJ, soft hyphen) between characters of a value, as copied
    from wrapped or hyphenated text; own RNG so the cases of earlier rounds stay the same. Not between
    two non-ASCII characters: there a ZWSP is a Thai word break and ZWNJ a Persian letter form, not a hint."""
    free = [index for index in range(1, len(plain)) if index not in inserts
            and (plain[index - 1].isascii() or plain[index].isascii())]
    if not free or rng.random() >= 0.25:
        return kinds
    for index in rng.sample(free, rng.randint(1, min(3, len(free)))):
        inserts[index] = ("zw_inside", rng.choice((ZWSP, WJ, SHY)))
    return sorted(kinds + ["zw_inside"])


def _typo_sep(rng, plain, slots, kinds):
    """Phone and date group separators typeset as another space or dash (06\u00a012\u00a034,
    1987\u201105\u201112); own RNG so the cases of earlier rounds stay the same."""
    free = [index for index, ch in enumerate(plain) if ch in " -" and slots[index] is None]
    if not free or rng.random() >= 0.25:
        return kinds
    density = rng.choice((1.0, 0.5))
    for index in free:
        if rng.random() < density:
            slots[index] = ("typo_sep", rng.choice(TYPO_SPACES if plain[index] == " " else TYPO_DASHES))
    return sorted(set(kinds) | {"typo_sep"}) if any(slot and slot[0] == "typo_sep" for slot in slots) else kinds


def _month_date(rng, plain, cores):
    """A day, a month name the date rule reads (février, so the "decomposed" kind can split its accent) and a
    year; own RNG so the cases of earlier rounds stay the same."""
    if rng.random() >= 0.3:
        return None
    day, name = str(rng.randint(1, 28)), rng.choice(sorted(constants.MONTH_NAMES_MULTILINGUAL))
    return f"{day} {name} {rng.randint(1950, 2030)}", [("str", f"{day} {name}")]


def _url_shape(rng, plain):
    """A host of two labels in one non-Latin script (例え.テスト, as IDNA allows) and a scheme-less
    www introducer, which URL_REGEX also accepts; own RNG so the cases of earlier rounds stay the same."""
    intro, host, path, rest = re.match(r"(https?://)([^/]+)(/[^?#]*)(.*)", plain).groups()
    if rng.random() < 0.25:
        script = rng.choice(("cyrillic", "arabic", "devanagari", "thai", "cjk"))
        host = _script_token(rng, script) + "." + _word(rng, script, 2, 4)
    if rng.random() < 0.25:
        intro, path = "www.", rng.choice((path, ""))
    if rng.random() < 0.2:  # a bare host, so glued CJK prose (，電話：06 ...) meets the host grammar itself
        path, rest = "", ""
    return intro + host + path + rest


def _paren_phone(rng, plain, cores):
    """An international number with the area code in parentheses and no spaces (+49(30)123456), so a
    styled parenthesis sits between digits; own RNG so the cases of earlier rounds stay the same."""
    if rng.random() >= 0.2:
        return plain, cores
    plain = f"+{rng.randint(30, 99)}({rng.randint(10, 99)}){_digits(rng, rng.randint(6, 8))}"
    return plain, [("digits", re.sub(r"\D", "", plain))]


def _userinfo(rng, plain, cores):
    """URL user info whose password ends in punctuation (https://alice:p4ss!@host/), so a styled @
    follows a non-alphanumeric character; own RNG so the cases of earlier rounds stay the same."""
    if "://" not in plain or rng.random() >= 0.2:
        return plain, cores
    user, password = _ascii_token(rng, 4, 7), _ascii_token(rng, 5, 8) + rng.choice("!$*+;=~")
    scheme, rest = plain.split("://", 1)
    return f"{scheme}://{user}:{password}@{rest}", cores + [("str", user), ("str", password[:-1])]


def _email_shape(rng, plain, cores):
    """A mail host of two labels in one non-Latin script (نامه.ایران, क्ष.भारत, as IDNA allows), so a
    join control in the label meets a styled dot with no ASCII beside it; own RNG so the cases of
    earlier rounds stay the same."""
    if rng.random() >= 0.25:
        return plain, cores
    script = rng.choice(("cyrillic", "arabic", "devanagari", "thai", "cjk"))
    label = _script_token(rng, script)
    return f"{plain.rsplit('@', 1)[0]}@{label}.{_word(rng, script, 2, 4)}", cores[:-1] + [("str", label)]


def _local_shape(rng, plain, cores):
    """A mail local part of up to three words joined by different separators, maybe ending in other
    punctuation the grammar takes (josé_smith.work@, anna.smith!@), so a styled separator sits beside
    an ASCII one; own RNG so the cases of earlier rounds stay the same."""
    if rng.random() >= 0.25:
        return plain, cores
    local, host = plain.rsplit("@", 1)
    parts = re.split(r"[.\-_+]", local) + [_ascii_token(rng, 4, 6)]
    local = parts[0] + "".join(rng.choice(".-_+") + part for part in parts[1:])
    local += rng.choice(("", "", "!", "#", "$", "%", "&", "'", "*", "+", "-"))
    return f"{local}@{host}", [("str", part) for part in parts] + cores[-1:]


def _local_sep(rng, plain, slots, kinds):
    """Local-part hyphens and dots typed as another dash or dot form (john.smith\u2011billing@), as
    copied from typeset text; own RNG so the cases of earlier rounds stay the same."""
    free = [index for index in range(plain.rindex("@") - 1) if plain[index] in "-." and slots[index] is None
            and plain[index + 1].isascii() and plain[index + 1].isalnum()]
    if not free or rng.random() >= 0.25:
        return kinds
    for index in free:
        if rng.random() < 0.7:
            slots[index] = ("local_sep", rng.choice(HOST_DASHES if plain[index] == "-" else HOST_DOTS))
    return sorted(set(kinds) | {"local_sep"}) if any(slot and slot[0] == "local_sep" for slot in slots) else kinds


def _upper(rng, plain):
    """An address written in capitals (JOHN@EXAMPLE.COM), so squared letters (🄹) have a target;
    own RNG so the cases of earlier rounds stay the same."""
    return plain.upper() if rng.random() < 0.2 else plain


def _host_span(plain, typ):
    """Start and end of the mail domain or the URL host in ``plain``."""
    if typ == "EMAIL":
        return plain.rindex("@") + 1, len(plain)
    start = plain.index("://") + 3 if "://" in plain else 0
    return start, re.search(r"[/?#]|$", plain[start:]).start() + start


def _idna_dot(rng, plain, slots, kinds, typ):
    """Host dots typed as an IDNA separator (example。com, as CJK input methods write them) where a
    Latin letter or digit follows; own RNG so the cases of earlier rounds stay the same."""
    start, end = _host_span(plain, typ)
    # the next character must render as ASCII too: a native digit after 。 is a native-digit host
    # label (rejected, Codex round 25 id 3)
    free = [index for index in range(start, end - 1) if plain[index] == "." and slots[index] is None
            and plain[index + 1].isascii() and plain[index + 1].isalnum()
            and (slots[index + 1] is None or slots[index + 1][0] != "native_digit")]
    if not free or rng.random() >= 0.25:
        return kinds
    for index in free:
        if rng.random() < 0.7:
            slots[index] = ("idna_dot", rng.choice(IDNA_DOT_CHARS))
    return sorted(set(kinds) | {"idna_dot"}) if any(slot and slot[0] == "idna_dot" for slot in slots) else kinds


def _host_hyphen(rng, plain, cores, typ):
    """A hyphen between two letters of a mail domain or URL host label (my-site.com), so a typeset dash
    can stand in for it; own RNG so the cases of earlier rounds stay the same."""
    start, end = _host_span(plain, typ)
    start += 4 if plain.startswith("www.") else 0  # keep the www introducer whole
    end = plain.rfind(".", start, end)  # a TLD takes no hyphen
    free = [index for index in range(start + 1, end) if plain[index - 1].isalpha() and plain[index].isalpha()]
    if not free or rng.random() >= 0.25:
        return plain, cores
    index = rng.choice(free)
    new_cores = []
    for kind, core in cores:
        at = plain.casefold().find(core.casefold(), max(0, index - len(core) + 1)) if kind == "str" else -1
        if 0 <= at < index < at + len(core):
            core = core[:index - at] + "-" + core[index - at:]
        new_cores.append((kind, core))
    return plain[:index] + "-" + plain[index:], new_cores


def _host_sep(rng, plain, slots, kinds, typ):
    """Host hyphens and dots typed as another dash or dot form (my\u2010site, example\u2024com), as copied from
    typeset text; own RNG so the cases of earlier rounds stay the same."""
    start, end = _host_span(plain, typ)
    # a dot before a native digit makes a native-digit host label (rejected, Codex round 25 id 3)
    free = [index for index in range(start, end - 1) if plain[index] in "-." and slots[index] is None
            and plain[index + 1].isascii() and plain[index + 1].isalnum()
            and (slots[index + 1] is None or slots[index + 1][0] != "native_digit")]
    if not free or rng.random() >= 0.25:
        return kinds
    for index in free:
        if rng.random() < 0.7:
            slots[index] = ("host_sep", rng.choice(HOST_DASHES if plain[index] == "-" else HOST_DOTS))
    return sorted(set(kinds) | {"host_sep"}) if any(slot and slot[0] == "host_sep" for slot in slots) else kinds


def _zw_domain(rng, plain, inserts, kinds, typ="EMAIL"):
    """A break hint (ZWSP, WJ, soft hyphen) between two non-Latin letters of a mail domain or a URL host
    (例\u200bえ.jp), copied from wrapped text; own RNG so the cases of earlier rounds stay the same."""
    if typ == "EMAIL":
        start, end = plain.rindex("@") + 1, len(plain)
    else:
        start, end = re.match(r"(?:https?://|www\.)([^/?#]*)", plain).span(1)
    free = [index for index in range(start + 1, end) if index not in inserts
            and not plain[index - 1].isascii() and not plain[index].isascii()
            and plain[index - 1].isalpha() and plain[index].isalpha()]
    if not free or rng.random() >= 0.25:
        return kinds
    for index in rng.sample(free, rng.randint(1, min(2, len(free)))):
        inserts[index] = ("zw_domain", rng.choice((ZWSP, WJ, SHY)))
    return sorted(kinds + ["zw_domain"])


def make_case(seed: int) -> dict:
    rng = random.Random(seed)  # noqa: S311  # reproducible test data, not crypto
    scripts = rng.sample(sorted(SCRIPT_RANGES), rng.choice((1, 2)))
    values = []
    for _ in range(rng.randint(1, 3)):
        typ = rng.choice(sorted(GENERATORS))
        plain, cores = GENERATORS[typ](rng)
        if typ == "URL":
            plain = _url_shape(random.Random(f"{seed}-{len(values)}-url"), plain)  # noqa: S311
            cores = [core for core in cores if core[1] in plain]  # a bare host has no query values
            plain, cores = _userinfo(random.Random(f"{seed}-{len(values)}-userinfo"), plain, cores)  # noqa: S311
        if typ == "PHONE":
            plain, cores = _paren_phone(random.Random(f"{seed}-{len(values)}-paren"), plain, cores)  # noqa: S311
        if typ == "EMAIL":
            plain, cores = _email_shape(random.Random(f"{seed}-{len(values)}-mail"), plain, cores)  # noqa: S311
            plain, cores = _local_shape(random.Random(f"{seed}-{len(values)}-local"), plain, cores)  # noqa: S311
            plain = _upper(random.Random(f"{seed}-{len(values)}-upper"), plain)  # noqa: S311
        if typ in ("EMAIL", "URL"):
            plain, cores = _host_hyphen(random.Random(f"{seed}-{len(values)}-hyphen"), plain, cores, typ)  # noqa: S311
        month = typ == "DATE" and _month_date(random.Random(f"{seed}-{len(values)}-month"), plain, cores)  # noqa: S311
        if month:
            plain, cores = month
        slots, inserts, kinds = _plan(rng, plain, _legal_join(plain, typ))
        if month:  # known gaps (round 29 notes): styled digits, circled letters, no space beside CJK/Thai
            slots = [None if slot and (plain[index].isdigit() or slot[0].startswith("enclosed")) else slot
                     for index, slot in enumerate(slots)]
            kinds = sorted({slot[0] for slot in slots if slot} | {insert[0] for insert in inserts.values()})
        kinds = _zw_inside(random.Random(f"{seed}-{len(values)}-zw"), plain, inserts, kinds)  # noqa: S311
        if typ in ("EMAIL", "URL"):
            kinds = _idna_dot(random.Random(f"{seed}-{len(values)}-idna"), plain, slots, kinds, typ)  # noqa: S311
            kinds = _host_sep(random.Random(f"{seed}-{len(values)}-hostsep"), plain, slots, kinds, typ)  # noqa: S311
        if typ == "EMAIL":
            kinds = _zw_domain(random.Random(f"{seed}-{len(values)}-zwd"), plain, inserts, kinds)  # noqa: S311
            kinds = _local_sep(random.Random(f"{seed}-{len(values)}-localsep"), plain, slots, kinds)  # noqa: S311
        if typ == "URL":
            kinds = _zw_domain(random.Random(f"{seed}-{len(values)}-zwh"), plain, inserts, kinds, typ)  # noqa: S311
        if typ in ("PHONE", "DATE"):
            kinds = _typo_sep(random.Random(f"{seed}-{len(values)}-sep"), plain, slots, kinds)  # noqa: S311
        glue = rng.random() < 0.5 and not month
        values.append({"type": typ, "plain": plain, "cores": cores, "slots": slots, "inserts": inserts,
                       "kinds": kinds + (["glue"] if glue else []), "glue": glue, "month": bool(month)})
    context_scripts = [rng.choice(scripts) for _ in range(len(values) + 1)]
    context = [_sentence(rng, script) for script in context_scripts]
    case = {"seed": seed, "scripts": scripts, "context": context, "context_scripts": context_scripts, "values": values}
    for value in values:
        assert fold(render(value)) == fold(value["plain"]), (seed, value["plain"], render(value))
    return case


def _escape(rng, text):
    """Each character as a literal backslash escape the cleaner's escape step decodes."""
    def forms(cp):
        return [f for f, ok in (("\\x%02x", cp < 0x100), ("\\u%04x", cp < 0x10000), ("\\U%08x", True)) if ok]
    return "".join(rng.choice(forms(ord(ch))) % ord(ch) for ch in text)


def _entity(rng, text):
    """Each character as an HTML entity the cleaner's ftfy step decodes, also escaped twice (``&amp;#51;``)."""
    forms = [rng.choice(("&#%d;", "&#x%x;", "&amp;#%d;")) % ord(ch) for ch in text]
    # ftfy keeps a few entities raw ("&#59;"), so those characters stay as they are
    return "".join(form if _unescape_html(form) == ch else ch for form, ch in zip(forms, text))


def _unescape_html(text):
    while (decoded := unescape_html(text)) != text:  # the cleaner's ftfy decode, to its fixpoint
        text = decoded
    return text


def _unescape_backslash(text):
    return text.encode("latin", "backslashreplace").decode("unicode-escape")


def escaped_case(case, kind="escaped", encode=_escape, decode=_unescape_backslash):
    """The case with a few characters of some values written as backslash escapes (``06 12\\u0033\\x34``) or, with
    ``kind="entity"``, HTML entities (``06 12&amp;#32;34``); own RNG per value and kind, so the plain case and the
    other kind stay the same. A value already encoded by an earlier pass is left as it is."""
    values = []
    salt = "escape" if kind == "escaped" else kind
    for index, value in enumerate(case["values"]):
        rng = random.Random(f"{case['seed']}-{index}-{salt}")  # noqa: S311
        if "escaped" in value["kinds"]:
            values.append(value)
            continue
        value = dict(value, slots=list(value["slots"]), inserts=dict(value["inserts"]))
        positions = [i for i, slot in enumerate(value["slots"]) if slot is None or slot[1]]
        if rng.random() < 0.3:
            for i in rng.sample(positions, min(len(positions), rng.randint(1, 3))):
                slot = value["slots"][i]
                value["slots"][i] = (kind, encode(rng, value["plain"][i] if slot is None else slot[1]))
            for i, (_, text) in list(value["inserts"].items()):
                if rng.random() < 0.5:
                    value["inserts"][i] = (kind, encode(rng, text))
            value["kinds"] = sorted({slot[0] for slot in value["slots"] if slot}
                                    | {insert[0] for insert in value["inserts"].values()}) + (
                ["glue"] if value["glue"] else [])
            decoded = decode(render(value))
            assert fold(decoded) == fold(value["plain"]), (case["seed"], value["plain"], render(value))
        values.append(value)
    return dict(case, values=values)


def render(value, disabled=frozenset()):
    out = []
    for index, ch in enumerate(value["plain"] + "\0"):
        insert = value["inserts"].get(index)
        if insert and insert[0] not in disabled:
            out.append(insert[1])
        if ch == "\0":
            break
        slot = value["slots"][index]
        out.append(ch if slot is None or slot[0] in disabled else slot[1])
    return "".join(out)


def build_doc(case, disabled=None) -> str:
    disabled = disabled or {}
    doc = case["context"][0]
    for index, value in enumerate(case["values"]):
        off = disabled.get(index, frozenset())
        glue = value["glue"] and "glue" not in off
        left, right = case["context_scripts"][index], case["context_scripts"][index + 1]
        # an address, a sentence and a mail local part glued all in one run have no boundary to find
        # (a@b.jpกขคฅก＋x@y.th: the prose may be either address), so the second address keeps its space
        prev = case["values"][index - 1] if index else None
        if (prev and prev["glue"] and "glue" not in disabled.get(index - 1, frozenset())
                and prev["type"] in ("EMAIL", "URL") and value["type"] == "EMAIL"):
            glue = False
        doc += ("" if glue and left in UNSPACED else " ") + render(value, off)
        doc += ("" if glue and right in UNSPACED else " ") + case["context"][index + 1]
    return doc


def steps(value):
    """Readable render steps: kind -> characters changed."""
    out = defaultdict(list)
    for index, slot in enumerate(value["slots"]):
        if slot:
            out[slot[0]].append(f"{value['plain'][index]}->{slot[1] or '(joined)'}")
    for index, (kind, ch) in sorted(value["inserts"].items()):
        out[kind].append(f"+U+{ord(ch):04X}@{index}" if len(ch) == 1 else f"+{ch}@{index}")  # escaped: as written
    if value["glue"]:
        out["glue"].append("no space beside CJK/Thai")
    return dict(out)


# --- oracle ---


@lru_cache(maxsize=None)
def cleaner(preset: str, fold_styled: bool, mode: str) -> TextCleaner:
    name, extra = PRESET_ARGS[preset]
    cfg = TextCleanerConfig.from_preset(name, check_ner_process=False, fold_styled_characters=fold_styled, **extra,
                                        replacement_mode="placeholder" if mode == "stream" else mode)
    return TextCleaner(cfg)


_DIGIT_RUN = re.compile(r"\d+(?:[^\w<>]{1,3}\d+)*")


def leaked(core, text_fold: str) -> bool:
    kind, value = core
    if kind == "str":
        return fold(value) in text_fold
    runs = [re.findall(r"\d+", m.group()) for m in _DIGIT_RUN.finditer(text_fold)]
    if kind == "ip":
        size = len(value)
        return ".".join(value) in text_fold or any(
            tuple(groups[i:i + size]) == tuple(value) for groups in runs for i in range(len(groups)))
    window = min(7, len(value))
    joined = ["".join(groups) for groups in runs]
    return any(value[i:i + window] in run for run in joined for i in range(len(value) - window + 1))


def run_stream(cl, doc, max_chars, overlap, seed):
    rng = random.Random(seed * 7919 + max_chars)  # noqa: S311  # reproducible chunking
    stream = WindowedStreamingRedactor(cl, max_buffer_chars=max_chars, overlap_chars=overlap)
    out, start = [], 0
    while start < len(doc):
        size = rng.randint(1, 20)
        out.append(stream.feed(doc[start:start + size]))
        start += size
    return "".join(out) + stream.finish()


def outputs(doc, preset, fold_styled, mode, seed):
    """Output text(s) of one mode; stream gives one per window size (None on BufferError)."""
    cl = cleaner(preset, fold_styled, mode)
    if mode != "stream":
        result = cl.process(doc)
        restored = result.metadata["anon_map"].deanonymize(result.lm_text) if mode == "reversible" else None
        return {"main": result.lm_text, "restored": restored}
    found = {}
    for max_chars, overlap in STREAM_PARAMS:
        try:
            found[f"{max_chars}/{overlap}"] = run_stream(cl, doc, max_chars, overlap, seed)
        except BufferError:
            found[f"{max_chars}/{overlap}"] = None
    return found


def active_cores(case, preset):
    """(value index, core) pairs whose type the config replaces and which are not in the prose."""
    cfg = cleaner(preset, True, "placeholder").cfg
    context_fold = fold(" ".join(case["context"]))
    return [(index, core) for index, value in enumerate(case["values"]) if getattr(cfg, TYPE_FLAG[value["type"]])
            for core in value["cores"] if not leaked(core, context_fold)]


def _leaks(case, preset, fold_styled, mode, index, core, disabled, key):
    out = outputs(build_doc(case, disabled), preset, fold_styled, mode, case["seed"])
    text = out.get(key)
    if text is None:
        return False
    return (not leaked(core, fold(text))) if key == "restored" else leaked(core, fold(text))


def _shrink(case, index, still_fails):
    """Greedy: drop each kind of value ``index`` while the failure remains."""
    kept = list(case["values"][index]["kinds"])
    for kind in list(kept):
        trial = [k for k in kept if k != kind]
        off = frozenset(set(case["values"][index]["kinds"]) - set(trial))
        if still_fails({index: off}):
            kept = trial
    return kept


def _shrink_all(case, still_fails):
    """Greedy over (value, kind) pairs for document-level failures."""
    pairs = [(i, k) for i, value in enumerate(case["values"]) for k in value["kinds"]]
    off = defaultdict(set)
    for index, kind in pairs:
        off[index].add(kind)
        if not still_fails({i: frozenset(s) for i, s in off.items()}):
            off[index].discard(kind)
    return sorted(f"{case['values'][i]['type']}:{k}" for i, k in pairs if k not in off[i])


def _record(case, doc, check, preset, fold_styled, mode, output, **extra):
    return {"seed": case["seed"], "check": check, "preset": preset, "fold_styled_characters": fold_styled,
            "mode": mode, "counts": fold_styled, "input": doc, "output": output, **extra}


def check_case(seed: int, modes=MODES, folds=(True, False)) -> dict:
    plain_case = make_case(seed)
    context_doc = " ".join(plain_case["context"])
    failures, stream_errors = [], 0
    for preset in PRESETS:
        decodes = cleaner(preset, True, "placeholder").cfg.check_decode_unicode_escapes
        case = escaped_case(escaped_case(plain_case) if decodes else plain_case, "entity", _entity, _unescape_html)
        doc = build_doc(case)
        # a window keeps a spaced token whole only when it fits in the overlap; a month-name date or a value
        # written with escapes (6-10 characters per escaped one) may not (48/12)
        short = {f"{size}/{overlap}" for size, overlap in STREAM_PARAMS
                 if any((value["month"] or {"escaped", "entity"} & set(value["kinds"])) and len(render(value)) > overlap
                        for value in case["values"])}
        cores = active_cores(case, preset)
        for fold_styled in folds:
            suffix = "" if fold_styled else "|fold_off"
            placeholder = cleaner(preset, fold_styled, "placeholder")
            # false positives: prose without PII comes out unchanged (training) or without PII tags (default)
            prose = placeholder.process(context_doc).lm_text
            if prose != context_doc if preset.startswith("training") else re.search(r"<[A-Z_]+>", prose):
                failures.append(_record(case, context_doc, "false_positive", preset, fold_styled, "placeholder", prose,
                                        signature=f"false_positive|{preset}|{'+'.join(case['scripts'])}{suffix}"))
            whole = placeholder.process(doc).lm_text
            for mode in modes:
                out = outputs(doc, preset, fold_styled, mode, seed)
                for key, text in out.items():
                    if text is None or (mode == "stream" and key in short):
                        stream_errors += text is None and mode == "stream"
                        continue
                    text_fold = fold(text)
                    is_restore = key == "restored"
                    for index, core in cores:
                        if leaked(core, text_fold) == is_restore:
                            continue
                        value = case["values"][index]
                        check = "restore" if is_restore else "leak"
                        kinds = _shrink(case, index, lambda off, index=index, core=core, mode=mode, key=key: _leaks(
                            case, preset, fold_styled, mode, index, core, off, key))
                        failures.append(_record(
                            case, doc, check, preset, fold_styled, mode, text, stream=key if mode == "stream" else None,
                            type=value["type"], plain=value["plain"], rendered=render(value), core=list(core),
                            steps=steps(value), minimal_kinds=kinds,
                            signature=f"{check}|{value['type']}|{'+'.join(kinds) or 'plain'}|{mode}{suffix}"))
                        break  # one record per output is enough to cluster
                    if is_restore and preset.startswith("training") and text_fold != fold(_unescape_html(doc)):
                        # training keeps the prose, so restore gives the whole input back (up to the fold and the
                        # cleaner's HTML entity decoding)
                        def restore_differs(off):
                            changed = build_doc(case, off)
                            got = outputs(changed, preset, fold_styled, "reversible", seed)["restored"]
                            return fold(got) != fold(_unescape_html(changed))
                        kinds = _shrink_all(case, restore_differs)
                        failures.append(_record(
                            case, doc, "restore_exact", preset, fold_styled, mode, text, minimal_kinds=kinds,
                            signature=f"restore_exact|{preset}|{'+'.join(kinds) or 'plain'}{suffix}"))
                    if mode == "stream" and text != whole:
                        def mismatch(off, key=key):
                            changed = build_doc(case, off)
                            got = outputs(changed, preset, fold_styled, "stream", seed)[key]
                            return got is not None and got != placeholder.process(changed).lm_text
                        kinds = _shrink_all(case, mismatch)
                        failures.append(_record(
                            case, doc, "stream_mismatch", preset, fold_styled, mode, text, stream=key, whole=whole,
                            minimal_kinds=kinds,
                            signature=f"stream_mismatch|{preset}|{'+'.join(kinds) or 'plain'}{suffix}"))
    return {"seed": seed, "failures": failures, "stream_errors": stream_errors,
            "types": [v["type"] for v in plain_case["values"]]}


def run(seeds, workers=1, modes=MODES):
    check = partial(check_case, modes=tuple(modes))
    if workers > 1:
        with multiprocessing.get_context("spawn").Pool(workers) as pool:
            return list(pool.imap(check, seeds, chunksize=8))
    return [check(seed) for seed in seeds]


def cluster(results):
    clusters = {}
    for result in results:
        for failure in result["failures"]:
            entry = clusters.setdefault(failure["signature"], {
                "count": 0, "counts": failure["counts"], "check": failure["check"], "seeds": [], "presets": set(),
                "example": failure})
            entry["count"] += 1
            entry["presets"].add(failure["preset"])
            if failure["seed"] not in entry["seeds"]:
                entry["seeds"].append(failure["seed"])
    return clusters


def summary(results, clusters) -> str:
    totals = Counter()
    for check in CHECKS:
        totals[check] = sum(c["count"] for c in clusters.values() if c["check"] == check and c["counts"])
        totals[check + "_fold_off"] = sum(c["count"] for c in clusters.values() if c["check"] == check and not c["counts"])
    lines = [
        f"cases: {len(results)}; values: {sum(len(r['types']) for r in results)} "
        f"({dict(Counter(t for r in results for t in r['types']))})",
        f"stream BufferError (documented, not counted): {sum(r['stream_errors'] for r in results)}",
        "counted (fold on): " + ", ".join(f"{k}={totals[k]}" for k in CHECKS),
        "recorded only (fold off): " + ", ".join(
            f"{k}={totals[k + '_fold_off']}" for k in CHECKS),
        "",
        "| cluster | counted | records | seeds | presets | example seed |",
        "|---|---|---|---|---|---|",
    ]
    for signature, entry in sorted(clusters.items(), key=lambda item: (not item[1]["counts"], -item[1]["count"])):
        lines.append(f"| `{signature}` | {'yes' if entry['counts'] else 'no'} | {entry['count']} | {len(entry['seeds'])} "
                     f"| {','.join(sorted(entry['presets']))} | {entry['seeds'][0]} |")
    lines += ["", "Examples (first record per counted cluster):", ""]
    for signature, entry in sorted(clusters.items(), key=lambda item: -item[1]["count"]):
        if entry["counts"]:
            example = entry["example"]
            lines.append(f"- `{signature}` seed {example['seed']} ({example['preset']}): "
                         f"value {example.get('rendered', '')!r} -> output {example['output'][:200]!r}")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seeds", type=int, default=200)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--out", type=Path, help="jsonl of failures")
    parser.add_argument("--report", type=Path, help="markdown summary")
    args = parser.parse_args(argv)
    results = run(range(args.start, args.start + args.seeds), args.workers, args.modes)
    clusters = cluster(results)
    if args.out:
        with args.out.open("w", encoding="utf-8") as handle:
            for result in results:
                for failure in result["failures"]:
                    handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
    text = summary(results, clusters)
    if args.report:
        args.report.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 1 if any(entry["counts"] for entry in clusters.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
