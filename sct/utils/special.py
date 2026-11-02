import json
import re
import unicodedata

from sct.utils import constants


def _drop_if_isolated(match):
    text, i = match.string, match.start()
    touches_word = any(
        0 <= j < len(text) and unicodedata.category(text[j])[0] in 'LNM' for j in (i - 1, i + 1)
    )
    return match.group() if touches_word else ''


_JSON = json.JSONDecoder()
# The library's own "<NUMBER>"-shaped tokens: earlier steps put them into JSON values, so they count as one.
_TOKEN = re.compile(r'<[A-Z][A-Z0-9_]*>')
_JSON_WINDOW = 4096


def _json_object_spans(text):
    """(start, end) of each outermost {...} that parses as a JSON object (replacement tokens read as values)."""
    last_close = text.rfind('}')  # a '{' with no '}' after it cannot start an object
    # Same length, so offsets map back to text: "<NUMBER>" -> "0       ".
    view = _TOKEN.sub(lambda m: '0'.ljust(len(m.group())), text)
    # Parse a slice from a recent base: a failed parse counts lines from the slice start, so this keeps
    # each failure O(window) instead of O(offset).
    spans, base, chunk = [], 0, view
    i = text.find('{')
    while -1 < i < last_close:
        if i - base > _JSON_WINDOW:
            base, chunk = i, view[i:last_close + 1]
        try:
            obj, end = _JSON.raw_decode(chunk, i - base)
            end += base
        except ValueError as err:
            # A '{' the failed parse already read through sits at a value position of the same broken
            # text, so it fails there too; resuming after it keeps malformed input linear.
            obj, end = None, max(i + 1, getattr(err, 'pos', 0) + base)
        except RecursionError:
            break  # shortcut: nesting past the recursion limit is not real JSON; later objects are not kept
        if isinstance(obj, dict):
            spans.append((i, end))
        i = text.find('{', end)
    return spans


def _remove_bracketed(text, remove_brackets, remove_braces, max_chars, max_words):
    if not (max_chars > 0 or max_words > 0):
        if remove_brackets:
            text = constants.BRACKET_CONTENT_REGEX.sub('', text)
        if remove_braces:
            text = constants.BRACE_CONTENT_REGEX.sub('', text)
        return text

    def drop(m):
        inner = m.group()[1:-1]
        too_long = (0 < max_chars < len(inner)) or (0 < max_words < len(inner.split()))
        return m.group() if too_long else ''
    if remove_brackets:
        text = constants.BRACKET_INNER_REGEX.sub(drop, text)
    if remove_braces:
        text = constants.BRACE_INNER_REGEX.sub(drop, text)
    return text


class ProcessSpecialSymbols:

    def __init__(self):
        pass

    def replace_currency_symbols(self, text, replace_with="<CUR>"):
        """
        Replace currency symbols in ``text`` str with string specified by ``replace_with`` str.
        Args:
            text (str): raw text
            replace_with (str): if None (default), replace symbols with
                their standard 3-letter abbreviations (e.g. '$' with 'USD', '£' with 'GBP');
                otherwise, pass in a string with which to replace all symbols
                (e.g. "*CURRENCY*")
        """
        if replace_with is None:
            return self.replace_currency_symbols_with_fn(text, constants.CURRENCIES.__getitem__)
        return constants.CURRENCY_REGEX.sub(replace_with, text)

    def replace_currency_symbols_with_fn(self, text, replace_fn):
        """Replace each single currency symbol: replace_fn(symbol) -> replacement."""
        return constants.CURRENCY_SYMBOL_REGEX.sub(lambda m: replace_fn(m.group()), text)


    def remove_isolated_letters(self, text):
        """
        Removes any isolated letters which doesn't add any value to the text.
        """
        cleaned_text = constants.ISOLATED_LETTERS_REGEX.sub('', text)

        return cleaned_text

    def remove_isolated_special_symbols(self, text, remove_brackets=True, remove_braces=True,
                                        max_chars=0, max_words=0, keep_json=False):
        """
        Removes any isolated symbols which shouldn't be present in the text.

        Args:
            text: Input text.
            remove_brackets: If True, remove [...] content (image/file references).
            remove_braces: If True, remove {...} content (HTML links/templates).
            max_chars, max_words: when either is > 0, only innermost spans within both limits
                are removed; 0 and 0 = remove every span (0.8.0).
            keep_json: never remove a {...} that parses as a JSON object, nor anything inside it.
        """
        spans = _json_object_spans(text) if keep_json and (remove_brackets or remove_braces) else []
        parts, last = [], 0
        for start, end in spans:
            parts += [_remove_bracketed(text[last:start], remove_brackets, remove_braces, max_chars, max_words),
                      text[start:end]]
            last = end
        parts.append(_remove_bracketed(text[last:], remove_brackets, remove_braces, max_chars, max_words))
        cleaned_text = ''.join(parts)
        cleaned_text = constants.ISOLATED_SPECIAL_SYMBOLS_REGEX.sub('', cleaned_text)
        cleaned_text = constants.ISOLATED_QUOTES_REGEX.sub(_drop_if_isolated, cleaned_text)

        return cleaned_text

    def remove_punctuation(self, text):
        return constants.PUNCTUATION_REGEX.sub('', text)
