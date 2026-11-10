import re
from collections.abc import Callable, Mapping
from typing import Any

from sct.utils import constants
from sct.utils.contact import DigitView


class ProcessDateTime:

    def __init__(self, date_regex: re.Pattern[str] | None = None, fuzzy_vocabulary: tuple[str, ...] | None = None,
                 fuzzy_vocabulary_by_lang: Mapping[str, tuple[str, ...]] | None = None):
        """
        Args:
            date_regex: Compiled regex for date matching. Defaults to constants.DATE_REGEX.
            fuzzy_vocabulary: Tuple of full month names for fuzzy matching.
            fuzzy_vocabulary_by_lang: Dict mapping language -> tuple of month names.
        """
        # Matched on the decimal-digit view, so ١٥/٠١/١٩٩٠ and १५/०१/१९९० are dates like 15/01/1990.
        self._date_regex = DigitView(date_regex or constants.DATE_REGEX)
        # Only None means "defaults": an empty vocabulary (date_builtin_month_languages=()) matches no month.
        self._fuzzy_vocabulary = (constants.FUZZY_MONTH_VOCABULARY if fuzzy_vocabulary is None
                                  else fuzzy_vocabulary)
        self._fuzzy_vocabulary_by_lang = (constants.FUZZY_MONTH_VOCABULARY_BY_LANG if fuzzy_vocabulary_by_lang is None
                                          else fuzzy_vocabulary_by_lang)

    def replace_years(self, text: str, replace_with: str | Callable[[re.Match[str]], str] = "<YEAR>") -> str:
        """
        Replaces years between 1900 to 2099 in the text with a special token.
        """
        return constants.YEAR_REGEX.sub(replace_with, text)

    def replace_dates(self, text: str, replace_with: str | Callable[[Any], str] = "<DATE>") -> str:
        """
        Replaces common date formats in the text with a special token.

        Supports ISO 8601, DD/MM/YYYY, MM-DD-YYYY, and multilingual month-name
        formats (EN, NL, DE, ES + any custom month names).
        Should be called before replace_years to avoid partial matches.
        """
        return self._date_regex.sub(replace_with, text)

    def fuzzy_replace_dates(self, text: str, replace_with: str | Callable[[str], str] = "<DATE>",
                            score_cutoff: float = 80, language: str | None = None) -> str:
        """
        Fuzzy post-regex pass: catches misspelled month names that DATE_REGEX missed.

        Requires ``rapidfuzz`` (install with ``pip install squeakycleantext[fuzzy]``).
        Scans for word+number patterns via FUZZY_DATE_CANDIDATE_REGEX, then
        fuzzy-matches candidate words against the month name vocabulary using
        RapidFuzz ``extractOne``.  Only replaces when score >= *score_cutoff*.

        When *language* is provided (e.g. ``"ENGLISH"``), only month names for
        that language (plus English as fallback) are checked — reducing false
        positive risk.  When *language* is ``None``, all supported languages
        are checked.

        Should be called AFTER replace_dates() so only near-misses remain.

        Args:
            text: Input text (ideally already processed by replace_dates).
            replace_with: Replacement token for detected dates, or a callable
                          ``fn(matched_text) -> replacement``.
            score_cutoff: Minimum RapidFuzz score (0-100) to accept a match.
            language: Detected language (e.g. "ENGLISH", "GERMAN"). If None,
                      checks all languages.

        Returns:
            Text with fuzzy-matched date expressions replaced.
        """
        parts: list[str] = []
        last = 0
        for match in self.fuzzy_date_matches(text, score_cutoff=score_cutoff, language=language):
            replacement = replace_with(match.group()) if callable(replace_with) else replace_with
            parts.extend((text[last:match.start()], replacement))
            last = match.end()
        return ''.join(parts) + text[last:]

    def fuzzy_date_matches(self, text: str, *, score_cutoff: float = 80,
                           language: str | None = None) -> list[re.Match[str]]:
        """Return validated fuzzy candidates with their original input positions."""
        try:
            from rapidfuzz import fuzz, process as rfprocess
        except ImportError as exc:
            raise ImportError(
                "rapidfuzz is required for fuzzy date detection. "
                "Install it with: pip install squeakycleantext[fuzzy]"
            ) from exc

        # Select vocabulary based on detected language
        if language and language in self._fuzzy_vocabulary_by_lang:
            # Use detected language + English fallback (deduplicated via set)
            lang_names = set(self._fuzzy_vocabulary_by_lang[language])
            if language != "ENGLISH":
                lang_names.update(self._fuzzy_vocabulary_by_lang.get("ENGLISH", ()))
            choices = tuple(lang_names)
        else:
            choices = self._fuzzy_vocabulary
        if not choices:
            return []

        def _is_fuzzy_month(word):
            """Check if word fuzzy-matches any known month name."""
            result = rfprocess.extractOne(
                word.lower(), choices, scorer=fuzz.ratio, score_cutoff=score_cutoff
            )
            return result is not None

        return [match for match in constants.FUZZY_DATE_CANDIDATE_REGEX.finditer(text)
                if (candidate := match.group(1) or match.group(2)) and _is_fuzzy_month(candidate)]
