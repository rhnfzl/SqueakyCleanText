from sct.utils import constants


class ProcessDateTime:

    def __init__(self):
        pass

    def replace_years(self, text, replace_with="<YEAR>"):
        """
        Replaces years between 1900 to 2099 in the text with a special token.
        """
        cleaned_string = constants.YEAR_REGEX.sub(replace_with, text)

        return cleaned_string

    def replace_dates(self, text, replace_with="<DATE>"):
        """
        Replaces common date formats in the text with a special token.

        Supports ISO 8601, DD/MM/YYYY, MM-DD-YYYY, and multilingual month-name
        formats (EN, NL, DE, ES).
        Should be called before replace_years to avoid partial matches.
        """
        cleaned_string = constants.DATE_REGEX.sub(replace_with, text)

        return cleaned_string

    def fuzzy_replace_dates(self, text, replace_with="<DATE>", score_cutoff=80,
                            language=None):
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
            replace_with: Replacement token for detected dates.
            score_cutoff: Minimum RapidFuzz score (0-100) to accept a match.
            language: Detected language (e.g. "ENGLISH", "GERMAN"). If None,
                      checks all languages.

        Returns:
            Text with fuzzy-matched date expressions replaced.
        """
        try:
            from rapidfuzz import fuzz, process as rfprocess
        except ImportError:
            raise ImportError(
                "rapidfuzz is required for fuzzy date detection. "
                "Install it with: pip install squeakycleantext[fuzzy]"
            )

        # Select vocabulary based on detected language
        if language and language in constants.FUZZY_MONTH_VOCABULARY_BY_LANG:
            # Use detected language + English fallback (deduplicated via set)
            lang_names = set(constants.FUZZY_MONTH_VOCABULARY_BY_LANG[language])
            if language != "ENGLISH":
                lang_names.update(constants.FUZZY_MONTH_VOCABULARY_BY_LANG.get("ENGLISH", ()))
            choices = tuple(lang_names)
        else:
            choices = constants.FUZZY_MONTH_VOCABULARY

        def _is_fuzzy_month(word):
            """Check if word fuzzy-matches any known month name."""
            result = rfprocess.extractOne(
                word.lower(), choices, scorer=fuzz.ratio, score_cutoff=score_cutoff
            )
            return result is not None

        # Replace each candidate match where the word is a fuzzy month name
        def _replace_match(match):
            # Group 1 = month-first pattern, Group 2 = day-first pattern
            candidate = match.group(1) or match.group(2)
            if candidate and _is_fuzzy_month(candidate):
                return replace_with
            return match.group(0)

        return constants.FUZZY_DATE_CANDIDATE_REGEX.sub(_replace_match, text)
