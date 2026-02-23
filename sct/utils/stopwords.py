from stop_words import get_stop_words as _get_pkg_stop_words

from sct.utils.resources import language_to_iso


class ProcessStopwords:

    def __init__(self, supported_languages=None, custom_stopwords=None):
        """
        Args:
            supported_languages: frozenset of language names (e.g. {'ENGLISH', 'POLISH'})
            custom_stopwords: dict mapping language name -> frozenset of stopwords.
                             Custom sets REPLACE auto-detected ones.
        """
        # Default to original 4 for backward compat
        languages = supported_languages or frozenset({'ENGLISH', 'DUTCH', 'GERMAN', 'SPANISH'})
        custom = custom_stopwords or {}

        self._lang_map = {}
        for lang in languages:
            if lang in custom:
                # Custom stopwords replace auto-detected
                self._lang_map[lang] = set(custom[lang])
            else:
                iso = language_to_iso(lang)
                try:
                    self._lang_map[lang] = set(_get_pkg_stop_words(iso))
                except KeyError:
                    # stop-words package doesn't support this language
                    self._lang_map[lang] = set()

        # Backward compat named attributes
        self.STOP_WORDS_EN = self._lang_map.get('ENGLISH', set())
        self.STOP_WORDS_NL = self._lang_map.get('DUTCH', set())
        self.STOP_WORDS_DE = self._lang_map.get('GERMAN', set())
        self.STOP_WORDS_ES = self._lang_map.get('SPANISH', set())

    def get_stop_words(self, language: str) -> set:
        """Get the stopword set for a language. Falls back to English."""
        return self._lang_map.get(language, self._lang_map.get('ENGLISH', set()))

    def remove_stopwords(self, text, lan):
        """Remove stopwords based on the detected language.

        Falls back to English stopwords for unknown languages.
        """
        stop_words = self._lang_map.get(lan, self._lang_map.get('ENGLISH', set()))
        return " ".join(word for word in text.split() if word not in stop_words)
