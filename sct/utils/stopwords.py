from stop_words import get_stop_words


# Language name to ISO 639-1 code mapping
_LANG_TO_ISO = {
    'ENGLISH': 'en',
    'DUTCH': 'nl',
    'GERMAN': 'de',
    'SPANISH': 'es',
}


class ProcessStopwords:

    def __init__(self):
        # Use sets for O(1) membership testing (was O(n) with lists)
        self.STOP_WORDS_EN = set(get_stop_words('en'))
        self.STOP_WORDS_NL = set(get_stop_words('nl'))
        self.STOP_WORDS_DE = set(get_stop_words('de'))
        self.STOP_WORDS_ES = set(get_stop_words('es'))

        self._lang_map = {
            'ENGLISH': self.STOP_WORDS_EN,
            'DUTCH': self.STOP_WORDS_NL,
            'GERMAN': self.STOP_WORDS_DE,
            'SPANISH': self.STOP_WORDS_ES,
        }

    def get_stop_words(self, language: str) -> set:
        """Get the stopword set for a language. Falls back to English."""
        return self._lang_map.get(language, self.STOP_WORDS_EN)

    def remove_stopwords(self, text, lan):
        """Remove stopwords based on the detected language.

        Falls back to English stopwords for unknown languages.
        """
        stop_words = self._lang_map.get(lan, self.STOP_WORDS_EN)
        return " ".join(word for word in text.split() if word not in stop_words)
