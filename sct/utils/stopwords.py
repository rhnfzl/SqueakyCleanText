from collections.abc import Collection, Mapping

from stop_words import StopWordError, get_stop_words as _get_pkg_stop_words

from sct.config import DEFAULT_LANGUAGES
from sct.utils.resources import language_to_iso


class ProcessStopwords:

    def __init__(self, supported_languages: Collection[str] | None = None,
                 custom_stopwords: Mapping[str, Collection[str]] | None = None):
        """
        Args:
            supported_languages: frozenset of language names (e.g. {'ENGLISH', 'POLISH'})
            custom_stopwords: dict mapping language name -> frozenset of stopwords.
                             Custom sets REPLACE auto-detected ones.
        """
        languages = supported_languages or DEFAULT_LANGUAGES
        custom = custom_stopwords or {}

        self._lang_map: dict[str, set[str]] = {}
        for lang in languages:
            if lang in custom:
                # Custom stopwords replace auto-detected
                self._lang_map[lang] = set(custom[lang])
            else:
                iso = language_to_iso(lang)
                try:
                    self._lang_map[lang] = set(_get_pkg_stop_words(iso))
                except (KeyError, StopWordError):
                    # stop-words package doesn't support this language
                    self._lang_map[lang] = set()

    def get_stop_words(self, language: str) -> set:
        """Get the stopword set for a language. Falls back to English."""
        return self._lang_map.get(language, self._lang_map.get('ENGLISH', set()))

    def remove_stopwords(self, text: str, language: str) -> str:
        """Remove stopwords based on the detected language.

        Falls back to English stopwords for unknown languages.
        """
        stop_words = self.get_stop_words(language)
        return " ".join(word for word in text.split() if word not in stop_words)
