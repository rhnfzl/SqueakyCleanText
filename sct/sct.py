"""
This code provides a comprehensive text cleaning and preprocessing pipeline. 
It includes functions to normalize, remove personal information and clean text data, 
which is crucial for natural language processing tasks.
"""
from sct import config
from sct.utils import contact, datetime, ner, normtext, resources, special, stopwords

class TextCleaner:
    
    def __init__(self):
        self.ProcessContacts = contact.ProcessContacts()
        self.ProcessDateTime = datetime.ProcessDateTime()
        self.ProcessSpecialSymbols = special.ProcessSpecialSymbols()
        self.NormaliseText = normtext.NormaliseText()
        self.ProcessStopwords = stopwords.ProcessStopwords()
        self.GeneralNER = ner.GeneralNER()
    
    def process(self, text):
        #--*
        text = str(text)
        
        #--*
        if config.CHECK_DETECT_LANGUAGE:
            language = str(resources.DETECTOR.detect_language_of(text)).split(".")[-1]
        #--*
        if config.CHECK_FIX_BAD_UNICODE:
            text = self.NormaliseText.fix_bad_unicode(text)

        if config.CHECK_TO_ASCII_UNICODE:
            text = self.NormaliseText.to_ascii_unicode(text)

        #--*
        if config.CHECK_REPLACE_HTML:
            text = self.ProcessContacts.replace_html(text, replace_with = config.REPLACE_WITH_HTML)
        if config.CHECK_REPLACE_URLS:
            text = self.ProcessContacts.replace_urls(text, replace_with = config.REPLACE_WITH_URL)
        if config.CHECK_REPLACE_EMAILS:
            text = self.ProcessContacts.replace_emails(text, replace_with = config.REPLACE_WITH_EMAIL)
        if config.CHECK_REPLACE_YEARS:
            text = self.ProcessDateTime.replace_years(text, replace_with = config.REPLACE_WITH_YEARS)
        if config.CHECK_REPLACE_PHONE_NUMBERS:
            text = self.ProcessContacts.replace_phone_numbers(text, replace_with = config.REPLACE_WITH_PHONE_NUMBERS)
        if config.CHECK_REPLACE_NUMBERS:
            text = self.ProcessContacts.replace_numbers(text, replace_with = config.REPLACE_WITH_NUMBERS)
        if config.CHECK_REPLACE_CURRENCY_SYMBOLS:
            text = self.ProcessSpecialSymbols.replace_currency_symbols(text, replace_with = config.REPLACE_WITH_CURRENCY_SYMBOLS)
        #--*
        if config.CHECK_NER_PROCESS:
            ner_words = self.GeneralNER.ner_process(text, config.POSITIONAL_TAGS, config.NER_CONFIDENCE_THRESHOLD, language)
            text = self.ProcessStopwords.remove_words_from_string(text, ner_words)
        #--*
        if config.CHECK_REMOVE_ISOLATED_LETTERS:
            text = self.ProcessSpecialSymbols.remove_isolated_letters(text)
        if config.CHECK_REMOVE_ISOLATED_SPECIAL_SYMBOLS:
            text = self.ProcessSpecialSymbols.remove_isolated_special_symbols(text)
        if config.CHECK_NORMALIZE_WHITESPACE:
            text = self.NormaliseText.normalize_whitespace(text, no_line_breaks=True)
        #--*
        if config.CHECK_STATISTICAL_MODEL_PROCESSING:
            if config.CHECK_CASEFOLD:
                stext = text.casefold() # lowercase
            if config.CHECK_REMOVE_STOPWORDS:
                stext = self.ProcessStopwords.remove_stopwords(stext, language)
            if config.CHECK_REMOVE_PUNCTUATION:
                stext = self.ProcessSpecialSymbols.remove_punctuation(stext)
            if config.CHECK_REMOVE_ISOLATED_LETTERS:
                stext = self.ProcessSpecialSymbols.remove_isolated_letters(stext)
            if config.CHECK_NORMALIZE_WHITESPACE:
                stext = self.NormaliseText.normalize_whitespace(stext)
        #--*
            return text, stext, language
        else:
            return text, language