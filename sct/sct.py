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
        self.pipeline = []
        self.language = None
        self.init_pipeline()
    
    def init_pipeline(self):
        # Initialize pipeline steps based on config
        language_config = config.LANGUAGE.lower() if config.LANGUAGE else None

        if language_config and language_config in resources.LANGUAGE_NAME:
            self.language = language_config.upper()
        elif any([config.CHECK_DETECT_LANGUAGE, config.CHECK_NER_PROCESS, config.CHECK_REMOVE_STOPWORDS]):
            self.pipeline.append(self.detect_language)
        
        if config.CHECK_FIX_BAD_UNICODE:
            self.pipeline.append(self.fix_bad_unicode)
        if config.CHECK_TO_ASCII_UNICODE:
            self.pipeline.append(self.to_ascii_unicode)
        
        if config.CHECK_REPLACE_HTML:
            self.pipeline.append(self.replace_html)
        if config.CHECK_REPLACE_URLS:
            self.pipeline.append(self.replace_urls)
        if config.CHECK_REPLACE_EMAILS:
            self.pipeline.append(self.replace_emails)
        if config.CHECK_REPLACE_YEARS:
            self.pipeline.append(self.replace_years)
        if config.CHECK_REPLACE_PHONE_NUMBERS:
            self.pipeline.append(self.replace_phone_numbers)
        if config.CHECK_REPLACE_NUMBERS:
            self.pipeline.append(self.replace_numbers)
        if config.CHECK_REPLACE_CURRENCY_SYMBOLS:
            self.pipeline.append(self.replace_currency_symbols)
        
        if config.CHECK_NER_PROCESS:
            self.pipeline.append(self.ner_process)
        
        if config.CHECK_REMOVE_ISOLATED_LETTERS:
            self.pipeline.append(self.remove_isolated_letters)
        if config.CHECK_REMOVE_ISOLATED_SPECIAL_SYMBOLS:
            self.pipeline.append(self.remove_isolated_special_symbols)
        if config.CHECK_NORMALIZE_WHITESPACE:
            self.pipeline.append(self.normalize_whitespace)
    
    def process(self, text):
        text = str(text)
        
        for step in self.pipeline:
            text = step(text)
            
        if config.CHECK_STATISTICAL_MODEL_PROCESSING:
            stext = self.statistical_model_processing(text)
            return text, stext, self.language
        elif config.CHECK_DETECT_LANGUAGE:
            return text, self.language
        else:
            return text

    def detect_language(self, text):
        self.language = str(resources.DETECTOR.detect_language_of(text)).split(".")[-1]
        return text

    def fix_bad_unicode(self, text):
        return self.NormaliseText.fix_bad_unicode(text)

    def to_ascii_unicode(self, text):
        return self.NormaliseText.to_ascii_unicode(text)

    def replace_html(self, text):
        return self.ProcessContacts.replace_html(text, replace_with=config.REPLACE_WITH_HTML)

    def replace_urls(self, text):
        return self.ProcessContacts.replace_urls(text, replace_with=config.REPLACE_WITH_URL)

    def replace_emails(self, text):
        return self.ProcessContacts.replace_emails(text, replace_with=config.REPLACE_WITH_EMAIL)

    def replace_years(self, text):
        return self.ProcessDateTime.replace_years(text, replace_with=config.REPLACE_WITH_YEARS)

    def replace_phone_numbers(self, text):
        return self.ProcessContacts.replace_phone_numbers(text, replace_with=config.REPLACE_WITH_PHONE_NUMBERS)

    def replace_numbers(self, text):
        return self.ProcessContacts.replace_numbers(text, replace_with=config.REPLACE_WITH_NUMBERS)

    def replace_currency_symbols(self, text):
        return self.ProcessSpecialSymbols.replace_currency_symbols(text, replace_with=config.REPLACE_WITH_CURRENCY_SYMBOLS)

    def ner_process(self, text):
        return self.GeneralNER.ner_process(text, config.POSITIONAL_TAGS, config.NER_CONFIDENCE_THRESHOLD, self.language)

    def remove_isolated_letters(self, text):
        return self.ProcessSpecialSymbols.remove_isolated_letters(text)

    def remove_isolated_special_symbols(self, text):
        return self.ProcessSpecialSymbols.remove_isolated_special_symbols(text)

    def normalize_whitespace(self, text):
        return self.NormaliseText.normalize_whitespace(text, no_line_breaks=True)

    def statistical_model_processing(self, text):
        if config.CHECK_CASEFOLD:
            stext = text.casefold()  # lowercase
        if config.CHECK_REMOVE_STOPWORDS:
            stext = self.ProcessStopwords.remove_stopwords(stext, self.language)
        if config.CHECK_REMOVE_PUNCTUATION:
            stext = self.ProcessSpecialSymbols.remove_punctuation(stext)
        if config.CHECK_REMOVE_ISOLATED_LETTERS:
            stext = self.ProcessSpecialSymbols.remove_isolated_letters(stext)
        if config.CHECK_NORMALIZE_WHITESPACE:
            stext = self.NormaliseText.normalize_whitespace(stext)
        return stext