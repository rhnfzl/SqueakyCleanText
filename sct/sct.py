"""
This code provides a comprehensive text cleaning and preprocessing pipeline. 
It includes functions to normalize, remove personal information and clean text data, 
which is crucial for natural language processing tasks.
"""
from sct import config
from sct.utils import contact, datetime, ner, normtext, resources, special, stopwords

class TextCleaner:
    
    def __init__(self):
        """
        Initialize the TextCleaner object and setup the pipeline.
        """
        
        # Initialize all the pipeline steps
        self.ProcessContacts = contact.ProcessContacts()  # Handles contact processing
        self.ProcessDateTime = datetime.ProcessDateTime()  # Handles datetime processing
        self.ProcessSpecialSymbols = special.ProcessSpecialSymbols()  # Handles special symbols processing
        self.NormaliseText = normtext.NormaliseText()  # Handles text normalization
        self.ProcessStopwords = stopwords.ProcessStopwords()  # Handles stopwords processing
        self.GeneralNER = ner.GeneralNER()  # Handles named entity recognition
        
        # Initialize the pipeline and language
        self.pipeline = []  # Stores the pipeline steps
        self.language = None  # Stores the language of the text
        
        # Initialize the pipeline
        self.init_pipeline()
    
    def init_pipeline(self):
        """
        Initialize pipeline steps based on config
        """
        
        # If the language is specified in the config, set it
        language_config = config.LANGUAGE.lower() if config.LANGUAGE else None
        if language_config and language_config in resources.LANGUAGE_NAME:
            self.language = language_config.upper()
        
        # Add the detect language step if any of the following are True
        if any([config.CHECK_DETECT_LANGUAGE, config.CHECK_NER_PROCESS, config.CHECK_REMOVE_STOPWORDS]):
            self.pipeline.append(self.detect_language)
        
        # Add the Unicode fix steps if any of the following are True
        if config.CHECK_FIX_BAD_UNICODE:
            self.pipeline.append(self.fix_bad_unicode)
        if config.CHECK_TO_ASCII_UNICODE:
            self.pipeline.append(self.to_ascii_unicode)
        
        # Add the replace steps if any of the following are True
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
        
        # Add the NER step if any of the following are True
        if config.CHECK_NER_PROCESS:
            self.pipeline.append(self.ner_process)
        
        # Add the remove isolated steps if any of the following are True
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
        """
        Detects the language of the input text and sets the language attribute.

        Args:
            text (str): The input text to detect the language of.

        Returns:
            str: The input text.
        """
        # Detect the language of the text
        detected_language = str(resources.DETECTOR.detect_language_of(text)).split(".")[-1]

        # Set the language attribute to the detected language
        self.language = detected_language

        # Return the input text
        return text

    def fix_bad_unicode(self, text):
        """
        Fixes broken unicode text using the `ftfy` library.

        This function fixes various issues with unicode text, including mojibake, HTML entities, and other code cruft.
        It also normalizes the text to a specified format, which can improve the readability and consistency of the text.

        Args:
            text (str): The raw text to be fixed.

        Returns:
            str: The fixed and normalized text.
        """
        # Call the fix_bad_unicode method of NormaliseText class to fix the text
        return self.NormaliseText.fix_bad_unicode(text)

    def to_ascii_unicode(self, text):
        """
        Transliterate unicode text into ascii characters.

        This function uses the `to_ascii_unicode` method of the `NormaliseText` class to transliterate the text.

        Args:
            text (str): The input text to transliterate.

        Returns:
            str: The transliterated text.
        """
        # Call the to_ascii_unicode method of the NormaliseText class to transliterate the text
        return self.NormaliseText.to_ascii_unicode(text)

    def replace_html(self, text):
        """
        Replace all HTML tags in the given text with the specified replacement string.

        Args:
            text (str): The input text to be processed.

        Returns:
            str: The processed text with all HTML tags replaced.
        """
        # Call the replace_html method of the ProcessContacts class to replace HTML tags in the text
        return self.ProcessContacts.replace_html(text, replace_with=config.REPLACE_WITH_HTML)

    def replace_urls(self, text):
        """
        Replace all URLs in the given text with the specified replacement string.

        This function uses the `replace_urls` method of the `ProcessContacts` class to replace URLs in the text.

        Args:
            text (str): The input text to be processed.

        Returns:
            str: The processed text with all URLs replaced.
        """
        # Call the replace_urls method of the ProcessContacts class to replace URLs in the text
        return self.ProcessContacts.replace_urls(text, replace_with=config.REPLACE_WITH_URL)

    def replace_emails(self, text):
        """
        Replace all email addresses in the given text with the specified replacement string.

        This function uses the `replace_emails` method of the `ProcessContacts` class to replace email addresses in the text.

        Args:
            text (str): The input text to be processed.

        Returns:
            str: The processed text with all email addresses replaced.
        """
        # Call the replace_emails method of the ProcessContacts class to replace email addresses in the text
        return self.ProcessContacts.replace_emails(text, replace_with=config.REPLACE_WITH_EMAIL)

    def replace_years(self, text):
        """
        Replace all years in the given text with the specified replacement string.

        This function uses the `replace_years` method of the `ProcessDateTime` class to replace years in the text.

        Args:
            text (str): The input text to be processed.

        Returns:
            str: The processed text with all years replaced.
        """
        # Call the replace_years method of the ProcessDateTime class to replace years in the text
        return self.ProcessDateTime.replace_years(text, replace_with=config.REPLACE_WITH_YEARS)

    def replace_phone_numbers(self, text):
        """
        Replace all phone numbers in the given text with the specified replacement string.

        This function uses the `replace_phone_numbers` method of the `ProcessContacts` class to replace phone numbers in the text.

        Args:
            text (str): The input text to be processed.

        Returns:
            str: The processed text with all phone numbers replaced.
        """
        # Call the replace_phone_numbers method of the ProcessContacts class to replace phone numbers in the text
        return self.ProcessContacts.replace_phone_numbers(text, replace_with=config.REPLACE_WITH_PHONE_NUMBERS)

    def replace_numbers(self, text):
        """
        Replace all numbers in the given text with the specified replacement string.

        This function uses the `replace_numbers` method of the `ProcessContacts` class to replace numbers in the text.

        Args:
            text (str): The input text to be processed.

        Returns:
            str: The processed text with all numbers replaced.
        """
        # Call the replace_numbers method of the ProcessContacts class to replace numbers in the text
        return self.ProcessContacts.replace_numbers(text, replace_with=config.REPLACE_WITH_NUMBERS)

    def replace_currency_symbols(self, text):
        """
        Replace all currency symbols in the given text with the specified replacement string.

        This function uses the `replace_currency_symbols` method of the `ProcessSpecialSymbols` class to replace currency symbols in the text.

        Args:
            text (str): The input text to be processed.

        Returns:
            str: The processed text with all currency symbols replaced.
        """
        # Call the replace_currency_symbols method of the ProcessSpecialSymbols class to replace currency symbols in the text
        return self.ProcessSpecialSymbols.replace_currency_symbols(text, replace_with=config.REPLACE_WITH_CURRENCY_SYMBOLS)

    def ner_process(self, text):
        """
        Execute named entity recognition (NER) to remove the positional tags from the text.

        Args:
            text (str): The input text to be processed.
            config.POSITIONAL_TAGS (list): The positional tags to be removed from the text.
            config.NER_CONFIDENCE_THRESHOLD (float): The confidence threshold for the NER model.
            self.language (str): The language of the text.

        Returns:
            str: The processed text with the positional tags removed.
        """
        return self.GeneralNER.ner_process(text, config.POSITIONAL_TAGS, config.NER_CONFIDENCE_THRESHOLD, self.language)

    def remove_isolated_letters(self, text):
        """
        Removes any isolated letters which doesn't add any value to the text.

        This method uses the `remove_isolated_letters` method of the `ProcessSpecialSymbols` class to remove isolated letters from the input text.

        Args:
            text (str): The input text to be processed.

        Returns:
            str: The processed text with isolated letters removed.
        """
        # Call the remove_isolated_letters method of the ProcessSpecialSymbols class to remove isolated letters from the text
        return self.ProcessSpecialSymbols.remove_isolated_letters(text)

    def remove_isolated_special_symbols(self, text):
        """
        Removes any isolated special symbols which doesn't add any value to the text.

        This method uses the `remove_isolated_special_symbols` method of the `ProcessSpecialSymbols` class to remove isolated special symbols from the input text.

        Args:
            text (str): The input text to be processed.

        Returns:
            str: The processed text with isolated special symbols removed.
        """
        # Call the remove_isolated_special_symbols method of the ProcessSpecialSymbols class to remove isolated special symbols from the text
        return self.ProcessSpecialSymbols.remove_isolated_special_symbols(text)

    def normalize_whitespace(self, text):
        """
        Normalize whitespace in the given text.

        This function replaces one or more spacings with a single space, and one
        or more line breaks with a single newline. Also, it strips leading/trailing whitespace.

        Args:
            text (str): The input text to be processed.

        Returns:
            str: The processed text with normalized whitespace.
        """
        # Call the normalize_whitespace method of the NormaliseText class to normalize whitespace in the text
        # Set no_line_breaks flag to True to remove all line breaks
        return self.NormaliseText.normalize_whitespace(text, no_line_breaks=True)

    def statistical_model_processing(self, text):
        """
        Process the given text for classical/statistical machine learning models.

        This function applies the following processing steps to the input text:
        1. Case folding (lowercase)
        2. Stopword removal
        3. Punctuation removal
        4. Removing isolated letters
        5. Normalizing whitespace

        Args:
            text (str): The input text to be processed.

        Returns:
            str: The processed text.
        """
        stext = text
        if config.CHECK_CASEFOLD:
            # Convert the text to lowercase
            stext = text.casefold()
        if config.CHECK_REMOVE_STOPWORDS:
            # Remove stopwords from the text
            stext = self.ProcessStopwords.remove_stopwords(stext, self.language)
        if config.CHECK_REMOVE_PUNCTUATION:
            # Remove punctuation from the text
            stext = self.ProcessSpecialSymbols.remove_punctuation(stext)
        if config.CHECK_REMOVE_ISOLATED_LETTERS:
            # Remove isolated letters from the text
            stext = self.ProcessSpecialSymbols.remove_isolated_letters(stext)
        if config.CHECK_NORMALIZE_WHITESPACE:
            # Normalize whitespace in the text
            stext = self.NormaliseText.normalize_whitespace(stext)
        return stext