"""
    detect_language : to detect the language automatically, but would consume more time if done on a batch
    fix_bad_unicode : if True, fix "broken" unicode such as mojibake and garbled HTML entities
    to_ascii_unicode : if True, convert non-to_ascii characters into their closest to_ascii equivalents
    replace_with_url : special URL token, default "",
    replace_with_email : special EMAIL token, default "",
    replace_years : replace year, default "",
    replace_with_phone_number : special PHONE token, default "",
    replace_with_number : special NUMBER token, default "",
    no_currency_symbols : if True, replace all currency symbols with the respective alphabetical ones,
    ner_process : To execute NER Process to remove the positpositional tags, PER, LOC, ORG, MISC
    remove_isolated_letters : remove any isolated letters which doesn't add any value to the text
    remove_isolated_symbols : remove any isolated symbols which shouldn't be present in the text, usually which isn't 
                            immediatly prefixed and suffixed by letter or number
    normalize_whitespace : remove any unnecessary whitespace
    statistical_model_processing : to get the statistical model text, like for fastText, SVM, LR etc
    casefold : to lower the text
    remove_stopwords : remove stopwords based on the language, usues NLTK stopwords
    remove_punctuation : removes all the special symbols
"""

CHECK_DETECT_LANGUAGE = True
CHECK_FIX_BAD_UNICODE = True
CHECK_TO_ASCII_UNICODE = True
CHECK_REPLACE_HTML = True
CHECK_REPLACE_URLS = True
CHECK_REPLACE_EMAILS = True
CHECK_REPLACE_YEARS = True
CHECK_REPLACE_PHONE_NUMBERS = True
CHECK_REPLACE_NUMBERS = True
CHECK_REPLACE_CURRENCY_SYMBOLS = True
CHECK_NER_PROCESS = True
CHECK_REMOVE_ISOLATED_LETTERS = True
CHECK_REMOVE_ISOLATED_SPECIAL_SYMBOLS = True
CHECK_NORMALIZE_WHITESPACE = True
CHECK_STATISTICAL_MODEL_PROCESSING = True
CHECK_CASEFOLD = True
CHECK_REMOVE_STOPWORDS = True
CHECK_REMOVE_PUNCTUATION = True
CHECK_REMOVE_STEXT_CUSTOM_STOP_WORDS = True
REPLACE_WITH_URL = ""
REPLACE_WITH_HTML = ""
REPLACE_WITH_EMAIL = ""
REPLACE_WITH_YEARS = ""
REPLACE_WITH_PHONE_NUMBERS = ""
REPLACE_WITH_NUMBERS = ""
REPLACE_WITH_CURRENCY_SYMBOLS = None
POSITIONAL_TAGS = ['PER', 'LOC', 'ORG']
NER_CONFIDENCE_THRESHOLD = 0.85
LANGUAGE = None