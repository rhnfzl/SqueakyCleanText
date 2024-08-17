"""
Module containing the configuration parameters for the SCT package.
"""

# Flag to detect the language automatically. If True, the language will be detected for each text.
CHECK_DETECT_LANGUAGE = True

# Flag to fix "broken" unicode such as mojibake and garbled HTML entities.
CHECK_FIX_BAD_UNICODE = True

# Flag to convert non-ASCII characters into their closest ASCII equivalents.
CHECK_TO_ASCII_UNICODE = True

# Flag to replace HTML tags with a special token.
CHECK_REPLACE_HTML = True

# Flag to replace URLs with a special token.
CHECK_REPLACE_URLS = True

# Flag to replace email addresses with a special token.
CHECK_REPLACE_EMAILS = True

# Flag to replace years with a special token.
CHECK_REPLACE_YEARS = True

# Flag to replace phone numbers with a special token.
CHECK_REPLACE_PHONE_NUMBERS = True

# Flag to replace numbers with a special token.
CHECK_REPLACE_NUMBERS = True

# Flag to replace currency symbols with their respective alphabetical equivalents.
CHECK_REPLACE_CURRENCY_SYMBOLS = True

# Flag to execute Named Entity Recognition (NER) to remove positional tags such as PER, LOC, ORG, MISC.
CHECK_NER_PROCESS = True

# Flag to remove any isolated letters which do not add any value to the text.
CHECK_REMOVE_ISOLATED_LETTERS = True

# Flag to remove any isolated symbols which should not be present in the text.
CHECK_REMOVE_ISOLATED_SPECIAL_SYMBOLS = True

# Flag to remove any unnecessary whitespace.
CHECK_NORMALIZE_WHITESPACE = True

# Flag to get the statistical model text, such as for fastText, SVM, LR.
CHECK_STATISTICAL_MODEL_PROCESSING = True

# Flag to convert all characters to lowercase.
CHECK_CASEFOLD = True

# Flag to remove stopwords based on the language. Uses NLTK stopwords.
CHECK_REMOVE_STOPWORDS = True

# Flag to remove all special symbols.
CHECK_REMOVE_PUNCTUATION = True

# Flag to remove custom stopwords specific to the SCT package.
CHECK_REMOVE_STEXT_CUSTOM_STOP_WORDS = True

# Special token to replace URLs.
REPLACE_WITH_URL = "<URL>"

# Special token to replace HTML tags.
REPLACE_WITH_HTML = "<HTML>"

# Special token to replace email addresses.
REPLACE_WITH_EMAIL = "<EMAIL>"

# Special token to replace years.
REPLACE_WITH_YEARS = "<YEAR>"

# Special token to replace phone numbers.
REPLACE_WITH_PHONE_NUMBERS = "<PHONE>"

# Special token to replace numbers.
REPLACE_WITH_NUMBERS = "<NUMBER>"

# Special token to replace currency symbols. If None, symbols will be replaced with their 3-letter abbreviations.
REPLACE_WITH_CURRENCY_SYMBOLS = None

# List of positional tags to be removed by NER.
POSITIONAL_TAGS = ['PER', 'LOC', 'ORG']

# Confidence threshold for NER.
NER_CONFIDENCE_THRESHOLD = 0.85

# Language to be used for NER. If None, the language will be detected automatically.
LANGUAGE = None

# List of pre-trained NER models in order of importance.
NER_MODELS_LIST = [
    "FacebookAI/xlm-roberta-large-finetuned-conll03-english",  # English Model
    "FacebookAI/xlm-roberta-large-finetuned-conll02-dutch",  # Dutch Model
    "FacebookAI/xlm-roberta-large-finetuned-conll03-german",  # German Model
    "FacebookAI/xlm-roberta-large-finetuned-conll02-spanish",  # Spanish Model
    "Babelscape/wikineural-multilingual-ner"  # Multilingual Model
]

