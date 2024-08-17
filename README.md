# `SqueakyCleanText` 

[![PyPI](https://img.shields.io/pypi/v/squeakycleantext.svg)](https://pypi.org/project/squeakycleantext/) [![PyPI - Downloads](https://img.shields.io/pypi/dm/squeakycleantext)](https://pypistats.org/packages/squeakycleantext)

In the world of machine learning and natural language processing, clean and well-structured text data is crucial for building effective downstream models and managing token limits in language models. 

SqueakyCleanText simplifies the process by automatically addressing common text issues, ensuring your data is clean and well-structured with minimal effort on your part.

### Key Features
- Encoding Issues: Corrects text encoding problems.
- HTML and URLs: Removes unnecessary long HTML tags and URLs, or replaces them with special tokens.
- Contact Information: Strips emails, phone numbers, and other contact details, or replaces them with special tokens.
- Isolated Characters: Eliminates isolated letters or symbols that add no value.
- NER Support: Uses a soft voting ensemble technique to handle named entities like location, person, and organization names, which can be replaced with special tokens if not needed in the text.
- Stopwords and Punctuation: For statistical models, it optimizes text by removing stopwords, special symbols, and punctuation.
- Currency Symbols: Replaces all currency symbols with their alphabetical equivalents.
- Whitespace Normalization: Removes unnecessary whitespace.
- Detects the language of the processed text, useful for downstream tasks.
- Supports English, Dutch, German, and Spanish languages.
- Provides text formatted for both Language Model processing and Statistical Model processing.

##### Benefits for Statistical Models
When working with statistical models, further optimization is often required, such as removing stopwords, special symbols, and punctuation. 
SqueakyCleanText streamlines this process, ensuring your text data is in optimal shape for classification and other downstream tasks.

##### Advantage for Ensemble NER Process
Relying on a single model for Named Entity Recognition (NER) may not be ideal, as there is a significant chance that it might miss some entities. Combining language-specific NER models increases specificity and reduces the risk of missing entities. 
The NER model in this package includes a chunking mechanism, enabling effective NER processing even when the text exceeds the model's token size limit.

By automating these text cleaning steps, SqueakyCleanText ensures your data is prepared efficiently and effectively, saving time and improving model performance.

## Installation

To install SqueakyCleanText, use the following pip command:

```sh
pip install SqueakyCleanText
```

## Usage

Here are a few examples of how to use the SqueakyCleanText package:

Examples:
```python
english_text = "Hey John Doe, wanna grab some coffee at Starbucks on 5th Avenue? I'm feeling a bit tired after last night's party at Jane's place. BTW, I can't make it to the meeting at 10:00 AM. LOL! Call me at +1-555-123-4567 or email me at john.doe@example.com. Check out this cool website: https://www.example.com."

dutch_text = "Hé Jan Jansen, wil je wat koffie halen bij Starbucks op de 5e Avenue? Ik voel me een beetje moe na het feest van gisteravond bij Annes huis. Btw, ik kan niet naar de vergadering om 10:00 uur. LOL! Bel me op +31-6-1234-5678 of mail me op jan.jansen@voorbeeld.com. Kijk eens naar deze coole website: https://www.voorbeeld.com."
```

- Using default configuration settings:

```python
# The first time you import the package, it may take some time because it will downloading the NER models. Please be patient.
from sct import sct

# Initialize the TextCleaner
sx = sct.TextCleaner()

# Process the text
# lmtext : Text for Language Models;
# cmtext : Text for Classical/Statistical ML;
# language : Processed text language

#### --- English Text
lmtext, cmtext, language = sx.process(english_text)
print(f"Language Model Text : {lmtext}")
print(f"Statistical Model Text : {cmtext}")
print(f"Language of the Text : {language}")

# Output the result
# Language Model Text : Hey <PERSON> wanna grab some coffee at Starbucks on <LOCATION> I'm feeling a bit tired after last night's party at <PERSON>'s place. BTW, can't make it to the meeting at <NUMBER><NUMBER> AM. LOL! Call me at <PHONE> or email me at <EMAIL> Check out this cool website: <URL>
# Statistical Model Text : hey person wanna grab coffee starbucks location im feeling bit tired last nights party persons place btw cant make meeting numbernumber am lol call phone email email check cool website url
# Language of the Text : ENGLISH

#### --- Dutch Text
lmtext, cmtext, language = sx.process(dutch_text)
print(f"Language Model Text : {lmtext}")
print(f"Statistical Model Text : {cmtext}")
print(f"Language of the Text : {language}")

# Output the result
# Language Model Text : He <PERSON> wil je wat koffie halen bij <ORGANISATION> op de <LOCATION> Ik voel me een beetje moe na het feest van gisteravond bij Annes huis. Btw, ik kan niet naar de vergadering om <NUMBER><NUMBER> uur. LOL! Bel me op <NUMBER><NUMBER><PHONE> of mail me op <EMAIL> Kijk eens naar deze coole website: <URL>
# Statistical Model Text : he person koffie halen organisation location voel beetje moe feest gisteravond annes huis btw vergadering numbernumber uur lol bel numbernumberphone mail email kijk coole website url
# Language of the Text : DUTCH
```

- Using the package with custom configuration:
You can modify the package’s functionality by changing settings in the configuration file before initializing TextCleaner().

    - Deactivating NER altogether:

    ```python

    from sct import sct, config

    config.CHECK_NER_PROCESS = False
    sx = sct.TextCleaner()

    lmtext, cmtext, language = sx.process(english_text)
    print(f"Language Model Text : {lmtext}")
    print(f"Statistical Model Text : {cmtext}")
    print(f"Language of the Text : {language}")

    # Output the result
    # Language Model Text : Hey John Doe, wanna grab some coffee at Starbucks on 5th Avenue? I'm feeling a bit tired after last night's party at Jane's place. BTW, can't make it to the meeting at <NUMBER><NUMBER> AM. LOL! Call me at <PHONE> or email me at <EMAIL> Check out this cool website: <URL>
    # Statistical Model Text : hey john doe wanna grab coffee starbucks 5th avenue im feeling bit tired last nights party janes place btw cant make meeting numbernumber am lol call phone email email check cool website url
    # Language of the Text : ENGLISH
    ```

    - Incase Statistical model text is not needed:

    ```python

    from sct import sct, config

    config.CHECK_STATISTICAL_MODEL_PROCESSING = False
    sx = sct.TextCleaner()

    lmtext, language = sx.process(english_text)
    print(f"Language Model Text : {lmtext}")
    print(f"Language of the Text : {language}")

    # Output the result

    # Output the result
    # Language Model Text : Hey John Doe, wanna grab some coffee at Starbucks on 5th Avenue? I'm feeling a bit tired after last night's party at Jane's place. BTW, can't make it to the meeting at <NUMBER><NUMBER> AM. LOL! Call me at <PHONE> or email me at <EMAIL> Check out this cool website: <URL>
    # Language of the Text : ENGLISH
    ```
### Full List of Configurable Settings:

    Similarly, other aspects of the configuration can be changed. Simply modify the settings before initializing TextCleaner(). Below is the full list of configurable settings:

    ```python

    from sct import sct, config
    # In case Language detection is not required as well as No NER and No Statistical Model stopwords are needed,
    # then only CHECK_DETECT_LANGUAGE will be considered False.
    config.CHECK_DETECT_LANGUAGE = True
    config.CHECK_FIX_BAD_UNICODE = True
    config.CHECK_TO_ASCII_UNICODE = True
    config.CHECK_REPLACE_HTML = True
    config.CHECK_REPLACE_URLS = True
    config.CHECK_REPLACE_EMAILS = True
    config.CHECK_REPLACE_YEARS = True
    config.CHECK_REPLACE_PHONE_NUMBERS = True
    config.CHECK_REPLACE_NUMBERS = True
    config.CHECK_REPLACE_CURRENCY_SYMBOLS = True
    config.CHECK_NER_PROCESS = True
    config.CHECK_REMOVE_ISOLATED_LETTERS = True
    config.CHECK_REMOVE_ISOLATED_SPECIAL_SYMBOLS = True
    config.CHECK_NORMALIZE_WHITESPACE = True
    config.CHECK_STATISTICAL_MODEL_PROCESSING = True
    config.CHECK_CASEFOLD = True
    config.CHECK_REMOVE_STOPWORDS = True
    config.CHECK_REMOVE_PUNCTUATION = True
    config.CHECK_REMOVE_SCT_CUSTOM_STOP_WORDS = True
    # Tags can be replaced if needed, like if no special tags are necessary "" can be passed
    config.REPLACE_WITH_URL = "<URL>"
    config.REPLACE_WITH_HTML = "<HTML>"
    config.REPLACE_WITH_EMAIL = "<EMAIL>"
    config.REPLACE_WITH_YEARS = "<YEAR>"
    config.REPLACE_WITH_PHONE_NUMBERS = "<PHONE>"
    config.REPLACE_WITH_NUMBERS = "<NUMBER>"
    config.REPLACE_WITH_CURRENCY_SYMBOLS = None
    # You can remove any of the tags
    config.POSITIONAL_TAGS = ['PER', 'LOC', 'ORG']
    config.NER_CONFIDENCE_THRESHOLD = 0.85
    # Pass it as ENGLISH, DUTCH, GERMAN etc. if you know the language of text beforehand.
    config.LANGUAGE = None

    # Order of the model is Important: English Model, Dutch Model, German Model, Spanish Model, MULTILINGUAL Model
    # All models passed need to support transformers AutoModel
    config.NER_MODELS_LIST = [
        "FacebookAI/xlm-roberta-large-finetuned-conll03-english",
        "FacebookAI/xlm-roberta-large-finetuned-conll02-dutch",
        "FacebookAI/xlm-roberta-large-finetuned-conll03-german",
        "FacebookAI/xlm-roberta-large-finetuned-conll02-spanish",
        "Babelscape/wikineural-multilingual-ner"
    ]
    
    sx = sct.TextCleaner()

    ```


## API

### `sct.TextCleaner`

#### `process(text: str) -> Tuple[str, str, str]`

Processes the input text and returns a tuple containing:
    - Cleaned text formatted for language models.
    - Cleaned text formatted for statistical models (stopwords removed).
    - Detected language of the text.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an issue.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

The package took inspirations from the following repo:

- [clean-text](https://github.com/jfilter/clean-text)