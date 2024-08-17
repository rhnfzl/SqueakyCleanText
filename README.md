
# `SqueakyCleanText` [![Build Status](https://img.shields.io/github/workflow/status/rhnfzl/SqueakyCleanText/Test)](https://github.com/rhnfzl/SqueakyCleanText/actions/workflows/test.yml) [![PyPI](https://img.shields.io/pypi/v/squeakycleantext.svg)](https://pypi.org/project/squeakycleantext/) [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/squeakycleantext.svg)](https://pypi.org/project/squeakycleantext/) [![PyPI - Downloads](https://img.shields.io/pypi/dm/squeakycleantext)](https://pypistats.org/packages/squeakycleantext)

In the world of machine learning and natural language processing, clean and well-structured text data is crucial for building effective downstream models and managing token limits in language models. 

SqueakyCleanText helps achieve this by addressing common text issues by doing most of the work for you.

### Key Features
- Encoding Issues: Corrects text encoding problems.
- HTML and URLs: Removes unnecessary long HTML tags and URLs, or replace them with special tokens.
- Contact Information: Strips emails, phone numbers, and other contact details, or replace them with special tokens.
- Isolated Characters: Eliminates isolated letters or symbols that adds no value.
- NER Support: Uses a soft voting ensemble technique to handle named entities like location, person and organisation names, which can be replaced with special tokens if not needed in the text.
- Stopwords and Punctuation: For statistical models, it optimizes text by removing stopwords, special symbols, and punctuation.
- Currency Symbols: Replaces all currency symbols with their alphabetical equivalents.
- Whitespace Normalization: Removes unnecessary whitespace.
- Detects the language of processed text if needed in downstream task.
- Provides text for both Lnaguage model processing and Statistical model processing.

##### Benefits for Statistical Models
When working with statistical models, further optimization is often required, such as removing stopwords, special symbols, and punctuation. 
SqueakyCleanText offers functionality to streamline this process, ensuring that your text data is in optimal shape for classification and other downstream tasks.


##### Advantage for ensemble NER process
Depending on sigle model for Name Entity recognition is not be ideal, as there is a high chance it might skip the entity all together. Also combining the language specific NER model makes it more specific for text and reduces the chance of missing out the entity.
The package NER model has the chunking mechanism which helps to do the NER process even if the text is longer than the model token size.

Important : Model 

By automating these text cleaning steps, SqueakyCleanText ensures your data is prepared efficiently and effectively, saving time and improving model performance.

## Installation

To install SqueakyCleanText, use the following pip command:

```sh
pip install SqueakyCleanText
```

## Usage

Few examples, how to use the SqueakyCleanText package:

- Uisng in it's default config settings:
```python
# first time import will take bit of time, so please have patience
from sct import sct

# Initialize the TextCleaner
sx = sct.TextCleaner()

# Process the text
#lmtext : Text for Language Models; cmtext : Text for Classical/Statistical ML, language : Processed text language

lmtext, cmtext, language = sx.process("Hello, My name is John!")
# Output the result
print(lmtext, cmtext, language)
# Hello, My name is hello name ENGLISH
```

## API

### `sct.TextCleaner`

#### `process(text: str) -> Tuple[str, str, str]`

Processes the input text and returns a tuple containing:
- Cleaned text with punctuation and unnecessary characters removed.
- Cleaned text with stopwords removed.
- Detected language of the text.

## TODO

- Add the ability to change the NER models from the config file, supporting AutoModel and AutoTokenizer.
- Expand language support for stopwords.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an issue.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

The package took inspirations from the following repo:

- [clean-text](https://github.com/jfilter/clean-text)