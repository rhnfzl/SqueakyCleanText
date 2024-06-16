
# SqueakyCleanText

In the world of machine learning and natural language processing, clean and well-structured text data is crucial for building effective downstream models and managing token limits in language models. 

SqueakyCleanText helps achieve this by addressing common text issues.

### Key Assumptions and Features
- Encoding Issues: Corrects text encoding problems.
- HTML and URLs: Removes unnecessary long HTML tags and URLs, or replace them with special tokens.
- Contact Information: Strips emails, phone numbers, and other contact details, or replace them with special tokens.
- Isolated Characters: Eliminates isolated letters or symbols that adds no value.
- NER Support: Uses a soft voting ensemble technique to handle named entities like location and person names, which can be omitted if not needed.
- Stopwords and Punctuation: For statistical models, it optimizes text by removing stopwords, special symbols, and punctuation.
- Currency Symbols: Replaces all currency symbols with their alphabetical equivalents.
- Whitespace Normalization: Removes unnecessary whitespace.

##### Benefits for Statistical Models
When working with statistical models, further optimization is often required, such as removing stopwords, special symbols, and punctuation. 
SqueakyCleanText offers functionality to streamline this process, ensuring that your text data is in optimal shape for classification and other downstream tasks.

By automating these text cleaning steps, SqueakyCleanText ensures your data is prepared efficiently and effectively, saving time and improving model performance.

## Installation

To install SqueakyCleanText, use the following pip command:

```sh
pip install SqueakyCleanText
```

## Usage

Here's a simple example to demonstrate how to use the SqueakyCleanText package:

```python
from sct import sct

# Initialize the TextCleaner
sx = sct.TextCleaner()

# Process the text
#lmtext : Text for Language Models, cmtext : Text for Classical ML, language : Language provided
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

- Thanks to the contributors and the community for their support.
