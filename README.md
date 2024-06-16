
# SqueakyCleanText

SqueakyCleanText is a handy text cleaning package designed to sanitize text for classical machine learning models and language models (such as BERT, RoBERTa) without altering the meaning of the text.

## Features

- Text sanitization for classical ML models and language models.
- Removes unnecessary characters and normalizes text.
- Supports Named Entity Recognition (NER).
- Identifies the language of the text.
- Provides cleaned text with stopwords removed.

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
result = sx.process("Hello, My name is John!")

# Output the result
print(result)
# Output: ('Hello, My name is', 'hello name', 'ENGLISH')
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
