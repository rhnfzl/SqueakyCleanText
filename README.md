# `SqueakyCleanText` 

[![PyPI](https://img.shields.io/pypi/v/squeakycleantext.svg)](https://pypi.org/project/squeakycleantext/) [![PyPI - Downloads](https://img.shields.io/pypi/dm/squeakycleantext)](https://pypistats.org/packages/squeakycleantext)

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
- Supports English, Dutch, German and Spanish language.
- Provides text for both Lnaguage model processing and Statistical model processing.

##### Benefits for Statistical Models
When working with statistical models, further optimization is often required, such as removing stopwords, special symbols, and punctuation. 
SqueakyCleanText offers functionality to streamline this process, ensuring that your text data is in optimal shape for classification and other downstream tasks.


##### Advantage for ensemble NER process
Depending on sigle model for Name Entity recognition is not be ideal, as there is a high chance it might skip the entity all together. Also combining the language specific NER model makes it more specific for text and reduces the chance of missing out the entity.
The package NER model has the chunking mechanism which helps to do the NER process even if the text is longer than the model token size.

By automating these text cleaning steps, SqueakyCleanText ensures your data is prepared efficiently and effectively, saving time and improving model performance.

## Installation

To install SqueakyCleanText, use the following pip command:

```sh
pip install SqueakyCleanText
```

## Usage

Few examples, how to use the SqueakyCleanText package:

Examples:
```python
english_text = "Hey John Doe, wanna grab some coffee at Starbucks on 5th Avenue? I'm feeling a bit tired after last night's party at Jane's place. BTW, I can't make it to the meeting at 10:00 AM. LOL! Call me at +1-555-123-4567 or email me at john.doe@example.com. Check out this cool website: https://www.example.com."

dutch_text = "Hé Jan Jansen, wil je wat koffie halen bij Starbucks op de 5e Avenue? Ik voel me een beetje moe na het feest van gisteravond bij Annes huis. Btw, ik kan niet naar de vergadering om 10:00 uur. LOL! Bel me op +31-6-1234-5678 of mail me op jan.jansen@voorbeeld.com. Kijk eens naar deze coole website: https://www.voorbeeld.com."
```

- Uisng in it's default config settings:

```python
# first time import will take bit of time, so please have patience
from sct import sct

# Initialize the TextCleaner
sx = sct.TextCleaner()

# Process the text
#lmtext : Text for Language Models;
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

- Uisng the package any of the functionality, lets take NER as an example

```python

from sct import sct, config

config.CHECK_NER_PROCESS = False
sx = sct.TextCleaner()

lmtext, cmtext, language = sx.process(english_text)
print(f"Language Model Text : {lmtext}")
print(f"Statistical Model Text : {cmtext}")
print(f"Language of the Text : {language}")

# Output the result
Language Model Text : Hey John Doe, wanna grab some coffee at Starbucks on 5th Avenue? I'm feeling a bit tired after last night's party at Jane's place. BTW, can't make it to the meeting at <NUMBER><NUMBER> AM. LOL! Call me at <PHONE> or email me at <EMAIL> Check out this cool website: <URL>
Statistical Model Text : hey john doe wanna grab coffee starbucks 5th avenue im feeling bit tired last nights party janes place btw cant make meeting numbernumber am lol call phone email email check cool website url
Language of the Text : ENGLISH
```

## API

### `sct.TextCleaner`

#### `process(text: str) -> Tuple[str, str, str]`

Processes the input text and returns a tuple containing:
- Cleaned text with punctuation and unnecessary characters removed.
- Cleaned text with stopwords removed.
- Detected language of the text.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an issue.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

The package took inspirations from the following repo:

- [clean-text](https://github.com/jfilter/clean-text)