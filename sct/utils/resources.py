import warnings
warnings.filterwarnings('ignore')
from nltk.corpus import stopwords as sw

#--- NER
import transformers
from transformers import pipeline
from lingua import Language, LanguageDetectorBuilder
from transformers import AutoTokenizer, AutoModelForTokenClassification
transformers.logging.set_verbosity_error() 

#---- NER Models
TOKENIZER_EN = AutoTokenizer.from_pretrained("xlm-roberta-large-finetuned-conll03-english") # based on conll2003 dataset
MODEL_EN = AutoModelForTokenClassification.from_pretrained("xlm-roberta-large-finetuned-conll03-english")

TOKENIZER_NL = AutoTokenizer.from_pretrained("xlm-roberta-large-finetuned-conll02-dutch") # based on CoNLL-2002 dataset
MODEL_NL = AutoModelForTokenClassification.from_pretrained("xlm-roberta-large-finetuned-conll02-dutch")

WIKITOKENIZER = AutoTokenizer.from_pretrained("Babelscape/wikineural-multilingual-ner") # based on wikineural multilingual dataset
WIKIMODEL = AutoModelForTokenClassification.from_pretrained("Babelscape/wikineural-multilingual-ner")

NLP_EN = pipeline("ner", model=MODEL_EN, tokenizer=TOKENIZER_EN, aggregation_strategy="simple")
NLP_NL = pipeline("ner", model=MODEL_NL, tokenizer=TOKENIZER_NL, aggregation_strategy="simple")
WIKINLP = pipeline("ner", model=WIKIMODEL, tokenizer=WIKITOKENIZER, aggregation_strategy="simple")

#---- Detect the Language / Also add languages to support in Future
LANGUAGES = [Language.DUTCH, Language.ENGLISH]
DETECTOR = LanguageDetectorBuilder.from_languages(*LANGUAGES).build()

### stop words
STOP_WORDS_EN = sw.words('english')
STOP_WORDS_NL = sw.words('dutch')
STOP_WORDS_DE = sw.words('german')