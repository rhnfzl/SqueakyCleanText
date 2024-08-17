import math
import torch
import itertools
from collections import defaultdict

import transformers
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import RecognizerResult

from sct.utils import constants
from sct import config
transformers.logging.set_verbosity_error() 

class GeneralNER:
    
    """
    To tag [PER, LOC, ORG, MISC] postional tags using ensemble technique
    """
    
    def __init__(self):
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.engine = AnonymizerEngine()
        
        if len(config.NER_MODELS_LIST) == 5:
            english_model_name = config.NER_MODELS_LIST[0]
            dutch_model_name = config.NER_MODELS_LIST[1]
            german_model_name = config.NER_MODELS_LIST[2]
            spanish_model_name = config.NER_MODELS_LIST[3]
            multilingual_model_name = config.NER_MODELS_LIST[4]
        else:
            # Load models
            model_name = ["FacebookAI/xlm-roberta-large-finetuned-conll03-english",
                          "FacebookAI/xlm-roberta-large-finetuned-conll02-dutch",
                        "FacebookAI/xlm-roberta-large-finetuned-conll03-german",
                        "FacebookAI/xlm-roberta-large-finetuned-conll02-spanish",
                        "Babelscape/wikineural-multilingual-ner"]
            
            english_model_name = model_name[0]
            dutch_model_name = model_name[1]
            german_model_name = model_name[2]
            spanish_model_name = model_name[3]
            multilingual_model_name = model_name[4]
        
        self.en_tokenizer = AutoTokenizer.from_pretrained(english_model_name)
        self.en_model = AutoModelForTokenClassification.from_pretrained(english_model_name).to(self.device)
        self.en_ner_pipeline = pipeline("ner", model=self.en_model, tokenizer=self.en_tokenizer, aggregation_strategy="simple")

        self.nl_tokenizer = AutoTokenizer.from_pretrained(dutch_model_name)
        self.nl_model = AutoModelForTokenClassification.from_pretrained(dutch_model_name).to(self.device)
        self.nl_ner_pipeline = pipeline("ner", model=self.nl_model, tokenizer=self.nl_tokenizer, aggregation_strategy="simple")

        self.de_tokenizer = AutoTokenizer.from_pretrained(german_model_name)
        self.de_model = AutoModelForTokenClassification.from_pretrained(german_model_name).to(self.device)
        self.de_ner_pipeline = pipeline("ner", model=self.de_model, tokenizer=self.de_tokenizer, aggregation_strategy="simple")

        self.es_tokenizer = AutoTokenizer.from_pretrained(spanish_model_name)
        self.es_model = AutoModelForTokenClassification.from_pretrained(spanish_model_name).to(self.device)
        self.es_ner_pipeline = pipeline("ner", model=self.es_model, tokenizer=self.es_tokenizer, aggregation_strategy="simple")

        self.multi_tokenizer = AutoTokenizer.from_pretrained(multilingual_model_name)
        self.multi_model = AutoModelForTokenClassification.from_pretrained(multilingual_model_name).to(self.device)
        self.multi_ner_pipeline = pipeline("ner", model=self.multi_model, tokenizer=self.multi_tokenizer, aggregation_strategy="simple")

        self.min_token_length = math.ceil(min(self.en_tokenizer.max_len_single_sentence, self.multi_tokenizer.max_len_single_sentence) * 0.9)

        self.tokenizer = self.en_tokenizer if self.en_tokenizer.max_len_single_sentence <= self.multi_tokenizer.max_len_single_sentence else self.multi_tokenizer

    def ner_data(self, data, pos):
        """
        Formats NER (Named Entity Recognition) files.
        """
        return [{'entity_group': ix['entity_group'], 'score': ix['score'], 'word': ix['word'], 'key': str(ix['start']) + str(ix['end']), 'start': ix['start'], 'end': ix['end']} for ix in data if ix['entity_group'] in pos]

    def filter_ner_data(self, data, keys):
        """
        Filters named entity recognition (NER) data based on a list of keys.

        Args:
            data (list): A list of dictionaries containing entity information.
            keys (list): A list of keys to filter the data by.

        Returns:
            list: A list of unique entity data dictionaries with the highest score for each key.
        """
        unique_data = {}
        for item in data:
            if item['key'] in keys:
                if item['key'] not in unique_data or item['score'] > unique_data[item['key']]['score']:
                    unique_data[item['key']] = item
        
        return list(unique_data.values())

    def anonymize_text(self, text, filtered_data):
        """
        Anonymizes a given text by replacing named entities with their corresponding type tags.

        Args:
            text (str): The input text to be anonymized.
            filtered_data (list): A list of dictionaries containing entity information.

        Returns:
            str: The anonymized text with entity type tags.
        """
        analyzer_result = list()
        for items in filtered_data:
            if items['entity_group'] == 'PER':
                analyzer_result.append(RecognizerResult(entity_type="PERSON", start=items['start'], end=items['end'], score=items['score']))
            elif items['entity_group'] == 'LOC':
                analyzer_result.append(RecognizerResult(entity_type="LOCATION", start=items['start'], end=items['end'], score=items['score']))
            elif items['entity_group'] == 'ORG':
                analyzer_result.append(RecognizerResult(entity_type="ORGANISATION", start=items['start'], end=items['end'], score=items['score']))
        
        text_length = len(text)
        analyzer_result = [entry for entry in analyzer_result if 0 <= entry.start < text_length and 0 < entry.end <= text_length]
        # Replace words in the text with their entity_type tags
        return self.engine.anonymize(text=text, analyzer_results=analyzer_result)

    def ner_ensemble(self, ner_results, t):
        """
        Applies an ensemble method for NER.

        Args:
            ner_results (List[Dict[str, Any]]): A list of dictionaries containing the NER results.
            t (float): The threshold value for filtering the NER results.

        Returns:
            List[Dict[str, Any]]: A sorted list of dictionaries containing the filtered NER results.
        """
        ner_keys = defaultdict(lambda: [0, 0])
        for entity in ner_results:
            ner_keys[entity['key']][0] += 1
            ner_keys[entity['key']][1] += entity['score']

        ner_keys = [key for key, val in ner_keys.items() if val[1] / val[0] >= t]

        filter_ner_results = self.filter_ner_data(ner_results, ner_keys)
        
        filter_ner_results.sort(key=lambda x: x['start'])
        
        return filter_ner_results
    
    def ner_process(self, text, positional_tags, ner_confidence_threshold, language):
        """_summary_
            Executes NER Process to remove the positional tags, PER, LOC, ORG, MISC.
        Args:
            text (string): text from which postional tags need to be recognised
            positional_tags (list): pass tags as ['PER', 'LOC'] (default), also supports 'ORG', 'MISC'
            ner_confidence_threshold (int): NER Confidence
            language (string): language model which need to be used, currently supports ENGLISH, DUTCH

        Returns:
            list: a list of words sorted based on length for the provided positional tags which meets the threshold
        """
        ner_clean_text = list()       
        # text length
        text_token_length = len(self.tokenizer.tokenize(text))
        
        # parts it need to get split
        num_parts = math.ceil(text_token_length/self.min_token_length)
        
        if num_parts == 0:
            texts = [text]
        else:
            texts = self.split_text(text, self.min_token_length, self.tokenizer)
            
        for text in texts:
            ner_results = []
            ner_results.append(self.ner_data(self.multi_ner_pipeline(text), positional_tags))
            ner_results.append(self.ner_data(self.en_ner_pipeline(text), positional_tags))
            if language == 'DUTCH':
                ner_results.append(self.ner_data(self.nl_ner_pipeline(text), positional_tags))
            elif language == 'GERMAN':
                ner_results.append(self.ner_data(self.de_ner_pipeline(text), positional_tags))
            elif language == 'SPANISH':
                ner_results.append(self.ner_data(self.es_ner_pipeline(text), positional_tags))
            
            # flat out the list
            ner_results = list(itertools.chain.from_iterable(ner_results))
            ner_results = self.ner_ensemble(ner_results, ner_confidence_threshold)
            ner_text = self.anonymize_text(text, ner_results).text
            ner_clean_text.append(ner_text)
            
        return ' '.join(ner_clean_text)
    
    def split_text(self, text, max_tokens, tokenizer):
        """
        Splits a given text into chunks based on a maximum token limit.

        Args:
            text (str): The input text to be split.
            max_tokens (int): The maximum number of tokens allowed in each chunk.
            tokenizer: A tokenizer object used to tokenize the input text.

        Returns:
            list: A list of text chunks, each containing a maximum of max_tokens tokens.
        """
        sentence_boundaries = [(m.start(), m.end()) for m in constants.SENTENCE_BOUNDARY_PATTERN.finditer(text)]
        
        chunks = []
        current_chunk = []
        current_token_count = 0
        current_position = 0
    
        for boundary_start, boundary_end in sentence_boundaries:
            sentence = text[current_position:boundary_start+1]
            current_position = boundary_end
    
            token_count = len(tokenizer(sentence)["input_ids"])
    
            if current_token_count + token_count <= max_tokens:
                current_chunk.append(sentence)
                current_token_count += token_count
            else:
                chunks.append(''.join(current_chunk))
                current_chunk = [sentence]
                current_token_count = token_count
    
        # Append the last sentence
        last_sentence = text[current_position:]
        current_chunk.append(last_sentence)
        chunks.append(''.join(current_chunk))
    
        return chunks