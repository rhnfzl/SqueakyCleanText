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
from typing import List, Dict, Any  # Add this line
transformers.logging.set_verbosity_error() 

class GeneralNER:
    
    """
    To tag [PER, LOC, ORG, MISC] postional tags using ensemble technique
    """
    
    def __init__(self):
        """
        Initializes the GeneralNER object.

        This function initializes the GeneralNER object by loading the necessary models and tokenizers.
        It sets the device to "cuda" if it is available, otherwise it sets it to "cpu".
        It initializes the AnonymizerEngine object.
        It loads the NER models and tokenizers based on the configuration.
        It sets the minimum token length based on the maximum length of the tokenizers.
        It sets the tokenizer based on the maximum length of the tokenizers.
        """

        # Set the device to "cuda" if it is available, otherwise set it to "cpu"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialize the AnonymizerEngine object
        self.engine = AnonymizerEngine()

        # Load the NER models and tokenizers based on the configuration
        if len(config.NER_MODELS_LIST) == 5:
            # Load models from the configuration
            english_model_name = config.NER_MODELS_LIST[0]
            dutch_model_name = config.NER_MODELS_LIST[1]
            german_model_name = config.NER_MODELS_LIST[2]
            spanish_model_name = config.NER_MODELS_LIST[3]
            multilingual_model_name = config.NER_MODELS_LIST[4]
        else:
            # Load default models
            model_name = [
                "FacebookAI/xlm-roberta-large-finetuned-conll03-english",
                "FacebookAI/xlm-roberta-large-finetuned-conll02-dutch",
                "FacebookAI/xlm-roberta-large-finetuned-conll03-german",
                "FacebookAI/xlm-roberta-large-finetuned-conll02-spanish",
                "Babelscape/wikineural-multilingual-ner"
            ]
            english_model_name = model_name[0]
            dutch_model_name = model_name[1]
            german_model_name = model_name[2]
            spanish_model_name = model_name[3]
            multilingual_model_name = model_name[4]

        # Load English model and tokenizer
        self.en_tokenizer = AutoTokenizer.from_pretrained(english_model_name)
        self.en_model = AutoModelForTokenClassification.from_pretrained(english_model_name).to(self.device)
        self.en_ner_pipeline = pipeline(
            "ner",
            model=self.en_model,
            tokenizer=self.en_tokenizer,
            aggregation_strategy="simple"
        )

        # Load Dutch model and tokenizer
        self.nl_tokenizer = AutoTokenizer.from_pretrained(dutch_model_name)
        self.nl_model = AutoModelForTokenClassification.from_pretrained(dutch_model_name).to(self.device)
        self.nl_ner_pipeline = pipeline(
            "ner",
            model=self.nl_model,
            tokenizer=self.nl_tokenizer,
            aggregation_strategy="simple"
        )

        # Load German model and tokenizer
        self.de_tokenizer = AutoTokenizer.from_pretrained(german_model_name)
        self.de_model = AutoModelForTokenClassification.from_pretrained(german_model_name).to(self.device)
        self.de_ner_pipeline = pipeline(
            "ner",
            model=self.de_model,
            tokenizer=self.de_tokenizer,
            aggregation_strategy="simple"
        )

        # Load Spanish model and tokenizer
        self.es_tokenizer = AutoTokenizer.from_pretrained(spanish_model_name)
        self.es_model = AutoModelForTokenClassification.from_pretrained(spanish_model_name).to(self.device)
        self.es_ner_pipeline = pipeline(
            "ner",
            model=self.es_model,
            tokenizer=self.es_tokenizer,
            aggregation_strategy="simple"
        )

        # Load Multilingual model and tokenizer
        self.multi_tokenizer = AutoTokenizer.from_pretrained(multilingual_model_name)
        self.multi_model = AutoModelForTokenClassification.from_pretrained(multilingual_model_name).to(self.device)
        self.multi_ner_pipeline = pipeline(
            "ner",
            model=self.multi_model,
            tokenizer=self.multi_tokenizer,
            aggregation_strategy="simple"
        )

        # Set the minimum token length based on the maximum length of the tokenizers
        self.min_token_length = math.ceil(min(
            self.en_tokenizer.max_len_single_sentence,
            self.multi_tokenizer.max_len_single_sentence
        ) * 0.9)

        # Set the tokenizer based on the maximum length of the tokenizers
        self.tokenizer = self.en_tokenizer if self.en_tokenizer.max_len_single_sentence <= self.multi_tokenizer.max_len_single_sentence else self.multi_tokenizer

    def ner_data(self, data, pos):
        """
        Formats the Named Entity Recognition (NER) data.

        Args:
            data (List[Dict[str, Any]]): A list of dictionaries containing entity information.
            pos (List[str]): A list of valid entity groups.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries with the formatted entity data.
        """
        # Filter the data based on the entity group and return the desired keys
        return [
            {
                'entity_group': ix['entity_group'],  # The entity group
                'score': ix['score'],  # The score of the entity
                'word': ix['word'],  # The word of the entity
                'key': f"{ix['start']}{ix['end']}",  # The unique key for the entity
                'start': ix['start'],  # The start position of the entity
                'end': ix['end']  # The end position of the entity
            }
            for ix in data  # Iterate over each entity in the data
            if ix['entity_group'] in pos  # Check if the entity group is in the valid entity groups
        ]

    def filter_ner_data(self, data, keys):
        """
        Filters named entity recognition (NER) data based on a list of keys.

        Args:
            data (list): A list of dictionaries containing entity information.
            keys (list): A list of keys to filter the data by.

        Returns:
            list: A list of unique entity data dictionaries with the highest score for each key.
        """
        # Dictionary to store the unique data with the highest score for each key
        unique_data = {}

        # Iterate over each item in the data
        for item in data:
            # Check if the key of the item is in the keys
            if item['key'] in keys:
                # Check if the key is already in unique_data
                # If it is, compare the scores and update the value if the new score is higher
                if item['key'] not in unique_data or item['score'] > unique_data[item['key']]['score']:
                    unique_data[item['key']] = item
        
        # Return the values of the unique_data dictionary
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
        # Initialize an empty list to store the RecognizerResult objects
        analyzer_result = []

        # Iterate over each item in the filtered_data
        for item in filtered_data:
            # Check the entity_group of the item
            if item['entity_group'] == 'PER':
                # Create a RecognizerResult object with the corresponding entity_type and other attributes
                analyzer_result.append(RecognizerResult(entity_type="PERSON",
                                                        start=item['start'],
                                                        end=item['end'],
                                                        score=item['score']))
            elif item['entity_group'] == 'LOC':
                # Create a RecognizerResult object with the corresponding entity_type and other attributes
                analyzer_result.append(RecognizerResult(entity_type="LOCATION",
                                                        start=item['start'],
                                                        end=item['end'],
                                                        score=item['score']))
            elif item['entity_group'] == 'ORG':
                # Create a RecognizerResult object with the corresponding entity_type and other attributes
                analyzer_result.append(RecognizerResult(entity_type="ORGANISATION",
                                                        start=item['start'],
                                                        end=item['end'],
                                                        score=item['score']))

        # Get the length of the text
        text_length = len(text)

        # Filter out the analyzer_result entries that have invalid start or end positions
        analyzer_result = [entry for entry in analyzer_result if 0 <= entry.start < text_length and 0 < entry.end <= text_length]

        # Replace words in the text with their entity_type tags using the anonymize function of the engine
        return self.engine.anonymize(text=text, analyzer_results=analyzer_result)

    def ner_ensemble(self, ner_results: List[Dict[str, Any]], t: float) -> List[Dict[str, Any]]:
        """
        Applies an ensemble method for NER.

        Args:
            ner_results (List[Dict[str, Any]]): A list of dictionaries containing the NER results.
            t (float): The threshold value for filtering the NER results.

        Returns:
            List[Dict[str, Any]]: A sorted list of dictionaries containing the filtered NER results.
        """
        # Initialize a dictionary to store the count and sum of scores for each key
        ner_keys = defaultdict(lambda: [0, 0])

        # Iterate over each item in the ner_results
        for entity in ner_results:
            # For each item, increment the count and add the score to the sum
            ner_keys[entity['key']][0] += 1
            ner_keys[entity['key']][1] += entity['score']

        # Filter out the keys that have a score below the threshold
        ner_keys = [key for key, val in ner_keys.items() if val[1] / val[0] >= t]

        # Filter out the NER results that have a key not in the ner_keys
        filter_ner_results = self.filter_ner_data(ner_results, ner_keys)
        
        # Sort the filtered NER results by start position
        filter_ner_results.sort(key=lambda x: x['start'])
        
        # Return the filtered NER results
        return filter_ner_results
    
    def ner_process(self, text: str, positional_tags: List[str], ner_confidence_threshold: float, language: str) -> str:
        """
        Executes NER Process to remove the positional tags, PER, LOC, ORG, MISC.

        Args:
            text (str): text from which postional tags need to be recognised
            positional_tags (list): pass tags as ['PER', 'LOC'] (default), also supports 'ORG', 'MISC'
            ner_confidence_threshold (float): NER Confidence
            language (str): language model which need to be used, currently supports ENGLISH, DUTCH

        Returns:
            str: a string of words sorted based on length for the provided positional tags which meets the threshold
        """
        # Initialize a list to store the cleaned text
        ner_clean_text = []
        
        # Get the length of the input text
        text_token_length = len(self.tokenizer.tokenize(text))
        
        # Calculate the number of parts the text needs to be split into
        num_parts = math.ceil(text_token_length / self.min_token_length)
        
        # If the text doesn't need to be split, just add it to the list
        if num_parts == 0:
            texts = [text]
        else:
            # Split the text into chunks
            texts = self.split_text(text, self.min_token_length, self.tokenizer)
            
        # Iterate over each chunk of text
        for text in texts:
            # Initialize a list to store the NER results
            ner_results = []
            
            # Get the NER results for the multi-lingual model
            ner_results.append(self.ner_data(self.multi_ner_pipeline(text), positional_tags))
            
            # Get the NER results for the English model
            ner_results.append(self.ner_data(self.en_ner_pipeline(text), positional_tags))
            
            # If the language is Dutch, get the NER results for the Dutch model
            if language == 'DUTCH':
                ner_results.append(self.ner_data(self.nl_ner_pipeline(text), positional_tags))
            
            # If the language is German, get the NER results for the German model
            elif language == 'GERMAN':
                ner_results.append(self.ner_data(self.de_ner_pipeline(text), positional_tags))
            
            # If the language is Spanish, get the NER results for the Spanish model
            elif language == 'SPANISH':
                ner_results.append(self.ner_data(self.es_ner_pipeline(text), positional_tags))
            
            # Flatten the list of NER results
            ner_results = list(itertools.chain.from_iterable(ner_results))
            
            # Apply the ensemble method to the NER results
            ner_results = self.ner_ensemble(ner_results, ner_confidence_threshold)
            
            # Anonymize the text using the filtered NER results
            ner_text = self.anonymize_text(text, ner_results).text
            
            # Add the anonymized text to the list
            ner_clean_text.append(ner_text)
        
        # Join the cleaned text into a single string and return it
        return ' '.join(ner_clean_text)
    
    def split_text(self, text: str, max_tokens: int, tokenizer) -> List[str]:
        """
        Splits a given text into chunks based on a maximum token limit.

        Args:
            text (str): The input text to be split.
            max_tokens (int): The maximum number of tokens allowed in each chunk.
            tokenizer: A tokenizer object used to tokenize the input text.

        Returns:
            list: A list of text chunks, each containing a maximum of max_tokens tokens.
        """
        # Find the sentence boundaries in the text
        sentence_boundaries = [(m.start(), m.end()) for m in constants.SENTENCE_BOUNDARY_PATTERN.finditer(text)]
        
        # Initialize a list to store the text chunks
        chunks = []
        
        # Initialize variables to keep track of the current chunk
        current_chunk = []
        current_token_count = 0
        current_position = 0
    
        # Iterate over the sentence boundaries
        for boundary_start, boundary_end in sentence_boundaries:
            # Get the sentence
            sentence = text[current_position:boundary_start+1]
            # Update the current position
            current_position = boundary_end
    
            # Get the number of tokens in the sentence
            token_count = len(tokenizer(sentence)["input_ids"])
    
            # If the sentence can fit in the current chunk, add it
            if current_token_count + token_count <= max_tokens:
                current_chunk.append(sentence)
                current_token_count += token_count
            # Otherwise, create a new chunk
            else:
                chunks.append(''.join(current_chunk))
                current_chunk = [sentence]
                current_token_count = token_count
    
        # Append the last sentence
        last_sentence = text[current_position:]
        current_chunk.append(last_sentence)
        chunks.append(''.join(current_chunk))
    
        # Return the list of chunks
        return chunks