import math
import itertools
from collections import defaultdict

import transformers
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

from sct.utils import constants
transformers.logging.set_verbosity_error() 

class GeneralNER:
    
    """
    To tag [PER, LOC, ORG, MISC] postional tags using ensemble technique
    """
    
    def __init__(self):
        
        #---- NER Models
        TOKENIZER_EN = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-large-finetuned-conll03-english")
        MODEL_EN = AutoModelForTokenClassification.from_pretrained("FacebookAI/xlm-roberta-large-finetuned-conll03-english")

        TOKENIZER_NL = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-large-finetuned-conll02-dutch")
        MODEL_NL = AutoModelForTokenClassification.from_pretrained("FacebookAI/xlm-roberta-large-finetuned-conll02-dutch")
        
        TOKENIZER_DE = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-large-finetuned-conll03-german")
        MODEL_DE = AutoModelForTokenClassification.from_pretrained("FacebookAI/xlm-roberta-large-finetuned-conll03-german")
        
        TOKENIZER_ES = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-large-finetuned-conll02-spanish")
        MODEL_ES = AutoModelForTokenClassification.from_pretrained("FacebookAI/xlm-roberta-large-finetuned-conll02-spanish")

        TOKENIZER_MULTI = AutoTokenizer.from_pretrained("Babelscape/wikineural-multilingual-ner")
        MODEL_MULTI = AutoModelForTokenClassification.from_pretrained("Babelscape/wikineural-multilingual-ner")
        
        # Length with buffer of 10%
        self.len_single_sentence = [TOKENIZER_EN.max_len_single_sentence, TOKENIZER_MULTI.max_len_single_sentence]
        self.min_token_length = math.ceil(min(self.len_single_sentence) * 0.9)
        tokenizer_indicator = self.len_single_sentence.index(min(self.len_single_sentence))
        
        if tokenizer_indicator == 0:
            self.tokenizer = TOKENIZER_EN
        elif tokenizer_indicator == 1:
            self.tokenizer = TOKENIZER_MULTI
            
        self.nlp_en = pipeline("ner", model=MODEL_EN, tokenizer=TOKENIZER_EN, aggregation_strategy="simple")
        self.nlp_nl = pipeline("ner", model=MODEL_NL, tokenizer=TOKENIZER_NL, aggregation_strategy="simple")
        self.nlp_de = pipeline("ner", model=MODEL_DE, tokenizer=TOKENIZER_DE, aggregation_strategy="simple")
        self.nlp_es = pipeline("ner", model=MODEL_ES, tokenizer=TOKENIZER_ES, aggregation_strategy="simple")
        self.nlp_multi = pipeline("ner", model=MODEL_MULTI, tokenizer=TOKENIZER_MULTI, aggregation_strategy="simple")

    def ner_data(self, data, pos):
        """
        Formats NER (Named Entity Recognition) files.
        """
        return [{'entity_group': ix['entity_group'], 'score': ix['score'], 'word': ix['word'], 'key': str(ix['start']) + str(ix['end'])} for ix in data if ix['entity_group'] in pos]

    def ner_ensemble(self, ner_results, t):
        """
        Applies an ensemble method for NER.
        """
        ner_keys = defaultdict(lambda: [0, 0])
        ner_words = set()

        for entity in ner_results:
            ner_keys[entity['key']][0] += 1
            ner_keys[entity['key']][1] += entity['score']

        ner_keys = [key for key, val in ner_keys.items() if val[1] / val[0] >= t]

        for entity in ner_results:
            if entity['key'] in ner_keys:
                ner_words.add(entity['word'])

        ner_words = sorted(list(ner_words), key=len, reverse=True)
        return ner_words
    
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
        ner_results = []
        
        # text length
        text_token_length = len(self.tokenizer.tokenize(text))
        
        # parts it need to get split
        num_parts = math.ceil(text_token_length/self.min_token_length)
        
        if num_parts == 0:
            texts = [text]
        else:
            texts = self.split_text(text, self.min_token_length, self.tokenizer)
            
        for text in texts:
            ner_results.append(self.ner_data(self.nlp_multi(text), positional_tags))
            ner_results.append(self.ner_data(self.nlp_en(text), positional_tags))

            if language == 'DUTCH':
                ner_results.append(self.ner_data(self.nlp_nl(text), positional_tags))
            elif language == 'GERMAN':
                ner_results.append(self.ner_data(self.nlp_de(text), positional_tags))
            elif language == 'SPANISH':
                ner_results.append(self.ner_data(self.nlp_es(text), positional_tags))
                
        # flat out the list
        ner_results = list(itertools.chain.from_iterable(ner_results))
        ner_words = self.ner_ensemble(ner_results, ner_confidence_threshold)
        return ner_words
    
    
    def split_text(self, text, max_tokens, tokenizer):
    
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