import math
import torch
import itertools
from collections import defaultdict
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import transformers
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import RecognizerResult

from sct.utils import constants
from sct import config
from sct.config import NER_MODELS_LIST

transformers.logging.set_verbosity_error() 

logger = logging.getLogger(__name__)

class ModelLoadError(Exception):
    """Raised when model loading fails"""
    pass

class GeneralNER:
    
    """
    To tag [PER, LOC, ORG, MISC] postional tags using ensemble technique
    """
    
    def __init__(self, cache_dir: Optional[Path] = None, device: str = None):
        """Initialize NER models.
        
        Args:
            cache_dir: Optional directory for caching models
            device: Device to use for inference ('cuda' or 'cpu'). If None, will auto-detect.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Default model names as fallback
        DEFAULT_MODELS = [
            "FacebookAI/xlm-roberta-large-finetuned-conll03-english",
            "FacebookAI/xlm-roberta-large-finetuned-conll02-dutch",
            "FacebookAI/xlm-roberta-large-finetuned-conll03-german", 
            "FacebookAI/xlm-roberta-large-finetuned-conll02-spanish",
            "Babelscape/wikineural-multilingual-ner"
        ]
        
        try:
            self.engine = AnonymizerEngine()
            
            # Use config if valid, otherwise fallback to defaults
            if len(NER_MODELS_LIST) == 5:
                model_names = NER_MODELS_LIST
                logger.info("Using models from config")
            else:
                model_names = DEFAULT_MODELS
                logger.warning("Invalid config model list, using default models")
            
            cache_args = {"cache_dir": str(cache_dir)} if cache_dir else {}
            self._load_models(model_names, cache_args)
            
            # Set tokenizer properties
            self.min_token_length = math.ceil(min(
                self.en_tokenizer.max_len_single_sentence,
                self.multi_tokenizer.max_len_single_sentence
            ) * 0.9)
            
            self.tokenizer = self.en_tokenizer if (
                self.en_tokenizer.max_len_single_sentence <= 
                self.multi_tokenizer.max_len_single_sentence
            ) else self.multi_tokenizer
                
        except Exception as e:
            logger.error(f"Failed to initialize NER: {e}")
            raise ModelLoadError(f"NER initialization failed: {e}")

    def _load_models(self, model_names: List[str], cache_args: Dict[str, str]) -> None:
        """Load NER models with caching support."""
        try:
            # Load models sequentially with proper error handling
            for i, model_name in enumerate(model_names):
                logger.info(f"Loading model {model_name}")
                if i == 0:  # English
                    self.en_tokenizer = AutoTokenizer.from_pretrained(model_name, **cache_args)
                    self.en_model = AutoModelForTokenClassification.from_pretrained(model_name, **cache_args).to(self.device)
                    self.en_ner_pipeline = pipeline("ner", model=self.en_model, tokenizer=self.en_tokenizer, 
                                                  aggregation_strategy="simple", device=self.device)
                elif i == 1:  # Dutch
                    self.nl_tokenizer = AutoTokenizer.from_pretrained(model_name, **cache_args)
                    self.nl_model = AutoModelForTokenClassification.from_pretrained(model_name, **cache_args).to(self.device)
                    self.nl_ner_pipeline = pipeline("ner", model=self.nl_model, tokenizer=self.nl_tokenizer,
                                                  aggregation_strategy="simple", device=self.device)
                elif i == 2:  # German
                    self.de_tokenizer = AutoTokenizer.from_pretrained(model_name, **cache_args)
                    self.de_model = AutoModelForTokenClassification.from_pretrained(model_name, **cache_args).to(self.device)
                    self.de_ner_pipeline = pipeline("ner", model=self.de_model, tokenizer=self.de_tokenizer,
                                                  aggregation_strategy="simple", device=self.device)
                elif i == 3:  # Spanish
                    self.es_tokenizer = AutoTokenizer.from_pretrained(model_name, **cache_args)
                    self.es_model = AutoModelForTokenClassification.from_pretrained(model_name, **cache_args).to(self.device)
                    self.es_ner_pipeline = pipeline("ner", model=self.es_model, tokenizer=self.es_tokenizer,
                                                  aggregation_strategy="simple", device=self.device)
                elif i == 4:  # Multilingual
                    self.multi_tokenizer = AutoTokenizer.from_pretrained(model_name, **cache_args)
                    self.multi_model = AutoModelForTokenClassification.from_pretrained(model_name, **cache_args).to(self.device)
                    self.multi_ner_pipeline = pipeline("ner", model=self.multi_model, tokenizer=self.multi_tokenizer,
                                                     aggregation_strategy="simple", device=self.device)
                
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise ModelLoadError(f"Model loading failed: {e}")

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
        """Anonymizes text while preserving whitespace."""
        analyzer_result = []
        for items in filtered_data:
            if items['entity_group'] == 'PER':
                analyzer_result.append(RecognizerResult(
                    entity_type="PERSON", 
                    start=items['start'], 
                    end=items['end'],
                    score=items['score']
                ))
            elif items['entity_group'] == 'LOC':
                analyzer_result.append(RecognizerResult(entity_type="LOCATION", start=items['start'], end=items['end'], score=items['score']))
            elif items['entity_group'] == 'ORG':
                analyzer_result.append(RecognizerResult(entity_type="ORGANISATION", start=items['start'], end=items['end'], score=items['score']))
        
        text_length = len(text)
        analyzer_result = [
            entry for entry in analyzer_result 
            if 0 <= entry.start < text_length and 0 < entry.end <= text_length
        ]
        
        # Return the text property from the anonymizer result
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
    
    @torch.no_grad()
    def ner_process(
        self, 
        text: str,
        positional_tags: List[str] = None,
        ner_confidence_threshold: float = None,
        language: str = None
    ) -> str:
        """Process text with NER models."""
        if not positional_tags:
            raise ValueError("Must provide at least one positional tag")
        
        ner_confidence_threshold = ner_confidence_threshold or 0.85
        
        # Split long text into chunks
        texts = self.split_text(text, self.min_token_length, self.tokenizer)
        ner_clean_text = []
        
        for text_chunk in texts:
            ner_results = []
            
            # First try language-specific pipeline if specified
            if language == 'DUTCH':
                ner_results.extend(self.ner_data(self.nl_ner_pipeline(text_chunk), positional_tags))
            elif language == 'GERMAN':
                ner_results.extend(self.ner_data(self.de_ner_pipeline(text_chunk), positional_tags))
            elif language == 'SPANISH':
                ner_results.extend(self.ner_data(self.es_ner_pipeline(text_chunk), positional_tags))
            else:
                # For English or unspecified, try English first then multilingual
                en_results = self.ner_data(self.en_ner_pipeline(text_chunk), positional_tags)
                if en_results:  # If English model finds entities, use those
                    ner_results.extend(en_results)
                else:  # Otherwise try multilingual model
                    ner_results.extend(self.ner_data(self.multi_ner_pipeline(text_chunk), positional_tags))
            
            # Apply confidence threshold before filtering
            confident_results = [r for r in ner_results if r['score'] >= ner_confidence_threshold]
            
            if confident_results:
                # Get unique entities with highest confidence
                keys = list(set(item['key'] for item in confident_results))
                filtered_data = self.filter_ner_data(confident_results, keys)
                
                # Anonymize text
                ner_text = self.anonymize_text(text_chunk, filtered_data).text
            else:
                # If no entities meet the confidence threshold, return original text
                ner_text = text_chunk
            
            ner_clean_text.append(ner_text)
        
        return ' '.join(ner_clean_text)
    
    def split_text(self, text: str, max_tokens: int, tokenizer) -> List[str]:
        """Split text into chunks optimized for model processing."""
        # Cache tokenizer results
        tokenized_text = tokenizer(text, return_offsets_mapping=True)
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        last_end = 0
        
        for start, end in tokenized_text.offset_mapping:
            if current_tokens >= max_tokens:
                chunks.append(text[last_end:start])
                current_chunk = []
                current_tokens = 0
                last_end = start
                
            current_chunk.append(text[start:end])
            current_tokens += 1
            
        if current_chunk:
            chunks.append(text[last_end:])
            
        return chunks

    def process_batch(
        self, 
        texts: List[str], 
        batch_size: int = 8,
        positional_tags: List[str] = None,
        ner_confidence_threshold: float = None,
        language: str = None
    ) -> List[str]:
        """Process multiple texts efficiently in batches.
        
        Args:
            texts: List of input texts
            batch_size: Number of texts to process simultaneously
            positional_tags: List of entity types to detect
            ner_confidence_threshold: Minimum confidence score for entity detection
            language: Language of the input texts
            
        Returns:
            List of processed texts with entities anonymized
        """
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = [
                self.ner_process(
                    text,
                    positional_tags=positional_tags,
                    ner_confidence_threshold=ner_confidence_threshold,
                    language=language
                ) for text in batch
            ]
            results.extend(batch_results)
        return results

    def __del__(self):
        """Cleanup GPU memory when object is destroyed."""
        if hasattr(self, 'device') and self.device == 'cuda':
            try:
                torch.cuda.empty_cache()
            except Exception as e:
                logger.warning(f"Failed to clear CUDA cache: {e}")