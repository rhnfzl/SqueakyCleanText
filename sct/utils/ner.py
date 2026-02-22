import math
import torch
import threading
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

transformers.logging.set_verbosity_error()

logger = logging.getLogger(__name__)

# Data-driven language-to-model mapping
LANG_MODEL_MAP = {
    'ENGLISH': 0,
    'DUTCH': 1,
    'GERMAN': 2,
    'SPANISH': 3,
    'MULTILINGUAL': 4,
}

# Entity group to Presidio entity type mapping
ENTITY_TYPE_MAP = {
    'PER': 'PERSON',
    'LOC': 'LOCATION',
    'ORG': 'ORGANISATION',
    'MISC': 'MISC',
}


class ModelLoadError(Exception):
    """Raised when model loading fails"""
    pass


class GeneralNER:
    """NER processor with lazy model loading and ensemble voting."""
    
    def __init__(self, cache_dir: Optional[Path] = None, device: str = None,
                 model_names: Optional[List[str]] = None):
        """Initialize NER processor.
        
        Args:
            cache_dir: Optional directory for caching models
            device: Device for inference ('cuda' or 'cpu'). Auto-detects if None.
            model_names: Optional list of model names. Uses config if None.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        self.engine = AnonymizerEngine()
        self._cache_args = {"cache_dir": str(cache_dir)} if cache_dir else {}
        
        # Build model mapping from config or provided list
        if model_names is not None:
            self._model_names = list(model_names)
        else:
            self._model_names = list(config.NER_MODELS_LIST)
        
        # Lazy-loaded pipelines, tokenizers, and models (keyed by language)
        self._pipelines = {}
        self._tokenizers = {}
        self._models = {}
        self._load_lock = threading.Lock()
        # HF fast tokenizers use Rust RefCell internally — not safe for
        # concurrent pipeline calls. Serialize inference across threads.
        self._inference_lock = threading.Lock()
        
        # Eagerly load English + multilingual to compute min_token_length
        # (needed for split_text before first ner_process call)
        self._ensure_loaded('ENGLISH')
        self._ensure_loaded('MULTILINGUAL')
        
        # Set tokenizer properties from loaded models
        en_tok = self._tokenizers['ENGLISH']
        multi_tok = self._tokenizers['MULTILINGUAL']
        self.min_token_length = math.ceil(min(
            en_tok.max_len_single_sentence,
            multi_tok.max_len_single_sentence
        ) * 0.9)
        
        self.tokenizer = en_tok if (
            en_tok.max_len_single_sentence <= multi_tok.max_len_single_sentence
        ) else multi_tok

    def _get_model_name(self, lang_key: str) -> str:
        """Get model name for a language using data-driven mapping."""
        idx = LANG_MODEL_MAP.get(lang_key)
        if idx is None or idx >= len(self._model_names):
            raise ModelLoadError(f"No model configured for language: {lang_key}")
        return self._model_names[idx]

    def _ensure_loaded(self, lang_key: str) -> None:
        """Lazily load a model and pipeline for the given language."""
        if lang_key in self._pipelines:
            return
        with self._load_lock:
            if lang_key in self._pipelines:
                return
            model_name = self._get_model_name(lang_key)
            logger.info(f"Loading model for {lang_key}: {model_name}")
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name, **self._cache_args)
                model = AutoModelForTokenClassification.from_pretrained(
                    model_name, **self._cache_args
                ).to(self.device)
                ner_pipeline = pipeline(
                    "ner", model=model, tokenizer=tokenizer,
                    aggregation_strategy="simple", device=self.device
                )
                self._tokenizers[lang_key] = tokenizer
                self._models[lang_key] = model
                self._pipelines[lang_key] = ner_pipeline
            except Exception as e:
                logger.error(f"Failed to load model for {lang_key}: {e}")
                raise ModelLoadError(f"Model loading failed for {lang_key}: {e}")

    def _get_pipeline(self, lang_key: str):
        """Get the NER pipeline for a language, loading it if needed."""
        self._ensure_loaded(lang_key)
        return self._pipelines[lang_key]

    def ner_data(self, data, pos):
        """Format NER results with collision-safe keys."""
        return [
            {
                'entity_group': ix['entity_group'],
                'score': ix['score'],
                'word': ix['word'],
                'key': f"{ix['start']}:{ix['end']}",  # Fixed: separator prevents collision
                'start': ix['start'],
                'end': ix['end'],
            }
            for ix in data if ix['entity_group'] in pos
        ]

    def filter_ner_data(self, data, keys):
        """Filter NER data, keeping highest-scoring entity per key."""
        unique_data = {}
        for item in data:
            if item['key'] in keys:
                if item['key'] not in unique_data or item['score'] > unique_data[item['key']]['score']:
                    unique_data[item['key']] = item
        return list(unique_data.values())

    def anonymize_text(self, text, filtered_data):
        """Anonymize text, replacing detected entities with type tokens."""
        analyzer_result = []
        for items in filtered_data:
            entity_type = ENTITY_TYPE_MAP.get(items['entity_group'])
            if entity_type:
                analyzer_result.append(RecognizerResult(
                    entity_type=entity_type,
                    start=items['start'],
                    end=items['end'],
                    score=items['score'],
                ))
        
        text_length = len(text)
        analyzer_result = [
            entry for entry in analyzer_result
            if 0 <= entry.start < text_length and 0 < entry.end <= text_length
        ]
        
        return self.engine.anonymize(text=text, analyzer_results=analyzer_result)

    def ner_ensemble(self, ner_results, t):
        """Apply ensemble voting across multiple model results.
        
        Groups entities by position key, averages their confidence scores,
        and filters by threshold. Returns the highest-scoring entity per position.
        """
        if not ner_results:
            return []
        
        ner_keys = defaultdict(lambda: [0, 0])
        for entity in ner_results:
            ner_keys[entity['key']][0] += 1
            ner_keys[entity['key']][1] += entity['score']

        passing_keys = [key for key, val in ner_keys.items() if val[1] / val[0] >= t]
        filter_ner_results = self.filter_ner_data(ner_results, passing_keys)
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
        """Process text with NER models using batched inference and ensemble voting.

        1. Split text into chunks (sentence-aware)
        2. Batch ALL chunks through language-specific pipeline (1 forward pass)
        3. Batch ALL chunks through multilingual pipeline (1 forward pass)
        4. Per-chunk: combine results, ensemble vote, anonymize
        """
        if not positional_tags:
            raise ValueError("Must provide at least one positional tag")

        ner_confidence_threshold = ner_confidence_threshold or 0.85

        # Select language-specific pipeline
        lang_key = language if language in LANG_MODEL_MAP and language != 'MULTILINGUAL' else 'ENGLISH'
        lang_pipe = self._get_pipeline(lang_key)
        multi_pipe = self._get_pipeline('MULTILINGUAL')

        # Serialized via lock — HF fast tokenizers use Rust RefCell internally,
        # so both tokenization (split_text) and pipeline inference must be
        # protected from concurrent access.
        with self._inference_lock:
            chunks = self.split_text(text, self.min_token_length, self.tokenizer)
            if not chunks:
                return text
            lang_batch = lang_pipe(chunks)
            multi_batch = multi_pipe(chunks)

        # Per-chunk: combine results, ensemble vote, anonymize
        ner_clean_text = []
        for i, chunk in enumerate(chunks):
            ner_results = []
            ner_results.extend(self.ner_data(lang_batch[i], positional_tags))
            ner_results.extend(self.ner_data(multi_batch[i], positional_tags))

            ensemble_results = self.ner_ensemble(ner_results, ner_confidence_threshold)

            if ensemble_results:
                ner_text = self.anonymize_text(chunk, ensemble_results).text
            else:
                ner_text = chunk

            ner_clean_text.append(ner_text)

        return ' '.join(ner_clean_text)
    
    def _token_count(self, text: str, tokenizer) -> int:
        """Count tokens in text."""
        return len(tokenizer(text).input_ids)

    def _merge_pieces(self, pieces: List[str], max_tokens: int, tokenizer,
                       separator: str = ' ') -> List[str]:
        """Greedily merge adjacent pieces into chunks that fit max_tokens.

        Uses *separator* to rejoin pieces that were split apart by a regex
        delimiter (which consumes the whitespace between them).
        """
        if not pieces:
            return []
        chunks = []
        current = pieces[0]
        for piece in pieces[1:]:
            merged = current + separator + piece
            if self._token_count(merged, tokenizer) <= max_tokens:
                current = merged
            else:
                if current:
                    chunks.append(current)
                current = piece
        if current:
            chunks.append(current)
        return chunks

    def split_text(self, text: str, max_tokens: int, tokenizer) -> List[str]:
        """Split text into token-bounded chunks using a delimiter hierarchy.

        Tries delimiters from coarsest (paragraph breaks) to finest (whitespace).
        For each delimiter level, splits oversized text, greedily merges small
        adjacent pieces, and recurses on any chunk that still exceeds the limit.
        Falls back to character-level splitting via token offsets as last resort.
        """
        if self._token_count(text, tokenizer) <= max_tokens:
            return [text]

        # Try each delimiter in priority order
        for delimiter in constants.CHUNK_DELIMITERS:
            pieces = delimiter.split(text)
            # Filter empty strings from split (e.g. leading/trailing delimiters)
            pieces = [p for p in pieces if p]
            if len(pieces) <= 1:
                continue

            # Greedily merge small adjacent pieces
            merged = self._merge_pieces(pieces, max_tokens, tokenizer)

            # Recurse on any chunk that still exceeds the limit
            result = []
            for chunk in merged:
                if self._token_count(chunk, tokenizer) > max_tokens:
                    result.extend(self.split_text(chunk, max_tokens, tokenizer))
                else:
                    result.append(chunk)
            return result

        # Last resort: character-level splitting via token offsets
        tokenized = tokenizer(text, return_offsets_mapping=True)
        offsets = tokenized.offset_mapping
        chunks = []
        last_end = 0
        current_tokens = 0

        for start, end in offsets:
            if current_tokens >= max_tokens:
                chunks.append(text[last_end:start])
                last_end = start
                current_tokens = 0
            current_tokens += 1

        if last_end < len(text):
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
        """Process multiple texts in batches."""
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
