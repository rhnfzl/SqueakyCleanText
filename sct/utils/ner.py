import math
import gc
import threading
from collections import defaultdict
import logging
from typing import Dict, List, Optional, Union
from pathlib import Path

import onnxruntime as ort

from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import RecognizerResult

from sct.utils import constants
from sct.utils.onnx_pipeline import load_onnx_ner_model
from sct.config import DEFAULT_NER_MODELS, LANG_KEYS

ort.set_default_logger_severity(3)  # Silence ONNX Runtime warnings

logger = logging.getLogger(__name__)

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
    """NER processor with lazy model loading, multi-backend support, and ensemble voting.

    Backends:
        onnx           — ONNX Runtime (default, torch-free)
        torch          — PyTorch/Transformers pipeline
        gliner         — GLiNER zero-shot NER with custom entity labels
        ensemble_onnx  — ONNX + GLiNER combined via ensemble voting
        ensemble_torch — Torch + GLiNER combined via ensemble voting
    """

    def __init__(self, cache_dir: Optional[Path] = None, device: str = None,
                 model_names: Optional[Union[Dict[str, str], List[str]]] = None,
                 ner_backend: str = 'onnx',
                 gliner_config: Optional[Dict] = None,
                 torch_model_names: Optional[Dict[str, str]] = None):
        """Initialize NER processor.

        Args:
            cache_dir: Optional directory for caching models
            device: Device for inference ('cuda' or 'cpu'). Auto-detects if None.
            model_names: Language-keyed dict of ONNX model repo IDs (preferred),
                or positional list for backward compat. Uses DEFAULT_NER_MODELS if None.
            ner_backend: Backend selection ('onnx', 'torch', 'gliner',
                'ensemble_onnx', 'ensemble_torch').
            gliner_config: Dict with keys: model, variant, labels, threshold, label_map.
                Required when ner_backend involves GLiNER.
            torch_model_names: Language-keyed dict of PyTorch model repo IDs.
                Required when ner_backend involves torch.
        """
        self._ner_backend = ner_backend
        self._gliner_pipe = None

        # Device detection
        if device:
            self.device = device
        elif ner_backend in ('torch', 'ensemble_torch'):
            try:
                import torch  # noqa: S404
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            except ImportError:
                self.device = 'cpu'
        elif 'CUDAExecutionProvider' in ort.get_available_providers():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        logger.info("Using device: %s (backend: %s)", self.device, ner_backend)

        self.engine = AnonymizerEngine()
        self._cache_args = {"cache_dir": str(cache_dir)} if cache_dir else {}

        # Common state for all backends
        self._pipelines: Dict = {}
        self._tokenizers: Dict = {}
        self._load_lock = threading.Lock()
        # Serialize inference across threads (tokenizers Rust internals, model access)
        self._inference_lock = threading.Lock()

        # --- Backend-specific initialization ---

        if ner_backend in ('onnx', 'ensemble_onnx'):
            # ONNX backend: build model mapping from dict, list, or defaults
            if model_names is None:
                self._model_names = dict(DEFAULT_NER_MODELS)
            elif isinstance(model_names, dict):
                self._model_names = dict(model_names)
            else:
                self._model_names = dict(zip(LANG_KEYS, model_names))

            self._ensure_loaded('ENGLISH')
            self._ensure_loaded('MULTILINGUAL')
            self._init_tokenizer_props()

        elif ner_backend in ('torch', 'ensemble_torch'):
            # Torch backend
            if torch_model_names is None:
                from sct.config import DEFAULT_TORCH_NER_MODELS
                self._model_names = dict(DEFAULT_TORCH_NER_MODELS)
            else:
                self._model_names = dict(torch_model_names)

            self._ensure_loaded('ENGLISH')
            self._ensure_loaded('MULTILINGUAL')
            self._init_tokenizer_props()

        elif ner_backend == 'gliner':
            # GLiNER-only: no token-classification models needed
            self._model_names = {}
            self.min_token_length = None
            self.tokenizer = None

        # --- GLiNER pipeline (for gliner/ensemble modes) ---
        if ner_backend in ('gliner', 'ensemble_onnx', 'ensemble_torch') and gliner_config:
            from sct.utils.gliner_adapter import GLiNERAdapter
            self._gliner_pipe = GLiNERAdapter(
                model_id=gliner_config['model'],
                variant=gliner_config.get('variant', 'gliner'),
                labels=gliner_config.get('labels'),
                threshold=gliner_config.get('threshold', 0.4),
                label_map=gliner_config.get('label_map'),
                device=self.device,
            )

    def _init_tokenizer_props(self):
        """Set min_token_length and tokenizer from loaded ENGLISH/MULTILINGUAL models."""
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
        """Get model name for a language via direct dict lookup."""
        model = self._model_names.get(lang_key)
        if model is None:
            raise ModelLoadError(f"No model configured for language: {lang_key}")
        return model

    def _ensure_loaded(self, lang_key: str) -> None:
        """Lazily load a model and pipeline for the given language.

        Dispatches to the appropriate loader based on the active backend.
        """
        if lang_key in self._pipelines:
            return
        with self._load_lock:
            if lang_key in self._pipelines:
                return
            model_name = self._get_model_name(lang_key)
            logger.info("Loading model for %s: %s", lang_key, model_name)
            try:
                if self._ner_backend in ('torch', 'ensemble_torch'):
                    from sct.utils.torch_pipeline import load_torch_ner_model
                    pipeline_obj, tok = load_torch_ner_model(
                        model_name, device=self.device,
                        cache_dir=self._cache_args.get("cache_dir"),
                    )
                else:
                    pipeline_obj, tok = load_onnx_ner_model(
                        model_name,
                        device=self.device,
                        cache_dir=self._cache_args.get("cache_dir"),
                    )
                self._pipelines[lang_key] = pipeline_obj
                self._tokenizers[lang_key] = tok
            except Exception as e:
                logger.error("Failed to load model for %s: %s", lang_key, e)
                raise ModelLoadError(
                    f"Model loading failed for {lang_key} ({model_name}): {e}"
                )

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
        """Anonymize text. Uses Presidio for standard tags, direct replacement for custom."""
        has_custom = any(
            items['entity_group'] not in ENTITY_TYPE_MAP
            for items in filtered_data
        )

        if has_custom:
            # Mixed labels: right-to-left string replacement (preserves offsets)
            sorted_data = sorted(filtered_data, key=lambda x: x['start'], reverse=True)
            for items in sorted_data:
                tag = ENTITY_TYPE_MAP.get(items['entity_group'], items['entity_group'])
                text = text[:items['start']] + f"<{tag}>" + text[items['end']:]
            return type('AnonymizeResult', (), {'text': text})()
        else:
            # Standard entities only: use Presidio (existing behavior)
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

    def _simple_chunk(self, text: str, max_chars: int = 1500) -> List[str]:
        """Split text at sentence boundaries by character count.

        Used for GLiNER-only mode where no ONNX/torch tokenizer is available.
        """
        if len(text) <= max_chars:
            return [text]

        for delimiter in constants.CHUNK_DELIMITERS:
            pieces = [p for p in delimiter.split(text) if p]
            if len(pieces) <= 1:
                continue
            chunks = []
            current = pieces[0]
            for piece in pieces[1:]:
                merged = current + ' ' + piece
                if len(merged) <= max_chars:
                    current = merged
                else:
                    if current:
                        chunks.append(current)
                    current = piece
            if current:
                chunks.append(current)
            return chunks

        # Last resort: split on whitespace
        words = text.split()
        chunks: List[str] = []
        current = ''
        for word in words:
            test = (current + ' ' + word).strip()
            if len(test) > max_chars and current:
                chunks.append(current)
                current = word
            else:
                current = test
        if current:
            chunks.append(current)
        return chunks or [text]

    def ner_process(
        self,
        text: str,
        positional_tags: List[str] = None,
        ner_confidence_threshold: float = None,
        language: str = None
    ) -> str:
        """Process text with NER models using the configured backend.

        Routes to ONNX, Torch, GLiNER, or ensemble depending on self._ner_backend.
        """
        if not positional_tags:
            raise ValueError("Must provide at least one positional tag")

        ner_confidence_threshold = ner_confidence_threshold or 0.85

        # --- Chunking ---
        if self.tokenizer is not None:
            with self._inference_lock:
                chunks = self.split_text(text, self.min_token_length, self.tokenizer)
        else:
            chunks = self._simple_chunk(text, max_chars=1500)

        if not chunks:
            return text

        # --- Inference + ensemble per chunk ---
        with self._inference_lock:
            ner_clean_text = []
            for chunk in chunks:
                ner_results = []

                # Primary backend: ONNX or Torch (lang-specific + multilingual)
                if self._ner_backend in ('onnx', 'torch', 'ensemble_onnx', 'ensemble_torch'):
                    lang_key = (language if language in self._model_names
                                and language != 'MULTILINGUAL' else 'MULTILINGUAL')
                    lang_batch = self._get_pipeline(lang_key)([chunk])
                    multi_batch = self._get_pipeline('MULTILINGUAL')([chunk])
                    ner_results.extend(self.ner_data(lang_batch[0], positional_tags))
                    ner_results.extend(self.ner_data(multi_batch[0], positional_tags))

                # GLiNER backend
                if self._ner_backend in ('gliner', 'ensemble_onnx', 'ensemble_torch') \
                        and self._gliner_pipe:
                    gliner_batch = self._gliner_pipe([chunk])
                    if self._ner_backend == 'gliner':
                        # GLiNER-only: include all mapped entity types
                        all_tags = set(positional_tags)
                        all_tags.update(self._gliner_pipe.label_map.values())
                        all_tags.update(
                            label.upper() for label in self._gliner_pipe.labels
                            if label not in self._gliner_pipe.label_map
                        )
                        ner_results.extend(self.ner_data(gliner_batch[0], all_tags))
                    else:
                        # Ensemble: filter to positional_tags only
                        ner_results.extend(self.ner_data(gliner_batch[0], positional_tags))

                # Ensemble vote + anonymize
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
        """Cleanup resources."""
        if hasattr(self, '_ner_backend') and self._ner_backend in ('torch', 'ensemble_torch'):
            for pipe in self._pipelines.values():
                if hasattr(pipe, 'cleanup'):
                    pipe.cleanup()
        gc.collect()
