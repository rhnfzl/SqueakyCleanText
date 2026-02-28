import math
import gc
import threading
from collections import defaultdict
import logging
from typing import Dict, List, Optional, Sequence, Union
from pathlib import Path

import onnxruntime as ort

from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import RecognizerResult

from typing import NamedTuple as _NamedTuple

from sct.utils import constants
from sct.utils.anonymization_map import AnonymizationMap
from sct.utils.onnx_pipeline import load_onnx_ner_model
from sct.config import DEFAULT_NER_MODELS, DEFAULT_NER_ENSEMBLE, NER_ENSEMBLE_DEFAULT_KEYS, LANG_KEYS


class AnonymizeResult(_NamedTuple):
    """Result of text anonymization — lightweight typed container."""
    text: str

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

    def __init__(self, cache_dir: Optional[Path] = None, device: Optional[str] = None,
                 model_names: Optional[Union[Dict[str, str], List[str]]] = None,
                 ner_backend: str = 'onnx',
                 gliner_config: Optional[Dict] = None,
                 torch_model_names: Optional[Dict[str, str]] = None,
                 ner_batch_size: int = 8,
                 ensemble_models: Optional[Dict] = None,
                 ensemble_default_keys: Optional[tuple] = None,
                 replacement_mode: str = 'placeholder',
                 synthetic_replacer=None):
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
            synthetic_replacer: Shared SyntheticReplacer instance (from TextCleaner).
        """
        self._ner_backend = ner_backend
        self._ner_batch_size = ner_batch_size
        self._replacement_mode = replacement_mode
        self._synthetic_replacer = synthetic_replacer
        self._gliner_pipe = None
        self._ensemble_models: Dict[str, tuple] = ensemble_models if ensemble_models is not None else DEFAULT_NER_ENSEMBLE
        self._ensemble_default_keys: tuple = (
            ensemble_default_keys
            if ensemble_default_keys is not None
            else NER_ENSEMBLE_DEFAULT_KEYS
        )

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
        # Per-model inference locks: enables parallel NER across different languages
        self._inference_locks: Dict[str, threading.Lock] = {}
        self._locks_lock = threading.Lock()  # Guards creation of per-model locks

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

        elif ner_backend == 'presidio_gliner':
            # Presidio handles everything: GLiNER recognition + anonymization
            self._model_names = {}
            self.min_token_length = None
            self.tokenizer = None
            if gliner_config:
                self._init_presidio_gliner(gliner_config)

        # --- GLiNER pipeline (for gliner/ensemble modes) ---
        if ner_backend in ('gliner', 'ensemble_onnx', 'ensemble_torch') and gliner_config:
            from sct.utils.gliner_adapter import GLiNERAdapter
            self._gliner_pipe = GLiNERAdapter(
                model_id=gliner_config['model'],
                variant=gliner_config.get('variant', 'gliner'),
                labels=gliner_config.get('labels'),
                threshold=gliner_config.get('threshold', 0.4),
                label_map=gliner_config.get('label_map'),
                label_descriptions=gliner_config.get('label_descriptions'),
                device=self.device,
                onnx=gliner_config.get('onnx', False),
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

    def _get_lock(self, lang_key: str) -> threading.Lock:
        """Get or create a per-model inference lock (double-checked locking)."""
        if lang_key not in self._inference_locks:
            with self._locks_lock:
                if lang_key not in self._inference_locks:
                    self._inference_locks[lang_key] = threading.Lock()
        return self._inference_locks[lang_key]

    def _init_presidio_gliner(self, gliner_config):
        """Initialize Presidio with GLiNERRecognizer (experimental beta backend)."""
        try:
            from presidio_analyzer import AnalyzerEngine  # noqa: S404
        except ImportError:
            raise ImportError(
                "presidio-analyzer is required for presidio_gliner backend. "
                "Install with: pip install presidio-analyzer"
            )
        self._analyzer = AnalyzerEngine()
        try:
            from presidio_analyzer.predefined_recognizers import GLiNERRecognizer  # noqa: S404
            gliner_recognizer = GLiNERRecognizer(
                model_name=gliner_config['model'],
                supported_entities=[
                    label.upper()
                    for label in gliner_config.get('labels', ['person', 'organization', 'location'])
                ],
                threshold=gliner_config.get('threshold', 0.4),
            )
            self._analyzer.registry.add_recognizer(gliner_recognizer)
        except ImportError:
            raise ImportError(
                "GLiNERRecognizer requires presidio-analyzer with GLiNER support. "
                "Install with: pip install presidio-analyzer gliner"
            )

    def _get_ensemble_keys(self, language: str) -> tuple:
        """Return ordered model keys to run for the given language."""
        return self._ensemble_models.get(language, self._ensemble_default_keys)

    def load_language(self, lang_key: str) -> None:
        """Pre-load NER model for the given language key (stable public API)."""
        self._ensure_loaded(lang_key)

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
            # Reuse existing pipeline/tokenizer if another key already loaded this model
            shared_pipe = next(
                (p for k, p in self._pipelines.items()
                 if self._model_names.get(k) == model_name), None
            )
            if shared_pipe is not None:
                self._pipelines[lang_key] = shared_pipe
                shared_tok = next(
                    (t for k, t in self._tokenizers.items()
                     if self._model_names.get(k) == model_name), None
                )
                if shared_tok is not None:
                    self._tokenizers[lang_key] = shared_tok
                return
            try:
                pipeline_obj: object
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

    def anonymize_text(self, text, filtered_data, replacement_mode='placeholder',
                       anon_map=None):
        """Anonymize text. Supports placeholder, synthetic, or reversible replacement."""
        if replacement_mode == 'reversible':
            return self._anonymize_reversible(text, filtered_data, anon_map)

        if replacement_mode == 'synthetic':
            result_text = self._synthetic_replacer.generate_for_entities(text, filtered_data)
            return AnonymizeResult(text=result_text)

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
            return AnonymizeResult(text=text)
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

            engine_result = self.engine.anonymize(text=text, analyzer_results=analyzer_result)
            return AnonymizeResult(text=engine_result.text)

    def _anonymize_reversible(self, text, filtered_data, anon_map=None):
        """Replace entities with indexed placeholders and populate the map.

        Uses right-to-left replacement to preserve character offsets.
        Returns an ``AnonymizeResult`` with the anonymized ``.text``.
        """
        if anon_map is None:
            anon_map = AnonymizationMap()

        sorted_data = sorted(filtered_data, key=lambda x: x['start'], reverse=True)
        for item in sorted_data:
            tag = ENTITY_TYPE_MAP.get(item['entity_group'], item['entity_group'])
            placeholder = anon_map.next_placeholder(tag)
            original = text[item['start']:item['end']]
            anon_map.add(
                placeholder=placeholder,
                original=original,
                entity_type=tag,
                start=item['start'],
                end=item['end'],
            )
            text = text[:item['start']] + placeholder + text[item['end']:]

        return AnonymizeResult(text=text)

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

    def _simple_chunk(self, text: str, max_tokens: int = 384) -> List[str]:
        """Split text into token-aware chunks for GLiNER.

        Used for GLiNER-only mode where no ONNX/torch tokenizer is available.
        Estimates token count at ~4 chars/token (safe average for European scripts).
        """
        max_chars = max_tokens * 2  # conservative: CJK=1, Arabic=2-3, Latin=4 chars/token
        if len(text) <= max_chars:
            return [text]

        for delimiter in constants.CHUNK_DELIMITERS:
            pieces = [p for p in delimiter.split(text) if p]
            if len(pieces) <= 1:
                continue
            chunks: List[str] = []
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
        chunks = []
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
        positional_tags: Optional[Sequence[str]] = None,
        ner_confidence_threshold: Optional[float] = None,
        language: Optional[str] = None,
        anon_map: Optional['AnonymizationMap'] = None,
    ) -> str:
        """Process text with NER models using the configured backend.

        Routes to ONNX, Torch, GLiNER, or ensemble depending on self._ner_backend.

        When *anon_map* is provided (reversible mode), entity-to-placeholder
        mappings are recorded in it for later deanonymization.
        """
        if not positional_tags:
            raise ValueError("Must provide at least one positional tag")

        ner_confidence_threshold = ner_confidence_threshold or 0.85

        # --- Presidio GLiNER backend: delegates everything to Presidio ---
        if self._ner_backend == 'presidio_gliner':
            analyzer_results = self._analyzer.analyze(text=text, language='en')
            result = self.engine.anonymize(text=text, analyzer_results=analyzer_results)
            return result.text

        # --- Chunking --- (lock-free: HF fast tokenizer Rust backend is thread-safe)
        if self.tokenizer is not None:
            chunks = self.split_text(text, self.min_token_length, self.tokenizer)
        else:
            # Use GLiNER adapter's dynamic context window when available
            gliner_max = self._gliner_pipe.max_context_length if self._gliner_pipe else 384
            chunks = self._simple_chunk(text, max_tokens=int(gliner_max * 0.9))

        if not chunks:
            return text

        # Pre-compute GLiNER-only tag set (constant across chunks)
        gliner_all_tags = None
        if self._ner_backend == 'gliner' and self._gliner_pipe:
            gliner_all_tags = set(positional_tags)
            gliner_all_tags.update(self._gliner_pipe.label_map.values())
            gliner_all_tags.update(
                label.upper() for label in self._gliner_pipe.labels
                if label not in self._gliner_pipe.label_map
            )

        # --- Inference + ensemble per chunk ---
        ner_clean_text = []
        for chunk in chunks:
            ner_results = []

            # Primary backend: ONNX or Torch — run each key in the ensemble
            if self._ner_backend in ('onnx', 'torch', 'ensemble_onnx', 'ensemble_torch'):
                for key in self._get_ensemble_keys(language or 'MULTILINGUAL'):
                    self._ensure_loaded(key)
                    model_name = self._model_names.get(key, key)
                    model_lock = self._get_lock(model_name)
                    with model_lock:
                        batch = self._pipelines[key]([chunk])
                    ner_results.extend(self.ner_data(batch[0], positional_tags))

            # GLiNER backend
            if self._ner_backend in ('gliner', 'ensemble_onnx', 'ensemble_torch') \
                    and self._gliner_pipe:
                gliner_lock = self._get_lock('gliner')
                with gliner_lock:
                    gliner_batch = self._gliner_pipe([chunk])
                if gliner_all_tags is not None:
                    ner_results.extend(self.ner_data(gliner_batch[0], gliner_all_tags))
                else:
                    # Ensemble: filter to positional_tags only
                    ner_results.extend(self.ner_data(gliner_batch[0], positional_tags))

            # Ensemble vote + anonymize
            ensemble_results = self.ner_ensemble(ner_results, ner_confidence_threshold)

            if ensemble_results:
                result = self.anonymize_text(
                    chunk, ensemble_results, self._replacement_mode,
                    anon_map=anon_map,
                )
                ner_text = result.text
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

        Tries sentence boundaries first (preferred: respects sentence integrity),
        then falls back to CHUNK_DELIMITERS from coarsest (paragraph) to finest
        (whitespace). Recurses on oversized chunks. Falls back to character-level
        splitting via token offsets as last resort.
        """
        if self._token_count(text, tokenizer) <= max_tokens:
            return [text]

        # Prefer sentence-boundary splits to avoid fragmenting entities mid-sentence
        sentence_pieces = constants.SENTENCE_BOUNDARY_PATTERN.split(text)
        sentence_pieces = [p for p in sentence_pieces if p]
        if len(sentence_pieces) > 1:
            merged = self._merge_pieces(sentence_pieces, max_tokens, tokenizer)
            result = []
            for chunk in merged:
                if self._token_count(chunk, tokenizer) > max_tokens:
                    result.extend(self.split_text(chunk, max_tokens, tokenizer))
                else:
                    result.append(chunk)
            return result

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
        batch_size: Optional[int] = None,
        positional_tags: Optional[List[str]] = None,
        ner_confidence_threshold: Optional[float] = None,
        language: Optional[str] = None
    ) -> List[str]:
        """Process multiple texts in batches."""
        batch_size = batch_size if batch_size is not None else self._ner_batch_size
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
        if hasattr(self, '_pipelines'):
            for pipe in self._pipelines.values():
                if hasattr(pipe, 'cleanup'):
                    pipe.cleanup()  # torch
                elif hasattr(pipe, '_session'):
                    del pipe._session  # ONNX
        gc.collect()
