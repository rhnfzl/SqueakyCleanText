import math
from bisect import bisect_left, bisect_right
import gc
import re
import threading
from collections import defaultdict
import logging
from typing import Dict, List, Optional, Sequence, Union
from pathlib import Path

import onnxruntime as ort

from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import RecognizerResult

from typing import NamedTuple as _NamedTuple

from sct.utils import constants, resources
from sct.utils.anonymization_map import AnonymizationMap
from sct.utils.onnx_pipeline import load_onnx_ner_model
from sct.config import DEFAULT_NER_MODELS, DEFAULT_NER_ENSEMBLE, NER_ENSEMBLE_DEFAULT_KEYS, LANG_KEYS
from sct.privacy import (
    DetectedEntity,
    EntityFinding,
    NERProcessingResult,
    PrivacyPolicy,
)


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

    _ner_batch_size = 8  # chunks per model call; __init__ sets the configured value
    _ner_chunk_overlap = 0  # tokens of context each window adds per side; __init__ sets it

    def __init__(self, cache_dir: Optional[Path] = None, device: Optional[str] = None,
                 model_names: Optional[Union[Dict[str, str], List[str]]] = None,
                 model_revisions: Optional[Dict[str, str]] = None,
                 ner_backend: str = 'onnx',
                 gliner_config: Optional[Dict] = None,
                 torch_model_names: Optional[Dict[str, str]] = None,
                 torch_model_revisions: Optional[Dict[str, str]] = None,
                 ner_batch_size: int = 8,
                 ensemble_models: Optional[Dict] = None,
                 ensemble_default_keys: Optional[tuple] = None,
                 replacement_mode: str = 'placeholder',
                 synthetic_replacer=None,
                 onnx_intra_op_threads: Optional[int] = None,
                 onnx_inter_op_threads: Optional[int] = None,
                 ner_chunk_overlap: int = 0):
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
            onnx_intra_op_threads, onnx_inter_op_threads: ONNX Runtime session
                thread counts; None keeps the ONNX Runtime default.
            ner_chunk_overlap: Tokens of neighbouring text each NER window adds on
                both sides of its chunk; 0 keeps the 0.7.1 chunks.
        """
        self._ner_backend = ner_backend
        self._ner_batch_size = ner_batch_size
        self._ner_chunk_overlap = ner_chunk_overlap
        self._replacement_mode = replacement_mode
        self._synthetic_replacer = synthetic_replacer
        self._model_revisions = dict(model_revisions or {})
        self._torch_model_revisions = dict(torch_model_revisions or {})
        self._onnx_intra_op_threads = onnx_intra_op_threads
        self._onnx_inter_op_threads = onnx_inter_op_threads
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
        # Per-model inference locks for torch models and GLiNER. ONNX runs lock-free:
        # InferenceSession.run and the Rust tokenizer are thread-safe.
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
                revision=gliner_config.get('revision'),
                variant=gliner_config.get('variant', 'gliner'),
                labels=gliner_config.get('labels'),
                threshold=gliner_config.get('threshold', 0.4),
                label_map=gliner_config.get('label_map'),
                label_descriptions=gliner_config.get('label_descriptions'),
                device=self.device,
                onnx=gliner_config.get('onnx', False),
            )

        if ner_backend != 'presidio_gliner' and ner_chunk_overlap:
            budget = self._chunk_budget()
            if not 0 < 2 * ner_chunk_overlap < budget // 2:
                raise ValueError(
                    f"ner_chunk_overlap must be >= 0 and 2 * overlap < {budget // 2} "
                    f"(half the {budget}-token window), got {ner_chunk_overlap}"
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

    def _model_key(self, lang_key: str) -> tuple:
        """(model name, revision) for a language key; same key = same loaded model."""
        revisions = (self._torch_model_revisions
                     if self._ner_backend in ('torch', 'ensemble_torch')
                     else self._model_revisions)
        return self._model_names.get(lang_key, lang_key), revisions.get(lang_key)

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
            # Reuse existing pipeline/tokenizer if another key already loaded this model + revision
            model_key = self._model_key(lang_key)
            shared_pipe = next(
                (p for k, p in self._pipelines.items()
                 if self._model_key(k) == model_key), None
            )
            if shared_pipe is not None:
                self._pipelines[lang_key] = shared_pipe
                shared_tok = next(
                    (t for k, t in self._tokenizers.items()
                     if self._model_key(k) == model_key), None
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
                        revision=self._torch_model_revisions.get(lang_key),
                    )
                else:
                    pipeline_obj, tok = load_onnx_ner_model(
                        model_name,
                        device=self.device,
                        cache_dir=self._cache_args.get("cache_dir"),
                        revision=self._model_revisions.get(lang_key),
                        intra_op_threads=self._onnx_intra_op_threads,
                        inter_op_threads=self._onnx_inter_op_threads,
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
        """Split text into chunks of at most ``max_tokens * 2`` characters for GLiNER.

        Used for GLiNER-only mode where no ONNX/torch tokenizer is available.
        Every chunk fits the budget: oversized pieces are split again with
        finer delimiters, and runs without any delimiter are cut hard.
        """
        max_chars = max_tokens * 2  # conservative: CJK=1, Arabic=2-3, Latin=4 chars/token
        if len(text) <= max_chars:
            return [text]

        for delimiter in constants.CHUNK_DELIMITERS:
            pieces = self._split_pieces(delimiter, text)
            if len(pieces) <= 1:
                continue
            chunks: List[str] = []
            for chunk in self._merge_pieces(pieces, lambda c: len(c) <= max_chars):
                chunks.extend(self._simple_chunk(chunk, max_tokens))
            return chunks

        # shortcut: hard cut can split an entity; ner_chunk_overlap > 0 covers it
        return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]

    @staticmethod
    def _split_pieces(pattern, text: str) -> List[tuple]:
        """Split *text* on *pattern* into ``(separator_before, piece)`` pairs.

        Keeping the matched separator lets chunks be rejoined exactly as in
        the source (no invented spaces between CJK sentences). Empty pieces
        are dropped; their separators fold into the next piece's separator.
        """
        pieces = []
        last = 0
        separator = ''
        for match in pattern.finditer(text):
            if match.start() > last:
                pieces.append((separator, text[last:match.start()]))
                separator = ''
            separator += match.group()
            last = max(last, match.end())
        if last < len(text):
            pieces.append((separator, text[last:]))
        return pieces

    @staticmethod
    def _align_chunk(
        source: str,
        chunk: str,
        search_start: int,
    ) -> tuple[str, int, list[int]]:
        """Locate a whitespace-normalized chunk and map its offsets to source."""
        # Chunks are normally exact source slices; a greedy \s+ match would eat
        # whitespace that belongs to the next chunk (hard cuts, blank chunks).
        start = source.find(chunk, search_start)
        if start >= 0:
            return chunk, start, list(range(len(chunk) + 1))
        parts = re.split(r'(\s+)', chunk)
        pattern = ''.join(
            r'\s+' if part.isspace() else re.escape(part)
            for part in parts if part
        )
        match = re.compile(pattern).search(source, search_start)
        if match is None:
            raise RuntimeError("Could not align NER chunk to its source text")

        source_chunk = match.group()
        offsets = [0] * (len(chunk) + 1)
        chunk_index = 0
        source_index = 0
        while chunk_index < len(chunk):
            if chunk[chunk_index].isspace():
                chunk_end = chunk_index
                while chunk_end < len(chunk) and chunk[chunk_end].isspace():
                    chunk_end += 1
                source_end = source_index
                while (
                    source_end < len(source_chunk)
                    and source_chunk[source_end].isspace()
                ):
                    source_end += 1
                chunk_width = chunk_end - chunk_index
                source_width = source_end - source_index
                for offset in range(chunk_width + 1):
                    offsets[chunk_index + offset] = (
                        source_index
                        + (offset * source_width // chunk_width)
                    )
                chunk_index = chunk_end
                source_index = source_end
            else:
                offsets[chunk_index] = source_index
                chunk_index += 1
                source_index += 1
        offsets[len(chunk)] = len(source_chunk)
        return source_chunk, match.start(), offsets

    def ner_process_detailed(
        self,
        text: str,
        positional_tags: Optional[Sequence[str]] = None,
        ner_confidence_threshold: Optional[float] = None,
        language: Optional[str] = None,
        anon_map: Optional['AnonymizationMap'] = None,
        privacy_policy: Optional[PrivacyPolicy] = None,
    ) -> NERProcessingResult:
        """Process text and return transformed text with entity findings.

        Routes to ONNX, Torch, GLiNER, or ensemble depending on self._ner_backend.

        When *anon_map* is provided (reversible mode), entity-to-placeholder
        mappings are recorded in it for later deanonymization.

        Finding and map-entry offsets index *text* as passed in, i.e. the text
        after the regex replacement stages (post-regex), not the raw input.
        """
        if not positional_tags:
            raise ValueError("Must provide at least one positional tag")

        if ner_confidence_threshold is None:
            ner_confidence_threshold = 0.85

        # --- Presidio GLiNER backend: delegates everything to Presidio ---
        if self._ner_backend == 'presidio_gliner':
            analyzer_language = resources.language_to_iso(language) if language else 'en'
            analyzer_results = self._analyzer.analyze(
                text=text,
                language=analyzer_language,
            )
            entities = tuple(
                DetectedEntity(
                    entity_type=result.entity_type,
                    score=result.score,
                    text=text[result.start:result.end],
                    start=result.start,
                    end=result.end,
                )
                for result in analyzer_results
            )
            if privacy_policy is not None:
                findings = privacy_policy.evaluate(entities)
                return NERProcessingResult(
                    text=privacy_policy.transform(text, findings),
                    findings=findings,
                )
            result = self.engine.anonymize(text=text, analyzer_results=analyzer_results)
            findings = tuple(
                EntityFinding(entity, self._replacement_mode, 'detector-threshold')
                for entity in entities
            )
            return NERProcessingResult(text=result.text, findings=findings)

        windows = self._plan_windows(text)
        if not windows:
            return NERProcessingResult(text=text, findings=())
        window_results = self._detect(
            [window[0] for window in windows], [language] * len(windows), positional_tags,
        )
        return self._finalize(
            text, windows, window_results, ner_confidence_threshold, anon_map, privacy_policy,
        )

    def ner_process_batch_detailed(
        self,
        texts: Sequence[str],
        languages: Sequence[Optional[str]],
        positional_tags: Optional[Sequence[str]] = None,
        ner_confidence_threshold: Optional[float] = None,
        anon_maps: Optional[Sequence[Optional['AnonymizationMap']]] = None,
        privacy_policy: Optional[PrivacyPolicy] = None,
        before_finalize=None,
        max_workers: int = 1,
    ) -> List[NERProcessingResult]:
        """``ner_process_detailed`` for many documents, batching model calls across them.

        Windows of all documents go through each model in calls of up to
        ``ner_batch_size`` windows; replacement then runs per document exactly as
        in ``ner_process_detailed``. *before_finalize(i)* runs just before
        document *i* is replaced (TextCleaner resets the synthetic cache there).
        *max_workers* ONNX batches run at once.
        """
        if not positional_tags:
            raise ValueError("Must provide at least one positional tag")
        if ner_confidence_threshold is None:
            ner_confidence_threshold = 0.85
        anon_maps = anon_maps if anon_maps is not None else [None] * len(texts)
        if self._ner_backend == 'presidio_gliner':
            results = []
            for i, text in enumerate(texts):
                if before_finalize is not None:
                    before_finalize(i)
                results.append(self.ner_process_detailed(
                    text, positional_tags, ner_confidence_threshold, languages[i],
                    anon_maps[i], privacy_policy,
                ))
            return results

        doc_windows = [self._plan_windows(text) for text in texts]
        flat_windows = [window[0] for windows in doc_windows for window in windows]
        flat_languages = [
            languages[i] for i, windows in enumerate(doc_windows) for _ in windows
        ]
        flat_results = self._detect(flat_windows, flat_languages, positional_tags, max_workers)

        results = []
        offset = 0
        for i, (text, windows) in enumerate(zip(texts, doc_windows)):
            window_results = flat_results[offset:offset + len(windows)]
            offset += len(windows)
            if before_finalize is not None:
                before_finalize(i)
            if not windows:
                results.append(NERProcessingResult(text=text, findings=()))
                continue
            results.append(self._finalize(
                text, windows, window_results, ner_confidence_threshold,
                anon_maps[i], privacy_policy,
            ))
        return results

    def _chunk_budget(self) -> int:
        """Token budget of one NER window (GLiNER-only: of its context window)."""
        if self.tokenizer is not None:
            return self.min_token_length
        # Use GLiNER adapter's dynamic context window when available
        gliner_max = self._gliner_pipe.max_context_length if self._gliner_pipe else 384
        return int(gliner_max * 0.9)

    def _plan_windows(self, text: str) -> List[tuple]:
        """Split *text* into model windows ``(window, win_start, offsets, own_start, own_end)``.

        Core chunks come from the chunker with ``2 * ner_chunk_overlap`` tokens
        less budget; each window adds up to ``ner_chunk_overlap`` tokens of the
        neighbouring text on both sides (GLiNER-only: 2 chars per token, as in
        ``_simple_chunk``). ``win_start + offsets[i]`` is the document position of
        window position *i*. A span belongs to the window whose core owns its
        start (``own_start <= start < own_end``); copies seen by other windows are
        dropped. With overlap 0 windows are the 0.7.1 chunks and own everything.
        """
        overlap = self._ner_chunk_overlap
        budget = self._chunk_budget() - 2 * overlap
        # lock-free: HF fast tokenizer Rust backend is thread-safe
        if self.tokenizer is not None:
            chunks = self.split_text(text, budget, self.tokenizer)
        else:
            chunks = self._simple_chunk(text, max_tokens=budget)

        cores = []
        search_start = 0
        for chunk in chunks:
            source_chunk, start, offsets = self._align_chunk(text, chunk, search_start)
            search_start = start + len(source_chunk)
            cores.append((chunk, start, start + len(source_chunk), offsets))
        if not overlap:
            return [(chunk, start, offsets, 0, len(text) + 1) for chunk, start, _, offsets in cores]

        if self.tokenizer is not None:
            tokens = [
                (start, end)
                for start, end in self.tokenizer(text, return_offsets_mapping=True).offset_mapping
                if end > start
            ]
            starts = [start for start, _ in tokens]
            ends = [end for _, end in tokens]

        windows = []
        for k, (_, core_start, core_end, _) in enumerate(cores):
            if self.tokenizer is not None:
                before = bisect_right(ends, core_start)  # tokens ending at or before the core
                after = bisect_left(starts, core_end)  # first token starting at or after its end
                win_start = min(core_start, starts[max(0, before - overlap)]) if before else core_start
                win_end = (
                    max(core_end, ends[min(len(ends), after + overlap) - 1])
                    if after < len(ends) else core_end
                )
            else:
                win_start = max(0, core_start - overlap * 2)
                win_end = min(len(text), core_end + overlap * 2)
            windows.append((
                text[win_start:win_end],
                win_start,
                range(win_end - win_start + 1),
                core_start if k else 0,
                cores[k + 1][1] if k + 1 < len(cores) else len(text) + 1,
            ))
        return windows

    def _detect(
        self,
        chunks: List[str],
        languages: List[Optional[str]],
        positional_tags: Sequence[str],
        max_workers: int = 1,
    ) -> List[list]:
        """Run every model on its chunks; return the raw NER results per chunk.

        The only place that calls model pipelines. Each model gets one call over
        all chunks it applies to, run ``ner_batch_size`` chunks per forward pass
        (ONNX: up to *max_workers* forward passes at once).
        Per chunk, results keep the order of that language's ensemble keys,
        then GLiNER, as in the one-chunk-at-a-time loop.
        """
        per_key: List[Dict[str, list]] = [{} for _ in chunks]
        chunk_keys: List[tuple] = [()] * len(chunks)

        # Primary backend: ONNX or Torch — run each key in the ensemble
        if self._ner_backend in ('onnx', 'torch', 'ensemble_onnx', 'ensemble_torch'):
            chunk_keys = [
                self._get_ensemble_keys(language or 'MULTILINGUAL') for language in languages
            ]
            for key in dict.fromkeys(k for keys in chunk_keys for k in keys):
                indices = [i for i, keys in enumerate(chunk_keys) if key in keys]
                self._ensure_loaded(key)
                key_chunks = [chunks[i] for i in indices]
                if self._ner_backend in ('onnx', 'ensemble_onnx'):
                    batch = self._pipelines[key](
                        key_chunks, batch_size=self._ner_batch_size, max_workers=max_workers,
                    )
                else:
                    with self._get_lock(self._model_key(key)):
                        batch = self._pipelines[key](key_chunks, batch_size=self._ner_batch_size)
                for i, data in zip(indices, batch):
                    per_key[i][key] = self.ner_data(data, positional_tags)

        results = [
            [item for key in keys for item in per_key[i][key]]
            for i, keys in enumerate(chunk_keys)
        ]

        # GLiNER backend
        if self._ner_backend in ('gliner', 'ensemble_onnx', 'ensemble_torch') \
                and self._gliner_pipe and chunks:
            # GLiNER-only keeps every label it was given; ensembles keep positional_tags.
            gliner_tags = set(positional_tags)
            if self._ner_backend == 'gliner':
                gliner_tags.update(self._gliner_pipe.label_map.values())
                gliner_tags.update(
                    label.upper() for label in self._gliner_pipe.labels
                    if label not in self._gliner_pipe.label_map
                )
            with self._get_lock('gliner'):
                gliner_batch = self._gliner_pipe(chunks, batch_size=self._ner_batch_size)
            for i, data in enumerate(gliner_batch):
                results[i].extend(self.ner_data(data, gliner_tags))
        return results

    def _finalize(
        self,
        text: str,
        windows: List[tuple],
        window_results: List[list],
        ner_confidence_threshold: float,
        anon_map: Optional['AnonymizationMap'],
        privacy_policy: Optional[PrivacyPolicy],
    ) -> NERProcessingResult:
        """Merge window spans in document coordinates, vote, and replace once."""
        spans = []
        for (_, win_start, offsets, own_start, own_end), ner_results in zip(windows, window_results):
            for item in ner_results:
                start = win_start + offsets[item['start']]
                if own_start <= start < own_end:
                    end = win_start + offsets[item['end']]
                    spans.append({
                        **item, 'start': start, 'end': end,
                        'word': text[start:end], 'key': f"{start}:{end}",
                    })

        ensemble_results = self.ner_ensemble(spans, ner_confidence_threshold)
        if not ensemble_results:
            return NERProcessingResult(text=text, findings=())
        entities = tuple(
            DetectedEntity(
                entity_type=ENTITY_TYPE_MAP.get(item['entity_group'], item['entity_group']),
                score=item['score'],
                text=item['word'],
                start=item['start'],
                end=item['end'],
            )
            for item in ensemble_results
        )
        if privacy_policy is not None:
            findings = privacy_policy.evaluate(entities)
            return NERProcessingResult(
                text=privacy_policy.transform(text, findings),
                findings=findings,
            )
        result = self.anonymize_text(
            text, ensemble_results, self._replacement_mode, anon_map=anon_map,
        )
        findings = tuple(
            EntityFinding(entity, self._replacement_mode, 'detector-threshold')
            for entity in entities
        )
        return NERProcessingResult(text=result.text, findings=findings)

    def ner_process(
        self,
        text: str,
        positional_tags: Optional[Sequence[str]] = None,
        ner_confidence_threshold: Optional[float] = None,
        language: Optional[str] = None,
        anon_map: Optional['AnonymizationMap'] = None,
    ) -> str:
        """Process text while preserving the legacy string return type."""
        return self.ner_process_detailed(
            text,
            positional_tags=positional_tags,
            ner_confidence_threshold=ner_confidence_threshold,
            language=language,
            anon_map=anon_map,
        ).text

    def _token_count(self, text: str, tokenizer) -> int:
        """Count tokens in text."""
        return len(tokenizer(text).input_ids)

    @staticmethod
    def _merge_pieces(pieces: List[tuple], fits) -> List[str]:
        """Greedily merge adjacent ``(separator, piece)`` pairs while ``fits(chunk)``.

        Pieces inside a chunk are rejoined with their original separator.
        """
        if not pieces:
            return []
        chunks = []
        current = pieces[0][1]
        for separator, piece in pieces[1:]:
            merged = current + separator + piece
            if fits(merged):
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
        def fits(chunk):
            return self._token_count(chunk, tokenizer) <= max_tokens

        sentence_pieces = self._split_pieces(constants.SENTENCE_BOUNDARY_PATTERN, text)
        if len(sentence_pieces) > 1:
            merged = self._merge_pieces(sentence_pieces, fits)
            result = []
            for chunk in merged:
                if self._token_count(chunk, tokenizer) > max_tokens:
                    result.extend(self.split_text(chunk, max_tokens, tokenizer))
                else:
                    result.append(chunk)
            return result

        # Try each delimiter in priority order
        for delimiter in constants.CHUNK_DELIMITERS:
            pieces = self._split_pieces(delimiter, text)
            if len(pieces) <= 1:
                continue

            # Greedily merge small adjacent pieces
            merged = self._merge_pieces(pieces, fits)

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
        # Reserve room for the special tokens ([CLS]/[SEP]) each chunk gets back.
        budget = max(1, max_tokens - self._token_count('', tokenizer))
        chunks = []
        last_end = 0
        current_tokens = 0

        for start, end in offsets:
            if end <= start:
                continue  # special token, no source text
            if current_tokens >= budget:
                chunks.append(text[last_end:start])
                last_end = start
                current_tokens = 0
            current_tokens += 1

        if last_end < len(text):
            chunks.append(text[last_end:])

        # Re-tokenizing a slice can add tokens (word-start markers), so halve any
        # slice that still does not fit until it does.
        result = []
        for chunk in chunks:
            if len(chunk) > 1 and not fits(chunk):
                mid = len(chunk) // 2
                result.extend(self.split_text(chunk[:mid], max_tokens, tokenizer))
                result.extend(self.split_text(chunk[mid:], max_tokens, tokenizer))
            else:
                result.append(chunk)
        return result

    def process_batch(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        positional_tags: Optional[List[str]] = None,
        ner_confidence_threshold: Optional[float] = None,
        language: Optional[str] = None
    ) -> List[str]:
        """Process multiple texts, *batch_size* documents per batched model call."""
        batch_size = batch_size if batch_size is not None else self._ner_batch_size
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            results.extend(result.text for result in self.ner_process_batch_detailed(
                batch,
                [language] * len(batch),
                positional_tags=positional_tags,
                ner_confidence_threshold=ner_confidence_threshold,
            ))
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
