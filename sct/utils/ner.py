import math
import os
import weakref
import gc
import re
import threading
from collections import defaultdict
import logging
from typing import TYPE_CHECKING, Any, cast
from collections.abc import Collection, Iterable, Mapping, Sequence

from sct.utils import constants
from sct.utils.anonymization_map import AnonymizationMap
from sct.utils.markers import PLACEHOLDER_TOKEN
from sct.utils.ner_chunking import Window, plan_windows
from sct.utils.ner_spans import (
    ENTITY_TYPE_MAP,
    finalize_spans,
    presidio_results,
    merge_overlaps,
)
from sct.config import (
    DEFAULT_NER_MODELS,
    DEFAULT_NER_ENSEMBLE,
    DEFAULT_TORCH_NER_MODELS,
    GLINER_ADAPTER_BACKENDS,
    GLINER_BACKENDS,
    NER_ENSEMBLE_DEFAULT_KEYS,
    ONNX_BACKENDS,
    ReplacementMode,
    TOKEN_MODEL_BACKENDS,
    TORCH_BACKENDS,
    TextCleanerConfig,
)
from sct.findings import Detection, DetectionContext
from sct.privacy import (
    DetectedEntity,
    EntityFinding,
    NERProcessingResult,
    PrivacyPolicy,
)

if TYPE_CHECKING:
    import regex

    from sct.utils.synthetic import SyntheticReplacer

logger = logging.getLogger(__name__)

_ENSEMBLE_LABEL_MAP = {'person': 'PER', 'organization': 'ORG', 'location': 'LOC'}

# An inference lock's key: 'gliner', or a token model's (name, revision).
ModelKey = tuple[str, str | None]

# Freeing an inherited ONNX session in a forked child crashes or hangs, so children pin every live
# GeneralNER for their whole life instead of ever destroying one.
_LIVE_NER: "weakref.WeakSet[GeneralNER]" = weakref.WeakSet()
_FORK_PINNED: list = []


def _pin_live_ner_in_child():
    _FORK_PINNED.extend(_LIVE_NER)


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_pin_live_ner_in_child)


class ModelLoadError(Exception):
    """Raised when model loading fails"""
    pass


def _presidio_type_map(legacy_type_names: bool, label_map: Mapping[str, str] | None) -> dict[str, str]:
    """Presidio entity type -> SCT type: the label map, and PHONE_NUMBER/IP_ADDRESS -> PHONE/IP."""
    types = {} if legacy_type_names else {old: new for new, old in constants.LEGACY_TYPE_NAMES.items()}
    types.update({label.upper(): tag for label, tag in (label_map or {}).items()})
    return types


def _mps_available() -> bool:
    try:
        import torch  # noqa: S404
        return bool(torch.backends.mps.is_available())
    except Exception:  # torch missing or built without MPS
        return False


def _torch_cuda_usable(device_id: int) -> bool:
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        count = torch.cuda.device_count()
    except Exception:  # torch missing or broken
        return False
    if device_id >= count:
        logger.warning("cuda_device_id %d is not visible to torch (%d GPUs)", device_id, count)
        return False
    return True


class GeneralNER:
    """NER processor with lazy model loading, multi-backend support, and ensemble voting.

    Backends:
        onnx           — ONNX Runtime (default, torch-free)
        torch          — PyTorch/Transformers pipeline
        gliner         — GLiNER zero-shot NER with custom entity labels
        ensemble_onnx  — ONNX + GLiNER combined via ensemble voting
        ensemble_torch — Torch + GLiNER combined via ensemble voting
    """

    def __init__(
        self,
        cfg: TextCleanerConfig,
        *,
        synthetic_replacer: 'SyntheticReplacer | None' = None,
        regex_markers: Sequence[str] = (),
        allow_regex: 'regex.Pattern | None' = None,
        privacy_policy: PrivacyPolicy | None = None,
    ):
        """Load the NER models *cfg* selects.

        Args:
            cfg: Every NER option (``ner_*``, ``gliner_*``, ``torch_ner_*``, ``onnx_*``, ``cuda_device_id``,
                ``positional_tags``, ``replacement_mode``) is read from here; the field comments in
                ``TextCleanerConfig`` document them.
            synthetic_replacer: Shared SyntheticReplacer instance (from TextCleaner).
            regex_markers: Custom regex-step markers (``replace_with_email="EMAIL"``) protected
                like ``<EMAIL>`` unless the source already holds them.
            allow_regex: Compiled custom_allow_terms; a span equal to or inside a match is kept as text.
            privacy_policy: Decides each entity's action in place of ``cfg.replacement_mode``; None = replace all.
        """
        _LIVE_NER.add(self)
        self.cfg = cfg
        self._allow_regex = allow_regex
        self._synthetic_replacer = synthetic_replacer
        self._regex_markers = tuple(regex_markers)
        self._privacy_policy = privacy_policy
        # Per language key: {"key", "id", "revision", "provider"} of the model that key loaded.
        self.provenance: dict[str, dict[str, str | None]] = {}
        self._gliner_pipe = None
        self._ensemble_models: dict[str, tuple] = (
            dict(cfg.ner_ensemble) if cfg.ner_ensemble is not None else DEFAULT_NER_ENSEMBLE
        )
        self._ensemble_default_keys: tuple = (
            tuple(cfg.ner_ensemble_default_keys)
            if cfg.ner_ensemble_default_keys is not None
            else NER_ENSEMBLE_DEFAULT_KEYS
        )
        ner_backend = cfg.ner_backend

        if cfg.ner_device:
            self.device = cfg.ner_device
        elif ner_backend in TORCH_BACKENDS:
            self.device = 'cuda' if _torch_cuda_usable(cfg.cuda_device_id) else 'cpu'
        else:
            import onnxruntime as ort
            self.device = 'cuda' if 'CUDAExecutionProvider' in ort.get_available_providers() else 'cpu'
        # Torch and GLiNER take an indexed device string; ONNX gets the index as a provider option.
        self._torch_device = f"cuda:{cfg.cuda_device_id}" if self.device == 'cuda' else self.device
        # The provider list says what ONNX Runtime was built with, not whether torch can use a GPU,
        # so GLiNER in torch follows torch: down to CPU without a usable GPU, up to CUDA with one.
        if not cfg.ner_device and ner_backend in GLINER_BACKENDS:
            if not _torch_cuda_usable(cfg.cuda_device_id):
                self._torch_device = 'cpu'
            elif not cfg.gliner_onnx:
                self._torch_device = f"cuda:{cfg.cuda_device_id}"
        # 'mps' (Apple GPU) moves torch and GLiNER only; ONNX models stay on CPU.
        if self.device == 'mps' and not _mps_available():
            logger.warning("NER runs on CPU: Apple GPU (mps) was requested but torch has no MPS backend")
            self._torch_device = 'cpu'
        logger.info("Using device: %s (backend: %s)", self._torch_device, ner_backend)

        self._engine: Any = None

        self._pipelines: dict = {}
        self._tokenizers: dict = {}
        self._load_lock = threading.Lock()
        # Per-model inference locks for torch models and GLiNER. ONNX runs lock-free:
        # InferenceSession.run and the Rust tokenizer are thread-safe.
        self._inference_locks: dict[str | ModelKey, threading.Lock] = {}
        self._locks_lock = threading.Lock()  # Guards creation of per-model locks

        if ner_backend in TOKEN_MODEL_BACKENDS:
            self._model_names = dict((cfg.torch_ner_models if ner_backend in TORCH_BACKENDS else cfg.ner_models) or {})
            self._ensure_loaded('ENGLISH')
            self._ensure_loaded('MULTILINGUAL')
            self._init_tokenizer_props()
        else:
            # GLiNER-only, or Presidio (GLiNER recognition + anonymization): no token-classification models
            self._model_names = {}
            self.min_token_length = None
            self.tokenizer = None
            if ner_backend == 'presidio_gliner':
                try:
                    self._init_presidio_gliner()
                except ImportError:
                    raise
                except Exception as e:
                    raise ModelLoadError(f"Model loading failed for GLINER ({cfg.gliner_model}): {e}") from e

        if ner_backend in GLINER_ADAPTER_BACKENDS:
            from sct.utils.gliner_adapter import GLiNERAdapter
            try:
                self._gliner_pipe = GLiNERAdapter(
                    model_id=cast(str, cfg.gliner_model),  # __post_init__ requires it for these backends
                    revision=cfg.gliner_revision,
                    variant=cfg.gliner_variant,
                    labels=cfg.gliner_labels,
                    threshold=cfg.effective_gliner_threshold,
                    label_map=dict(cfg.gliner_label_map or {}),
                    label_descriptions=dict(cfg.gliner_label_descriptions or {}),
                    device=self._torch_device,
                    onnx=cfg.gliner_onnx,
                )
            except ImportError:
                raise
            except Exception as e:
                raise ModelLoadError(f"Model loading failed for GLINER ({cfg.gliner_model}): {e}") from e
        if ner_backend in GLINER_BACKENDS:
            onnx_gliner = self._gliner_pipe is not None and self._gliner_pipe._onnx
            # After the load: GLiNER ONNX ignores the device, but missing ONNX weights fall back to torch.
            if self.device == 'cuda' and self._torch_device == 'cpu' and not onnx_gliner:
                logger.warning(
                    "Torch models run on CPU: ONNX Runtime has CUDA but torch cannot use GPU %d. "
                    "Check CUDA_VISIBLE_DEVICES and that the torch CUDA version matches your driver.",
                    cfg.cuda_device_id,
                )
            self.provenance['GLINER'] = {
                'key': 'GLINER', 'id': cfg.gliner_model,
                'revision': cfg.gliner_revision,
                'provider': 'onnx' if onnx_gliner else self._torch_device,
            }

        if ner_backend != 'presidio_gliner' and cfg.ner_chunk_overlap:
            self._check_overlap(self._chunk_budget())

    def _check_overlap(self, budget: int) -> None:
        """Reject an overlap that leaves no core inside a *budget*-token window."""
        overlap = self.cfg.ner_chunk_overlap
        if not 0 < 2 * overlap < budget // 2:
            raise ValueError(
                f"ner_chunk_overlap must be >= 0 and 2 * overlap < {budget // 2} "
                f"(half the {budget}-token window), got {overlap}"
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

    def _get_lock(self, model_key: str | ModelKey) -> threading.Lock:
        """Get or create a per-model inference lock (double-checked locking)."""
        if model_key not in self._inference_locks:
            with self._locks_lock:
                if model_key not in self._inference_locks:
                    self._inference_locks[model_key] = threading.Lock()
        return self._inference_locks[model_key]

    def _init_presidio_gliner(self):
        """Initialize Presidio with GLiNERRecognizer (experimental beta backend)."""
        cfg = self.cfg
        try:
            from presidio_analyzer import AnalyzerEngine  # noqa: S404
        except ImportError as exc:
            raise ImportError(
                "presidio-analyzer is required for presidio_gliner backend. "
                "Install with: pip install squeakycleantext[presidio]"
            ) from exc
        self._analyzer = AnalyzerEngine()
        self._presidio_types = _presidio_type_map(cfg.legacy_type_names, cfg.gliner_label_map)
        try:
            from presidio_analyzer.predefined_recognizers import GLiNERRecognizer  # noqa: S404
            gliner_recognizer = GLiNERRecognizer(
                model_name=cfg.gliner_model,
                supported_entities=[label.upper() for label in cfg.gliner_labels],
                threshold=cfg.effective_gliner_threshold,
                map_location=self._torch_device,  # Presidio loads GLiNER in torch
            )
            self._analyzer.registry.add_recognizer(gliner_recognizer)
        except ImportError as exc:
            raise ImportError(
                "GLiNERRecognizer requires presidio-analyzer with GLiNER support. "
                "Install with: pip install squeakycleantext[presidio]"
            ) from exc

    def _get_ensemble_keys(self, language: str) -> tuple:
        """Return ordered model keys to run for the given language."""
        if language in self._ensemble_models:
            return self._ensemble_models[language]
        if self._has_own_model(language):
            return (language,)
        return self._ensemble_default_keys

    def _has_own_model(self, language: str | None) -> bool:
        """True when the user brought a model for *language* (e.g. ``ner_models={"HINDI": ...}``).

        Built-in FR/PT/IT keys point at the default multilingual model and keep the fallback.
        """
        if language == 'MULTILINGUAL' or language not in self._model_names:
            return False
        defaults = DEFAULT_TORCH_NER_MODELS if self.cfg.ner_backend in TORCH_BACKENDS else DEFAULT_NER_MODELS
        return self._model_names[language] != defaults.get(language)

    def uses_fallback_models(self, language: str | None) -> bool:
        """True when *language* has no dedicated NER route and runs the default keys."""
        return (
            bool(self._model_names)
            and language not in self._ensemble_models
            and not self._has_own_model(language)
        )

    def load_language(self, lang_key: str) -> None:
        """Pre-load NER model for the given language key (stable public API)."""
        self._ensure_loaded(lang_key)

    def _model_key(self, lang_key: str) -> ModelKey:
        """(model name, revision) for a language key; same key = same loaded model."""
        revisions = (self.cfg.torch_ner_model_revisions
                     if self.cfg.ner_backend in TORCH_BACKENDS
                     else self.cfg.ner_model_revisions)
        return self._model_names.get(lang_key, lang_key), (revisions or {}).get(lang_key)

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
            # Tokenizer first: a pipeline in _pipelines marks the key ready for lock-free readers.
            if shared_pipe is not None:
                shared_tok = next(
                    (t for k, t in self._tokenizers.items()
                     if self._model_key(k) == model_key), None
                )
                if shared_tok is not None:
                    self._tokenizers[lang_key] = shared_tok
                self._record_provenance(lang_key, shared_pipe)
                self._pipelines[lang_key] = shared_pipe
                return
            try:
                pipeline_obj: object
                if self.cfg.ner_backend in TORCH_BACKENDS:
                    from sct.utils.torch_pipeline import load_torch_ner_model
                    pipeline_obj, tok = load_torch_ner_model(
                        model_name, device=self._torch_device,
                        revision=(self.cfg.torch_ner_model_revisions or {}).get(lang_key),
                    )
                else:
                    from sct.utils.onnx_pipeline import load_onnx_ner_model
                    pipeline_obj, tok = load_onnx_ner_model(
                        model_name,
                        device=self.device,
                        revision=(self.cfg.ner_model_revisions or {}).get(lang_key),
                        intra_op_threads=self.cfg.onnx_intra_op_threads,
                        inter_op_threads=self.cfg.onnx_inter_op_threads,
                        cuda_device_id=self.cfg.cuda_device_id,
                        cuda_provider_options=dict(self.cfg.onnx_cuda_provider_options or {}),
                    )
                self._tokenizers[lang_key] = tok
                self._record_provenance(lang_key, pipeline_obj)
                self._pipelines[lang_key] = pipeline_obj
            except Exception as e:  # a missing extra too: docs/API.md promises this type, the message names the install
                raise ModelLoadError(
                    f"Model loading failed for {lang_key} ({model_name}): {e}"
                ) from e

    def _record_provenance(self, lang_key: str, pipe) -> None:
        """Remember which model, revision and provider *lang_key* runs (own dict per key)."""
        model_name, revision = self._model_key(lang_key)
        self.provenance[lang_key] = {
            'key': lang_key,
            'id': model_name,
            'revision': getattr(pipe, 'revision', None) or revision,
            # ONNX pipelines report their execution provider; torch runs on its device.
            'provider': getattr(pipe, 'provider', None) or self._torch_device,
        }

    def models_for(self, language: str | None) -> list[dict[str, str | None]]:
        """Provenance of the loaded models that NER runs for *language* (copies)."""
        keys: tuple = ()
        if self.cfg.ner_backend in TOKEN_MODEL_BACKENDS:
            # tuple(): a configured list route must not grow on every call.
            keys = tuple(self._get_ensemble_keys(language or 'MULTILINGUAL'))
        if 'GLINER' in self.provenance:
            keys += ('GLINER',)
        return [dict(self.provenance[k]) for k in keys if k in self.provenance]

    def entities_with_position_key(self, results: Iterable[dict[str, Any]], labels: Collection[str]) -> list[dict[str, Any]]:
        """Keep the entities whose group is in *labels*, with collision-safe position keys."""
        return [
            {
                'entity_group': ix['entity_group'],
                'score': ix['score'],
                'word': ix['word'],
                'key': f"{ix['start']}:{ix['end']}",  # Fixed: separator prevents collision
                'start': ix['start'],
                'end': ix['end'],
            }
            for ix in results if ix['entity_group'] in labels
        ]

    def filter_ner_data(self, data: Iterable[dict[str, Any]], keys: Collection[str]) -> list[dict[str, Any]]:
        """Filter NER data, keeping highest-scoring entity per key."""
        unique_data: dict[str, dict[str, Any]] = {}
        for item in data:
            if item['key'] in keys:
                if item['key'] not in unique_data or item['score'] > unique_data[item['key']]['score']:
                    unique_data[item['key']] = item
        return list(unique_data.values())

    def anonymize_text(self, text: str, filtered_data: list[dict[str, Any]], replacement_mode: ReplacementMode = 'placeholder',
                       anon_map: AnonymizationMap | None = None) -> str:
        """Anonymize text. Supports placeholder, synthetic, or reversible replacement."""
        has_custom = any(
            items['entity_group'] not in ENTITY_TYPE_MAP
            for items in filtered_data
        )
        if has_custom or replacement_mode in ('reversible', 'synthetic'):
            # Right-to-left slicing keeps offsets only for disjoint spans.
            filtered_data = merge_overlaps(text, filtered_data)

        if replacement_mode == 'reversible':
            return self._anonymize_reversible(text, filtered_data, anon_map)

        if replacement_mode == 'synthetic':
            return cast('SyntheticReplacer', self._synthetic_replacer).generate_for_entities(text, filtered_data)

        if has_custom:
            # Mixed labels: right-to-left string replacement (preserves offsets)
            sorted_data = sorted(filtered_data, key=lambda x: x['start'], reverse=True)
            for items in sorted_data:
                tag = ENTITY_TYPE_MAP.get(items['entity_group'], items['entity_group'])
                text = text[:items['start']] + f"<{tag}>" + text[items['end']:]
            return text
        # Standard entities only: use Presidio (existing behavior)
        from presidio_anonymizer.entities import RecognizerResult
        analyzer_result = [
            RecognizerResult(
                entity_type=ENTITY_TYPE_MAP[items['entity_group']],
                start=items['start'],
                end=items['end'],
                score=items['score'],
            )
            for items in filtered_data
        ]

        text_length = len(text)
        analyzer_result = [
            entry for entry in analyzer_result
            if 0 <= entry.start < text_length and 0 < entry.end <= text_length
        ]

        return self.engine.anonymize(text=text, analyzer_results=analyzer_result).text

    @property
    def engine(self) -> Any:
        # Lazy: presidio_anonymizer imports presidio_analyzer (and torch via GLiNER) when installed.
        if self._engine is None:
            from presidio_anonymizer import AnonymizerEngine
            self._engine = AnonymizerEngine()
        return self._engine

    def _anonymize_reversible(self, text, filtered_data, anon_map=None):
        """Replace entities with indexed placeholders and populate the map.

        Uses right-to-left replacement to preserve character offsets.
        Returns the anonymized text.
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

        return text

    def vote_entities(self, entities: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
        """Apply ensemble voting across multiple model results.

        Groups entities by position key and keeps a key whose vote passes *threshold*:
        the best model score ('max'), or the mean score ('mean', 0.8.0, where a second
        weak vote can sink a strong one). Returns the highest-scoring entity per position.
        """
        if not entities:
            return []

        ner_keys = defaultdict(list)
        for entity in entities:
            ner_keys[entity['key']].append(entity['score'])

        vote = max if self.cfg.ner_ensemble_vote == 'max' else (lambda s: sum(s) / len(s))
        passing_keys = [key for key, scores in ner_keys.items() if vote(scores) >= threshold]
        kept_entities = self.filter_ner_data(entities, passing_keys)
        kept_entities.sort(key=lambda x: x['start'])
        return kept_entities

    def _tags_and_cut(self, positional_tags, ner_confidence_threshold) -> tuple:
        """A direct caller's tags and score cut, else the config's."""
        tags = self.cfg.positional_tags if positional_tags is None else positional_tags
        if not tags:
            raise ValueError("Must provide at least one positional tag")
        cut = ner_confidence_threshold
        return tags, self.cfg.effective_ner_confidence_threshold if cut is None else cut

    def analyze(self, text: str, *, language: str | None, context: DetectionContext) -> tuple[Detection, ...]:
        """Return value-free findings in canonical input coordinates, without replacement."""
        if self.cfg.ner_backend == 'presidio_gliner':
            results = presidio_results(self, text, language, protect_markers=False)
            return tuple(Detection(r.entity_type, r.start, r.end, 'builtin_ner',
                                   None if r.score is None else float(r.score), 'canonical')
                         for r in results)
        tags, cutoff = self._tags_and_cut(None, None)
        self._load_models([language])
        windows = self._plan_windows(text, self._window_keys(language))
        results = self._detect([w.text for w in windows], [language] * len(windows), tags) if windows else []
        spans = finalize_spans(self, text, windows, results, cutoff, protect_markers=False)
        return tuple(Detection(ENTITY_TYPE_MAP.get(s['entity_group'], s['entity_group']),
                               s['start'], s['end'], 'builtin_ner', float(s['score']), 'canonical') for s in spans)

    def ner_process_detailed(
        self,
        text: str,
        positional_tags: Sequence[str] | None = None,
        ner_confidence_threshold: float | None = None,
        language: str | None = None,
        anon_map: 'AnonymizationMap | None' = None,
        source_markers: Collection[Any] | None = None,
    ) -> NERProcessingResult:
        """Process text and return transformed text with entity findings.

        Routes to ONNX, Torch, GLiNER, or ensemble depending on ``cfg.ner_backend``.

        When *anon_map* is provided (reversible mode), entity-to-placeholder
        mappings are recorded in it for later deanonymization.

        Finding and map-entry offsets index *text* as passed in, i.e. the text
        after the regex replacement stages (post-regex), not the raw input.
        """
        positional_tags, ner_confidence_threshold = self._tags_and_cut(positional_tags, ner_confidence_threshold)

        if self.cfg.ner_backend == 'presidio_gliner':
            return self._process_presidio(text, language, anon_map, source_markers)

        self._load_models([language])
        windows = self._plan_windows(text, self._window_keys(language))
        if not windows:
            return NERProcessingResult(text=text, findings=())
        window_results = self._detect(
            [window.text for window in windows], [language] * len(windows), positional_tags,
        )
        return self._finalize(text, windows, window_results, ner_confidence_threshold, anon_map, source_markers)

    def _process_presidio(self, text, language, anon_map, source_markers=None) -> NERProcessingResult:
        """The presidio_gliner backend: Presidio detects; the replacement mode replaces.

        Presidio's own analyzer decides what is found: the caller's tags and confidence threshold
        do not filter its results. Only the tags this pipeline wrote do: a result inside one is
        dropped and a result that cuts one takes it whole. Marker-shaped source text
        (*source_markers*) gets no protection, and neither do fakes, so real text equal to a
        fake is still masked.
        """
        analyzer_results = presidio_results(self, text, language, source_markers)
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

        def anonymize():
            if self.cfg.replacement_mode == 'placeholder':
                return self.engine.anonymize(text=text, analyzer_results=analyzer_results).text
            spans = [
                {'entity_group': e.entity_type, 'score': e.score, 'word': e.text, 'start': e.start,
                 'end': e.end, 'key': f"{e.start}:{e.end}"}
                for e in entities
            ]
            return self.anonymize_text(text, spans, self.cfg.replacement_mode, anon_map=anon_map)
        return self._policy_or_default(text, entities, anonymize)

    def _policy_or_default(self, text, entities, anonymize) -> NERProcessingResult:
        """Apply the privacy policy to *entities*; without one, replace them through *anonymize()*."""
        if self._privacy_policy is not None:
            findings = self._privacy_policy.evaluate(entities)
            return NERProcessingResult(
                text=self._privacy_policy.transform(text, findings),
                findings=findings,
            )
        anonymized = anonymize()
        findings = tuple(
            EntityFinding(entity, self.cfg.replacement_mode, 'detector-threshold')
            for entity in entities
        )
        return NERProcessingResult(text=anonymized, findings=findings)

    def ner_process_batch_detailed(
        self,
        texts: Sequence[str],
        languages: Sequence[str | None],
        positional_tags: Sequence[str] | None = None,
        ner_confidence_threshold: float | None = None,
        anon_maps: Sequence['AnonymizationMap | None'] | None = None,
        synthetic_caches: Sequence[dict[tuple, str] | None] | None = None,
        max_workers: int = 1,
        source_markers: Sequence[Collection[Any] | None] | None = None,
    ) -> list[NERProcessingResult]:
        """``ner_process_detailed`` for many documents, batching model calls across them.

        Windows of all documents go through each model in calls of up to
        ``ner_batch_size`` windows; replacement then runs per document exactly as
        in ``ner_process_detailed``. *synthetic_caches* carries each document's
        replacements from earlier cleaning stages into entity replacement.
        *max_workers* ONNX batches run at once.
        """
        positional_tags, ner_confidence_threshold = self._tags_and_cut(positional_tags, ner_confidence_threshold)
        anon_maps = anon_maps if anon_maps is not None else [None] * len(texts)
        source_markers = source_markers if source_markers is not None else [None] * len(texts)
        if self.cfg.ner_backend == 'presidio_gliner':
            results = []
            for i, text in enumerate(texts):
                if synthetic_caches is not None and self._synthetic_replacer is not None:
                    self._synthetic_replacer.swap_cache(synthetic_caches[i])
                results.append(self.ner_process_detailed(
                    text, positional_tags, ner_confidence_threshold, languages[i], anon_maps[i], source_markers[i],
                ))
            return results

        self._load_models(languages)
        doc_windows = [
            self._plan_windows(text, self._window_keys(language))
            for text, language in zip(texts, languages)
        ]
        flat_windows = [window.text for windows in doc_windows for window in windows]
        flat_languages = [
            languages[i] for i, windows in enumerate(doc_windows) for _ in windows
        ]
        flat_results = self._detect(flat_windows, flat_languages, positional_tags, max_workers)

        results = []
        offset = 0
        for i, (text, windows) in enumerate(zip(texts, doc_windows)):
            window_results = flat_results[offset:offset + len(windows)]
            offset += len(windows)
            if synthetic_caches is not None and self._synthetic_replacer is not None:
                self._synthetic_replacer.swap_cache(synthetic_caches[i])
            if not windows:
                results.append(NERProcessingResult(text=text, findings=()))
                continue
            results.append(self._finalize(
                text, windows, window_results, ner_confidence_threshold, anon_maps[i], source_markers[i],
            ))
        return results

    def _load_models(self, languages) -> None:
        """Load every model these languages route to, so windows are sized for all of them."""
        if self.cfg.ner_backend in TOKEN_MODEL_BACKENDS:
            for language in dict.fromkeys(languages):
                for key in self._get_ensemble_keys(language or 'MULTILINGUAL'):
                    self._ensure_loaded(key)

    def _window_keys(self, language):
        """Model keys a document's windows must fit (None in GLiNER-only mode)."""
        return self._get_ensemble_keys(language or 'MULTILINGUAL') if self.tokenizer is not None else None

    def _model_tokenizers(self, keys=None) -> list:
        """Distinct tokenizers of the loaded models for *keys* (None: every loaded model)."""
        loaded = self._tokenizers
        if keys is not None:
            loaded = {key: loaded[key] for key in keys if key in loaded}
        return list({id(tok): tok for tok in loaded.values()}.values())

    def _window_tokenizer(self, keys=None):
        """Tokenizer the chunker counts with: a window must fit every model in *keys*.

        Counts (and offsets) come from whichever of those models' tokenizers gives
        the most tokens, so no model truncates a window; ``self.tokenizer`` is used
        only when none is loaded, so a route without it is not split by its count.
        """
        tokenizers = self._model_tokenizers(keys) or [self.tokenizer]
        if len(tokenizers) == 1:
            return tokenizers[0]

        def strictest(text, **kwargs):
            return max((tok(text, **kwargs) for tok in tokenizers), key=lambda r: len(r.input_ids))
        return strictest

    def _chunk_budget(self, keys=None) -> int:
        """Token budget of one NER window (GLiNER-only: of its context window)."""
        if self.tokenizer is not None:
            return min([self.min_token_length] + [
                math.ceil(tok.max_len_single_sentence * 0.9) for tok in self._model_tokenizers(keys)
            ])
        # Use GLiNER adapter's dynamic context window when available
        gliner_max = self._gliner_pipe.max_context_length if self._gliner_pipe else 384
        return int(gliner_max * 0.9)

    def _plan_windows(self, text: str, keys=None) -> list[Window]:
        """Size windows for this document's models, independent of other loaded routes."""
        overlap = self.cfg.ner_chunk_overlap
        budget = self._chunk_budget(keys)
        if overlap:
            self._check_overlap(budget)  # a model loaded after construction can shrink it
        tokenizer = self._window_tokenizer(keys) if self.tokenizer is not None else None
        return plan_windows(text, budget, tokenizer, overlap)

    def _detect(
        self,
        chunks: list[str],
        languages: list[str | None],
        positional_tags: Sequence[str],
        max_workers: int = 1,
    ) -> list[list]:
        """Run every model on its chunks; return the raw NER results per chunk.

        The only place that calls model pipelines. Each model gets one call over
        all chunks it applies to, run ``ner_batch_size`` chunks per forward pass
        (ONNX: up to *max_workers* forward passes at once).
        Per chunk, results keep the order of that language's ensemble keys,
        then GLiNER, as in the one-chunk-at-a-time loop.
        """
        per_key: list[dict[str, list]] = [{} for _ in chunks]
        chunk_keys: list[tuple] = [()] * len(chunks)

        # Primary backend: ONNX or Torch — run each key in the ensemble
        if self.cfg.ner_backend in TOKEN_MODEL_BACKENDS:
            chunk_keys = [
                self._get_ensemble_keys(language or 'MULTILINGUAL') for language in languages
            ]
            for key in dict.fromkeys(k for keys in chunk_keys for k in keys):
                indices = [i for i, keys in enumerate(chunk_keys) if key in keys]
                self._ensure_loaded(key)
                key_chunks = [chunks[i] for i in indices]
                if self.cfg.ner_backend in ONNX_BACKENDS:
                    batch = self._pipelines[key](
                        key_chunks, batch_size=self.cfg.ner_batch_size, max_workers=max_workers,
                        subword_widen=self.cfg.ner_subword_widen,
                        subword_same_script=self.cfg.ner_subword_widen_same_script,
                    )
                else:
                    with self._get_lock(self._model_key(key)):
                        batch = self._pipelines[key](key_chunks, batch_size=self.cfg.ner_batch_size)
                for i, data in zip(indices, batch):
                    per_key[i][key] = [
                        {**item, 'model': key} for item in self.entities_with_position_key(data, positional_tags)
                    ]

        results = [
            [item for key in keys for item in per_key[i][key]]
            for i, keys in enumerate(chunk_keys)
        ]

        # GLiNER backend
        if self.cfg.ner_backend in GLINER_ADAPTER_BACKENDS and self._gliner_pipe and chunks:
            # GLiNER-only keeps every label it was given; ensembles keep positional_tags.
            gliner_tags = set(positional_tags)
            short_tags: dict[str, str] = {}
            if self.cfg.ner_backend != 'gliner' and self.cfg.gliner_label_map is None:
                # A tag requested under its long name keeps reaching the vote under that name.
                short_tags = {
                    label.upper(): tag for label, tag in _ENSEMBLE_LABEL_MAP.items()
                    if label in self._gliner_pipe.labels and label.upper() not in gliner_tags
                }
            if self.cfg.ner_backend == 'gliner':
                gliner_tags.update(self._gliner_pipe.label_map.values())
                gliner_tags.update(
                    label.upper() for label in self._gliner_pipe.labels
                    if label not in self._gliner_pipe.label_map
                )
            with self._get_lock('gliner'):
                gliner_batch = self._gliner_pipe(chunks, batch_size=self.cfg.ner_batch_size)
            for i, data in enumerate(gliner_batch):
                data = [{**item, 'entity_group': short_tags.get(item['entity_group'], item['entity_group'])} for item in data]
                results[i].extend({**item, 'model': 'gliner'} for item in self.entities_with_position_key(data, gliner_tags))
        return results

    def _finalize(
        self,
        text: str,
        windows: list[Window],
        window_results: list[list],
        ner_confidence_threshold: float,
        anon_map: 'AnonymizationMap | None',
        source_markers: Collection[Any] | None = None,
    ) -> NERProcessingResult:
        """Merge window spans in document coordinates, vote, and replace once."""
        ensemble_results = finalize_spans(
            self, text, windows, window_results, ner_confidence_threshold, source_markers)
        if not ensemble_results:
            return NERProcessingResult(text=text, findings=())
        entities = tuple(self._entity(item) for item in ensemble_results)
        return self._policy_or_default(
            text, entities,
            lambda: self.anonymize_text(text, ensemble_results, self.cfg.replacement_mode, anon_map=anon_map),
        )

    def _marker_spans(self, text, source_markers) -> list[tuple[int, int]]:
        """Protect tags the pipeline emitted, including custom markers.

        Marker-shaped source text must remain detectable, even when another
        occurrence of the same marker was emitted during cleaning.
        """
        source_markers = source_markers or ()
        source_spans = [s for s in source_markers if isinstance(s, tuple)]

        def emitted(m):
            return m.group() not in source_markers and not any(
                a <= m.start() and m.end() <= b for a, b in source_spans)
        markers = [m.span() for m in PLACEHOLDER_TOKEN.finditer(text) if emitted(m)]
        for marker in self._regex_markers:
            markers += [m.span() for m in re.finditer(re.escape(marker), text) if emitted(m)]
        return markers

    @staticmethod
    def _entity(item) -> DetectedEntity:
        return DetectedEntity(
            entity_type=ENTITY_TYPE_MAP.get(item['entity_group'], item['entity_group']),
            score=item['score'],
            text=item['word'],
            start=item['start'],
            end=item['end'],
        )

    def ner_process(
        self,
        text: str,
        positional_tags: Sequence[str] | None = None,
        ner_confidence_threshold: float | None = None,
        language: str | None = None,
        anon_map: 'AnonymizationMap | None' = None,
    ) -> str:
        """Process text while preserving the legacy string return type."""
        return self.ner_process_detailed(
            text,
            positional_tags=positional_tags,
            ner_confidence_threshold=ner_confidence_threshold,
            language=language,
            anon_map=anon_map,
        ).text

    def ner_process_batch(
        self,
        texts: list[str],
        positional_tags: Sequence[str] | None = None,
        batch_size: int | None = None,
        ner_confidence_threshold: float | None = None,
        language: str | None = None
    ) -> list[str]:
        """Process multiple texts, *batch_size* documents per batched model call."""
        batch_size = batch_size if batch_size is not None else self.cfg.ner_batch_size
        results: list[str] = []
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
