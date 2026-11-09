"""GLiNER adapter — normalizes GLiNER output to match ONNXNERPipeline format.

Supports both original GLiNER (urchade) and GLiNER2 (knowledgator/fastino).
Auto-detects bi-encoder models (ModernBERT, etc.) and caches label embeddings.
Optional backend. Requires: pip install squeakycleantext[gliner] or [gliner2]
"""
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class GLiNERAdapter:
    """Adapts GLiNER/GLiNER2 to the same callable interface as ONNXNERPipeline.

    __call__(texts) returns List[List[dict]] with keys:
        entity_group, score, word, start, end

    Automatically detects bi-encoder models (via ``encode_labels``) and caches
    label embeddings for efficient repeated inference.
    """

    def __init__(
        self,
        model_id: str,
        revision: str | None = None,
        variant: str = 'gliner',
        labels: tuple[str, ...] | None = None,
        threshold: float = 0.4,
        label_map: dict[str, str] | None = None,
        label_descriptions: dict[str, str] | None = None,
        device: str = 'cpu',
        onnx: bool = False,
    ):
        self.variant = variant
        self.labels = list(labels) if labels else ['person', 'organization', 'location']
        self.threshold = threshold
        self.label_map = label_map or {}
        self.label_descriptions = label_descriptions or {}
        self.model_id = model_id
        self._onnx = onnx
        load_kwargs = {"revision": revision} if revision else {}

        if variant == 'gliner':
            self.model = self._load_gliner(model_id, load_kwargs)
        elif variant == 'gliner2':
            self.model = self._load_gliner2(model_id, load_kwargs)
        else:
            raise ValueError(f"Unknown GLiNER variant: {variant!r}. Use 'gliner' or 'gliner2'.")
        # The effective backend decides: an ONNX load that fell back to torch still moves.
        if device.startswith(('cuda', 'mps')) and not self._onnx:
            self.model = self.model.to(device)

        # Bi-encoder detection: ModernBERT and similar architectures expose encode_labels()
        self._is_bi_encoder = hasattr(self.model, 'encode_labels')
        self._label_embeddings = None
        self._cached_labels_key: tuple | None = None

        logger.info(
            "Loaded GLiNER model: %s (variant=%s, bi_encoder=%s)",
            model_id, variant, self._is_bi_encoder,
        )

    def _load_gliner(self, model_id: str, load_kwargs: dict[str, str]) -> Any:
        try:
            from gliner import GLiNER  # noqa: S404
        except ImportError as exc:
            raise ImportError(
                "gliner is required for GLiNER backend. "
                "Install with: pip install squeakycleantext[gliner]"
            ) from exc
        if not self._onnx:
            return GLiNER.from_pretrained(model_id, **load_kwargs)
        try:
            model = GLiNER.from_pretrained(
                model_id, load_onnx_model=True, load_tokenizer=True,
                **load_kwargs,
            )
            logger.info("Loaded GLiNER model in ONNX mode: %s", model_id)
            return model
        except FileNotFoundError:
            logger.warning(
                "ONNX model not found for %s (GLiNER issue #314: most models "
                "don't ship model.onnx at repo root). Falling back to PyTorch.",
                model_id,
            )
            self._onnx = False
            return GLiNER.from_pretrained(model_id, **load_kwargs)

    def _load_gliner2(self, model_id: str, load_kwargs: dict[str, str]) -> Any:
        try:
            from gliner2 import GLiNER2  # noqa: S404
        except ImportError as exc:
            raise ImportError(
                "gliner2 is required for GLiNER2 backend. "
                "Install with: pip install squeakycleantext[gliner2]"
            ) from exc
        self._onnx = False  # GLiNER2 has no ONNX loader; it always runs in torch
        if load_kwargs.get('revision') and not Path(model_id).is_dir():
            from huggingface_hub import snapshot_download
            # GLiNER2 ignores revision kwargs, so bind every bundled file before SDK loading.
            model_id = snapshot_download(model_id, revision=load_kwargs['revision'])
        return GLiNER2.from_pretrained(model_id, **load_kwargs)

    def _get_inference_labels(self) -> list:
        """Return labels for inference. Uses descriptions if provided, falls back to names."""
        if self.label_descriptions:
            return [self.label_descriptions.get(label, label) for label in self.labels]
        return list(self.labels)

    def _ensure_label_embeddings(self) -> None:
        """Pre-compute and cache label embeddings for bi-encoder models.

        Thread-safe: called under the GLiNER inference lock in ner.py.
        """
        labels_key = tuple(self.labels)
        if self._label_embeddings is not None and self._cached_labels_key == labels_key:
            return
        inference_labels = self._get_inference_labels()
        self._label_embeddings = self.model.encode_labels(inference_labels, batch_size=8)
        self._cached_labels_key = labels_key
        logger.info(
            "Cached label embeddings for %d labels (bi-encoder: %s)",
            len(self.labels), self.model_id,
        )

    def invalidate_label_cache(self) -> None:
        """Clear cached label embeddings. Call after changing self.labels."""
        self._label_embeddings = None
        self._cached_labels_key = None

    def _map_label(self, label: str) -> str:
        """Map a GLiNER label to entity_group tag."""
        return self.label_map.get(label, label.upper())

    def __call__(self, texts, batch_size: int = 1) -> list[list[dict]]:
        """Run NER. Returns List[List[dict]] matching ONNXNERPipeline format.

        GLiNER runs ``batch_size`` texts per forward pass; ``1`` keeps the
        one-text-per-call pattern. GLiNER2 always runs one text per call.
        """
        if isinstance(texts, str):
            texts = [texts]

        if self.variant == 'gliner':
            return self._call_gliner(texts, batch_size)
        return [self._call_gliner2(text) for text in texts]

    def _predict_gliner(self, texts, batch_size: int) -> Any:
        """GLiNER's raw entity lists, one per text as far as the model returned them."""
        inference_labels = self._get_inference_labels()
        if self._is_bi_encoder:
            self._ensure_label_embeddings()
            if batch_size > 1:
                return self.model.batch_predict_with_embeds(
                    texts, self._label_embeddings, self.labels,
                    threshold=self.threshold, batch_size=batch_size,
                )
            return [
                (self.model.batch_predict_with_embeds(
                    [text], self._label_embeddings, self.labels,
                    threshold=self.threshold,
                ) or [[]])[0]
                for text in texts
            ]
        if batch_size > 1 and hasattr(self.model, 'inference'):
            return self.model.inference(
                texts, inference_labels, threshold=self.threshold,
                batch_size=batch_size,
            )
        return [
            self.model.predict_entities(
                text, inference_labels, threshold=self.threshold,
            )
            for text in texts
        ]

    def _call_gliner(self, texts, batch_size: int) -> list[list[dict]]:
        # Description-to-label reverse mapping for ZERONER-style labels
        desc_to_label = {self.label_descriptions.get(label, label): label for label in self.labels} \
            if self.label_descriptions else {}
        raw_batch = self._predict_gliner(texts, batch_size)
        return [
            [
                {
                    'entity_group': self._map_label(
                        desc_to_label.get(e['label'], e['label'])
                    ),
                    'score': e['score'],
                    'word': e['text'],
                    'start': e['start'],
                    'end': e['end'],
                }
                for e in (raw_batch[index] if index < len(raw_batch) else [])
            ]
            for index in range(len(texts))
        ]

    def _call_gliner2(self, text: str) -> list[dict]:
        raw = self.model.extract_entities(
            text, self.labels, include_spans=True
        )
        entities = []
        for label, spans in raw.get('entities', {}).items():
            mapped = self._map_label(label)
            for span in spans:
                entities.append({
                    'entity_group': mapped,
                    'score': span.get('score', 1.0),
                    'word': span.get('text', text[span['start']:span['end']]),
                    'start': span['start'],
                    'end': span['end'],
                })
        return entities

    @property
    def max_context_length(self) -> int:
        """Derive max context window from the loaded model's config.

        Works for all architectures: DeBERTa (512), ModernBERT (2048-8192),
        GLiNER2 models, etc.
        """
        config = getattr(self.model, 'config', None)
        if config is not None:
            max_pos = getattr(config, 'max_position_embeddings', None)
            if max_pos is not None:
                return max_pos
        # Fallback for GLiNER2 or unknown models
        if self.variant == 'gliner2':
            return 2048
        return 512
