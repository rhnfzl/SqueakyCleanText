"""GLiNER adapter — normalizes GLiNER output to match ONNXNERPipeline format.

Supports both original GLiNER (urchade) and GLiNER2 (knowledgator/fastino).
Auto-detects bi-encoder models (ModernBERT, etc.) and caches label embeddings.
Optional backend. Requires: pip install squeakycleantext[gliner] or [gliner2]
"""
import logging
from typing import Dict, List, Optional, Tuple

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
        variant: str = 'gliner',
        labels: Optional[Tuple[str, ...]] = None,
        threshold: float = 0.4,
        label_map: Optional[Dict[str, str]] = None,
        label_descriptions: Optional[Dict[str, str]] = None,
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

        if variant == 'gliner':
            try:
                from gliner import GLiNER  # noqa: S404
            except ImportError:
                raise ImportError(
                    "gliner is required for GLiNER backend. "
                    "Install with: pip install squeakycleantext[gliner]"
                )
            if onnx:
                try:
                    self.model = GLiNER.from_pretrained(
                        model_id, load_onnx_model=True, load_tokenizer=True,
                    )
                    logger.info("Loaded GLiNER model in ONNX mode: %s", model_id)
                except FileNotFoundError:
                    logger.warning(
                        "ONNX model not found for %s (GLiNER issue #314: most models "
                        "don't ship model.onnx at repo root). Falling back to PyTorch.",
                        model_id,
                    )
                    self.model = GLiNER.from_pretrained(model_id)
                    self._onnx = False
            else:
                self.model = GLiNER.from_pretrained(model_id)
            if device == 'cuda' and not onnx:
                self.model = self.model.to('cuda')

        elif variant == 'gliner2':
            try:
                from gliner2 import GLiNER2  # noqa: S404
            except ImportError:
                raise ImportError(
                    "gliner2 is required for GLiNER2 backend. "
                    "Install with: pip install squeakycleantext[gliner2]"
                )
            self.model = GLiNER2.from_pretrained(model_id)
        else:
            raise ValueError(f"Unknown GLiNER variant: {variant!r}. Use 'gliner' or 'gliner2'.")

        # Bi-encoder detection: ModernBERT and similar architectures expose encode_labels()
        self._is_bi_encoder = hasattr(self.model, 'encode_labels')
        self._label_embeddings = None
        self._cached_labels_key: Optional[tuple] = None

        logger.info(
            "Loaded GLiNER model: %s (variant=%s, bi_encoder=%s)",
            model_id, variant, self._is_bi_encoder,
        )

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

    def __call__(self, texts) -> List[List[dict]]:
        """Run NER. Returns List[List[dict]] matching ONNXNERPipeline format."""
        if isinstance(texts, str):
            texts = [texts]

        all_results: List[List[dict]] = []

        # Build description-to-label reverse mapping for ZERONER-style labels
        desc_to_label: Dict[str, str] = {}
        if self.label_descriptions:
            for label in self.labels:
                desc = self.label_descriptions.get(label, label)
                desc_to_label[desc] = label

        for text in texts:
            if self.variant == 'gliner':
                inference_labels = self._get_inference_labels()

                if self._is_bi_encoder:
                    self._ensure_label_embeddings()
                    raw_batch = self.model.batch_predict_with_embeds(
                        [text], self._label_embeddings, self.labels,
                        threshold=self.threshold,
                    )
                    raw = raw_batch[0] if raw_batch else []
                else:
                    raw = self.model.predict_entities(
                        text, inference_labels, threshold=self.threshold,
                    )

                entities = [
                    {
                        'entity_group': self._map_label(
                            desc_to_label.get(e['label'], e['label'])
                        ),
                        'score': e['score'],
                        'word': e['text'],
                        'start': e['start'],
                        'end': e['end'],
                    }
                    for e in raw
                ]
            else:
                # gliner2 variant
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

            all_results.append(entities)

        return all_results

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
