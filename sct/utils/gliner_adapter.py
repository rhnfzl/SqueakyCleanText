"""GLiNER adapter — normalizes GLiNER output to match ONNXNERPipeline format.

Supports both original GLiNER (urchade) and GLiNER2 (knowledgator/fastino).
Optional backend. Requires: pip install squeakycleantext[gliner] or [gliner2]
"""
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class GLiNERAdapter:
    """Adapts GLiNER/GLiNER2 to the same callable interface as ONNXNERPipeline.

    __call__(texts) returns List[List[dict]] with keys:
        entity_group, score, word, start, end
    """

    def __init__(
        self,
        model_id: str,
        variant: str = 'gliner',
        labels: Optional[Tuple[str, ...]] = None,
        threshold: float = 0.4,
        label_map: Optional[Dict[str, str]] = None,
        device: str = 'cpu',
    ):
        self.variant = variant
        self.labels = list(labels) if labels else ['person', 'organization', 'location']
        self.threshold = threshold
        self.label_map = label_map or {}
        self.model_id = model_id

        if variant == 'gliner':
            try:
                from gliner import GLiNER  # noqa: S404
            except ImportError:
                raise ImportError(
                    "gliner is required for GLiNER backend. "
                    "Install with: pip install squeakycleantext[gliner]"
                )
            self.model = GLiNER.from_pretrained(model_id)
            if device == 'cuda':
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

        logger.info("Loaded GLiNER model: %s (variant=%s)", model_id, variant)

    def _map_label(self, label: str) -> str:
        """Map a GLiNER label to entity_group tag."""
        return self.label_map.get(label, label.upper())

    def __call__(self, texts) -> List[List[dict]]:
        """Run NER. Returns List[List[dict]] matching ONNXNERPipeline format."""
        if isinstance(texts, str):
            texts = [texts]

        all_results: List[List[dict]] = []

        for text in texts:
            if self.variant == 'gliner':
                raw = self.model.predict_entities(
                    text, self.labels, threshold=self.threshold
                )
                entities = [
                    {
                        'entity_group': self._map_label(e['label']),
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
        """Max context window: 512 (GLiNER/DeBERTa), 2048 (GLiNER2)."""
        return 2048 if self.variant == 'gliner2' else 512
