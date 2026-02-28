"""GLiClass adapter — zero-shot document classification.

Supports PyTorch (``gliclass`` package) and ONNX backends.
Optional dependency. Requires: ``pip install squeakycleantext[classify]``
or ``pip install squeakycleantext[classify-onnx]`` for torch-free path.
"""

import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class GLiClassAdapter:
    """Zero-shot document classifier using GLiClass models.

    __call__(text) returns a list of ``{"label": str, "score": float}`` dicts.
    """

    def __init__(
        self,
        model_id: str,
        labels: Tuple[str, ...] = (),
        threshold: float = 0.5,
        classification_type: str = 'single-label',
        onnx: bool = False,
    ):
        self.model_id = model_id
        self.labels = list(labels)
        self.threshold = threshold
        self.classification_type = classification_type
        self._onnx = onnx
        self._pipeline = None

        self._init_model(model_id)

        logger.info(
            "Loaded GLiClass model: %s (onnx=%s, labels=%d)",
            model_id, onnx, len(self.labels),
        )

    def _init_model(self, model_id: str) -> None:
        """Load GLiClass model via gliclass package.

        Note: gliclass does not yet expose a native ONNX loader, so the
        ``onnx`` flag is recorded but both paths use the same PyTorch-backed
        ``GLiClassModel.from_pretrained``.  When gliclass adds ONNX support,
        this method should branch on ``self._onnx``.
        """
        try:
            from gliclass import GLiClassModel, ZeroShotClassificationPipeline  # noqa: S404
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError(
                "gliclass is required for GLiClass classification backend. "
                "Install with: pip install squeakycleantext[classify]"
            )

        model = GLiClassModel.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._pipeline = ZeroShotClassificationPipeline(
            model=model,
            tokenizer=tokenizer,
            classification_type=self.classification_type,
            device='cpu',
        )

    def classify(self, text: str) -> List[Dict[str, float]]:
        """Classify text against configured labels.

        Returns:
            List of ``{"label": str, "score": float}`` dicts,
            filtered by threshold and sorted by score descending.
        """
        if self._pipeline is None:
            return []

        result = self._pipeline(
            text,
            labels=self.labels,
        )

        # Pipeline returns list[list[dict]] — one list per input text,
        # each containing {"label": str, "score": float} dicts.
        classifications = []
        entries = result[0] if result else []
        for entry in entries:
            if entry['score'] >= self.threshold:
                classifications.append({'label': entry['label'], 'score': entry['score']})

        classifications.sort(key=lambda x: x['score'], reverse=True)
        return classifications

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded and ready for inference."""
        return self._pipeline is not None
