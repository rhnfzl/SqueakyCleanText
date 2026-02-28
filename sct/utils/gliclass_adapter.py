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

        if onnx:
            self._init_onnx(model_id)
        else:
            self._init_pytorch(model_id)

        logger.info(
            "Loaded GLiClass model: %s (onnx=%s, labels=%d)",
            model_id, onnx, len(self.labels),
        )

    def _init_pytorch(self, model_id: str) -> None:
        """Load GLiClass model via PyTorch (gliclass package)."""
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

    def _init_onnx(self, model_id: str) -> None:
        """Load GLiClass model via ONNX Runtime (torch-free)."""
        try:
            from gliclass import GLiClassModel, ZeroShotClassificationPipeline  # noqa: S404
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError(
                "gliclass + onnxruntime are required for ONNX GLiClass backend. "
                "Install with: pip install squeakycleantext[classify] squeakycleantext[classify-onnx]"
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
            candidate_labels=self.labels,
        )

        # Pipeline returns {"sequence": ..., "labels": [...], "scores": [...]}
        classifications = []
        labels = result.get('labels', [])
        scores = result.get('scores', [])
        for label, score in zip(labels, scores):
            if score >= self.threshold:
                classifications.append({'label': label, 'score': score})

        classifications.sort(key=lambda x: x['score'], reverse=True)
        return classifications

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded and ready for inference."""
        return self._pipeline is not None
