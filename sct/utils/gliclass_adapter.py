"""GLiClass adapter — zero-shot document classification.

Optional dependency. Requires: ``pip install squeakycleantext[classify]``
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class GLiClassAdapter:
    """Zero-shot document classifier using GLiClass models.

    classify(text) returns a list of ``{"label": str, "score": float}`` dicts.
    """

    def __init__(
        self,
        model_id: str,
        revision: str | None = None,
        labels: tuple[str, ...] = (),
        threshold: float = 0.5,
        classification_type: str = 'single-label',
        onnx: bool = False,
        device: str = 'cpu',
    ):
        self.model_id = model_id
        self._device = device
        self.labels = list(labels)
        self.threshold = threshold
        self.classification_type = classification_type
        self._onnx = onnx
        self._pipeline = None
        self._revision = revision
        if onnx:
            raise ValueError(
                "GLiClass ONNX is not supported by the gliclass package."
            )

        self._init_model(model_id)

        logger.info(
            "Loaded GLiClass model: %s (onnx=%s, labels=%d)",
            model_id, onnx, len(self.labels),
        )

    def _init_model(self, model_id: str) -> None:
        """Load GLiClass model via gliclass package.

        The gliclass package currently exposes a PyTorch loader only.
        """
        try:
            from gliclass import GLiClassModel, ZeroShotClassificationPipeline  # noqa: S404
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "gliclass is required for GLiClass classification backend. "
                "Install with: pip install squeakycleantext[classify]"
            ) from exc

        load_kwargs = {"revision": self._revision} if self._revision else {}
        model = GLiClassModel.from_pretrained(model_id, **load_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_id, **load_kwargs)
        self._pipeline = ZeroShotClassificationPipeline(
            model=model,
            tokenizer=tokenizer,
            classification_type=self.classification_type,
            device=self._device,
        )

    def classify(self, text: str) -> list[dict[str, Any]]:
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
