"""PyTorch/Transformers NER pipeline — wraps HuggingFace pipeline("ner").

Optional backend. Requires: pip install squeakycleantext[torch]
"""
import logging
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


class TorchNERPipeline:
    """PyTorch-backed NER pipeline matching ONNXNERPipeline's interface.

    Uses HuggingFace transformers.pipeline("ner", aggregation_strategy="simple")
    which produces the same output format as ONNXNERPipeline.
    """

    def __init__(self, model_name: str, device: str = 'cpu',
                 cache_dir: Optional[str] = None):
        try:
            import torch  # noqa: S404
            from transformers import (  # noqa: S404
                AutoTokenizer, AutoModelForTokenClassification,
                pipeline as hf_pipeline,
            )
        except ImportError:
            raise ImportError(
                "torch and transformers are required for the torch NER backend. "
                "Install with: pip install squeakycleantext[torch]"
            )

        import transformers
        transformers.logging.set_verbosity_error()

        cache_kwargs = {"cache_dir": cache_dir} if cache_dir else {}

        self._tokenizer = AutoTokenizer.from_pretrained(model_name, **cache_kwargs)
        model = AutoModelForTokenClassification.from_pretrained(
            model_name, **cache_kwargs
        ).to(device)

        self._pipeline = hf_pipeline(
            "ner", model=model, tokenizer=self._tokenizer,
            aggregation_strategy="simple", device=device,
        )
        self._device = device
        self._torch = torch

        logger.info("Loaded Torch NER model: %s (device=%s)", model_name, device)

    def __call__(self, texts) -> List[List[dict]]:
        """Run NER inference. Same interface as ONNXNERPipeline."""
        if isinstance(texts, str):
            texts = [texts]

        with self._torch.no_grad():
            results = self._pipeline(texts)

        # HF pipeline returns List[dict] for single text, List[List[dict]] for batch
        if texts and not isinstance(results[0], list):
            results = [results]
        return results

    def cleanup(self):
        """Release GPU memory."""
        if self._device == 'cuda':
            try:
                self._torch.cuda.empty_cache()
            except Exception:  # noqa: S110
                pass


def load_torch_ner_model(
    model_name: str,
    device: str = 'cpu',
    cache_dir: Optional[str] = None,
) -> Tuple['TorchNERPipeline', object]:
    """Load a PyTorch NER model and return (pipeline, tokenizer).

    Returns the same tuple shape as load_onnx_ner_model for interface parity.
    The tokenizer is an HF AutoTokenizer (already has max_len_single_sentence).
    """
    pipe = TorchNERPipeline(model_name, device=device, cache_dir=cache_dir)
    return pipe, pipe._tokenizer
