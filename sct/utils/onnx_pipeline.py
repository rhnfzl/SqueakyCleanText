"""ONNX Runtime NER pipeline — drop-in replacement for HuggingFace pipeline("ner").

Reimplements ``aggregation_strategy="simple"`` using raw onnxruntime + tokenizers,
eliminating the torch and transformers dependencies (~2 GB) while producing
byte-identical entity span and label output.
"""

import json
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TokenizerWrapper — adapts ``tokenizers.Tokenizer`` to the interface that
# ``split_text()`` and ``_token_count()`` in ner.py expect.
# ---------------------------------------------------------------------------

class _EncodingResult:
    """Lightweight wrapper around a tokenizers.Encoding.

    Exposes ``.input_ids`` and optionally ``.offset_mapping`` with the same
    attribute names that the rest of the NER code uses (matching the HF
    ``AutoTokenizer`` return convention).
    """

    __slots__ = ('input_ids', 'offset_mapping')

    def __init__(self, ids: List[int],
                 offsets: Optional[List[Tuple[int, int]]] = None):
        self.input_ids = ids
        self.offset_mapping = offsets


class TokenizerWrapper:
    """Wraps ``tokenizers.Tokenizer`` to expose the subset of the HF
    ``AutoTokenizer`` interface used by ``split_text()`` and ``_token_count()``.

    Key mappings:
        encoding.ids → result.input_ids
        encoding.offsets → result.offset_mapping
        tokenizer_config["model_max_length"] → .max_len_single_sentence
    """

    def __init__(self, tokenizer: Tokenizer, model_max_length: int = 512):
        self._tokenizer = tokenizer
        self._model_max_length = model_max_length

    @property
    def max_len_single_sentence(self) -> int:
        """Maximum sequence length minus special tokens (CLS + SEP)."""
        return self._model_max_length - 2

    def __call__(self, text: str, *,
                 return_offsets_mapping: bool = False) -> _EncodingResult:
        """Tokenize *text* and return an HF-compatible result object."""
        encoding = self._tokenizer.encode(text)
        offsets = encoding.offsets if return_offsets_mapping else None
        return _EncodingResult(ids=encoding.ids, offsets=offsets)

    def encode_for_inference(self, text: str):
        """Return the full ``tokenizers.Encoding`` for ONNX inference."""
        return self._tokenizer.encode(text)


# ---------------------------------------------------------------------------
# ONNXNERPipeline — callable pipeline matching HF output format
# ---------------------------------------------------------------------------

class ONNXNERPipeline:
    """ONNX-backed NER pipeline producing the same output format as
    ``transformers.pipeline("ner", aggregation_strategy="simple")``.

    Parameters
    ----------
    session : ort.InferenceSession
        ONNX Runtime session loaded from ``model.onnx``.
    tokenizer_wrapper : TokenizerWrapper
        Wrapped tokenizer for inference.
    id2label : Dict[int, str]
        Mapping from model output indices to label strings (e.g. ``{0: "O", 1: "B-PER", …}``).
    """

    def __init__(self, session: ort.InferenceSession,
                 tokenizer_wrapper: TokenizerWrapper,
                 id2label: Dict[int, str]):
        self._session = session
        self._tokenizer = tokenizer_wrapper
        self._id2label = id2label

        # Detect which inputs the model expects (BERT needs token_type_ids,
        # XLM-RoBERTa does not).
        self._input_names = [inp.name for inp in session.get_inputs()]
        self._needs_token_type_ids = 'token_type_ids' in self._input_names

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __call__(self, texts) -> List[List[dict]]:
        """Run NER on one or more texts.

        Parameters
        ----------
        texts : str | List[str]
            A single string or a list of strings.

        Returns
        -------
        List[List[dict]]
            One list of entity dicts per input text, each with keys:
            ``entity_group``, ``score``, ``word``, ``start``, ``end``.
        """
        if isinstance(texts, str):
            texts = [texts]

        all_results: List[List[dict]] = []
        for text in texts:
            encoding = self._tokenizer.encode_for_inference(text)

            # Build ONNX input feeds
            input_ids = np.array([encoding.ids], dtype=np.int64)
            attention_mask = np.array([encoding.attention_mask], dtype=np.int64)

            feeds: Dict[str, np.ndarray] = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
            }
            if self._needs_token_type_ids:
                feeds['token_type_ids'] = np.array(
                    [encoding.type_ids], dtype=np.int64
                )

            # Run inference — returns list of output arrays
            logits = self._session.run(None, feeds)[0]  # (1, seq_len, num_labels)

            scores = self._softmax(logits[0])  # (seq_len, num_labels)
            entities = self._aggregate_simple(
                text, encoding, scores
            )
            all_results.append(entities)

        return all_results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        """Numerically stable softmax along the last axis."""
        shifted = logits - logits.max(axis=-1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=-1, keepdims=True)

    def _aggregate_simple(self, text: str, encoding, scores: np.ndarray) -> List[dict]:
        """Reimplements HuggingFace ``aggregation_strategy="simple"``.

        For each non-special token:
        1. label = argmax(softmax(logits))
        2. score = softmax[argmax_idx]
        3. Group consecutive B-TAG + I-TAG tokens with matching base tag
        4. Group score = mean of per-token scores
        5. Word = text[start:end] using character offsets
        """
        special_tokens_mask = encoding.special_tokens_mask
        offsets = encoding.offsets

        # Per-token predictions (skip special tokens and "O" labels)
        pre_entities = []
        for idx in range(len(scores)):
            if special_tokens_mask[idx]:
                continue

            label_id = int(np.argmax(scores[idx]))
            label = self._id2label[label_id]
            if label == 'O':
                continue

            token_score = float(scores[idx][label_id])
            start, end = offsets[idx]

            # Parse BIO prefix
            if '-' in label:
                bio_prefix, entity_type = label.split('-', 1)
            else:
                bio_prefix, entity_type = 'B', label

            pre_entities.append({
                'bio': bio_prefix,
                'entity_type': entity_type,
                'score': token_score,
                'start': start,
                'end': end,
            })

        # Group consecutive tokens with matching entity type
        if not pre_entities:
            return []

        groups: List[List[dict]] = []
        current_group = [pre_entities[0]]

        for token in pre_entities[1:]:
            prev = current_group[-1]
            # Continue group if: same entity type AND (I-tag OR adjacent B-tag)
            # HF "simple" merges B+I of same type, and consecutive same-type tokens
            if (token['entity_type'] == prev['entity_type']
                    and token['bio'] in ('I', 'B')
                    and token['start'] <= prev['end'] + 1):
                # Only continue with I-tags or adjacent tokens of same type
                if token['bio'] == 'I' or token['start'] == prev['end']:
                    current_group.append(token)
                    continue
            # Start new group
            groups.append(current_group)
            current_group = [token]

        groups.append(current_group)

        # Convert groups to output entities
        entities = []
        for group in groups:
            entity_type = group[0]['entity_type']
            start = group[0]['start']
            end = group[-1]['end']
            avg_score = float(np.mean([t['score'] for t in group]))
            word = text[start:end]

            entities.append({
                'entity_group': entity_type,
                'score': avg_score,
                'word': word,
                'start': start,
                'end': end,
            })

        return entities


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def load_onnx_ner_model(
    model_name: str,
    device: str = 'cpu',
    cache_dir: Optional[str] = None,
) -> Tuple[ONNXNERPipeline, TokenizerWrapper]:
    """Download and load an ONNX NER model from HuggingFace Hub.

    Parameters
    ----------
    model_name : str
        HuggingFace repo ID (e.g. ``"rhnfzl/bert-base-NER-onnx"``).
    device : str
        ``"cpu"`` or ``"cuda"``.
    cache_dir : str, optional
        Local directory for caching downloaded files.

    Returns
    -------
    (ONNXNERPipeline, TokenizerWrapper)
        Ready-to-use pipeline and tokenizer.
    """
    download_kwargs = {}
    if cache_dir:
        download_kwargs['cache_dir'] = cache_dir

    # Download required files
    model_path = hf_hub_download(model_name, 'model.onnx', **download_kwargs)
    config_path = hf_hub_download(model_name, 'config.json', **download_kwargs)
    tokenizer_path = hf_hub_download(model_name, 'tokenizer.json', **download_kwargs)

    # Large models (XLM-RoBERTa-large ~2.1GB) store weights in an external
    # data file. ONNX Runtime finds it automatically if it's in the same
    # directory as model.onnx — hf_hub_download caches to the same dir.
    try:
        hf_hub_download(model_name, 'model.onnx_data', **download_kwargs)
    except Exception:  # noqa: S110
        pass  # Smaller models don't have this file

    # Try to get tokenizer_config.json for model_max_length
    model_max_length = 512  # safe default
    try:
        tok_config_path = hf_hub_download(
            model_name, 'tokenizer_config.json', **download_kwargs
        )
        with open(tok_config_path) as f:
            tok_config = json.load(f)
        model_max_length = tok_config.get('model_max_length', 512)
        # Some models set absurdly large values (e.g. 1e30)
        if model_max_length > 100_000:
            model_max_length = 512
    except Exception:
        logger.debug("tokenizer_config.json not found, using default max_length=512")

    # Load id2label from config.json
    with open(config_path) as f:
        model_config = json.load(f)
    id2label = {int(k): v for k, v in model_config['id2label'].items()}

    # Load tokenizer
    tokenizer = Tokenizer.from_file(tokenizer_path)
    wrapper = TokenizerWrapper(tokenizer, model_max_length=model_max_length)

    # Create ONNX session with appropriate providers
    providers = []
    if device == 'cuda' and 'CUDAExecutionProvider' in ort.get_available_providers():
        providers.append('CUDAExecutionProvider')
    providers.append('CPUExecutionProvider')

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(
        model_path,
        sess_options=session_options,
        providers=providers,
    )

    pipeline = ONNXNERPipeline(session, wrapper, id2label)

    logger.info(f"Loaded ONNX NER model: {model_name} (providers: {session.get_providers()})")

    return pipeline, wrapper
