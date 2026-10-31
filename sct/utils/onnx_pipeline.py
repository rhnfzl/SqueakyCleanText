"""ONNX Runtime NER pipeline — drop-in replacement for HuggingFace pipeline("ner").

Reimplements ``aggregation_strategy="simple"`` using raw onnxruntime + tokenizers,
eliminating the torch and transformers dependencies (~2 GB) while producing
byte-identical entity span and label output.
"""

import ctypes
import hashlib
import json
import logging
import os
import re
import shutil
import sys
import unicodedata
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download, try_to_load_from_cache
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
        # A saved truncation shorter than model_max_length still cuts inference input,
        # so windows are planned within the smaller of the two.
        truncation = tokenizer.truncation
        if truncation and truncation.get("max_length"):
            model_max_length = min(model_max_length, truncation["max_length"])
        self._model_max_length = model_max_length
        # Chunk planning must see every token: a truncating tokenizer.json would
        # hide the text past max_length and leave it out of model input.
        self._counter = Tokenizer.from_str(tokenizer.to_str())
        self._counter.no_truncation()
        self._counter.no_padding()  # fixed padding would count every text as max_length tokens

    @property
    def max_len_single_sentence(self) -> int:
        """Maximum sequence length minus special tokens (CLS + SEP)."""
        return self._model_max_length - 2

    def __call__(self, text: str, *,
                 return_offsets_mapping: bool = False) -> _EncodingResult:
        """Tokenize *text* and return an HF-compatible result object."""
        encoding = self._counter.encode(text)
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

    provider: Optional[str] = None  # set by load_onnx_ner_model
    revision: Optional[str] = None

    def __init__(self, session: ort.InferenceSession,
                 tokenizer_wrapper: TokenizerWrapper,
                 id2label: Dict[int, str],
                 pad_id: int = 0):
        self._session = session
        self._tokenizer = tokenizer_wrapper
        self._id2label = id2label
        self._pad_id = pad_id
        # WordPiece marks a word's continuation pieces ("##e"); entities end on whole words.
        model = getattr(getattr(tokenizer_wrapper, '_tokenizer', None), 'model', None)
        self._subword_prefix = getattr(model, 'continuing_subword_prefix', None) or None

        # Detect which inputs the model expects (BERT needs token_type_ids,
        # XLM-RoBERTa does not).
        inputs = session.get_inputs()
        self._input_names = [inp.name for inp in inputs]
        self._needs_token_type_ids = 'token_type_ids' in self._input_names
        # Models exported with a fixed batch dimension of 1 take one row per run.
        self._single_row = any(inp.shape and inp.shape[0] == 1 for inp in inputs)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __call__(self, texts, batch_size: int = 1, max_workers: int = 1,
                 subword_widen: Optional[float] = 1.0,
                 subword_same_script: bool = True) -> List[List[dict]]:
        """Run NER on one or more texts.

        Parameters
        ----------
        texts : str | List[str]
            A single string or a list of strings.
        batch_size : int
            Texts per ``session.run``. Rows are right-padded to the longest text
            in the batch; ``1`` runs each text alone with no padding.
        max_workers : int
            Threads running batches at once (``session.run`` is thread-safe).
        subword_widen : float | None
            WordPiece only: widen an entity to its whole word when the added text is at
            most this many times the tagged span; ``None`` always, ``0`` never.
        subword_same_script : bool
            WordPiece only: never widen over a piece whose letters are in another Unicode
            script than the entity's edge letter (``David Miller`` + ``##は``).

        Returns
        -------
        List[List[dict]]
            One list of entity dicts per input text (input order), each with keys:
            ``entity_group``, ``score``, ``word``, ``start``, ``end``.
        """
        if isinstance(texts, str):
            texts = [texts]

        encodings = [self._tokenizer.encode_for_inference(text) for text in texts]
        all_results: List[List[dict]] = [[] for _ in texts]
        # Similar lengths share a batch, which keeps padding small.
        order = sorted(range(len(texts)), key=lambda i: len(encodings[i].ids))
        batch_size = 1 if self._single_row else max(1, batch_size)

        def run_batch(rows):
            width = max(len(encodings[i].ids) for i in rows)

            def padded(field, pad):
                return np.array(
                    [getattr(encodings[i], field) + [pad] * (width - len(encodings[i].ids))
                     for i in rows],
                    dtype=np.int64,
                )

            feeds: Dict[str, np.ndarray] = {
                'input_ids': padded('ids', self._pad_id),
                'attention_mask': padded('attention_mask', 0),
            }
            if self._needs_token_type_ids:
                feeds['token_type_ids'] = padded('type_ids', 0)

            logits = self._session.run(None, feeds)[0]  # (rows, width, num_labels)

            for row, i in enumerate(rows):
                scores = self._softmax(logits[row, :len(encodings[i].ids)])
                all_results[i] = self._aggregate_simple(
                    texts[i], encodings[i], scores, subword_widen, subword_same_script)

        batches = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
        if max_workers > 1 and len(batches) > 1:
            with ThreadPoolExecutor(max_workers=min(max_workers, len(batches))) as pool:
                list(pool.map(run_batch, batches))
        else:
            for rows in batches:
                run_batch(rows)

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

    def _aggregate_simple(self, text: str, encoding, scores: np.ndarray,
                          subword_widen: Optional[float] = 1.0,
                          subword_same_script: bool = True) -> List[dict]:
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
                'idx': idx,
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
            tokens = getattr(encoding, 'tokens', None) if self._subword_prefix and subword_widen != 0 else None
            if tokens:
                # Widen to whole words so "John Do" + "##e" never leaves "e" behind.
                first, last = group[0]['idx'], group[-1]['idx'] + 1
                # Letters of an added piece must share the script of the entity's edge letter.
                span = _letter_scripts(text[start:end]) if subword_same_script else None
                left, right = ({span[0]}, {span[-1]}) if span else (None, None)

                def same(idx, edge):
                    return not edge or set(_letter_scripts(text[slice(*offsets[idx])])) <= edge

                while first > 0 and tokens[first].startswith(self._subword_prefix) \
                        and not special_tokens_mask[first - 1] and same(first - 1, left):
                    first -= 1
                while last < len(tokens) and not special_tokens_mask[last] \
                        and tokens[last].startswith(self._subword_prefix) and same(last, right):
                    last += 1
                wide_start, wide_end = offsets[first][0], offsets[last - 1][1]
                # Unspaced scripts (Thai, kana) make a whole clause one word: widen only a short tail.
                if subword_widen is None or (wide_end - wide_start) - (end - start) <= subword_widen * (end - start):
                    start, end = wide_start, wide_end
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


def _letter_scripts(text: str) -> List[str]:
    """Script of each letter, from its Unicode name after NFKC ("LATIN", "HIRAGANA", "THAI").

    Marks, digits, punctuation and modifier letters (ー, ๆ) belong to no script here.
    """
    return [unicodedata.name(c, ' ').split(' ', 1)[0]
            for c in unicodedata.normalize('NFKC', text)
            if unicodedata.category(c) in ('Lu', 'Ll', 'Lt', 'Lo')]


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def _quantized_dir(model_name: str, revision: Optional[str]) -> str:
    """Cache dir name for a quantized model; distinct ids and revisions never share one.

    Keeps the 0.7.1 ``org__name`` form unless the id holds ``__`` itself.
    """
    name = (
        quote(model_name, safe="").replace("_", "%5F") if "__" in model_name
        else model_name.replace("/", "__")
    )
    # Lowercase digest: revisions differing only in case must not alias on case-insensitive disks.
    return name + (f"@{hashlib.sha256(revision.encode()).hexdigest()[:16]}" if revision else "")


def _link(src: Path, dst: Path) -> None:
    """Hard-link *src* to *dst*; on macOS make an APFS clone (own inode, shared blocks) instead.

    macOS may report an opened hard link under its other name (``F_GETPATH``), so ONNX Runtime
    then sees ``model.onnx_data`` in the blob store, outside the model directory, and refuses it.
    """
    if sys.platform != 'darwin':
        os.link(src, dst)
    elif ctypes.CDLL(None, use_errno=True).clonefile(os.fsencode(src), os.fsencode(dst), 0):
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err), str(dst))


def _colocate_external_data(model_path: str, data_path: str, link_dir: Path) -> str:
    """Return a model path whose external data file resolves next to it.

    ONNX Runtime only accepts ``model.onnx_data`` whose real path is in the real directory of
    ``model.onnx``. A huggingface-hub cache that stores each blob in its own directory (the
    sharded ``hub/blobs/<xx>/<sha256>`` store) breaks that, so both files are hard-linked
    (cloned on macOS, copied if that fails) into ``link_dir``. Other layouts are returned unchanged.
    """
    real_model, real_data = Path(model_path).resolve(), Path(data_path).resolve()
    if real_model.parent == real_data.parent:
        return model_path
    pairs = [(real_model, 'model.onnx'), (real_data, 'model.onnx_data')]

    def same(src: Path, dst: Path) -> bool:
        if os.path.samefile(src, dst):
            return True
        # A copy is published only when complete; a source blob newer than it means the snapshot moved.
        s, d = src.stat(), dst.stat()
        return s.st_size == d.st_size and d.st_mtime >= s.st_mtime

    def ready() -> bool:
        try:
            return all(same(src, link_dir / name) for src, name in pairs)
        except OSError:
            return False

    if not ready():
        tmp = link_dir.parent / f'.{link_dir.name}.{uuid.uuid4().hex}'
        tmp.mkdir(parents=True)
        try:
            for src, name in pairs:
                try:
                    _link(src, tmp / name)
                except OSError:
                    logger.warning("Cannot hard-link %s, copying it to %s", src, link_dir)
                    shutil.copyfile(src, tmp / name)
            if link_dir.exists() and not ready():
                shutil.rmtree(link_dir, ignore_errors=True)  # stale; a concurrent loader may remove it first
            try:
                os.rename(tmp, link_dir)  # complete directory appears atomically
            except OSError:
                if not ready():  # lost a race only if the winner's directory is good
                    raise
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    return str(link_dir / 'model.onnx')


def load_onnx_ner_model(
    model_name: str,
    device: str = 'cpu',
    cache_dir: Optional[str] = None,
    quantize: bool = False,
    revision: Optional[str] = None,
    intra_op_threads: Optional[int] = None,
    inter_op_threads: Optional[int] = None,
    cuda_device_id: int = 0,
    cuda_provider_options: Optional[Dict[str, Any]] = None,
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
    quantize : bool
        If ``True``, apply INT8 dynamic quantization to the model weights before
        loading.  The quantized model is cached next to the original
        (``model_quantized.onnx``) so quantization only runs once per model.
        Reduces model size ~4× with a small accuracy trade-off.  Default: ``False``.
    intra_op_threads, inter_op_threads : int, optional
        ONNX Runtime session thread counts. ``None`` keeps the ONNX Runtime default.
    cuda_device_id, cuda_provider_options
        GPU index and extra ``CUDAExecutionProvider`` options (passed to ORT unvalidated).

    Returns
    -------
    (ONNXNERPipeline, TokenizerWrapper)
        Ready-to-use pipeline and tokenizer. The pipeline carries ``provider`` (the
        execution provider that runs it) and ``revision`` (resolved snapshot commit).
    """
    download_kwargs = {}
    if cache_dir:
        download_kwargs['cache_dir'] = cache_dir
    if revision:
        download_kwargs['revision'] = revision

    # Download required files
    model_path = hf_hub_download(model_name, 'model.onnx', **download_kwargs)
    # hub cache layout: .../snapshots/<commit>/model.onnx
    commit = Path(model_path).parent.name
    if re.fullmatch(r'[0-9a-f]{40}', commit):
        # The rest comes from this snapshot: a commit hash is served from the cache with no
        # HTTP, so a forked worker does not resolve DNS (which can hang after fork).
        download_kwargs['revision'] = commit

    def optional_download(filename):
        # The Hub's cached "no such file" answer counts too, so the probe needs no HEAD request.
        cached = try_to_load_from_cache(
            model_name, filename, cache_dir=cache_dir, revision=download_kwargs.get('revision'))
        if cached is not None and not isinstance(cached, str):
            raise FileNotFoundError(filename)
        return cached or hf_hub_download(model_name, filename, **download_kwargs)
    config_path = hf_hub_download(model_name, 'config.json', **download_kwargs)
    tokenizer_path = hf_hub_download(model_name, 'tokenizer.json', **download_kwargs)

    # Large models (XLM-RoBERTa-large ~2.1GB) store weights in an external
    # data file. ONNX Runtime needs it in the real directory of model.onnx.
    try:
        data_path = optional_download('model.onnx_data')
    except Exception:  # noqa: S110
        pass  # Smaller models don't have this file
    else:
        link_base = Path(cache_dir) if cache_dir else Path.home() / ".cache"
        model_path = _colocate_external_data(
            model_path, data_path,
            # Keyed by snapshot commit: a moving branch must not reuse another commit's files.
            link_base / "sct_linked" / _quantized_dir(model_name, commit),
        )

    # Try to get tokenizer_config.json for model_max_length
    model_max_length = 512  # safe default
    try:
        tok_config_path = optional_download('tokenizer_config.json')
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

    # Optional INT8 dynamic quantization
    if quantize:
        from onnxruntime.quantization import QuantType, quantize_dynamic  # noqa: PLC0415
        _qcache = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "sct_quantized"
        # Keyed by snapshot commit: a moved branch must not reuse another commit's weights.
        quantized_path = _qcache / _quantized_dir(model_name, commit) / "model_quantized.onnx"
        quantized_path.parent.mkdir(parents=True, exist_ok=True)
        if not quantized_path.exists():
            logger.info("Quantizing model to INT8: %s", quantized_path)
            quantize_dynamic(model_path, str(quantized_path), weight_type=QuantType.QInt8)
        model_path = str(quantized_path)

    # Create ONNX session with appropriate providers
    providers: List[Any] = []
    cuda_requested = device == 'cuda' and 'CUDAExecutionProvider' in ort.get_available_providers()
    if cuda_requested:
        providers.append(('CUDAExecutionProvider',
                          {'device_id': cuda_device_id, **(cuda_provider_options or {})}))
    providers.append('CPUExecutionProvider')

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if intra_op_threads is not None:
        session_options.intra_op_num_threads = intra_op_threads
    if inter_op_threads is not None:
        session_options.inter_op_num_threads = inter_op_threads

    session = ort.InferenceSession(
        model_path,
        sess_options=session_options,
        providers=providers,
    )

    # Pad id for batched inference: config, else the tokenizer's pad token, else 0.
    pad_id = model_config.get('pad_token_id')
    if pad_id is None:
        pad_id = next(
            (tok_id for tok_id in map(tokenizer.token_to_id, ('[PAD]', '<pad>'))
             if tok_id is not None),
            0,
        )
    pipeline = ONNXNERPipeline(session, wrapper, id2label, pad_id=pad_id)
    pipeline.provider = session.get_providers()[0]
    pipeline.revision = commit if re.fullmatch(r'[0-9a-f]{40}', commit) else revision

    if cuda_requested and pipeline.provider != 'CUDAExecutionProvider':
        # ORT drops CUDA silently when the CUDA/cuDNN libraries or the driver do not match.
        logger.warning(
            "ONNX NER model %s runs on CPU: CUDA was requested but could not be loaded. "
            "Check that the onnxruntime-gpu CUDA version matches your driver and that CUDA/cuDNN "
            "libraries are found (onnxruntime-gpu[cuda,cudnn] plus onnxruntime.preload_dlls()).",
            model_name,
        )
    logger.info(f"Loaded ONNX NER model: {model_name} (providers: {session.get_providers()})")

    return pipeline, wrapper
