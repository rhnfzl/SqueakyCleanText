"""SCT adapter: entries ``sct-default``, ``sct-gliner`` (``{"mode": "gliner"}`` -> ``ner_mode='pii'``),
``sct-training`` (``{"preset": "training"}``) and ``sct-gliner-nvidia`` (gliner mode, ``gliner_model`` override).

Spans come from reversible mode, as in ``benchmarks.run``: each ``<TYPE_n>`` placeholder is
mapped back to the source span it replaced. ``masked_text`` is SCT's own output (the native
pipeline track). SCT gives no per-span score, so ``score`` is None; its operating point is
moved with ``overrides`` (``ner_confidence_threshold``, ``gliner_threshold``).

Config: ``mode`` (default|gliner), ``preset`` (a ``sct.presets`` name, applied under the scoring settings),
``language`` (``gold``: one cleaner per gold language, pinned;
``auto``: SCT's own detection), ``threads``, ``workers`` (process_batch concurrency),
``device`` (cpu|cuda), ``gpu_mem_gb``, ``overrides`` (extra TextCleanerConfig fields).
"""

import difflib
from bisect import bisect_left
from collections import Counter
import os
import re
import sys
import unicodedata
from collections.abc import Callable
from pathlib import Path
from typing import Any

# Repo root goes AFTER site-packages: the venv's pinned sct wins over the checkout's sources.
sys.path.append(str(Path(__file__).resolve().parents[3]))

from _protocol import cap_gpu, serve  # noqa: E402


def sct_language(lang: str | None) -> str | None:
    """Gold code -> a language SCT accepts (``ar-mixed_en`` -> ``ar``), or None (auto) when it has none."""
    from sct.utils.resources import resolve_language

    for code in (lang, (lang or "").replace("_", "-").split("-")[0]):
        try:
            return resolve_language(code) if code else None
        except ValueError:
            continue
    return None


def sct_view(source: str, normalise) -> tuple[str, list[int]]:
    """SCT's normalised view of *source* (NFKC of *normalise*) and the source offset of each view char.

    *normalise* is SCT's own pre-regex normalisation (unicode fix, entity decoding, invisible-char
    removal, ``unescape_markup``), so values SCT stored after decoding ("ana@x" from "ana&#64;x")
    are found and map back to the source range that produced them. SCT's steps read a text split
    at whitespace as the whole (sct/streaming.py relies on it), so each whitespace-separated token
    is normalised on its own and a changed token is aligned to its view char by char.
    shortcut: the alignment is difflib's, so a token rewritten beyond recognition maps its view chars
    to the start of the changed run; fine for decoding and dropping chars, revisit if a step reorders text.
    """
    pieces, offsets = [], []
    for match in re.finditer(r"\S+|\s+", source):
        token, at = match.group(), match.start()
        piece = token if token.isspace() else unicodedata.normalize("NFKC", normalise(token))
        pieces.append(piece)
        if piece == token:
            offsets.extend(range(at, at + len(token)))
            continue
        for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, token, piece, autojunk=False).get_opcodes():
            # chars inserted after the token's last char belong to that char, not to the next token
            offsets.extend(range(at + i1, at + i2) if op == "equal" else [at + min(i1, len(token) - 1)] * (j2 - j1))
    offsets.append(len(source))
    return "".join(pieces), offsets


def unplaced_reason(value: str, view: str) -> str:
    """Why a value was not placed: ``no_value`` (no anon_map entry), ``in_view`` (in the view as written:
    placement refused it, the cursor or window excluded it or only unmasked copies matched), ``separators``
    (its letters and digits are in the view, a cleanup changed what lies between them) or ``absent``."""
    if not value:
        return "no_value"
    if " ".join(value.split()) in " ".join(view.split()):
        return "in_view"
    alnum = re.sub(r"\W", "", value)
    return "separators" if alnum and alnum in re.sub(r"\W", "", view) else "absent"


def misplaced(context: str, view: tuple[str, list[int]], start: int) -> bool | None:
    """Whether a placed span looks misplaced: the last 2+ char word of the output *context* differs from
    the last one before the span in *view*. None when the context has no such word to check."""
    words = re.findall(r"\w{2,}", context)
    if not words:
        return None
    before = re.findall(r"\w{2,}", view[0][:bisect_left(view[1], start)][-200:])
    return not before or before[-1] != words[-1]


def cleaner_fields(config: dict[str, Any]) -> dict[str, Any]:
    """Adapter config -> TextCleanerConfig fields."""
    from benchmarks.score import SCORING

    base = dict(SCORING)
    if config.get("preset"):
        from sct.presets import PRESETS

        base = {**PRESETS[config["preset"]], **SCORING}
    if config.get("mode", "default") == "gliner":
        base["ner_mode"] = "pii"
        cap_gpu(config)  # GLiNER runs on torch: the ONNX gpu_mem_limit below does not reach it
    if config.get("threads"):
        base["onnx_intra_op_threads"] = int(config["threads"])
    if config.get("device"):
        base["ner_device"] = config["device"]
    if config.get("device") == "cuda":
        import onnxruntime

        if hasattr(onnxruntime, "preload_dlls"):  # CUDA 12 / cuDNN from the pip nvidia-* wheels (onnxruntime-gpu[cuda,cudnn])
            onnxruntime.preload_dlls()
    if config.get("device") == "cuda" and config.get("gpu_mem_gb"):
        base["onnx_cuda_provider_options"] = {"gpu_mem_limit": int(config["gpu_mem_gb"] * 2**30)}
    base.update(config.get("overrides", {}))
    return base


def load(config: dict[str, Any]) -> Callable[[list[dict]], list[dict]]:
    from sct.config import TextCleanerConfig
    from sct.sct import TextCleaner

    from sct.utils.normtext import unescape_markup

    from benchmarks.score import _NUMBERED_PLACEHOLDER as PLACEHOLDER_REGEX, predicted_spans

    gold = config.get("language", "gold") == "gold"
    workers = config.get("workers")
    # One cleaner for every language: a pinned ``language`` is only read when a doc is
    # processed (sct.TextCleaner._detect_language), so switching it per batch equals a
    # cleaner per language without loading the NER models once per language.
    model = TextCleaner(cfg=TextCleanerConfig(**cleaner_fields(config)))

    def normalise(token: str) -> str:
        return unescape_markup(model.normalise(token))

    warned: set = set()
    # Set SCT_ADAPTER_UNPLACED_DEBUG=1 for per-batch counts of why spans stay unplaced (stderr, no text).
    debug = os.environ.get("SCT_ADAPTER_UNPLACED_DEBUG")

    def predict(docs: list[dict]) -> list[dict]:
        groups: dict = {}
        for i, doc in enumerate(docs):
            key = sct_language(doc.get("lang")) if gold else None
            if gold and key is None and doc.get("lang") not in warned:
                warned.add(doc.get("lang"))
                print(f"sct_adapter: no SCT language for {doc.get('lang')!r}, using detection", file=sys.stderr)
            groups.setdefault(key, []).append(i)
        out: list = [None] * len(docs)
        reasons: Counter = Counter()
        for key, indices in groups.items():
            object.__setattr__(model.cfg, "language", key)
            # Adapter environments pin SCT releases that still call this argument batch_size.
            results = model.process_batch([docs[i]["text"] for i in indices], batch_size=workers)
            for i, result in zip(indices, results, strict=True):
                anon_map = result.metadata.get("anon_map")
                originals = {e.placeholder: e.original for e in anon_map.entries} if anon_map else {}
                view = sct_view(docs[i]["text"], normalise)
                spans = predicted_spans(docs[i]["text"], result.lm_text, originals, view, model.cfg)
                placed = [{"start": s.start, "end": s.end, "label": s.label, "score": None} for s in spans if s.start >= 0]
                out[i] = {"spans": placed, "masked_text": result.lm_text, "unplaced": len(spans) - len(placed)}
                if debug:
                    tags = [m.group(0) for m in PLACEHOLDER_REGEX.finditer(result.lm_text)]
                    reasons.update(f"{s.label}:{unplaced_reason(originals.get(tag, ''), view[0])}"
                                   for tag, s in zip(tags, spans, strict=True) if s.start < 0)
                    contexts = re.split(PLACEHOLDER_REGEX.pattern, unicodedata.normalize("NFKC", result.lm_text))
                    reasons.update(f"{s.label}:placed_{'unchecked' if bad is None else 'suspect' if bad else 'ok'}"
                                   for s, context in zip(spans, contexts[::PLACEHOLDER_REGEX.groups + 1])
                                   if s.start >= 0 for bad in [misplaced(context, view, s.start)])
        if reasons:
            print(f"sct_adapter: unplaced {dict(reasons)}", file=sys.stderr)
        return out

    return predict


if __name__ == "__main__":
    serve(load)
