"""SCT adapter: entries ``sct-default`` and ``sct-gliner`` (``{"mode": "gliner"}`` -> ``ner_mode='pii'``).

Spans come from reversible mode, as in ``benchmarks.run``: each ``<TYPE_n>`` placeholder is
mapped back to the source span it replaced. ``masked_text`` is SCT's own output (the native
pipeline track). SCT gives no per-span score, so ``score`` is None; its operating point is
moved with ``overrides`` (``ner_confidence_threshold``, ``gliner_threshold``).

Config: ``mode`` (default|gliner), ``language`` (``gold``: one cleaner per gold language, pinned;
``auto``: SCT's own detection), ``threads``, ``workers`` (process_batch concurrency),
``device`` (cpu|cuda), ``gpu_mem_gb``, ``overrides`` (extra TextCleanerConfig fields).
"""

import sys
from pathlib import Path

# Repo root goes AFTER site-packages: the venv's pinned sct wins over the checkout's sources.
sys.path.append(str(Path(__file__).resolve().parents[3]))

from _protocol import serve  # noqa: E402

SCORING = {"replacement_mode": "reversible", "check_to_ascii_unicode": False}


def sct_language(lang: str | None) -> str | None:
    """Gold code -> a language SCT accepts (``ar-mixed_en`` -> ``ar``), or None (auto) when it has none."""
    from sct.utils.resources import resolve_language

    for code in (lang, (lang or "").replace("_", "-").split("-")[0]):
        try:
            return resolve_language(code) if code else None
        except ValueError:
            continue
    return None


def load(config: dict):
    from sct.config import TextCleanerConfig
    from sct.sct import TextCleaner

    from scripts.benchmark_release import predicted_spans

    base = dict(SCORING)
    if config.get("mode", "default") == "gliner":
        base["ner_mode"] = "pii"
    if config.get("threads"):
        base["onnx_intra_op_threads"] = int(config["threads"])
    if config.get("device"):
        base["ner_device"] = config["device"]
    if config.get("device") == "cuda" and config.get("gpu_mem_gb"):
        base["onnx_cuda_provider_options"] = {"gpu_mem_limit": int(config["gpu_mem_gb"] * 2**30)}
    base.update(config.get("overrides", {}))
    gold = config.get("language", "gold") == "gold"
    workers = config.get("workers")
    # One cleaner for every language: a pinned ``language`` is only read when a doc is
    # processed (sct.TextCleaner._detect_language), so switching it per batch equals a
    # cleaner per language without loading the NER models once per language.
    model = TextCleaner(cfg=TextCleanerConfig(**base))
    warned: set = set()

    def predict(docs: list[dict]) -> list[dict]:
        groups: dict = {}
        for i, doc in enumerate(docs):
            key = sct_language(doc.get("lang")) if gold else None
            if gold and key is None and doc.get("lang") not in warned:
                warned.add(doc.get("lang"))
                print(f"sct_adapter: no SCT language for {doc.get('lang')!r}, using detection", file=sys.stderr)
            groups.setdefault(key, []).append(i)
        out: list = [None] * len(docs)
        for key, indices in groups.items():
            object.__setattr__(model.cfg, "language", key)
            results = model.process_batch([docs[i]["text"] for i in indices], batch_size=workers)
            for i, result in zip(indices, results, strict=True):
                anon_map = result.metadata.get("anon_map")
                originals = {e.placeholder: e.original for e in anon_map.entries} if anon_map else {}
                spans = predicted_spans(docs[i]["text"], result.lm_text, originals)
                placed = [{"start": s.start, "end": s.end, "label": s.label, "score": None} for s in spans if s.start >= 0]
                out[i] = {"spans": placed, "masked_text": result.lm_text, "unplaced": len(spans) - len(placed)}
        return out

    return predict


if __name__ == "__main__":
    serve(load)
