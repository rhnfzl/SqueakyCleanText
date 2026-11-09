"""perplexity-ai/PII-Tracer adapter: entry ``pii-tracer`` (bidirectional Qwen3 encoder, BIOES + Viterbi, MIT).

The model card's quick start: ``AutoModel.from_pretrained(repo, trust_remote_code=True)`` at the pinned
revision (the remote code, ``modeling_pii_masking.py``, is pinned with it) and ``spans, sensitivity =
model.predict(text)`` per doc. The bundled decoder already returns whitespace-trimmed character spans
(the fast tokenizer's code-point offsets) with a score (sigmoid of the mean decoded logit); the
document sensitivity is not a span and is dropped. ``masked_text`` is the vendor's ``mask()`` format
(``[PRIVATE_EMAIL]``), built from the same spans so the model runs once.

Long docs, the vendor's way: ``predict`` truncates at ``max_seq_len`` = 4096 tokens; spans past it are
never seen (PROTOCOL section 11). No batch API: docs run one by one. ``lang`` is ignored.

Config: ``device`` (cpu|cuda), ``threads``, ``revision``, ``gpu_mem_gb``.
"""

from _metadata import PII_TRACER as LABELS
from _metadata import PII_TRACER_MODEL as MODEL, PII_TRACER_REVISION as REVISION
from _protocol import cap_gpu, serve


OUT_OF_SCOPE = ("secret", "other_pii")


def to_result(text: str, spans) -> dict:
    """Vendor ``PredictedSpan`` objects -> contract spans + the vendor's ``[LABEL]`` masked text."""
    out = sorted(({"start": s.start, "end": s.end, "label": s.label, "score": float(s.score)} for s in spans),
                 key=lambda s: (s["start"], s["end"]))
    masked = text
    for s in sorted(out, key=lambda s: -s["start"]):
        masked = masked[:s["start"]] + f"[{s['label'].upper()}]" + masked[s["end"]:]
    return {"spans": out, "masked_text": masked}


def load(config: dict):
    import torch
    from transformers import AutoModel

    if config.get("threads"):
        torch.set_num_threads(int(config["threads"]))
    device = config.get("device") or "cpu"
    cap_gpu(config)
    model = AutoModel.from_pretrained(MODEL, revision=config.get("revision", REVISION), trust_remote_code=True)
    model = model.to(device).eval()
    types = set(getattr(model.viterbi, "pii_types", ()))
    if types != set(LABELS) | set(OUT_OF_SCOPE):
        raise RuntimeError(f"PII-Tracer labels changed: {sorted(types ^ (set(LABELS) | set(OUT_OF_SCOPE)))}")

    def predict(docs: list[dict]) -> list[dict]:
        return [to_result(d["text"], model.predict(d["text"])[0]) for d in docs]

    return predict


if __name__ == "__main__":
    serve(load)
