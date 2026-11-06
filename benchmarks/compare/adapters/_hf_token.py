"""Shared runner for BIO / IO token-classification checkpoints on ``transformers`` (piiranha, ai4privacy).

Standard library at import time; transformers and torch load inside ``load``. The vendor path for these
cards is the transformers ``token-classification`` pipeline with ``aggregation_strategy="simple"``
(B-/I- tags merged into one span per entity, the span score the mean of its token scores). Long docs
use the pipeline's own windowing: ``model_max_length`` tokens per window with ``stride`` tokens of
overlap, spans remapped to document offsets by the pipeline. Offsets are the fast tokenizer's
character offsets (code points); whitespace that a subword token carries at a span edge is trimmed.
"""

from _protocol import cap_gpu


def label_set(id2label: dict) -> set[str]:
    """Entity types of a BIO / IO head: ``{"B-CITY", "I-CITY", "O"}`` -> ``{"CITY"}``."""
    return {label.split("-", 1)[1] for label in id2label.values() if label != "O"}


def to_spans(text: str, found: list[dict], threshold: float = 0.0) -> list[dict]:
    """Pipeline entities (``entity_group``, ``start``, ``end``, ``score``) -> contract spans."""
    spans = []
    for ent in found:
        start, end = int(ent["start"]), int(ent["end"])
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if end > start and float(ent["score"]) >= threshold:
            spans.append({"start": start, "end": end, "label": ent["entity_group"], "score": float(ent["score"])})
    return sorted(spans, key=lambda s: (s["start"], s["end"]))


def load(model: str, revision: str, labels: set[str], config: dict, max_len: int, stride: int):
    """``predict`` for *model*; refuses to run when the checkpoint's types differ from *labels* (the frozen map)."""
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

    if config.get("threads"):
        torch.set_num_threads(int(config["threads"]))
    device = config.get("device") or "cpu"
    cap_gpu(config)
    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
    tokenizer.model_max_length = int(config.get("max_len") or max_len)
    net = AutoModelForTokenClassification.from_pretrained(model, revision=revision)
    found = label_set(net.config.id2label)
    if found != labels:
        raise RuntimeError(f"{model} labels changed: {sorted(found ^ labels)}")
    nlp = pipeline("token-classification", model=net, tokenizer=tokenizer, aggregation_strategy="simple",
                   stride=int(config.get("stride") or stride), device=device)
    threshold = float(config.get("threshold") or 0.0)

    def predict(docs: list[dict]) -> list[dict]:
        out = nlp([d["text"] for d in docs], batch_size=len(docs))
        return [{"spans": to_spans(d["text"], ents, threshold)} for d, ents in zip(docs, out, strict=True)]

    return predict
