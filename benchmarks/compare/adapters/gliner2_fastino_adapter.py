"""GLiNER2-PII adapter: entry ``gliner2-fastino`` (gliner2 2.1.0, fastino/gliner2-privacy-filter-PII-multi).

Default track = the model card quick start: ``GLiNER2.from_pretrained(...)``,
``extract_entities(text, labels, threshold=0.5, include_confidence=True, include_spans=True)``,
with the protocol's fixed ``GLINER_PROMPTS`` list as the labels (entries.py, PROTOCOL section 3).
Long docs go through the library's own ``batch_extract_entities_long`` (vendor chunker: 384
whitespace words per window, 64-word overlap, chunk spans remapped to document offsets and
merged by the library). A doc of at most 384 words is one window, i.e. the plain quick start.
Context limit: 384 words per window (the library default ``chunk_size``); the DeBERTa-v3
encoder itself has relative positions and no hard cap. The model has no masking output, so
no ``masked_text``: the runner applies the common mask.

Config (tuned track may change only ``threshold`` and ``labels``, PROTOCOL section 3):
``threshold`` (0.5), ``labels`` (``gliner``: GLINER_PROMPTS keys; ``card``: the 42 card labels,
see CARD_LABELS), ``device`` (cpu|cuda), ``threads``, ``encoder_batch`` (4: entries.py "batch<=4",
GPU memory cap), ``chunk_size`` / ``chunk_overlap`` (vendor 384 / 64; recorded, not tuned),
``fp16`` (False; speed only). ``gpu_mem_gb`` is ignored (torch has no hard cap; batch caps it).
"""

from _metadata import GLINER2_FASTINO_MODEL as CHECKPOINT, GLINER2_FASTINO_REVISION as REVISION
from _metadata import FASTINO_CARD_LABELS as CARD_LABELS, GLINER_PROMPTS
from _protocol import serve


def prompts(choice: str) -> list[str]:
    if choice == "card":
        return list(CARD_LABELS)
    if choice == "gliner":
        return list(GLINER_PROMPTS)
    raise ValueError(f"labels must be 'gliner' or 'card', got {choice!r}")


def to_spans(result: dict) -> dict:
    """Library result (``{"entities": {label: [{"text", "confidence", "start", "end"}]}}``) -> contract."""
    spans, unplaced = [], 0
    for label, items in result.get("entities", {}).items():
        for item in items:
            start, end = item.get("start"), item.get("end")
            if start is None or end is None:
                unplaced += 1  # no offsets: never guessed with text.find (the card's redact() does that)
                continue
            spans.append({"start": start, "end": end, "label": label, "score": item.get("confidence")})
    spans.sort(key=lambda s: (s["start"], s["end"]))
    return {"spans": spans, "unplaced": unplaced}


def load(config: dict):
    import torch
    from gliner2 import GLiNER2

    if config.get("threads"):
        torch.set_num_threads(int(config["threads"]))
    device = config.get("device") or "cpu"
    model = GLiNER2.from_pretrained(CHECKPOINT, revision=config.get("revision", REVISION),
                                    map_location=device, quantize=bool(config.get("fp16", False)))
    model.to(device)
    labels = prompts(config.get("labels", "gliner"))
    threshold = float(config.get("threshold", 0.5))
    chunk_size, chunk_overlap = int(config.get("chunk_size", 384)), int(config.get("chunk_overlap", 64))
    encoder_batch = int(config.get("encoder_batch", 4))

    def predict(docs: list[dict]) -> list[dict]:
        texts = [d["text"] for d in docs]
        with torch.inference_mode():
            results = model.batch_extract_entities_long(
                texts, labels, batch_size=encoder_batch, threshold=threshold, include_confidence=True,
                include_spans=True, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return [to_spans(result) for result in results]

    return predict


if __name__ == "__main__":
    serve(load)
