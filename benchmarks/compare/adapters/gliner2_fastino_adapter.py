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

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3]))  # repo root, for the shared prompt list

from _protocol import serve  # noqa: E402

CHECKPOINT = "fastino/gliner2-privacy-filter-PII-multi"
REVISION = "1cb4166094dc58fa8d836429f060d6c95f62b495"

# The model card's 42 labels -> SCT type (None = out of scope: credentials, usernames).
# shortcut: entries.py maps only GLINER_PROMPTS; copy this into entries.py labels if the tuned
# track picks ``card`` (one choice per model, PROTOCOL section 3 rule 4).
CARD_LABELS = {
    **dict.fromkeys(("person", "full_name", "first_name", "middle_name", "last_name"), "PERSON"),
    **dict.fromkeys(("address", "street_address", "city", "state_or_region", "country"), "LOCATION"),
    "email": "EMAIL", "phone_number": "PHONE", "ip_address": "IP",
    **dict.fromkeys(("date_of_birth", "sensitive_date", "document_date", "expiration_date",
                     "transaction_date", "card_expiry"), "DATE"),
    **dict.fromkeys(("postal_code", "government_id", "national_id_number", "passport_number",
                     "drivers_license_number", "license_number", "tax_id", "tax_number", "bank_account",
                     "account_number", "routing_number", "iban", "payment_card", "card_number", "card_cvv",
                     "account_id", "sensitive_account_id"), "NUMBER"),
    **dict.fromkeys(("username", "password", "secret", "api_key", "access_token", "recovery_code"), None),
}


def prompts(choice: str) -> list[str]:
    if choice == "card":
        return list(CARD_LABELS)
    if choice == "gliner":
        from benchmarks.compare.entries import GLINER_PROMPTS

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
