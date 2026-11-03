"""nvidia/gliner-PII adapter (entry ``gliner-nvidia``), run as the model card's quick start.

Default track: ``GLiNER.from_pretrained`` at the pinned revision, ``inference`` with the
library defaults (``flat_ner=True``, ``multi_label=False``) and the quick start's
``threshold=0.5``. The model is English-only, so ``lang`` is ignored.

Long docs, the vendor's way: the whole doc goes in and gliner truncates it at ``max_len``
= 384 whitespace words (``gliner_config.json``; DeBERTa-v3-large encoder, 512 positions).
Words past that limit are never seen; gliner warns on stderr and the doc is scored whole
anyway (PROTOCOL section 11). The common chunker is a separate track.

Config (tuned-track knobs per PROTOCOL section 3 in caps):
  THRESHOLD ``threshold`` (default 0.5; the card's own eval used 0.3);
  PROMPT LIST ``labels``: ``common`` (GLINER_PROMPTS, default) or ``card`` (Nemotron-PII names);
  runtime only: ``device`` (cpu|cuda), ``dtype`` (default fp16 on cuda for the 6 GB GPU cap,
  fp32 on cpu), ``threads``, ``revision``.
"""

from _protocol import serve

MODEL = "nvidia/gliner-PII"
REVISION = "bd23e8ef4425fd04e34c5204ab49ffaa706eae79"

# Same keys as entries.GLINER_PROMPTS (kept literal: this script runs in the tool's own venv).
COMMON = ["person", "organization", "location", "address", "email", "phone number", "ip address", "url",
          "date", "date of birth", "credit card number", "iban", "social security number", "passport number",
          "account number", "tax id", "national id"]
# The model's own training labels (Nemotron-PII) for the 9 scored types; max_types is 25.
CARD = ["first_name", "last_name", "company_name", "street_address", "city", "state", "country", "postcode",
        "email", "phone_number", "fax_number", "ipv4", "ipv6", "url", "date", "date_of_birth", "date_time",
        "account_number", "credit_debit_card", "ssn", "bank_routing_number", "swift_bic"]


def load(config: dict):
    import torch
    from gliner import GLiNER

    device = config.get("device") or "cpu"
    if config.get("threads"):
        torch.set_num_threads(int(config["threads"]))
    dtype = config.get("dtype") or ("float16" if device == "cuda" else "float32")
    model = GLiNER.from_pretrained(MODEL, revision=config.get("revision", REVISION), map_location=device,
                                   dtype=dtype)
    model.to(device)
    labels = {"common": COMMON, "card": CARD}[config.get("labels", "common")]
    threshold = float(config.get("threshold", 0.5))

    def predict(docs: list[dict]) -> list[dict]:
        found = model.inference([d["text"] for d in docs], labels, threshold=threshold, batch_size=len(docs))
        return [{"spans": [{"start": e["start"], "end": e["end"], "label": e["label"], "score": e["score"]}
                           for e in ents]} for ents in found]

    return predict


if __name__ == "__main__":
    serve(load)
