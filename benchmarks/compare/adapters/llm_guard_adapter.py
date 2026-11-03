"""LLM Guard ``Anonymize`` adapter: entry ``llm-guard`` (llm-guard 0.3.16, the last release; repo archived).

Default track = the vendor quick start: ``Anonymize(Vault())`` with its defaults: ``DEFAULT_ENTITY_TYPES``,
the default recognizer ``DEBERTA_AI4PRIVACY_v2_CONF`` (Isotonic/deberta-v3-base_finetuned_ai4privacy_v2 at
the revision llm-guard pins), Presidio regex recognizers, ``threshold=0.5``, ``language="en"`` (the scanner
only takes en / zh), PyTorch (no ONNX).

``scan()`` returns only the anonymised text, so spans are taken from the same steps ``scan()`` runs, in
order, on the same analyzer: ``analyze`` (on the text with single quotes blanked, same length), conflict
removal, merge of same-type neighbours split by spaces. ``masked_text`` is ``_anonymize`` on those spans
(``[REDACTED_<TYPE>_<n>]``) with a fresh ``Vault`` per doc, so numbering never carries across docs.
These are private methods of the pinned release, read from its source (llm_guard/input_scanners/anonymize.py
at v0.3.16). A span that conflict removal shrank to zero width covers no source character, so it is not a
span (its placeholder, which ``scan()`` inserts too, stays in ``masked_text``).

Config: ``threshold`` (default 0.5); ``device`` / ``gpu_mem_gb`` only cap the torch share of the card
(the recognizer picks cuda when it is visible).
"""

from _protocol import cap_gpu, serve

# DEFAULT_ENTITY_TYPES of 0.3.16 (+ CUSTOM, added by the scanner); UUID, CRYPTO and CUSTOM are out of scope.
LABELS = {"PERSON": "PERSON", "EMAIL_ADDRESS": "EMAIL", "EMAIL_ADDRESS_RE": "EMAIL", "PHONE_NUMBER": "PHONE",
          "IP_ADDRESS": "IP",
          **dict.fromkeys(("CREDIT_CARD", "CREDIT_CARD_RE", "IBAN_CODE", "US_SSN", "US_SSN_RE", "US_BANK_NUMBER"),
                          "NUMBER")}
OUT_OF_SCOPE = ("CRYPTO", "UUID", "CUSTOM")


def to_spans(results) -> list[dict]:
    return sorted(({"start": r.start, "end": r.end, "label": r.entity_type, "score": float(r.score)}
                   for r in results if r.end > r.start), key=lambda s: (s["start"], s["end"]))


def load(config: dict):
    from llm_guard.input_scanners import Anonymize
    from llm_guard.vault import Vault

    cap_gpu(config)  # the recognizer takes cuda when visible
    scanner = Anonymize(Vault(), threshold=float(config.get("threshold", 0.5)), language="en")
    if set(scanner._entity_types) != set(LABELS) | set(OUT_OF_SCOPE):
        raise RuntimeError(f"llm-guard entity types changed: {sorted(scanner._entity_types)}")

    def one(text: str) -> dict:
        if text.strip() == "":  # scan() returns such a prompt unchanged
            return {"spans": [], "masked_text": text}
        results = scanner._analyzer.analyze(text=Anonymize.remove_single_quotes(text), language=scanner._language,
                                            entities=scanner._entity_types, allow_list=scanner._allowed_names,
                                            score_threshold=scanner._threshold)
        results = scanner._remove_conflicts_and_get_text_manipulation_data(results)
        merged = Anonymize._merge_entities_with_whitespace_between(text, results)
        masked, _ = Anonymize._anonymize(text, merged, Vault(), False)
        return {"spans": to_spans(merged), "masked_text": masked}

    return lambda docs: [one(d["text"]) for d in docs]


if __name__ == "__main__":
    serve(load)
