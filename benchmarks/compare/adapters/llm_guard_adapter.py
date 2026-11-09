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

Long texts: llm-guard's own 600-character chunks with 40 overlap (``chunk_long_texts``: it never chunks at
the shipped tokenizer limit).

Config: ``threshold`` (default 0.5); ``device`` / ``gpu_mem_gb`` only cap the torch share of the card
(the recognizer picks cuda when it is visible).
"""

from _metadata import LLM_GUARD as LABELS
from _protocol import cap_gpu, serve

# DEFAULT_ENTITY_TYPES of 0.3.16 (+ CUSTOM, added by the scanner); UUID, CRYPTO and CUSTOM are out of scope.
OUT_OF_SCOPE = ("CRYPTO", "UUID", "CUSTOM")


def to_spans(results) -> list[dict]:
    return sorted(({"start": r.start, "end": r.end, "label": r.entity_type, "score": float(r.score)}
                   for r in results if r.end > r.start), key=lambda s: (s["start"], s["end"]))


def chunk_long_texts(scanner) -> None:
    """Give the recognizer's tokenizer the model's real length, so llm-guard's own chunking runs.

    The DeBERTa tokenizer ships the "no limit" sentinel (1e30) as ``model_max_length``, and the recognizer
    chunks a text (``CHUNK_SIZE`` 600 chars, 40 overlap, its own defaults) only when the text is longer than
    that, so it never chunks and puts a whole document through one attention pass (smoke run 2026-11-03:
    CUDA OOM on 6 of 20 RedactionBench dev docs under the 5 GB cap).
    """
    for recognizer in scanner._analyzer.registry.recognizers:
        pipe = getattr(recognizer, "pipeline", None)
        if pipe is not None and pipe.tokenizer.model_max_length > 10**6:
            pipe.tokenizer.model_max_length = pipe.model.config.max_position_embeddings


def load(config: dict):
    from llm_guard.input_scanners import Anonymize
    from llm_guard.vault import Vault

    cap_gpu(config)  # the recognizer takes cuda when visible
    scanner = Anonymize(Vault(), threshold=float(config.get("threshold", 0.5)), language="en")
    if set(scanner._entity_types) != set(LABELS) | set(OUT_OF_SCOPE):
        raise RuntimeError(f"llm-guard entity types changed: {sorted(scanner._entity_types)}")
    chunk_long_texts(scanner)

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
