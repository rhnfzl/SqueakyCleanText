"""DataFog adapter: entry ``datafog`` (datafog 4.9.0, regex engine, no model, CPU).

Default track = the vendor quick start, ``datafog.sanitize(text, engine="regex")``, which is
``scan_and_redact(text, engine="regex")`` with the ``token`` strategy, no locale, no entity
filter. We call ``scan_and_redact`` directly to get the entities (code-point offsets from
Python ``re``) and the redacted text in one pass. Regex gives no real confidence (DataFog
sets 1.0 as a compatibility value), so ``score`` is None.

Labels are DataFog's canonical output names, not the regex annotator's: EMAIL, PHONE, SSN,
CREDIT_CARD, IP_ADDRESS, DATE (annotator DOB), ZIP_CODE (annotator ZIP), and with the ``de``
locale DE_VAT_ID, DE_IBAN, DE_TAX_ID, DE_SOCIAL_SECURITY_NUMBER, DE_POSTAL_CODE,
DE_PASSPORT_NUMBER, DE_RESIDENCE_PERMIT_NUMBER.

Long documents: regex has no context limit; every doc is scanned whole (PROTOCOL section 11).

Config (tuned-track knobs, PROTOCOL section 3):
- ``locales``: None (default), ``"gold"`` (DataFog's own locale pack for the gold language:
  only German exists, so ``de*`` docs get ``["de"]``, others none), or a fixed list.
- ``entity_types``: None (vendor default set) or a list of DataFog types to enable.
- ``engine``: ``regex`` (default). ``smart``/``gliner``/``spacy`` need the ``nlp`` /
  ``nlp-advanced`` extras, which ``envs/datafog.txt`` does not install.
- ``strategy``: redaction strategy for ``masked_text`` (vendor default ``token``).
- ``backend``: ``python`` (default) or ``rust`` (needs the ``rust`` extra; not installed).
"""

import os

os.environ.setdefault("DATAFOG_NO_TELEMETRY", "1")  # telemetry is opt-in in 4.9.0; force it off anyway

from _protocol import serve  # noqa: E402

RUNTIME = ("threads", "device", "gpu_mem_gb")  # set by runner.py and speed.py for every entry
DEFAULTS = {"engine": "regex", "locales": None, "entity_types": None, "strategy": "token", "backend": "python"}


def doc_locales(setting, lang: str | None) -> list[str] | None:
    """``"gold"`` -> DataFog's locale for the gold language (German only), else the setting as given."""
    if setting != "gold":
        return setting
    return ["de"] if (lang or "").lower().replace("_", "-").split("-")[0] == "de" else None


def settings(config: dict) -> dict:
    """Adapter config -> DataFog settings; the runners' runtime keys mean nothing to a regex engine."""
    config = {k: v for k, v in config.items() if k not in RUNTIME}
    unknown = set(config) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown datafog config keys: {sorted(unknown)}")
    return {**DEFAULTS, **config}


def load(config: dict):
    from datafog.engine import scan_and_redact

    cfg = settings(config)

    def predict(docs: list[dict]) -> list[dict]:
        out = []
        for doc in docs:
            result = scan_and_redact(
                doc["text"], engine=cfg["engine"], entity_types=cfg["entity_types"], strategy=cfg["strategy"],
                locales=doc_locales(cfg["locales"], doc.get("lang")), backend=cfg["backend"],
            )
            # NER engines place values by str.find and give -1 when they cannot: count those, never guess.
            placed = [e for e in result.entities if e.start >= 0]
            out.append({
                "spans": [{"start": e.start, "end": e.end, "label": e.type, "score": None} for e in placed],
                "masked_text": result.redacted_text,
                "unplaced": len(result.entities) - len(placed),
            })
        return out

    return predict


if __name__ == "__main__":
    serve(load)
