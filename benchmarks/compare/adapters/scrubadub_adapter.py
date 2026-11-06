"""scrubadub adapter: entry ``scrubadub`` (scrubadub 2.0.1 + scrubadub_spacy 2.0.0, both unmaintained).

Default track = the scrubadub_spacy quick start: ``scrubadub.Scrubber()`` (locale ``en_US``, the default
detector list: email, phone, url, credit card, SSN, driving licence, NINO, postcode, tax reference,
vehicle plate, credential, twitter) plus ``SpacyEntityDetector()`` (its default model for en,
``en_core_web_trf``, and its default entities PERSON / PER / ORG). ``lang`` is ignored (English locale).

Spans: ``Scrubber.iter_filth(text)`` after scrubadub's own merging; a ``MergedFilth`` is expanded into the
detections it holds, each with its own type (overlaps of different types are kept, as in PROTOCOL section 6).
Offsets are Python str offsets; the spaCy detector re-finds each entity in the original text (it runs on a
whitespace-collapsed copy for transformer models). scrubadub gives no score. ``masked_text`` is ``clean()``'s
default form, ``{{TYPE}}`` per merged filth (types joined by ``+`` when a merge holds several), built from
the same filth so the detectors run once.

Config: ``locale`` (default ``en_US``), ``spacy`` (default true: add the spaCy name detector), ``model``
(spaCy model; default the detector's own choice). Runner keys are ignored.
"""

from _protocol import serve

LABELS = {"name": "PERSON", "organization": "ORGANISATION", "location": "LOCATION", "address": "LOCATION",
          "email": "EMAIL", "phone": "PHONE", "url": "URL", "date_of_birth": "DATE",
          **dict.fromkeys(("credit_card", "drivers_licence", "national_insurance_number", "postalcode",
                           "social_security_number", "tax_reference_number", "vehicle_licence_plate"), "NUMBER")}
OUT_OF_SCOPE = ("credential", "twitter", "skype", "unknown", "tagged")


def to_result(text: str, filths) -> dict:
    spans, masked, last = [], [], 0
    for f in sorted(filths, key=lambda f: f.beg):
        parts = getattr(f, "filths", None) or [f]
        spans += [{"start": p.beg, "end": p.end, "label": p.type, "score": None} for p in parts]
        masked += [text[last:f.beg], "{{" + "+".join(sorted({p.type.upper() for p in parts})) + "}}"]
        last = f.end
    return {"spans": sorted(spans, key=lambda s: (s["start"], s["end"])), "masked_text": "".join([*masked, text[last:]])}


def load(config: dict):
    import scrubadub
    from scrubadub_spacy.detectors.spacy import SpacyEntityDetector

    locale = config.get("locale", "en_US")
    scrubber = scrubadub.Scrubber(locale=locale)
    if config.get("spacy", True):
        kwargs = {"model": config["model"]} if config.get("model") else {}
        scrubber.add_detector(SpacyEntityDetector(locale=locale, **kwargs))

    return lambda docs: [to_result(d["text"], list(scrubber.iter_filth(d["text"]))) for d in docs]


if __name__ == "__main__":
    serve(load)
