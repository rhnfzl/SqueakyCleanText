"""Presidio adapter: entry ``presidio`` (presidio-analyzer + spaCy, CPU only).

default track (empty config): the vendor quick start, ``AnalyzerEngine()`` with its bundled
``default.yaml`` (spaCy ``en_core_web_lg``, ORGANIZATION in ``labels_to_ignore``) and
``analyze(text, language="en")`` for every doc, vendor ``score_threshold`` (0). ``masked_text``
is ``AnonymizerEngine().anonymize`` with its default ``replace`` operator (``<ENTITY_TYPE>``),
the native pipeline track. Spans are Presidio's own char offsets (code points) and scores.

tuned-track knobs (PROTOCOL section 3), all optional:
``score_threshold`` (float), ``language`` ("en" | "gold"), ``models`` ({lang: spaCy model},
used with ``language="gold"``; each model must be installed in the venv), ``phone_regions``
(list of regions for ``PhoneRecognizer``), ``enable`` (predefined recognizer class names that
are off by default, e.g. ``UkNinoRecognizer``). Gold languages without a model in ``models``
are analysed as ``en`` and named once on stderr. Runner keys (threads, device...) are ignored.

Long docs: spaCy has no model window; ``nlp.max_length`` (default 1,000,000 chars) is raised
for a longer doc so it is analysed whole, as PROTOCOL section 11 says.
"""

import sys

from _protocol import serve


def load(config: dict):
    from presidio_analyzer import AnalyzerEngine, RecognizerRegistry, predefined_recognizers
    from presidio_anonymizer import AnonymizerEngine

    gold = config.get("language", "en") == "gold"
    models = {"en": "en_core_web_lg", **config.get("models", {})} if gold else {"en": "en_core_web_lg"}
    if not gold and not config.get("phone_regions") and not config.get("enable"):
        analyzer = AnalyzerEngine()  # the vendor quick start, untouched
    else:
        from pathlib import Path

        import yaml
        from presidio_analyzer.nlp_engine import NlpEngineProvider

        conf = yaml.safe_load(Path(NlpEngineProvider._get_full_conf_path()).read_text())
        conf["models"] = [{"lang_code": lang, "model_name": name} for lang, name in models.items()]
        nlp_engine = NlpEngineProvider(nlp_configuration=conf).create_engine()
        langs = list(models)
        registry = RecognizerRegistry(supported_languages=langs)
        registry.load_predefined_recognizers(languages=langs, nlp_engine=nlp_engine)
        if config.get("phone_regions"):
            registry.remove_recognizer("PhoneRecognizer")
            for lang in langs:
                registry.add_recognizer(predefined_recognizers.PhoneRecognizer(
                    supported_language=lang, supported_regions=tuple(config["phone_regions"])))
        for name in config.get("enable", ()):
            for lang in langs:
                registry.add_recognizer(getattr(predefined_recognizers, name)(supported_language=lang))
        analyzer = AnalyzerEngine(registry=registry, nlp_engine=nlp_engine, supported_languages=langs)
    anonymizer = AnonymizerEngine()
    threshold = config.get("score_threshold")
    warned: set = set()

    def language(doc: dict) -> str:
        lang = (doc.get("lang") or "").replace("_", "-").split("-")[0]
        if not gold or lang in models:
            return lang if gold else "en"
        if lang not in warned:
            warned.add(lang)
            print(f"presidio_adapter: no spaCy model for {doc.get('lang')!r}, analysing as en", file=sys.stderr)
        return "en"

    def one(doc: dict) -> dict:
        text = doc["text"]
        lang = language(doc)
        nlp = analyzer.nlp_engine.nlp[lang]
        nlp.max_length = max(nlp.max_length, len(text) + 1)
        results = analyzer.analyze(text=text, language=lang, score_threshold=threshold)
        spans = [{"start": r.start, "end": r.end, "label": r.entity_type, "score": r.score} for r in results]
        masked = anonymizer.anonymize(text=text, analyzer_results=results).text
        return {"spans": sorted(spans, key=lambda s: (s["start"], s["end"])), "masked_text": masked}

    return lambda docs: [one(doc) for doc in docs]


if __name__ == "__main__":
    serve(load)
