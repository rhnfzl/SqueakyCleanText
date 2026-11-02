"""Comparison entries: each one is a complete configuration (PROTOCOL.md section 2).

Frozen before any test run. ``labels`` maps the tool's own output label to an SCT type;
a tool label missing from it is out of scope (dropped before scoring, not a false
positive). ``supports`` is the claimed SCT-type coverage used for the "tool-supported"
view. Revisions are HF commit shas; package versions are exact pins for that entry's venv.
"""

from benchmarks.labels import DATE, EMAIL, IP, LOCATION, NUMBER, ORGANISATION, PERSON, PHONE, URL

SCT_TYPES = (PERSON, LOCATION, ORGANISATION, EMAIL, PHONE, IP, URL, DATE, NUMBER)

# One fixed prompt list for every GLiNER-family entry (free-text labels), mapped to SCT types.
GLINER_PROMPTS = {
    "person": PERSON, "organization": ORGANISATION, "location": LOCATION, "address": LOCATION,
    "email": EMAIL, "phone number": PHONE, "ip address": IP, "url": URL, "date": DATE,
    "date of birth": DATE, "credit card number": NUMBER, "iban": NUMBER, "social security number": NUMBER,
    "passport number": NUMBER, "account number": NUMBER, "tax id": NUMBER, "national id": NUMBER,
}

ENTRIES = {
    "sct-default": {
        "package": "SqueakyCleanText==0.8.1", "checkpoint": "sct DEFAULT_NER_MODELS (xlm-r conll ONNX, wikineural ONNX)",
        "revision": "sct/config.py DEFAULT_NER_REVISIONS", "config": "TextCleanerConfig() + reversible scoring mode",
        "threshold": "SCT default", "mask": "SCT placeholder", "device": "cpu", "gpu_gb": 0,
        "labels": {t: t for t in SCT_TYPES}, "supports": SCT_TYPES,
    },
    "sct-gliner": {
        "package": "SqueakyCleanText[gliner]==0.8.1", "checkpoint": "knowledgator/gliner-pii-base-v1.0 (ONNX)",
        "revision": "61726e0ad791dcab3e29339bbec3ad42ded65641", "config": "TextCleanerConfig(ner_mode='pii')",
        "threshold": "SCT default", "mask": "SCT placeholder", "device": "cpu|cuda", "gpu_gb": 1,
        "labels": {t: t for t in SCT_TYPES}, "supports": SCT_TYPES,
    },
    "presidio": {
        "package": "presidio-analyzer==2.2.366 spacy==3.8.18 en_core_web_lg==3.8.0", "checkpoint": "en_core_web_lg",
        "revision": "spaCy model wheel 3.8.0", "config": "AnalyzerEngine() defaults, language='en'",
        "threshold": "0 (score_threshold default)", "mask": "common", "device": "cpu", "gpu_gb": 0,
        "labels": {"PERSON": PERSON, "LOCATION": LOCATION, "ORGANIZATION": ORGANISATION, "EMAIL_ADDRESS": EMAIL,
                   "PHONE_NUMBER": PHONE, "IP_ADDRESS": IP, "URL": URL, "DATE_TIME": DATE, "IBAN_CODE": NUMBER,
                   "CREDIT_CARD": NUMBER, "US_SSN": NUMBER, "US_PASSPORT": NUMBER, "US_DRIVER_LICENSE": NUMBER,
                   "US_BANK_NUMBER": NUMBER, "US_ITIN": NUMBER, "UK_NHS": NUMBER, "UK_NINO": NUMBER},
        "supports": (PERSON, LOCATION, EMAIL, PHONE, IP, URL, DATE, NUMBER),  # ORG only with a non-spaCy engine
    },
    "gliner-nvidia": {
        "package": "gliner==0.2.30", "checkpoint": "nvidia/gliner-PII",
        "revision": "bd23e8ef4425fd04e34c5204ab49ffaa706eae79", "config": "GLINER_PROMPTS, fp16, batch<=4, 384-token chunks",
        "threshold": 0.5, "mask": "common", "device": "cuda|cpu", "gpu_gb": 3,
        "labels": GLINER_PROMPTS, "supports": SCT_TYPES,
    },
    "gliner2-fastino": {
        "package": "gliner2==2.1.0", "checkpoint": "fastino/gliner2-privacy-filter-PII-multi",
        "revision": "1cb4166094dc58fa8d836429f060d6c95f62b495", "config": "GLINER_PROMPTS, batch<=4",
        "threshold": 0.5, "mask": "common", "device": "cuda|cpu", "gpu_gb": 2,
        "labels": GLINER_PROMPTS, "supports": SCT_TYPES,
    },
    "opf": {
        "package": "transformers==5.22.1 torch==2.15.0", "checkpoint": "openai/privacy-filter",
        "revision": "7ffa9a043d54d1be65afb281eddf0ffbe629385b",
        "config": "token-classification pipeline, aggregation_strategy='simple', default decoder biases",
        "threshold": "default operating point", "mask": "common", "device": "cuda|cpu", "gpu_gb": 5,
        "labels": {"private_person": PERSON, "private_address": LOCATION, "private_email": EMAIL,
                   "private_phone": PHONE, "private_url": URL, "private_date": DATE, "account_number": NUMBER},
        "supports": (PERSON, LOCATION, EMAIL, PHONE, URL, DATE, NUMBER),  # no ORG, no IP class
    },
    "bardsai-v3": {
        "package": "transformers==5.22.1 onnxruntime==1.31.1", "checkpoint": "bardsai/eu-pii-anonimization-multilang-v3",
        "revision": "091984b5fbce4c161abc4776ed0f4e3206e1b374",
        "config": "bundled pii_detector.py (read first), onnx/model_quantized.onnx, 512-token windows, 128 overlap",
        "threshold": "card default", "mask": "common", "device": "cpu", "gpu_gb": 0,
        # Filled from the model's config.json id2label at adapter time; target types listed here.
        "labels": {}, "supports": SCT_TYPES,
    },
    "datafog": {
        "package": "datafog==4.9.0", "checkpoint": "regex engine (no model)", "revision": "-",
        "config": "DataFog() quick-start defaults", "threshold": "-", "mask": "common", "device": "cpu", "gpu_gb": 0,
        # shortcut: label names from the shortlist notes; check against the 4.9.0 wheel when the adapter is written.
        "labels": {"EMAIL": EMAIL, "PHONE": PHONE, "IP_ADDRESS": IP, "CREDIT_CARD": NUMBER, "SSN": NUMBER,
                   "DOB": DATE, "ZIP": NUMBER},
        "supports": (EMAIL, PHONE, IP, DATE, NUMBER),
    },
}

RESERVE = ("scrubadub==2.0.1", "datatrove==0.10.1 PIIFormatter", "urchade/gliner_multi_pii-v1",
           "OpenMed/privacy-filter-multilingual-v2", "iiiorg/piiranha-v1-detect-personal-information")

# Runner registry: adapter script (in ``adapters/``) and its config. The pinned install spec
# is ``envs/<entry>.txt``; every venv uses PYTHON. ``adapter`` None = not written yet.
PYTHON = "3.12"
RUNS = {
    "sct-default": {"adapter": "sct_adapter.py", "config": {"mode": "default", "language": "gold"}},
    "sct-gliner": {"adapter": "sct_adapter.py", "config": {"mode": "gliner", "language": "gold"}},
    **{name: {"adapter": None, "config": {}} for name in
       ("presidio", "gliner-nvidia", "gliner2-fastino", "opf", "bardsai-v3", "datafog")},
}

# Not a comparison entry: a stdlib-only adapter for tests and smoke runs (runs in any Python).
TEST_ENTRIES = {
    "dummy": {
        "package": "-", "checkpoint": "-", "revision": "-", "config": "regex e-mail + digit runs",
        "threshold": "-", "mask": "common", "device": "cpu", "gpu_gb": 0,
        "labels": {"EMAIL": EMAIL, "NUMBER": NUMBER}, "supports": (EMAIL, NUMBER),
        "adapter": "dummy_adapter.py", "adapter_config": {},
    },
}


def tool(name: str) -> dict:
    """Entry + run settings: ``adapter``, ``adapter_config`` and ``requirements`` (None = current Python)."""
    from pathlib import Path

    if name in TEST_ENTRIES:
        return {**TEST_ENTRIES[name], "requirements": None}
    if name not in ENTRIES:
        raise ValueError(f"unknown entry {name!r}; known: {', '.join([*ENTRIES, *TEST_ENTRIES])}")
    run = RUNS[name]
    return {**ENTRIES[name], "adapter": run["adapter"], "adapter_config": run["config"],
            "requirements": Path(__file__).resolve().parent / "envs" / f"{name}.txt"}


# PROTOCOL.md section 17, per entry: dataset -> flag ("*" = every other dataset).
# T trained on it, E vendor evaluates on it, D SCT developed against it, U training data undisclosed, - none known.
CONTAMINATION = {
    "sct-default": {"*": "D", "gretel_en": "-", "argilla_pii": "-"},
    "sct-gliner": {"*": "U, D", "gretel_en": "U", "argilla_pii": "U"},
    "presidio": {"*": "-"},
    "gliner-nvidia": {"*": "-", "openpii": "E", "gretel_en": "E", "nemotron": "T", "argilla_pii": "E"},
    "gliner2-fastino": {"*": "-"},
    "opf": {"*": "U"},
    "bardsai-v3": {"*": "-", "openpii": "T", "gretel_en": "T", "nemotron": "T", "argilla_pii": "T-family"},
    "datafog": {"*": "-"},
}


def contamination(entry: str, dataset: str) -> str:
    flags = CONTAMINATION.get(entry, {"*": "-"})
    return flags.get(dataset, flags["*"])
