"""Comparison entries: each one is a complete configuration (PROTOCOL.md section 2).

Frozen before any test run. ``labels`` maps the tool's own output label to an SCT type;
a tool label missing from it is out of scope (dropped before scoring, not a false
positive). ``supports`` is the claimed SCT-type coverage used for the "tool-supported"
view. Revisions are HF commit shas; package versions are exact pins for that entry's venv.
"""

from benchmarks.labels import DATE, EMAIL, IP, LOCATION, NUMBER, ORGANISATION, PERSON, PHONE, PRED_ALIASES, URL

SCT_TYPES = (PERSON, LOCATION, ORGANISATION, EMAIL, PHONE, IP, URL, DATE, NUMBER)
# SCT's own placeholder names (YEAR, IBAN, ...) are aliases of the canonical types: map them
# here, before map_spans drops unknown labels.
SCT_LABELS = {**{t: t for t in SCT_TYPES}, **PRED_ALIASES}

# One fixed prompt list for every GLiNER-family entry (free-text labels), mapped to SCT types.
GLINER_PROMPTS = {
    "person": PERSON, "organization": ORGANISATION, "location": LOCATION, "address": LOCATION,
    "email": EMAIL, "phone number": PHONE, "ip address": IP, "url": URL, "date": DATE,
    "date of birth": DATE, "credit card number": NUMBER, "iban": NUMBER, "social security number": NUMBER,
    "passport number": NUMBER, "account number": NUMBER, "tax id": NUMBER, "national id": NUMBER,
}
# Tuned-track alternative per GLiNER model (adapter config labels='card'): the model card's own label
# names. Merged into that entry's labels so card-mode spans are scored, not dropped.
NVIDIA_CARD = {
    "first_name": PERSON, "last_name": PERSON, "company_name": ORGANISATION,
    **dict.fromkeys(("street_address", "city", "state", "country"), LOCATION),
    "postcode": NUMBER,  # as gold postcode (nemotron, gretel_en) and Fastino postal_code
    "email": EMAIL, "phone_number": PHONE, "fax_number": PHONE, "ipv4": IP, "ipv6": IP, "url": URL,
    "date": DATE, "date_of_birth": DATE, "date_time": DATE,
    **dict.fromkeys(("account_number", "credit_debit_card", "ssn", "bank_routing_number", "swift_bic"), NUMBER),
}
FASTINO_CARD = {  # credentials and usernames are out of scope (absent)
    **dict.fromkeys(("person", "full_name", "first_name", "middle_name", "last_name"), PERSON),
    **dict.fromkeys(("address", "street_address", "city", "state_or_region", "country"), LOCATION),
    "email": EMAIL, "phone_number": PHONE, "ip_address": IP,
    **dict.fromkeys(("date_of_birth", "sensitive_date", "document_date", "expiration_date",
                     "transaction_date", "card_expiry"), DATE),
    **dict.fromkeys(("postal_code", "government_id", "national_id_number", "passport_number",
                     "drivers_license_number", "license_number", "tax_id", "tax_number", "bank_account",
                     "account_number", "routing_number", "iban", "payment_card", "card_number", "card_cvv",
                     "account_id", "sensitive_account_id"), NUMBER),
}

ENTRIES = {
    "sct-default": {
        "package": "SqueakyCleanText==0.8.1", "checkpoint": "sct DEFAULT_NER_MODELS (xlm-r conll ONNX, wikineural ONNX)",
        "revision": "sct/config.py DEFAULT_NER_REVISIONS", "config": "TextCleanerConfig() + reversible scoring mode",
        "threshold": "SCT default", "mask": "SCT placeholder", "device": "cpu", "gpu_gb": 0,
        "labels": SCT_LABELS, "supports": SCT_TYPES, "recover_dropped": True,
    },
    "sct-gliner": {
        "package": "SqueakyCleanText[gliner]==0.8.1",
        "checkpoint": "knowledgator/gliner-pii-base-v1.0 (PyTorch: no root model.onnx in the repo, SCT falls back)",
        "revision": "61726e0ad791dcab3e29339bbec3ad42ded65641", "config": "TextCleanerConfig(ner_mode='pii')",
        "threshold": "SCT default", "mask": "SCT placeholder", "device": "cpu|cuda", "gpu_gb": 1,
        "labels": SCT_LABELS, "supports": SCT_TYPES, "recover_dropped": True,
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
        "package": "gliner==0.2.30 torch==2.15.0 transformers==5.22.1",
        "checkpoint": "nvidia/gliner-PII",
        "revision": "bd23e8ef4425fd04e34c5204ab49ffaa706eae79",
        "config": "GLINER_PROMPTS, fp16 on cuda (fp32 cpu), batch<=4, vendor truncation at 384 words",
        "threshold": 0.5, "mask": "common", "device": "cuda|cpu", "gpu_gb": 3,
        "labels": {**GLINER_PROMPTS, **NVIDIA_CARD}, "supports": SCT_TYPES,
    },
    "gliner2-fastino": {
        "package": "gliner2[local]==2.1.0 torch==2.15.0 transformers==5.22.1",
        "checkpoint": "fastino/gliner2-privacy-filter-PII-multi",
        "revision": "1cb4166094dc58fa8d836429f060d6c95f62b495", "config": "GLINER_PROMPTS, batch<=4",
        "threshold": 0.5, "mask": "common", "device": "cuda|cpu", "gpu_gb": 2,
        "labels": {**GLINER_PROMPTS, **FASTINO_CARD}, "supports": SCT_TYPES,
    },
    "opf": {
        "package": "transformers==5.22.1 torch==2.15.0", "checkpoint": "openai/privacy-filter",
        "revision": "7ffa9a043d54d1be65afb281eddf0ffbe629385b",
        "config": "vendor BIOES constrained Viterbi (checkpoint viterbi_calibration biases), bf16, 4096-token windows",
        "threshold": "default operating point", "mask": "common", "device": "cuda|cpu", "gpu_gb": 5,
        "labels": {"private_person": PERSON, "private_address": LOCATION, "private_email": EMAIL,
                   "private_phone": PHONE, "private_url": URL, "private_date": DATE, "account_number": NUMBER},
        "supports": (PERSON, LOCATION, EMAIL, PHONE, URL, DATE, NUMBER),  # no ORG, no IP class
        # Its score is a mean token probability, not the decoder's operating point (bias presets):
        # filtering on it is no OPF operating point, so no curve; presets are separate --config runs.
        "curve_by_filter": False,
    },
    "bardsai-v3": {
        "package": "transformers==5.22.1 onnxruntime==1.31.1 torch==2.15.0",
        "checkpoint": "bardsai/eu-pii-anonimization-multilang-v3",
        "revision": "091984b5fbce4c161abc4776ed0f4e3206e1b374",
        "config": "bundled pii_detector.py (read first), onnx/model_quantized.onnx, 512-token windows, 128 overlap",
        "threshold": "card default", "mask": "common", "device": "cpu", "gpu_gb": 0,
        # Frozen from config.json id2label at the pinned revision (adapter LABELS; the rest out of scope).
        "labels": {"PERSON_NAME": PERSON, "PERSON_ALIAS": PERSON, "LOCATION": LOCATION, "POSTAL_ADDRESS": LOCATION,
                   "GEO_LOCATION": LOCATION, "ORGANIZATION_NAME": ORGANISATION, "EMAIL_ADDRESS": EMAIL,
                   "PHONE_NUMBER": PHONE, "IP_ADDRESS": IP, "URL": URL, "IDENTIFYING_LINK": URL,
                   "DATE_TIME": DATE, "DATE_OF_BIRTH": DATE,
                   **dict.fromkeys(("PERSON_IDENTIFIER", "ORGANIZATION_IDENTIFIER", "BANK_ACCOUNT_IDENTIFIER",
                                    "ACCOUNT_IDENTIFIER", "PAYMENT_CARD", "PAYMENT_CARD_SECURITY",
                                    "DOCUMENT_IDENTIFIER", "VEHICLE_IDENTIFIER", "DEVICE_IDENTIFIER"), NUMBER)},
        "supports": SCT_TYPES,
    },
    "datafog": {
        "package": "datafog==4.9.0", "checkpoint": "regex engine (no model)", "revision": "-",
        "config": "DataFog() quick-start defaults", "threshold": "-", "mask": "common", "device": "cpu", "gpu_gb": 0,
        # Checked against the 4.9.0 wheel; DE_* come from the German locale pack (tuned track).
        "labels": {"EMAIL": EMAIL, "PHONE": PHONE, "IP_ADDRESS": IP, "CREDIT_CARD": NUMBER, "SSN": NUMBER,
                   "DATE": DATE, "ZIP_CODE": NUMBER,
                   **dict.fromkeys(("DE_IBAN", "DE_VAT_ID", "DE_TAX_ID", "DE_SOCIAL_SECURITY_NUMBER", "DE_POSTAL_CODE",
                                    "DE_PASSPORT_NUMBER", "DE_RESIDENCE_PERMIT_NUMBER"), NUMBER)},
        "supports": (EMAIL, PHONE, IP, DATE, NUMBER),
    },
}

RESERVE = ("scrubadub==2.0.1", "datatrove==0.10.1 PIIFormatter", "urchade/gliner_multi_pii-v1",
           "OpenMed/privacy-filter-multilingual-v2", "iiiorg/piiranha-v1-detect-personal-information")

# Runner registry: adapter script (in ``adapters/``) and its config. The pinned install spec
# is ``envs/<entry>.txt``; every venv uses PYTHON. Config {} = the vendor default track.
PYTHON = "3.12"
RUNS = {
    "sct-default": {"adapter": "sct_adapter.py", "config": {"mode": "default", "language": "auto"}},
    "sct-gliner": {"adapter": "sct_adapter.py", "config": {"mode": "gliner", "language": "auto"}},
    **{name: {"adapter": f"{name.replace('-', '_')}_adapter.py", "config": {}} for name in
       ("presidio", "gliner-nvidia", "gliner2-fastino", "opf", "bardsai-v3", "datafog")},
}
# Language tracks (PROTOCOL.md section 12), applied to every entry the same way. ``native``:
# each tool routes as shipped (SCT's own LID, Presidio en, DataFog no locale). ``gold``: every
# tool that takes a language gets the gold code. Entries absent here take no language input.
TRACKS = ("gold", "native")
GOLD_TRACK = {
    "sct-default": {"language": "gold"}, "sct-gliner": {"language": "gold"},
    "presidio": {"language": "gold"}, "datafog": {"locales": "gold"},
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


def tool(name: str, track: str = "native") -> dict:
    """Entry + run settings: ``adapter``, ``adapter_config`` and ``requirements`` (None = current Python)."""
    from pathlib import Path

    if name in TEST_ENTRIES:
        return {**TEST_ENTRIES[name], "requirements": None}
    if name not in ENTRIES:
        raise ValueError(f"unknown entry {name!r}; known: {', '.join([*ENTRIES, *TEST_ENTRIES])}")
    if track not in TRACKS:
        raise ValueError(f"unknown track {track!r}; known: {', '.join(TRACKS)}")
    run = RUNS[name]
    config = {**run["config"], **(GOLD_TRACK.get(name, {}) if track == "gold" else {})}
    return {**ENTRIES[name], "adapter": run["adapter"], "adapter_config": config,
            "requirements": Path(__file__).resolve().parent / "envs" / f"{name}.txt"}


# PROTOCOL.md section 17, per entry: dataset -> direct flag ("*" = every other dataset).
# T trained on it, E vendor evaluates on it, D SCT developed against it, U training data undisclosed, - none known.
# Family flags are not listed here: ``contamination`` derives them from FAMILIES for every entry alike.
CONTAMINATION = {
    "sct-default": {"*": "D", "gretel_en": "-", "argilla_pii": "-"},
    "sct-gliner": {"*": "U, D", "gretel_en": "U", "argilla_pii": "U"},
    "presidio": {"*": "-"},
    "gliner-nvidia": {"*": "-", "openpii": "E", "gretel_en": "E", "nemotron": "T", "argilla_pii": "E"},
    "gliner2-fastino": {"*": "-"},
    "opf": {"*": "U"},
    "bardsai-v3": {"*": "-", "openpii": "T", "gretel_en": "T", "nemotron": "T"},
    "datafog": {"*": "-"},
}
# Generator families (decision 21: a dataset AND its family). A T, E or D flag on one member puts
# "<flag>-family" on the others. Gretel: gretel finance and gretel_en (Gretel Navigator) and nemotron
# (NeMo Data Designer, Navigator's successor after NVIDIA took over Gretel; the gretel_en and Nemotron-PII
# cards both carry the datadesigner tag). ai4privacy: openpii and argilla_pii (a slice of ai4privacy-200k).
FAMILIES = (("gretel", "gretel_en", "nemotron"), ("openpii", "argilla_pii"))


def _flags(entry: str, dataset: str) -> list[str]:
    flags = CONTAMINATION.get(entry, {"*": "-"})
    return [f.strip() for f in flags.get(dataset, flags["*"]).split(",") if f.strip() not in ("", "-")]


def contamination(entry: str, dataset: str) -> str:
    """Direct flags of the cell, then ``<X>-family`` for each T/E/D on another member of its family."""
    own = _flags(entry, dataset)
    family = [f"{x}-family" for x in "TED"
              if x not in own and any(x in _flags(entry, d) for fam in FAMILIES if dataset in fam
                                      for d in fam if d != dataset)]
    return ", ".join(own + family) or "-"


def status(entry: str, dataset: str) -> str:
    """Decision 21 (PROTOCOL.md section 17): ``excluded`` (T, E or D: the tool or a model in it was trained,
    evaluated or developed on the set or its family), ``unknown`` (U only: training data undisclosed), else ``clean``.
    """
    flags = {flag.strip()[:1] for flag in contamination(entry, dataset).split(",")}
    return "excluded" if flags & {"T", "E", "D"} else "unknown" if "U" in flags else "clean"


# PROTOCOL.md section 12, rule 2: languages each entry claims (model card language tags at the pinned
# revision; None = every language: datafog's regexes, and sct-default, which claims multilingual NER through
# its WikiNEural fallback, so SCT is held to every language). Scores outside the claim are reported apart.
EU24 = ("bg", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr", "ga", "hr", "hu", "it", "lt", "lv", "mt", "nl",
        "pl", "pt", "ro", "sk", "sl", "sv")
CLAIMED_LANGUAGES = {
    "sct-default": None,
    "sct-gliner": ("en",),  # knowledgator/gliner-pii-base-v1.0 @61726e0 card metadata language: [en] (its
    # "multilingual" is a free topic tag, not a language tag)
    "presidio": ("en",),  # en_core_web_lg only, in both tracks
    "gliner-nvidia": ("en",), "gliner2-fastino": ("en", "fr", "es", "de", "it", "pt", "nl"),
    "opf": ("en",),  # "primarily English"
    "bardsai-v3": EU24, "datafog": None,
}


def claims(entry: str, lang: str) -> bool:
    """Is a gold language code (``pt_br``, ``ar-mixed_en``) inside the entry's claimed languages?"""
    claimed = CLAIMED_LANGUAGES.get(entry)
    return claimed is None or lang.replace("_", "-").split("-")[0] in claimed
