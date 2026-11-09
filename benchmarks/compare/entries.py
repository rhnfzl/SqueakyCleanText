"""Comparison entries: each one is a complete configuration (PROTOCOL.md section 2).

Frozen before any test run. ``labels`` maps the tool's own output label to an SCT type;
a tool label missing from it is out of scope (dropped before scoring, not a false
positive). ``supports`` is the claimed SCT-type coverage used for the "tool-supported"
view. Revisions are HF commit shas; package versions are exact pins for that entry's venv.
"""

from benchmarks.compare.adapters._metadata import (
    GLINER_PROMPTS, NVIDIA_CARD, FASTINO_CARD, PIIRANHA, AI4PRIVACY, OPENMED,
    BARDSAI, OPF, PII_TRACER, LLM_GUARD, SCRUBADUB,
    GLINER_NVIDIA_MODEL, GLINER_NVIDIA_REVISION, GLINER2_FASTINO_REVISION,
    GLINER_KNOWLEDGATOR_REVISION, OPF_REVISION, BARDSAI_V3_REVISION,
    PIIRANHA_REVISION, AI4PRIVACY_REVISION, OPENMED_REVISION, PII_TRACER_REVISION,
)
from benchmarks.datasets import ROUND2
from pathlib import Path
from typing import Any

from benchmarks.labels import DATE, EMAIL, IP, LOCATION, NUMBER, ORGANISATION, PERSON, PHONE, PRED_ALIASES, URL

SCT_TYPES = (PERSON, LOCATION, ORGANISATION, EMAIL, PHONE, IP, URL, DATE, NUMBER)
# SCT's own placeholder names (YEAR, IBAN, ...) are aliases of the canonical types: map them
# here, before map_spans drops unknown labels.
# GPU entries use CUDA 12 venvs (envs/*.txt): torch cu126 runs on a CUDA 12.6 driver.
TORCH_CU126 = "torch==2.14.1+cu126"
SCT_LABELS = {**{t: t for t in SCT_TYPES}, **PRED_ALIASES}

# Presidio language packs (PROTOCOL.md section 3.2): spaCy's lg pipeline where one exists,
# else the multilingual xx_ent_wiki_sm; keyed by the gold code prefix the adapter routes on.
SPACY_LG = ("da", "de", "el", "en", "es", "fi", "fr", "hr", "it", "ja", "ko", "lt", "nl", "pl", "pt", "ro", "ru",
            "sl", "sv")
SPACY_XX = ("ar", "bg", "bn", "cs", "et", "he", "hi", "hu", "id", "lv", "ms", "sk", "sr", "ta", "th", "tl", "tr", "vi")
PRESIDIO_MODELS = {**{c: f"{c}_core_news_lg" for c in SPACY_LG}, "en": "en_core_web_lg", "zh": "zh_core_web_lg",
                   "no": "nb_core_news_lg",
                   **dict.fromkeys(SPACY_XX, "xx_ent_wiki_sm")}
# Vendor default regions plus one region per scored language (fixed for every dataset).
PHONE_REGIONS = ("US", "GB", "DE", "FR", "IL", "IN", "CA", "BR", "AE", "BD", "BG", "CN", "CZ", "DK", "EE", "EG",
                 "ES", "FI", "GR", "HR", "HU", "ID", "IT", "JP", "KR", "LK", "LT", "LV", "MY", "NL", "NO", "PH", "PL",
                 "PT", "RO", "RS", "RU", "SA", "SE", "SI", "SK", "TH", "TR", "TW", "VN")

ENTRIES = {
    "sct-default": {
        "package": "SqueakyCleanText==0.8.2", "checkpoint": "sct DEFAULT_NER_MODELS (xlm-r conll ONNX, wikineural ONNX)",
        "revision": "sct/config.py DEFAULT_NER_REVISIONS", "config": "TextCleanerConfig() + reversible scoring mode",
        "threshold": "SCT default", "mask": "SCT placeholder", "device": "cpu", "gpu_gb": 0,
        "labels": SCT_LABELS, "supports": SCT_TYPES, "recover_dropped": True,
    },
    "sct-gliner": {
        "package": "SqueakyCleanText[gliner]==0.8.2",
        "checkpoint": "knowledgator/gliner-pii-base-v1.0 (PyTorch: no root model.onnx in the repo, SCT falls back)",
        "revision": GLINER_KNOWLEDGATOR_REVISION, "config": "TextCleanerConfig(ner_mode='pii')",
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
        "package": f"gliner==0.2.30 {TORCH_CU126} transformers==5.22.1",
        "checkpoint": "nvidia/gliner-PII",
        "revision": GLINER_NVIDIA_REVISION,
        "config": "GLINER_PROMPTS, fp16 on cuda (fp32 cpu), batch<=4, vendor truncation at 384 words",
        "threshold": 0.5, "mask": "common", "device": "cuda|cpu", "gpu_gb": 3,
        "labels": {**GLINER_PROMPTS, **NVIDIA_CARD}, "supports": SCT_TYPES,
    },
    "gliner2-fastino": {
        "package": f"gliner2[local]==2.1.0 {TORCH_CU126} transformers==5.22.1",
        "checkpoint": "fastino/gliner2-privacy-filter-PII-multi",
        "revision": GLINER2_FASTINO_REVISION, "config": "GLINER_PROMPTS, batch<=4",
        "threshold": 0.5, "mask": "common", "device": "cuda|cpu", "gpu_gb": 2,
        "labels": {**GLINER_PROMPTS, **FASTINO_CARD}, "supports": SCT_TYPES,
    },
    "opf": {
        "package": f"transformers==5.22.1 {TORCH_CU126}", "checkpoint": "openai/privacy-filter",
        "revision": OPF_REVISION,
        "config": "vendor BIOES constrained Viterbi (checkpoint viterbi_calibration biases), bf16, 4096-token windows",
        "threshold": "default operating point", "mask": "common", "device": "cuda|cpu", "gpu_gb": 5,
        "labels": OPF,
        "supports": (PERSON, LOCATION, EMAIL, PHONE, URL, DATE, NUMBER),  # no ORG, no IP class
        # Its score is a mean token probability, not the decoder's operating point (bias presets):
        # filtering on it is no OPF operating point, so no curve; presets are separate --config runs.
        "curve_by_filter": False,
    },
    "bardsai-v3": {
        "package": f"transformers==5.22.1 onnxruntime==1.31.1 {TORCH_CU126}",
        "checkpoint": "bardsai/eu-pii-anonimization-multilang-v3",
        "revision": BARDSAI_V3_REVISION,
        "config": "bundled pii_detector.py (read first), onnx/model_quantized.onnx, 512-token windows, 128 overlap",
        "threshold": "card default", "mask": "common", "device": "cpu", "gpu_gb": 0,
        # Frozen from config.json id2label at the pinned revision (adapter LABELS; the rest out of scope).
        "labels": BARDSAI,
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

# Label maps are frozen at each checkpoint revision.
ROUND2_ENTRIES = {
    "gliner-knowledgator": {
        "package": f"gliner==0.2.30 transformers==5.22.1 {TORCH_CU126}",
        "checkpoint": "knowledgator/gliner-pii-base-v1.0 (standalone; the sct-gliner model)",
        "revision": GLINER_KNOWLEDGATOR_REVISION,
        "config": "GLINER_PROMPTS, card threshold 0.3, fp16 on cuda (fp32 cpu), vendor truncation at 2048 words",
        "threshold": 0.3, "mask": "common", "device": "cuda|cpu", "gpu_gb": 1,
        "labels": dict(GLINER_PROMPTS), "supports": SCT_TYPES,
    },
    "piiranha": {
        "package": f"transformers==5.22.1 {TORCH_CU126}", "checkpoint": "iiiorg/piiranha-v1-detect-personal-information",
        "revision": PIIRANHA_REVISION,
        # weights CC-BY-NC-ND-4.0: benchmark only, never shipped
        "config": "token-classification pipeline, aggregation simple, 256-token windows, 64 overlap",
        "threshold": "0 (vendor)", "mask": "common", "device": "cuda|cpu", "gpu_gb": 1,
        "labels": PIIRANHA, "supports": (PERSON, LOCATION, EMAIL, PHONE, DATE, NUMBER),
    },
    "llm-guard": {
        "package": f"llm-guard==0.3.16 (presidio 2.2.358, transformers 4.51.3) {TORCH_CU126}",
        "checkpoint": "Isotonic/deberta-v3-base_finetuned_ai4privacy_v2 + Presidio regex (llm-guard defaults)",
        "revision": "9ea992753ab2686be4a8f64605ccc7be197ad794 (pinned by llm-guard)",
        "config": "Anonymize(Vault()) defaults: DEFAULT_ENTITY_TYPES, language en, PyTorch",
        "threshold": 0.5, "mask": "llm-guard [REDACTED_TYPE_n]", "device": "cuda|cpu", "gpu_gb": 1,
        "labels": LLM_GUARD,
        "supports": (PERSON, EMAIL, PHONE, IP, NUMBER),  # no LOCATION/ORG/DATE/URL in the default entity types
    },
    "scrubadub": {
        "package": f"scrubadub==2.0.1 scrubadub_spacy==2.0.0 spacy==3.8.18 en_core_web_trf==3.8.0 {TORCH_CU126}",
        "checkpoint": "en_core_web_trf (SpacyEntityDetector default for en)", "revision": "spaCy model wheel 3.8.0",
        "config": "Scrubber() default detectors + SpacyEntityDetector() (PERSON, PER, ORG), locale en_US",
        "threshold": "-", "mask": "scrubadub {{TYPE}}", "device": "cpu", "gpu_gb": 0,  # spaCy stays on CPU (no cupy)
        "labels": SCRUBADUB,
        "supports": (PERSON, ORGANISATION, EMAIL, PHONE, URL, NUMBER),  # location / dob detectors are off by default
    },
    "ai4privacy": {
        "package": f"transformers==5.22.1 {TORCH_CU126}",
        "checkpoint": "ai4privacy/llama-ai4privacy-multilingual-categorical-anonymiser-openpii",
        "revision": AI4PRIVACY_REVISION,
        "config": "token-classification pipeline, aggregation simple, 512-token windows, 128 overlap",
        "threshold": "0 (vendor)", "mask": "common", "device": "cuda|cpu", "gpu_gb": 1,
        "labels": AI4PRIVACY, "supports": (PERSON, LOCATION, EMAIL, PHONE, DATE, NUMBER),
    },
    "openmed": {
        "package": f"transformers==5.22.1 {TORCH_CU126}", "checkpoint": "OpenMed/privacy-filter-multilingual-v2",
        "revision": OPENMED_REVISION,
        "config": "opf decoder: BIOES constrained Viterbi, all biases 0 (no calibration file), bf16, 4096-token windows",
        "threshold": "default operating point", "mask": "common", "device": "cuda|cpu", "gpu_gb": 5,
        "labels": OPENMED, "supports": SCT_TYPES, "curve_by_filter": False,  # same score as opf: no operating point
    },
    "pii-tracer": {
        "package": f"transformers==5.22.1 {TORCH_CU126}", "checkpoint": "perplexity-ai/PII-Tracer (remote code)",
        "revision": PII_TRACER_REVISION,
        "config": "AutoModel(trust_remote_code).predict(text): bundled BIOES Viterbi, vendor truncation at 4096 tokens",
        "threshold": "default operating point", "mask": "pii-tracer [LABEL]", "device": "cuda|cpu", "gpu_gb": 3,
        "labels": PII_TRACER,
        "supports": (PERSON, LOCATION, EMAIL, PHONE, URL, DATE, NUMBER),  # no ORG, no IP class (as opf)
        "curve_by_filter": False,  # sigmoid of the mean decoded logit: not the decoder's operating point
    },
    "presidio-full": {
        "package": "presidio-analyzer==2.2.366 spacy==3.8.18 + every presidio language pack",
        "checkpoint": "PRESIDIO_MODELS (spaCy lg per language, else xx_ent_wiki_sm)", "revision": "spaCy model wheels 3.8.0",
        "config": "gold-language engines, PHONE_REGIONS, every predefined pattern recognizer, ORGANIZATION kept",
        "threshold": "0 (score_threshold default)", "mask": "common", "device": "cpu", "gpu_gb": 0,
        "labels": {**ENTRIES["presidio"]["labels"]}, "supports": SCT_TYPES,
    },
    "sct-training": {
        "package": "SqueakyCleanText==0.8.2 onnxruntime-gpu==1.31.1",
        "checkpoint": "sct DEFAULT_NER_MODELS (xlm-r conll ONNX, wikineural ONNX)",
        "revision": "sct/config.py DEFAULT_NER_REVISIONS", "config": "preset 'training' + reversible scoring mode",
        "threshold": "SCT default", "mask": "SCT placeholder", "device": "cuda|cpu", "gpu_gb": 2,
        # the preset turns dates, years and free numbers off; IBAN/card stay on, so NUMBER is partly covered
        "labels": SCT_LABELS, "supports": tuple(t for t in SCT_TYPES if t != DATE), "recover_dropped": True,
    },
    "sct-gliner-nvidia": {
        "package": f"SqueakyCleanText[gliner]==0.8.2 {TORCH_CU126}",
        "checkpoint": "nvidia/gliner-PII through SCT's GLiNER backend (gliner_model)",
        "revision": GLINER_NVIDIA_REVISION,
        "config": "TextCleanerConfig(ner_mode='pii', ner_backend='gliner', gliner_model='nvidia/gliner-PII') + reversible "
                  "scoring mode; SCT's PII prompts and threshold (tuned on knowledgator) as shipped",
        "threshold": "SCT default", "mask": "SCT placeholder", "device": "cuda|cpu", "gpu_gb": 3,
        "labels": SCT_LABELS, "supports": SCT_TYPES, "recover_dropped": True,
    },
}
ENTRIES.update(ROUND2_ENTRIES)


# Runner registry: adapter script (in ``adapters/``) and its config. The pinned install spec
# is ``envs/<entry>.txt``; every venv uses runtime.PYTHON. Config {} = the vendor default track.
RUNS = {
    "sct-default": {"adapter": "sct_adapter.py", "config": {"mode": "default", "language": "auto"}},
    "sct-gliner": {"adapter": "sct_adapter.py", "config": {"mode": "gliner", "language": "auto"}},
    **{name: {"adapter": f"{name.replace('-', '_')}_adapter.py", "config": {}} for name in
       ("presidio", "gliner-nvidia", "gliner2-fastino", "opf", "bardsai-v3", "datafog", "gliner-knowledgator",
        "piiranha", "llm-guard", "scrubadub", "ai4privacy", "openmed", "pii-tracer")},
    # presidio-full is defined by per-language engines, so it takes the gold language in both tracks
    "presidio-full": {"adapter": "presidio_full_adapter.py",
                      "config": {"language": "gold", "models": PRESIDIO_MODELS, "phone_regions": list(PHONE_REGIONS),
                                 "enable": "all", "organizations": True}},
    "sct-training": {"adapter": "sct_adapter.py", "config": {"mode": "default", "language": "auto", "preset": "training"}},
    "sct-gliner-nvidia": {"adapter": "sct_adapter.py", "config": {
        "mode": "gliner", "language": "auto",
        "overrides": {"gliner_model": GLINER_NVIDIA_MODEL, "gliner_revision": GLINER_NVIDIA_REVISION,
                      # explicit backend: pii mode on the onnx default forces gliner_onnx=True (nvidia ships no ONNX)
                      "ner_backend": "gliner"}}},
}
# Language tracks (PROTOCOL.md section 12), applied to every entry the same way. ``native``:
# each tool routes as shipped (SCT's own LID, Presidio en, DataFog no locale). ``gold``: every
# tool that takes a language gets the gold code. Entries absent here take no language input.
TRACKS = ("gold", "native")
GOLD_TRACK = {
    "sct-default": {"language": "gold"}, "sct-gliner": {"language": "gold"},
    "presidio": {"language": "gold"}, "datafog": {"locales": "gold"},
    "sct-training": {"language": "gold"}, "sct-gliner-nvidia": {"language": "gold"},
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


def tool(name: str, track: str = "native") -> dict[str, Any]:
    """Entry + run settings: ``adapter``, ``adapter_config`` and ``requirements`` (None = current Python)."""
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
# Round 2 (sct-integration-research round2-datasets/SHORTLIST.md section 2, after its U-cell pass): SCT was not
# developed against any round-2 set, and no entry trained on or was vendor-evaluated on one. Third-party published
# scores (E3) and clean-by-inference (c*) count as clean, so every round-2 cell is "-" except the one residual U,
# openai/privacy-filter x KDPII, which also stays U for openmed (OPF is the model inside it: decision 21).
CONTAMINATION = {
    "sct-default": {"*": "D", "gretel_en": "-", "argilla_pii": "-", **dict.fromkeys(ROUND2, "-")},
    "sct-gliner": {"*": "U, D", "gretel_en": "U", "argilla_pii": "U", **dict.fromkeys(ROUND2, "-")},
    "presidio": {"*": "-"},
    "gliner-nvidia": {"*": "-", "openpii": "E", "gretel_en": "E", "nemotron": "T", "argilla_pii": "E"},
    "gliner2-fastino": {"*": "-"},
    "opf": {"*": "U", **dict.fromkeys(ROUND2, "-"), "kdpii": "U"},
    "bardsai-v3": {"*": "-", "openpii": "T", "gretel_en": "T", "nemotron": "T"},
    "datafog": {"*": "-"},
    # Round 2 entries. Round-2 cells: the shortlist has no T or vendor E for any of these tools (E3 / c* = "-").
    # Training sets outside the harness are named by their family member (FAMILIES), so the family rule flags
    # openpii / argilla_pii without a hand-set family flag.
    "gliner-knowledgator": {"*": "U", **dict.fromkeys(ROUND2, "-")},  # the sct-gliner model without SCT's D
    "piiranha": {"*": "-", "ai4privacy-400k": "T"},  # card: datasets ai4privacy/pii-masking-400k
    "llm-guard": {"*": "-", "argilla_pii": "T"},  # Isotonic deberta: ai4privacy/pii-masking-200k (Argilla's source split)
    "scrubadub": {"*": "-"},  # rules + en_core_web_trf (OntoNotes 5), as presidio
    "ai4privacy": {"*": "-", "ai4privacy-500k": "T, E"},  # trained and evaluated on open-pii-masking-500k
    # base openai/privacy-filter is undisclosed (opf: U); the fine-tune names ai4privacy 200k/400k/500k/openpii-1m,
    # Nemotron-PII, gretel-pii-masking-en-v1, privy
    "openmed": {"*": "U", "openpii": "T", "gretel_en": "T", "nemotron": "T", "argilla_pii": "T",
                **dict.fromkeys(ROUND2, "-"), "kdpii": "U"},
    # PII-TRACE paper 3.7: in-house conversations + ai4privacy samples (set unnamed); 4.7 evaluates on ai4privacy,
    # Nemotron-PII, SPY, Gretel (gretel-pii-masking-en-v1), TAB
    "pii-tracer": {"*": "-", "ai4privacy-unnamed": "T, E", "nemotron": "E", "gretel_en": "E", "tab": "E"},
    "presidio-full": {"*": "-"},  # spaCy packs (OntoNotes / WikiNER / national corpora), none of ours
    "sct-training": {"*": "D", "gretel_en": "-", "argilla_pii": "-", **dict.fromkeys(ROUND2, "-")},
    "sct-gliner-nvidia": {"*": "D", "openpii": "E, D", "gretel_en": "E", "nemotron": "T, D", "argilla_pii": "E",
                          **dict.fromkeys(ROUND2, "-")},
}
# Generator families (decision 21: a dataset AND its family). A T, E or D flag on one member puts
# "<flag>-family" on the others. Gretel: gretel finance and gretel_en (Gretel Navigator) and nemotron
# (NeMo Data Designer, Navigator's successor after NVIDIA took over Gretel; the gretel_en and Nemotron-PII
# cards both carry the datadesigner tag). ai4privacy: openpii and argilla_pii (a slice of ai4privacy-200k).
# The ai4privacy family also lists the generator's training sets that are not loaded here (names only), so a model
# trained on one of them gets the family flag on openpii and argilla_pii.
FAMILIES = (("gretel", "gretel_en", "nemotron"),
            ("openpii", "argilla_pii", "ai4privacy-200k", "ai4privacy-400k", "ai4privacy-500k", "ai4privacy-openpii-1m",
             "ai4privacy-unnamed"))


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
    "gliner-knowledgator": ("en",), "piiranha": ("en", "it", "fr", "de", "nl", "es"),
    "llm-guard": ("en",), "scrubadub": ("en",),  # English locale / spaCy model only
    "ai4privacy": ("fr", "en", "de", "te", "hi", "it", "es", "nl"),
    "openmed": ("ar", "bn", "de", "en", "es", "fr", "hi", "it", "ja", "ko", "nl", "pt", "te", "tr", "vi", "zh"),
    "pii-tracer": None,  # card language field: [en, multilingual] (multilingual is in the language field itself)
    "presidio-full": None,  # every gold language gets an engine (xx_ent_wiki_sm where no lg pack exists)
    "sct-training": None, "sct-gliner-nvidia": ("en",),  # nvidia/gliner-PII is English-only
}


def claims(entry: str, lang: str) -> bool:
    """Is a gold language code (``pt_br``, ``ar-mixed_en``) inside the entry's claimed languages?"""
    claimed = CLAIMED_LANGUAGES.get(entry)
    return claimed is None or lang.replace("_", "-").split("-")[0] in claimed
