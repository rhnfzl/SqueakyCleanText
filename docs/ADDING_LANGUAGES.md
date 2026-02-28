# Adding Language Support to SqueakyCleanText

This guide covers how to add new language support to SqueakyCleanText, both as a **user** (via config) and as a **maintainer** (adding built-in support).

## Overview

SqueakyCleanText has 4 language-dependent subsystems:

| Subsystem | What it does | Default languages |
|-----------|-------------|-------------------|
| **Lingua detection** | Auto-detects input language | EN, NL, DE, ES |
| **Stopwords** | Removes common words for statistical model | EN, NL, DE, ES |
| **Date patterns** | Regex + fuzzy matching for month names | EN, NL, DE, ES |
| **NER models** | Named entity recognition (ONNX models) | EN, NL, DE, ES + Multilingual |

Each subsystem can be extended independently. A language doesn't need support in all 4 to be useful.

### v0.6.0 Notes

- **PII mode** (`ner_mode='pii'`) applies to all supported languages — PII labels are language-agnostic
- **Replacement modes** (`replacement_mode='placeholder'|'synthetic'|'reversible'`) work for all languages
- **ModernBERT** is available as an alternative English-only backend with 8192-token context (vs 512 for XLM-RoBERTa). Set via `ner_models={'ENGLISH': 'rhnfzl/modernbert-base-ner-conll03-english-onnx'}` once exported
- **GLiClass** document classification is language-independent (zero-shot, works with any input language)

---

## How to Specify Languages

All language parameters accept any of these formats (case-insensitive):

| Format | Example | Resolves to |
|--------|---------|-------------|
| Lingua name | `'ENGLISH'`, `'english'` | `'ENGLISH'` |
| ISO 639-1 | `'en'`, `'EN'` | `'ENGLISH'` |
| ISO 639-3 | `'eng'`, `'ENG'` | `'ENGLISH'` |

This applies to: `language`, `extra_languages`, `custom_stopwords` keys, and `custom_month_names` keys.

### Single language vs. tuple of languages

```python
# Pin to one language (skip auto-detection)
cfg = TextCleanerConfig(language='en')             # → detects nothing, always ENGLISH

# Restrict detection to listed languages (auto-detect among them)
cfg = TextCleanerConfig(language=('en', 'nl'))     # → detect per text, restricted to EN/NL

# Full auto-detection (default — all supported languages)
cfg = TextCleanerConfig(language=None)             # → detect among EN, NL, DE, ES + extras
```

---

## Option A: User-Level Extension (via TextCleanerConfig)

Users can add language support without modifying source code:

```python
from sct.config import TextCleanerConfig
from sct.sct import TextCleaner

cfg = TextCleanerConfig(
    # 1. Extend Lingua detection to include Polish (ISO codes also work: 'pl')
    extra_languages=('POLISH',),

    # 2. Custom stopwords (optional — auto-detected from stop-words package if available)
    custom_stopwords={
        'POLISH': frozenset({'i', 'w', 'na', 'z', 'do', 'nie', 'to', 'jest'}),
    },

    # 3. Custom month names for date detection (optional)
    custom_month_names={
        'POLISH': (
            'styczen', 'luty', 'marzec', 'kwiecien', 'maj', 'czerwiec',
            'lipiec', 'sierpien', 'wrzesien', 'pazdziernik', 'listopad', 'grudzien',
        ),
    },

    # 4. Custom NER model (optional — falls back to MULTILINGUAL if not provided)
    ner_models={
        'ENGLISH': 'rhnfzl/xlm-roberta-large-conll03-english-onnx',
        'MULTILINGUAL': 'rhnfzl/wikineural-multilingual-ner-onnx',
        'POLISH': 'my-org/polish-ner-onnx',  # Must be ONNX format
    },
)

cleaner = TextCleaner(cfg=cfg)
result = cleaner.process("Spotkanie z Janem Kowalskim 15 styczen 2025 w Warszawie")
```

### What each field does

**`extra_languages`**: Tuple of language identifiers (e.g. `'POLISH'`, `'pl'`, `'pol'`). Accepts Lingua names, ISO 639-1, or ISO 639-3 codes. This extends the Lingua detector to consider these languages during auto-detection.

**`custom_stopwords`**: Dict mapping language name to frozenset of stopwords. If provided for a language, **replaces** (not merges with) any auto-detected stopwords. If not provided, the `stop-words` package is queried using the language's ISO 639-1 code.

**`custom_month_names`**: Dict mapping language name to tuple of 12 month names (full form, in order Jan-Dec). These are compiled into the date regex at config time, enabling date detection for that language.

**`ner_models`**: Dict mapping language name to HuggingFace ONNX model repo ID. `ENGLISH` and `MULTILINGUAL` are always required. Languages without a dedicated NER model fall back to the `MULTILINGUAL` model.

### Language discovery

The `supported_languages` field is automatically computed as the union of:
- Default 4: `{ENGLISH, DUTCH, GERMAN, SPANISH}`
- `extra_languages` keys
- `ner_models` keys (minus `MULTILINGUAL`)
- `custom_stopwords` keys
- `custom_month_names` keys

You don't need to add a language to `extra_languages` if it's already declared in any other field.

### Minimal examples

**Just detection** (no stopwords/dates/NER for the new language):
```python
cfg = TextCleanerConfig(
    extra_languages=('FRENCH',),
    check_ner_process=False,
)
```

**Detection + stopwords** (auto-loaded from stop-words package):
```python
cfg = TextCleanerConfig(
    extra_languages=('FRENCH',),  # stop-words has French built-in
    check_ner_process=False,
)
```

**Pin language** (skip auto-detection — ISO codes accepted):
```python
cfg = TextCleanerConfig(
    language='fr',                # ISO 639-1 → resolves to 'FRENCH'
    extra_languages=('FRENCH',),  # Must be in supported set
    check_ner_process=False,
)
```

---

## Option B: Adding Built-In Language Support (Maintainer Guide)

There are two tiers of built-in support:

| Tier | What it means | Steps required |
|------|--------------|----------------|
| **Opt-in** | Users enable via `extra_languages=('FRENCH',)` | Steps 1, 3 (NER only), 7 |
| **Full default** | Language auto-detected for all users by default | Steps 1, 2, 3, 7 |

**FR/PT/IT were added as opt-in (v0.5.0)**: users set `extra_languages=('FRENCH',)` etc. — no changes to `resources.py` `DEFAULT_LANGUAGES` required. `build_detector()` is already fully dynamic and handles any Lingua language name.

To permanently add a new language (e.g. French) to the package defaults, follow these steps in dependency order.

### Step 1: `sct/utils/constants.py` — Add month names

Add the language's month names to `MONTH_NAMES_MULTILINGUAL`:

```python
MONTH_NAMES_MULTILINGUAL = {
    # ... existing entries ...
    # French (accented characters are supported — keys are lowercase canonical forms)
    "janvier": frozenset({"fr"}), "février": frozenset({"fr"}),
    "mars": frozenset({"fr"}),  # French March — NOT shared with English "march"
    "avril": frozenset({"fr"}), "juin": frozenset({"fr"}),
    "juillet": frozenset({"fr"}), "août": frozenset({"fr"}),
    "septembre": frozenset({"fr"}), "octobre": frozenset({"fr"}),
    "novembre": frozenset({"fr", "it"}),  # shared: FR novembre = IT novembre
    "décembre": frozenset({"fr"}),
    # Update existing entries when month names are shared:
    "mai": frozenset({"de", "fr"}),    # German Mai = French mai (both = May)
}
```

**Shared month names** — check these before adding new entries:
- `"agosto"` (ES/PT/IT August): add new ISO codes to the existing frozenset
- `"marzo"` (ES/IT March): already tagged `{"es", "it"}`
- `"abril"` (ES/PT April): already tagged `{"es", "pt"}`
- `"mai"` (DE/FR May): already tagged `{"de", "fr"}`
- `"novembre"` (FR/IT November): already tagged `{"fr", "it"}`

**Rules**:
- Keys must be lowercase (include accented characters as they appear in text)
- Values are frozensets of ISO 639-1 codes
- If a month name string is identical across languages, add all ISO codes to one entry
- Abbreviations (≤ 4 chars) are caught by exact DATE_REGEX; include unambiguous ones only
- Full names (> 4 chars) are used by fuzzy matching via `FUZZY_MONTH_VOCABULARY_BY_LANG`

Update `_LANG_TO_VOCAB`:
```python
_LANG_TO_VOCAB = {
    "ENGLISH": "en",
    "DUTCH": "nl",
    "GERMAN": "de",
    "SPANISH": "es",
    "FRENCH": "fr",     # ADD
    "PORTUGUESE": "pt", # ADD
    "ITALIAN": "it",    # ADD
}
```

The module-level `DATE_REGEX`, `FUZZY_MONTH_VOCABULARY`, and `FUZZY_MONTH_VOCABULARY_BY_LANG` are automatically recomputed from `MONTH_NAMES_MULTILINGUAL` at import time. No further changes to `datetime.py` needed.

### Step 2: `sct/utils/resources.py` — Update default languages (full-default tier only)

> **Skip this step for opt-in languages** — `build_detector()` dynamically handles any language in `supported_languages`. Only needed to add the language to the *automatic* detection set for all users.

Add the language to `LANGUAGES` (module-level Lingua preset) and `DEFAULT_LANGUAGES`:

```python
LANGUAGES = [Language.DUTCH, Language.ENGLISH, Language.FRENCH, Language.GERMAN, Language.SPANISH]
LANGUAGE_NAME = [(language.name).lower() for language in LANGUAGES]

DEFAULT_LANGUAGES = frozenset({'ENGLISH', 'DUTCH', 'FRENCH', 'GERMAN', 'SPANISH'})
```

### Step 3: `sct/config.py` — Update defaults

**For full-default tier** — add to `DEFAULT_LANGUAGES`:
```python
DEFAULT_LANGUAGES = frozenset({'ENGLISH', 'DUTCH', 'FRENCH', 'GERMAN', 'SPANISH'})
```

**For NER model fallback** — add to `DEFAULT_NER_MODELS` and `DEFAULT_TORCH_NER_MODELS`.
Start with the MULTILINGUAL fallback while a dedicated model is being prepared:
```python
DEFAULT_NER_MODELS: dict[str, str] = {
    # ... existing language-specific ONNX models ...
    'FRENCH': 'rhnfzl/wikineural-multilingual-ner-onnx',     # MULTILINGUAL fallback (update after Step 3.5)
    'PORTUGUESE': 'rhnfzl/wikineural-multilingual-ner-onnx', # MULTILINGUAL fallback (update after Step 3.5)
    'ITALIAN': 'rhnfzl/wikineural-multilingual-ner-onnx',    # MULTILINGUAL fallback (update after Step 3.5)
    'MULTILINGUAL': 'rhnfzl/wikineural-multilingual-ner-onnx',
}
```

> **Note on LANG_KEYS**: Do NOT update `LANG_KEYS` for opt-in languages. `LANG_KEYS` is tied to the deprecated `ner_models_list` tuple API and its length must stay fixed for backward compatibility. New languages are accessed via the `ner_models` dict API only.

### Step 3.5: `scripts/export_onnx_models.py` — Create dedicated ONNX model (optional)

> **FR/PT/IT (v0.5.0+) use `wikineural-multilingual-ner-onnx` as their NER model.** This is the accepted design — per-language ONNX models for these languages are hard to source (few CoNLL/WikiNER-style XLM-RoBERTa models exist for FR/PT/IT on HuggingFace). The multilingual model provides adequate PER/LOC/ORG/MISC coverage across all three languages.

This step is only needed if you specifically want a dedicated ONNX model for a language and have identified a suitable source model.
Skip this step entirely if the MULTILINGUAL fallback is acceptable (it is for FR/PT/IT).

**1. Find a suitable source model on HuggingFace:**

For languages with CoNLL-02/03 coverage (EN, NL, DE, ES), the canonical source models are the `FacebookAI/xlm-roberta-large-finetuned-conll0{2,3}-*` series. For other languages, search HuggingFace for XLM-RoBERTa-based NER models fine-tuned on a standard NER corpus (WikiNER, Evalita, HAREM, etc.):

```bash
# Search HuggingFace for French NER models
# Filter: task=token-classification, architecture=XLM-RoBERTa
# Prefer: high downloads, standard NER labels (PER/LOC/ORG/MISC), CoNLL-style evaluation
```

**2. Add the source → target mapping to `MODEL_MAP`:**

```python
# scripts/export_onnx_models.py
MODEL_MAP = {
    # ... existing entries ...
    # Add when a suitable source model is identified:
    "your-org/source-french-ner-model": "rhnfzl/xlm-roberta-large-ner-french-onnx",
    "your-org/source-portuguese-ner-model": "rhnfzl/xlm-roberta-large-ner-portuguese-onnx",
    "your-org/source-italian-ner-model": "rhnfzl/xlm-roberta-large-ner-italian-onnx",
}
```

**3. Export and upload the model:**

```bash
# Install export dependencies (separate from runtime deps)
pip install torch transformers "optimum[onnxruntime]" huggingface_hub

# Authenticate to HuggingFace Hub
huggingface-cli login

# Export a single language model
python scripts/export_onnx_models.py --models your-org/source-french-ner-model

# Or export all models in MODEL_MAP (skips already-uploaded ones)
python scripts/export_onnx_models.py

# Preview without uploading
python scripts/export_onnx_models.py --dry-run
```

The script handles: ONNX export via Optimum CLI, verification of required files, repo creation on HuggingFace Hub, file upload, and cleanup.

**4. Update `config.py` with the new ONNX repo ID:**

After a successful upload, replace the MULTILINGUAL fallback in `DEFAULT_NER_MODELS`:
```python
'FRENCH': 'rhnfzl/xlm-roberta-large-ner-french-onnx',  # dedicated model
```

### Step 4: `sct/utils/stopwords.py` — No changes needed

The `stop-words` package supports 33 languages. If the new language is among them, stopwords are auto-loaded via `language_to_iso()`. Check coverage:

```python
from stop_words import get_stop_words
print(get_stop_words('fr'))  # French: works
print(get_stop_words('ja'))  # Japanese: KeyError
```

If the package doesn't support the language, users can provide `custom_stopwords`, or you can add a built-in set.

### Step 5: `sct/utils/datetime.py` — No changes needed

Date regex is auto-built from `MONTH_NAMES_MULTILINGUAL` (Step 1). Fuzzy vocabulary is auto-built from the same data. No code changes needed.

### Step 6: `sct/utils/ner.py` — No changes needed

NER models are loaded from the `ner_models` dict. If a default model was added in Step 3, it will be available automatically. Languages without dedicated models fall back to `MULTILINGUAL`.

### Step 7: `tests/test_sct.py` — Add language-specific tests

Add date regex tests:
```python
def test_date_regex_french(self):
    dt = datetime.ProcessDateTime()
    cases = [
        ("Reunion le 15 janvier 2024 a Paris", "15 janvier 2024"),
        ("Le 3 mars 2023 debut du projet", "3 mars 2023"),
    ]
    for input_text, date_str in cases:
        result = dt.replace_dates(input_text, "<DATE>")
        self.assertIn("<DATE>", result, f"Failed for French date: {date_str}")
```

Add NER test (if dedicated model added):
```python
@requires_ner
def test_ner_process_french(self):
    processed = self.ner.ner_process(
        "Emmanuel Macron habite a Paris.",
        positional_tags=['PER', 'LOC'],
        ner_confidence_threshold=0.85,
        language='FRENCH',
    )
    self.assertNotIn("Emmanuel Macron", processed)
    self.assertNotIn("Paris", processed)
```

### Step 8: Version bump

Bump version in `sct/__init__.py` (single source of truth for version).

---

## Creating ONNX NER Models for New Languages

Use `scripts/export_onnx_models.py` to convert a PyTorch NER model from HuggingFace to ONNX format and upload it to the `rhnfzl` organization. This script is the standard tool for all ONNX model creation in this project.

### Prerequisites

```bash
# Install export dependencies (not in the runtime requirements)
pip install torch transformers "optimum[onnxruntime]" huggingface_hub

# Log in to HuggingFace Hub (requires write access to rhnfzl org)
huggingface-cli login
```

### Workflow

**Step 1: Identify a suitable source model**

For languages with CoNLL-02/03 data, prefer the `FacebookAI/xlm-roberta-large-finetuned-*` series (consistent architecture with existing EN/NL/DE/ES models). For other languages:

- Search HuggingFace for XLM-RoBERTa models fine-tuned on a standard NER corpus
- Prefer: high download counts, CoNLL-style labels (PER/LOC/ORG/MISC), public evaluation results
- CoNLL-02 covers Spanish, Dutch; CoNLL-03 covers English, German; other languages typically use WikiNER, Evalita (Italian), HAREM (Portuguese), QUAERO (French), etc.

**Step 2: Add the mapping to `MODEL_MAP`**

Edit `scripts/export_onnx_models.py`:

```python
MODEL_MAP = {
    # Existing entries (CoNLL-02/03):
    "FacebookAI/xlm-roberta-large-finetuned-conll03-english": "rhnfzl/xlm-roberta-large-conll03-english-onnx",
    # ... other existing entries ...

    # New language — add your chosen source model:
    "your-org/xlm-roberta-large-french-ner": "rhnfzl/xlm-roberta-large-ner-french-onnx",
}
```

Target repo IDs follow the pattern `rhnfzl/xlm-roberta-large-ner-{language}-onnx`.

**Step 3: Export and upload**

```bash
# Export a specific model only
python scripts/export_onnx_models.py --models your-org/xlm-roberta-large-french-ner

# Dry run first (exports but doesn't upload)
python scripts/export_onnx_models.py --models your-org/xlm-roberta-large-french-ner --dry-run

# Keep the export directory for inspection
python scripts/export_onnx_models.py --models your-org/xlm-roberta-large-french-ner --keep-exports
```

The script exports via `optimum.exporters.onnx` (token-classification task), verifies required files (`model.onnx`, `config.json`, `tokenizer.json`, `tokenizer_config.json`), creates the HuggingFace repo if it doesn't exist, uploads all files, and cleans up the local export directory.

**Step 4: Update `config.py`**

After a successful upload, replace the MULTILINGUAL fallback:

```python
# sct/config.py
DEFAULT_NER_MODELS: dict[str, str] = {
    # ...
    'FRENCH': 'rhnfzl/xlm-roberta-large-ner-french-onnx',  # was: wikineural-multilingual fallback
}
```

The ONNX pipeline (`sct/utils/onnx_pipeline.py`) handles download, caching, and inference automatically. INT8 dynamic quantization is also available at load time via the `quantize=True` parameter, reducing model size ~4× with a small accuracy trade-off.

---

## Language Support Matrix

Current built-in support (v0.5.0+):

| Language | Lingua (default) | Stopwords | Date Regex | Fuzzy Dates | NER Model | Opt-in via |
|----------|-----------------|-----------|------------|-------------|-----------|------------|
| English | Yes (auto) | Yes | Yes | Yes | `rhnfzl/xlm-roberta-large-conll03-english-onnx` | — |
| Dutch | Yes (auto) | Yes | Yes | Yes | `rhnfzl/xlm-roberta-large-conll02-dutch-onnx` | — |
| German | Yes (auto) | Yes | Yes | Yes | `rhnfzl/xlm-roberta-large-conll03-german-onnx` | — |
| Spanish | Yes (auto) | Yes | Yes | Yes | `rhnfzl/xlm-roberta-large-conll02-spanish-onnx` | — |
| French | No | Yes (stop-words) | Yes | Yes | `rhnfzl/wikineural-multilingual-ner-onnx` (multilingual) | `extra_languages=('FRENCH',)` |
| Portuguese | No | Yes (stop-words) | Yes | Yes | `rhnfzl/wikineural-multilingual-ner-onnx` (multilingual) | `extra_languages=('PORTUGUESE',)` |
| Italian | No | Yes (stop-words) | Yes | Yes | `rhnfzl/wikineural-multilingual-ner-onnx` (multilingual) | `extra_languages=('ITALIAN',)` |

User-extensible via `TextCleanerConfig` (no source changes needed):

| Feature | How to enable | Auto-detected? |
|---------|--------------|----------------|
| Lingua detection | `extra_languages=('POLISH',)` | N/A |
| Stopwords | Auto from `stop-words` pkg | Yes (33 languages) |
| Custom stopwords | `custom_stopwords={'POLISH': frozenset({...})}` | No |
| Date regex | `custom_month_names={'POLISH': (...)}` | No |
| NER model | `ner_models={'POLISH': 'model-id'}` | Falls back to MULTILINGUAL |

### Languages supported by the `stop-words` package

Arabic, Bulgarian, Catalan, Czech, Danish, Dutch, English, Finnish, French, German, Greek, Hindi, Hungarian, Indonesian, Irish, Italian, Japanese, Korean, Lithuanian, Latvian, Norwegian, Persian, Polish, Portuguese, Romanian, Russian, Slovak, Slovenian, Spanish, Swedish, Thai, Turkish, Ukrainian.

---

## Checklist for Adding a New Built-In Language

### Opt-in tier (users activate via `extra_languages`)
- [ ] Add month names to `MONTH_NAMES_MULTILINGUAL` in `constants.py` (check for shared names!)
- [ ] Add ISO code to `_LANG_TO_VOCAB` in `constants.py`
- [ ] Add to `DEFAULT_NER_MODELS` (dedicated ONNX model or MULTILINGUAL fallback) in `config.py`
- [ ] Add to `DEFAULT_TORCH_NER_MODELS` (dedicated Torch model or MULTILINGUAL fallback) in `config.py`
- [ ] **Create dedicated ONNX model** (optional — see Step 3.5; skip if MULTILINGUAL fallback is acceptable):
  - [ ] Find a suitable source model on HuggingFace (XLM-RoBERTa-based, CoNLL/WikiNER-style labels)
  - [ ] Add `"source-model": "rhnfzl/xlm-roberta-large-ner-{lang}-onnx"` to `MODEL_MAP` in `scripts/export_onnx_models.py`
  - [ ] Run `python scripts/export_onnx_models.py --models <source-model>` to export and upload to rhnfzl Hub
  - [ ] Update `DEFAULT_NER_MODELS` in `config.py` from MULTILINGUAL fallback to the new ONNX repo ID
- [ ] Add date regex tests in `test_sct.py`
- [ ] (Optional) Add NER tests if dedicated model added
- [ ] Verify: `ruff check sct/ tests/` clean
- [ ] Verify: `pytest tests/ -v` all pass
- [ ] Bump version in `pyproject.toml`

### Full-default tier (language auto-detected for all users)
All of the above, plus:
- [ ] Add `Language.XXX` to `LANGUAGES` list in `resources.py`
- [ ] Add to `DEFAULT_LANGUAGES` in both `resources.py` and `config.py`

### Do NOT change (backward-compat constraint)
- `LANG_KEYS` in `config.py` — tied to the deprecated `ner_models_list` tuple; changing its length breaks existing users
