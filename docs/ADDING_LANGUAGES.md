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

---

## Option A: User-Level Extension (via TextCleanerConfig)

Users can add language support without modifying source code:

```python
from sct.config import TextCleanerConfig
from sct.sct import TextCleaner

cfg = TextCleanerConfig(
    # 1. Extend Lingua detection to include Polish
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

**`extra_languages`**: Tuple of Lingua language names (e.g. `'POLISH'`, `'FRENCH'`, `'PORTUGUESE'`). Must be valid members of `lingua.Language`. This extends the Lingua detector to consider these languages during auto-detection.

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

**Pin language** (skip auto-detection):
```python
cfg = TextCleanerConfig(
    language='FRENCH',
    extra_languages=('FRENCH',),  # Must be in supported set
    check_ner_process=False,
)
```

---

## Option B: Adding Built-In Language Support (Maintainer Guide)

To permanently add a new language (e.g. French) to the package defaults, follow these steps in dependency order.

### Step 1: `sct/utils/constants.py` — Add month names

Add the language's month names to `MONTH_NAMES_MULTILINGUAL`:

```python
MONTH_NAMES_MULTILINGUAL = {
    # ... existing entries ...
    # French
    "janvier": frozenset({"fr"}), "fevrier": frozenset({"fr"}),
    "mars": frozenset({"fr", "en"}),  # "mars" shared with EN "march"? No, keep separate
    "avril": frozenset({"fr"}), "mai": frozenset({"fr", "de"}),  # shared with DE
    "juin": frozenset({"fr"}), "juillet": frozenset({"fr"}),
    "aout": frozenset({"fr"}), "septembre": frozenset({"fr", "en"}),
    "octobre": frozenset({"fr"}), "novembre": frozenset({"fr", "en"}),
    "decembre": frozenset({"fr"}),
}
```

**Rules**:
- Keys must be lowercase
- Values are frozensets of ISO 639-1 codes
- If a month name is shared across languages (e.g. "november"), add the new ISO code to the existing frozenset
- Include common abbreviations (3-4 chars) if they exist

Update `_LANG_TO_VOCAB`:
```python
_LANG_TO_VOCAB = {
    "ENGLISH": "en",
    "DUTCH": "nl",
    "GERMAN": "de",
    "SPANISH": "es",
    "FRENCH": "fr",  # ADD
}
```

The module-level `DATE_REGEX`, `FUZZY_MONTH_VOCABULARY`, and `FUZZY_MONTH_VOCABULARY_BY_LANG` are automatically recomputed from `MONTH_NAMES_MULTILINGUAL` at import time.

### Step 2: `sct/utils/resources.py` — Update default languages

Add the language to `LANGUAGES` and `DEFAULT_LANGUAGES`:

```python
LANGUAGES = [Language.DUTCH, Language.ENGLISH, Language.FRENCH, Language.GERMAN, Language.SPANISH]
LANGUAGE_NAME = [(language.name).lower() for language in LANGUAGES]

DEFAULT_LANGUAGES = frozenset({'ENGLISH', 'DUTCH', 'FRENCH', 'GERMAN', 'SPANISH'})
```

### Step 3: `sct/config.py` — Update defaults

Add to `DEFAULT_LANGUAGES`:
```python
DEFAULT_LANGUAGES = frozenset({'ENGLISH', 'DUTCH', 'FRENCH', 'GERMAN', 'SPANISH'})
```

If providing a default NER model, add to `DEFAULT_NER_MODELS`:
```python
DEFAULT_NER_MODELS: dict[str, str] = {
    'ENGLISH': 'rhnfzl/xlm-roberta-large-conll03-english-onnx',
    'DUTCH': 'rhnfzl/xlm-roberta-large-conll02-dutch-onnx',
    'FRENCH': 'rhnfzl/xlm-roberta-large-conll02-french-onnx',  # ADD
    'GERMAN': 'rhnfzl/xlm-roberta-large-conll03-german-onnx',
    'SPANISH': 'rhnfzl/xlm-roberta-large-conll02-spanish-onnx',
    'MULTILINGUAL': 'rhnfzl/wikineural-multilingual-ner-onnx',
}
```

Update `LANG_KEYS` (order matters for backward-compat `ner_models_list` tuple):
```python
LANG_KEYS = ('ENGLISH', 'DUTCH', 'FRENCH', 'GERMAN', 'SPANISH', 'MULTILINGUAL')
```

**WARNING**: Changing `LANG_KEYS` length breaks anyone using the deprecated `ner_models_list` tuple with positional arguments. Prefer the `ner_models` dict API. If you must change `LANG_KEYS`, also update `ner_models_list` default tuple in the dataclass.

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

Update `pyproject.toml`:
```toml
version = "0.5.0"  # Minor version bump for new language support
```

---

## Creating ONNX NER Models for New Languages

To add a dedicated NER model for a language:

1. **Find a HuggingFace NER model** trained on that language (e.g. CoNLL-02/03 datasets)
2. **Export to ONNX format** using the scripts in `scripts/` or Optimum:
   ```bash
   optimum-cli export onnx --model <hf-model-name> <output-dir>/
   ```
3. **Upload to HuggingFace Hub** with these files:
   - `model.onnx` — the ONNX model
   - `config.json` — model config
   - `tokenizer.json` — fast tokenizer
4. **Reference in config**:
   ```python
   ner_models={'FRENCH': 'your-org/french-ner-onnx', ...}
   ```

The ONNX pipeline (`sct/utils/onnx_pipeline.py`) handles download, caching, and inference automatically.

---

## Language Support Matrix

Current built-in support (v0.4.1+):

| Language | Lingua | Stopwords | Date Regex | Fuzzy Dates | NER Model |
|----------|--------|-----------|------------|-------------|-----------|
| English | Yes | Yes | Yes | Yes | Yes (conll03) |
| Dutch | Yes | Yes | Yes | Yes | Yes (conll02) |
| German | Yes | Yes | Yes | Yes | Yes (conll03) |
| Spanish | Yes | Yes | Yes | Yes | Yes (conll02) |

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

- [ ] Add month names to `MONTH_NAMES_MULTILINGUAL` in `constants.py`
- [ ] Add ISO code to `_LANG_TO_VOCAB` in `constants.py`
- [ ] Add `Language.XXX` to `LANGUAGES` list in `resources.py`
- [ ] Add to `DEFAULT_LANGUAGES` in both `resources.py` and `config.py`
- [ ] (Optional) Add ONNX NER model to `DEFAULT_NER_MODELS` in `config.py`
- [ ] (Optional) Update `LANG_KEYS` and default `ner_models_list` in `config.py`
- [ ] Add date regex tests in `test_sct.py`
- [ ] (Optional) Add NER tests if dedicated model added
- [ ] Verify: `ruff check sct/ tests/` clean
- [ ] Verify: `pytest tests/ -v` all pass
- [ ] Bump version in `pyproject.toml`
