<div align="center">

# SqueakyCleanText

[![PyPI](https://img.shields.io/pypi/v/squeakycleantext.svg)](https://pypi.org/project/squeakycleantext/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/squeakycleantext)](https://pypistats.org/packages/squeakycleantext)
[![Python package](https://github.com/rhnfzl/SqueakyCleanText/actions/workflows/python-package.yml/badge.svg)](https://github.com/rhnfzl/SqueakyCleanText/actions/workflows/python-package.yml)
[![Python Versions](https://img.shields.io/badge/Python-3.12%20|%203.13%20|%203.14-blue)](https://pypi.org/project/squeakycleantext/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A comprehensive text cleaning and preprocessing pipeline for machine learning and NLP tasks.
</div>

> **Using an AI coding assistant?** This repo includes an [`llms.txt`](https://github.com/rhnfzl/SqueakyCleanText/blob/main/llms.txt) with the full API surface, config reference, and Q&A - optimised for Claude, Cursor, Copilot, and ChatGPT.

In the world of machine learning and natural language processing, clean and well-structured text data is crucial for building effective downstream models and managing token limits in language models.

SqueakyCleanText simplifies the process by automatically addressing common text issues - removing PII, anonymizing named entities (persons, organisations, locations), and ensuring your data is clean and well-structured for language models and classical ML pipelines with minimal effort on your part.

### Key Features

- **Named Entity Recognition (NER)**:
  - Multi-backend: ONNX (default, torch-free), PyTorch, GLiNER, and ensemble modes
  - Zero-shot custom entities via GLiNER (e.g., PRODUCT, EVENT, SKILL)
  - Multi-language support (English, Dutch, German, Spanish, French, Portuguese, Italian)
  - Ensemble voting across backends for improved accuracy
  - Configurable confidence thresholds
  - Lazy model loading (models load on demand per language)
  - Shared ONNX sessions across same-model languages (~600 MB RAM saved)
  - Automatic text chunking for long documents (CJK/Arabic safe)
  - GPU acceleration support (CUDA for ONNX and PyTorch)
  - Model warm-up API to pre-load on startup
- **Text Normalization**:
  - Corrects text encoding problems and handles bad Unicode characters
  - Removes or replaces HTML tags and URLs with configurable tokens
  - Handles emails, phone numbers, and other contact details
  - Multilingual date detection and replacement (ISO 8601, month names, common formats)
  - Fuzzy date matching for misspelled months (requires `[fuzzy]` extra)
  - Year and number standardization
  - Configurable emoji removal
  - Configurable bracket/brace content removal
  - Removes isolated letters and symbols
  - Normalizes whitespace and handles currency symbols
  - Smart case folding (preserves NER tokens like `<PERSON>`)
- **Language Support**:
  - Automatic language detection over all 75 Lingua languages, with a confidence score
  - Language-specific NER models; French, Portuguese, Italian via multilingual model
  - Language-aware stopword removal
  - Extensible: add custom languages with stopwords, month names, and NER models
- **Pipelines and tools**:
  - `training` preset: remove only PII, keep code, numbers, case and non-Latin text as written
  - Value-free `result.signals` (language, PII counts and types, warnings, model provenance)
  - Adapters for Hugging Face `datasets`, DataTrove, LangChain and Haystack; recipes for Ray, Spark, Polars, DuckDB, dlt and Daft
  - Score columns, data cards and the `sct-report` command (PII share, optional quality score and PII-normalised MinHash, k-anonymity)
  - MCP server (`sct-mcp`) with warnings-first output and optional file input
- **Dual Output Formats**:
  - Language Model format (preserves structure with tokens)
  - Statistical Model format (optimized for classical ML)
- **Performance**:
  - ONNX Runtime inference (torch-free base install, ~3-5x faster than PyTorch)
  - Thread-parallel batch processing with NER batched across documents (`process_batch` 3-4x faster than 0.7.1 on CPU, see [docs/PERFORMANCE.md](docs/PERFORMANCE.md))
  - Async batch processing (`aprocess_batch`) for FastAPI / aiohttp
  - Lazy model loading (only loads models as needed)
  - Shared ONNX sessions for same-model languages (saves ~600 MB for FR/PT/IT)
  - Memory-efficient processing of large texts
  - GPU acceleration (CUDA) for both ONNX and PyTorch backends

![Default Flow of cleaning Text](resources/sct_flow.png)

### Benefits

#### For Language Models
- Maintains text structure while anonymizing sensitive information
- Configurable token replacements
- Preserves context while removing noise
- Handles long documents through intelligent chunking

#### For Statistical Models
- Removes stopwords and punctuation
- Case normalization
- Special symbol removal
- Optimized for classification tasks

#### Advanced NER Processing
- Ensemble approach reduces missed entities
- Language-specific models improve accuracy
- Confidence thresholds for precision control
- Efficient batch processing for large datasets
- Automatic handling of long documents

## Installation

```sh
pip install SqueakyCleanText
```

The base install uses **ONNX Runtime** for NER inference - no PyTorch or Transformers required.

### Optional Extras

| Extra | Command | What it adds |
|-------|---------|--------------|
| GPU | `pip install SqueakyCleanText[gpu]` | CUDA-accelerated ONNX inference (see [docs/PERFORMANCE.md](docs/PERFORMANCE.md) for CUDA setup and speed by hardware) |
| Fuzzy dates | `pip install SqueakyCleanText[fuzzy]` | Fuzzy month name matching ([rapidfuzz](https://github.com/rapidfuzz/RapidFuzz)) |
| PyTorch NER | `pip install SqueakyCleanText[torch]` | PyTorch/Transformers NER backend |
| GLiNER | `pip install SqueakyCleanText[gliner]` | [GLiNER](https://github.com/urchade/GLiNER) zero-shot NER |
| GLiNER2 | `pip install SqueakyCleanText[gliner2]` | [GLiNER2](https://github.com/Knowledgator/GLiNER) (knowledgator) backend |
| Synthetic | `pip install SqueakyCleanText[synthetic]` | Faker-based synthetic replacement (realistic fake values instead of `<TAG>` tokens) |
| Presidio | `pip install SqueakyCleanText[presidio]` | Presidio-analyzer for `presidio_gliner` backend |
| Classify | `pip install SqueakyCleanText[classify]` | GLiClass document-level pre-classification |
| MCP | `pip install SqueakyCleanText[mcp]` | FastMCP server with text and JSON cleaning tools |
| Datasets | `pip install SqueakyCleanText[datasets]` | Hugging Face `datasets` adapter ([docs/INTEGRATIONS.md](docs/INTEGRATIONS.md)) |
| DataTrove | `pip install SqueakyCleanText[datatrove]` | DataTrove pipeline step |
| LangChain | `pip install SqueakyCleanText[langchain]` | LangChain document transformer |
| Haystack | `pip install SqueakyCleanText[haystack]` | Haystack document cleaner component |
| Score | `pip install SqueakyCleanText[score]` | MinHash signatures for score columns and `sct-report` ([docs/SCORE.md](docs/SCORE.md)) |
| OCR | `pip install SqueakyCleanText[ocr]` | Pillow image redaction and Tesseract OCR adapter |
| All NER | `pip install SqueakyCleanText[all-ner]` | All NER backends combined |
| Development | `pip install SqueakyCleanText[dev]` | Testing and linting tools |

You can combine extras: `pip install SqueakyCleanText[gpu,fuzzy,gliner]`

## Usage

### Basic Usage

```python
from sct import TextCleaner

# Initialize the TextCleaner
cleaner = TextCleaner()

# Input text
text = "Contact John Doe at john.doe@company.com. Meeting on 2023-10-01."

# Process the text
lm_text, stat_text, lang = cleaner.process(text)

print(f"Language Model format:    {lm_text}")
# Output: "Contact <PERSON> at <EMAIL>. Meeting on <YEAR><NUMBER><NUMBER>."
# (the date step is off by default; check_replace_dates=True keeps the date as one token)

print(f"Statistical Model format: {stat_text}")
# Output: "contact person email meeting yearnumbernumber"

print(f"Detected Language: {lang}")
# Output: "ENGLISH"
```

### Using TextCleanerConfig

```python
from sct import TextCleaner, TextCleanerConfig

# Create an immutable configuration
cfg = TextCleanerConfig(
    check_ner_process=True,
    ner_confidence_threshold=0.85,
    positional_tags=('PER', 'LOC', 'ORG', 'MISC'),
    replace_with_url="<URL>",
    replace_with_email="<EMAIL>",
    replace_with_phone_numbers="<PHONE>",
    language="en",  # Pin to English (also accepts 'ENGLISH', 'eng')
)

# Initialize with config
cleaner = TextCleaner(cfg=cfg)
```

#### Presets

`preset_config("training")` (also `TextCleanerConfig.from_preset()`) only removes PII and
repairs broken unicode, for LLM training and RAG corpora: URLs, emails, phones, IPs and NER
entities become placeholders; numbers, years, dates, HTML, brackets, case, punctuation,
line breaks, indentation and backslash escapes in code are kept, and no `stat_text` is made.
`"sct"` is the library default config. Any config field overrides the preset:

```python
from sct import TextCleaner, preset_config

cleaner = TextCleaner(cfg=preset_config("training", check_replace_numbers=True))
```

The data-pipeline adapters use `"training"` when no config is given. GLiNER PII stays opt-in
(`ner_mode="pii"`).

#### Language Specification

All language parameters accept Lingua names (`'ENGLISH'`), ISO 639-1 (`'en'`), or ISO 639-3 (`'eng'`) codes:

```python
from sct import TextCleanerConfig

# Pin to one language (skip auto-detection)
cfg = TextCleanerConfig(language='de', check_ner_process=False)

# Restrict detection to specific languages (auto-detect among them)
cfg = TextCleanerConfig(language=('en', 'nl', 'de'), check_ner_process=False)

# Add extra languages (stopwords/dates/NER keys)
cfg = TextCleanerConfig(extra_languages=('fr', 'pt'), check_ner_process=False)

# Pin any of the 75 Lingua languages, e.g. Hindi
cfg = TextCleanerConfig(language='hi', check_ner_process=False)

# Bring your own NER model for a language: Hindi text runs only this model
cfg = TextCleanerConfig(ner_models={'ENGLISH': '...', 'MULTILINGUAL': '...', 'HINDI': 'your/hindi-ner-onnx'})
```

Auto-detection covers all 75 Lingua languages (`detect_all_languages=True`). Languages
without their own NER model run the MULTILINGUAL + ENGLISH fallback; stopwords exist for
EN/NL/DE/ES (others use English) and month names for 7 languages plus `custom_month_names`.
`ner_backend='gliner'` is language independent, so any pinned language works with it.

### GLiNER: Zero-Shot Custom NER

Use [GLiNER](https://github.com/urchade/GLiNER) to recognize any entity type without retraining:

```python
from sct import TextCleaner, TextCleanerConfig

cfg = TextCleanerConfig(
    ner_backend='gliner',
    gliner_model='urchade/gliner_large-v2.1',
    gliner_labels=('person', 'organization', 'location', 'product', 'event'),
    gliner_label_map={
        'person': 'PER', 'organization': 'ORG', 'location': 'LOC',
        # 'product' and 'event' are unmapped - they become <PRODUCT>, <EVENT> tokens
    },
    gliner_threshold=0.4,
)

cleaner = TextCleaner(cfg=cfg)
lm_text, stat_text, lang = cleaner.process(
    "John bought an iPhone at the Apple Store in Berlin during CES 2025."
)
# lm_text: "<PERSON> bought an <PRODUCT> at the <ORGANISATION> in <LOCATION> during <EVENT>."
```

### Ensemble NER

Combine ONNX/Torch models with GLiNER for improved recall via ensemble voting:

```python
from sct import TextCleaner, TextCleanerConfig

cfg = TextCleanerConfig(
    ner_backend='ensemble_onnx',  # or 'ensemble_torch'
    gliner_model='urchade/gliner_large-v2.1',
    gliner_labels=('person', 'organization', 'location'),
    gliner_label_map={'person': 'PER', 'organization': 'ORG', 'location': 'LOC'},
)

cleaner = TextCleaner(cfg=cfg)
lm_text, stat_text, lang = cleaner.process("Angela Merkel visited the Bundestag in Berlin.")
```

### PII Detection Mode

Automatically configure GLiNER for comprehensive PII detection with 60+ entity types (personal, financial, healthcare, identity, digital):

```python
from sct import TextCleaner, TextCleanerConfig

cfg = TextCleanerConfig(ner_mode='pii')

cleaner = TextCleaner(cfg=cfg)
lm_text, stat_text, lang = cleaner.process(
    "John Smith's SSN is 123-45-6789, email john@example.com, DOB 1990-01-15"
)
# Entities are anonymized: names, SSNs, emails, dates of birth, and 50+ more PII types
```

PII mode auto-configures: `ner_backend='gliner'`, uses [`knowledgator/gliner-pii-base-v1.0`](https://huggingface.co/knowledgator/gliner-pii-base-v1.0), uses `gliner_threshold` 0.3 and the same final `ner_confidence_threshold` (recall-focused), and expands positional tags. Both thresholds default to `None` ("not set"); any value you pass, even 0.85 or 0.4, is used as given. `cfg.effective_ner_confidence_threshold` and `cfg.effective_gliner_threshold` give the values that run.

**Alternative PII models** (pass as `gliner_model`):

| Model | Type | Size | Labels | F1 |
|-------|------|------|--------|-----|
| [`knowledgator/gliner-pii-base-v1.0`](https://huggingface.co/knowledgator/gliner-pii-base-v1.0) | Uni-encoder | 330MB (ONNX FP16) | 60+ | 80.99% |
| [`nvidia/gliner-PII`](https://huggingface.co/nvidia/gliner-PII) | Bi-encoder | 570MB | 55+ | — |
| [`gretelai/gretel-gliner-bi-base-v1.0`](https://huggingface.co/gretelai/gretel-gliner-bi-base-v1.0) | Bi-encoder | ~800MB | 40+ | 95% |
| [`urchade/gliner_multi_pii-v1`](https://huggingface.co/urchade/gliner_multi_pii-v1) | Multilingual | — | — | — |

### Synthetic Replacement

Replace detected entities with realistic fake values (via [Faker](https://faker.readthedocs.io/)) instead of `<TAG>` placeholder tokens:

```python
from sct import TextCleaner, TextCleanerConfig

cfg = TextCleanerConfig(
    ner_mode='pii',
    replacement_mode='synthetic',  # pip install squeakycleantext[synthetic]
)

cleaner = TextCleaner(cfg=cfg)
lm_text, stat_text, lang = cleaner.process(
    "Contact John Smith at john.smith@company.com or +1-555-0123"
)
# Output: "Contact Jennifer Williams at lisa45@example.net or +1-555-0198"
# Same entity always maps to same fake value within a document
```

> **Note**: Synthetic replacement preserves data utility for downstream ML tasks but is NOT GDPR-compliant anonymization. Same-document consistency is maintained (same entity text always maps to the same fake value).

### Reversible Anonymization

Replace entities with indexed placeholders (`<PERSON_0>`, `<LOCATION_1>`) and get a mapping for round-trip deanonymization:

```python
from sct import TextCleaner, TextCleanerConfig

cfg = TextCleanerConfig(
    ner_mode='pii',
    replacement_mode='reversible',
)

cleaner = TextCleaner(cfg=cfg)
result = cleaner.process("John Smith works at Google in London.")

print(result.lm_text)
# "<PERSON_0> works at <ORGANISATION_0> in <LOCATION_0>."

# Access the anonymization map via metadata
anon_map = result.metadata['anon_map']
restored = anon_map.deanonymize(result.lm_text)
# "John Smith works at Google in London."

# Serialize the map for storage
import json
json.dumps(anon_map.to_dict())
```

> **Note**: `ProcessResult` from `process()` unpacks as a 3-tuple (`lm_text, stat_text, language`) for backward compatibility, but also exposes `.metadata` for reversible maps and document classification. `metadata` is always a dict with value-free counts: `metadata['regex_counts']` (e.g. `{"EMAIL": {"count": 1, "chars": 17}}`, one entry per regex replacement type) and `metadata['entity_counts']` (NER findings per type, same shape). It also holds `language_confidence` (Lingua top-1 confidence, `1.0` when pinned, `None` when unknown), `language_source` (`pinned`, `detected`, `fallback` when a `language` tuple fell back to its first entry, or `none`) and `warnings`, a sorted list of codes: `language_unknown`, `language_fallback`, `language_low_confidence`, `ner_fallback_models` (the language has no own NER model and ran MULTILINGUAL + ENGLISH), `cuda_fallback` (CUDA was requested but a model ran on CPU), `empty_input`. `metadata['provenance']` is `{"backend", "models"}`, one `{"key", "id", "revision", "provider"}` entry per NER model that runs for the document's language (`revision` is the resolved Hub commit for ONNX models, `provider` the ONNX execution provider or the torch device); `{"backend": None, "models": []}` when NER is off.

### Document Classification (GLiClass)

Classify documents before processing using zero-shot classification with [GLiClass](https://github.com/Knowledgator/GLiClass):

```python
from sct import TextCleaner, TextCleanerConfig

cfg = TextCleanerConfig(
    check_classify_document=True,
    gliclass_labels=('email', 'code', 'legal', 'medical'),
    # gliclass_model defaults to 'knowledgator/gliclass-edge-v3.0' (32.7M params)
)

cleaner = TextCleaner(cfg=cfg)  # pip install squeakycleantext[classify]
result = cleaner.process("Dear Sir, please find attached the contract...")

# Classification results in metadata
print(result.metadata['classes'])
# [{"label": "email", "score": 0.92}, {"label": "legal", "score": 0.78}]
```

### Bi-Encoder GLiNER Models

Bi-encoder models (ModernBERT, etc.) are auto-detected and leverage pre-computed label embeddings for faster inference with larger context windows:

```python
from sct import TextCleaner, TextCleanerConfig

cfg = TextCleanerConfig(
    ner_backend='gliner',
    gliner_model='knowledgator/gliner-bi-base-v2.0',
    gliner_labels=('person', 'organization', 'location'),
)

cleaner = TextCleaner(cfg=cfg)
# Auto-detects bi-encoder → caches label embeddings → uses 2048+ token context window
```

### Entity Description Labels (ZERONER-Style)

Provide natural-language descriptions for labels to improve zero-shot recognition accuracy:

```python
from sct import TextCleaner, TextCleanerConfig

cfg = TextCleanerConfig(
    ner_backend='gliner',
    gliner_model='knowledgator/gliner-bi-base-v2.0',
    gliner_label_descriptions={
        'person': "a person's full legal name",
        'location': "a geographical place or address",
        'organization': "a company, institution, or government body",
    },
)

cleaner = TextCleaner(cfg=cfg)
# Descriptions are used for inference, results are mapped back to original label names
```

`result.signals` is the value-free view of that metadata, with the same keys for every input:
`language`, `language_confidence`, `language_source`, `chars`, `pii_count`, `pii_chars`,
`pii_share`, `pii_types`, `regex_counts`, `entity_counts`, `sensitive`, `warnings` (adds
`ner_off`) and `models`. It never holds entity text, so it is safe to log or store next to
cleaned data. Score columns, `sct-report`, the adapters and the MCP server read it.

### Batch Processing

```python
from sct import TextCleaner, TextCleanerConfig

cfg = TextCleanerConfig(
    check_remove_stopwords=True,
    check_remove_punctuation=True,
    check_ner_process=True,
    positional_tags=('PER', 'ORG', 'LOC'),
    ner_confidence_threshold=0.90,
)

cleaner = TextCleaner(cfg=cfg)

# Sample texts
texts = [
    "Email maria.garcia@example.es for more info.",  # Spanish
    "Besuchen Sie uns im Büro in Berlin.",           # German
    "Voor vragen, bel +31 20 123 4567.",             # Dutch
]

# Process texts in batch (uses ThreadPoolExecutor for parallel processing)
results = cleaner.process_batch(texts, batch_size=2)

for lm_text, stat_text, lang in results:
    print(f"Language: {lang}")
    print(f"LM Format:    {lm_text}")
    print(f"Stat Format:  {stat_text}")
    print("-" * 40)
```

<details>
<summary>Legacy Configuration (backward compatible)</summary>

```python
from sct import sct, config

# Customize settings via module-level variables
config.CHECK_NER_PROCESS = True
config.NER_CONFIDENCE_THRESHOLD = 0.85
config.POSITIONAL_TAGS = ['PER', 'LOC', 'ORG']
config.REPLACE_WITH_URL = "<URL>"
config.REPLACE_WITH_EMAIL = "<EMAIL>"
config.LANGUAGE = "ENGLISH"

# Initialize (reads from module-level config)
cleaner = sct.TextCleaner()
```

> **Note**: The legacy module-level configuration is not thread-safe. For concurrent processing, use `TextCleanerConfig` instead.

</details>

## NER Backends

SqueakyCleanText supports six NER backends, selectable via the `ner_backend` config field:

| Backend | Description | Dependencies | Best for |
|---------|-------------|-------------|----------|
| `onnx` (default) | ONNX Runtime inference with quantized XLM-RoBERTa models | Base install | Production: fast, torch-free |
| `torch` | PyTorch/Transformers pipeline with full XLM-RoBERTa models | `[torch]` extra | Compatibility with existing PyTorch workflows |
| `gliner` | GLiNER zero-shot NER with custom entity labels | `[gliner]` or `[gliner2]` extra | Custom entity types, PII detection, bi-encoder models |
| `ensemble_onnx` | ONNX + GLiNER ensemble voting | `[gliner]` extra | Maximum recall with custom entities |
| `ensemble_torch` | Torch + GLiNER ensemble voting | `[torch,gliner]` extra | Maximum recall with PyTorch |
| `presidio_gliner` | Presidio + GLiNER recognizer (beta) | `presidio-analyzer`, `[gliner]` | Context-aware NER via Presidio's pipeline |

### Default NER Models (ONNX)

| Language | Model |
|----------|-------|
| English | [`rhnfzl/xlm-roberta-large-conll03-english-onnx`](https://huggingface.co/rhnfzl/xlm-roberta-large-conll03-english-onnx) |
| Dutch | [`rhnfzl/xlm-roberta-large-conll02-dutch-onnx`](https://huggingface.co/rhnfzl/xlm-roberta-large-conll02-dutch-onnx) |
| German | [`rhnfzl/xlm-roberta-large-conll03-german-onnx`](https://huggingface.co/rhnfzl/xlm-roberta-large-conll03-german-onnx) |
| Spanish | [`rhnfzl/xlm-roberta-large-conll02-spanish-onnx`](https://huggingface.co/rhnfzl/xlm-roberta-large-conll02-spanish-onnx) |
| French / Portuguese / Italian | [`rhnfzl/wikineural-multilingual-ner-onnx`](https://huggingface.co/rhnfzl/wikineural-multilingual-ner-onnx) (shared session) |
| Multilingual (fallback) | [`rhnfzl/wikineural-multilingual-ner-onnx`](https://huggingface.co/rhnfzl/wikineural-multilingual-ner-onnx) |

### GLiNER Model Recommendations

| Model | Architecture | Context | Languages | Best for |
|-------|-------------|---------|-----------|----------|
| `knowledgator/gliner-bi-base-v2.0` | Bi-encoder (ModernBERT) | 2048 | Multi | General NER, long documents |
| `knowledgator/gliner-pii-base-v1.0` | Bi-encoder | 2048 | Multi | PII detection (60+ entity types) |
| `urchade/gliner_large-v2.1` | Uni-encoder (DeBERTa) | 512 | Multi | Legacy, high accuracy on short texts |
| `MatteoFasulo/ModernBERT-base-NER` | ModernBERT | 8192 | English | English-only, very long context |

> **GLiNER2 note**: `pip install squeakycleantext[gliner2]` installs [Knowledgator's gliner2 package](https://github.com/Knowledgator/GLiNER), not Fastino AI's GLiNER2 from EMNLP 2025 (different API).

### GLiNER Label Mapping

GLiNER uses lowercase free-text labels (e.g., `'person'`, `'product'`). To map them to standard NER tags used by the anonymizer, use `gliner_label_map`:

```python
gliner_label_map={
    'person': 'PER',          # → <PERSON>
    'organization': 'ORG',    # → <ORGANISATION>
    'location': 'LOC',        # → <LOCATION>
}
# Unmapped labels are uppercased automatically:
# 'product' → <PRODUCT>, 'event' → <EVENT>, 'skill' → <SKILL>
```

## API

### `TextCleaner`

#### `process(text: str) -> Tuple[str, Optional[str], Optional[str]]`

Processes the input text and returns a tuple containing:
  - Cleaned text formatted for language models.
  - Cleaned text formatted for statistical models (`None` if `check_statistical_model_processing` is `False`).
  - Detected language of the text (`None` if language detection is disabled).

#### `process_batch(texts: List[str], batch_size: int = None) -> List[Tuple[str, Optional[str], Optional[str]]]`

Processes multiple texts using thread-parallel execution. Each result follows the same format as `process()`.

#### `aprocess_batch(texts: List[str], batch_size: int = None) -> List[Tuple[str, Optional[str], Optional[str]]]`

Async version of `process_batch` for use with asyncio-based frameworks (FastAPI, aiohttp). Runs the batch in a thread-pool executor so it does not block the event loop:

```python
from sct import TextCleaner

cleaner = TextCleaner()

# In an async context (FastAPI route, aiohttp handler, etc.)
async def handler(texts):
    return await cleaner.aprocess_batch(texts)
```

#### `warmup(languages: Optional[List[str]] = None) -> None`

Pre-loads NER models to avoid first-request latency. Call once during application startup:

```python
from sct import TextCleaner

cleaner = TextCleaner()
cleaner.warmup(['ENGLISH', 'DUTCH'])  # or warmup() for all supported languages
```

### `TextCleanerConfig`

Immutable (frozen) dataclass. Create modified copies with `dataclasses.replace()`:

```python
import dataclasses
from sct import TextCleanerConfig

cfg = TextCleanerConfig()
new_cfg = dataclasses.replace(cfg, check_ner_process=False)
```

<details>
<summary>Full configuration reference</summary>

**Pipeline toggles** (all `bool`, default shown):

| Field | Default | Description |
|-------|---------|-------------|
| `check_detect_language` | `True` | Auto-detect language |
| `check_fix_bad_unicode` | `True` | Fix encoding issues via ftfy |
| `check_decode_unicode_escapes` | `True` | Part of `check_fix_bad_unicode`: decode literal backslash escapes (`\u00e9` to `é`, `\n` to a line break). `False` keeps them as written (code, JSON) |
| `check_to_ascii_unicode` | `True` | Transliterate to ASCII |
| `check_replace_html` | `True` | Strip/replace HTML tags |
| `check_replace_urls` | `True` | Replace URLs with token |
| `check_replace_emails` | `True` | Replace emails with token |
| `check_replace_ips` | `True` | Replace IPv4/IPv6 addresses (runs before dates, phones, years) |
| `replace_ips_public_only` | `False` | Keep private, loopback and reserved IPs; replace only public ones |
| `check_replace_years` | `True` | Replace years (1900-2099) |
| `check_replace_dates` | `False` | Replace full dates (ISO 8601, month names) |
| `check_fuzzy_replace_dates` | `False` | Fuzzy match misspelled months (requires `[fuzzy]`) |
| `check_replace_phone_numbers` | `True` | Replace phone numbers |
| `check_replace_numbers` | `True` | Replace standalone numbers |
| `replace_numbers_in_words` | `False` | Also replace digits inside Latin/Greek/Cyrillic words (`GPT4` to `GPT<NUMBER>`) |
| `check_replace_currency_symbols` | `True` | Replace currency symbols |
| `check_ner_process` | `True` | Run NER entity recognition |
| `check_remove_isolated_letters` | `True` | Remove single letters |
| `check_remove_isolated_special_symbols` | `True` | Remove isolated symbols |
| `check_remove_bracket_content` | `True` | Remove `[...]` content |
| `check_remove_brace_content` | `True` | Remove `{...}` content |
| `check_normalize_whitespace` | `True` | Normalize whitespace |
| `check_statistical_model_processing` | `True` | Generate stat model output |
| `check_casefold` | `True` | Lowercase stat output |
| `check_smart_casefold` | `False` | Lowercase but preserve NER tokens |
| `check_remove_stopwords` | `True` | Remove stopwords from stat output |
| `check_remove_punctuation` | `True` | Remove punctuation from stat output |
| `check_remove_stext_custom_stop_words` | `True` | Remove custom stop words from stat output |
| `check_remove_emoji` | `False` | Remove emoji characters |

**Replacement tokens** (all `str`):

| Field | Default |
|-------|---------|
| `replace_with_url` | `"<URL>"` |
| `replace_with_html` | `"<HTML>"` |
| `replace_with_email` | `"<EMAIL>"` |
| `replace_with_ips` | `"<IP>"` |
| `replace_with_years` | `"<YEAR>"` |
| `replace_with_dates` | `"<DATE>"` |
| `replace_with_phone_numbers` | `"<PHONE>"` |
| `replace_with_numbers` | `"<NUMBER>"` |
| `replace_with_currency_symbols` | `None` |

**NER settings**:

| Field | Default | Description |
|-------|---------|-------------|
| `ner_backend` | `'onnx'` | Backend: `onnx`, `torch`, `gliner`, `ensemble_onnx`, `ensemble_torch`, `presidio_gliner` |
| `ner_mode` | `'standard'` | `'standard'` or `'pii'` (auto-configures GLiNER for PII detection) |
| `replacement_mode` | `'placeholder'` | `'placeholder'`, `'synthetic'` (Faker), or `'reversible'` (indexed placeholders + deanonymize map) |
| `synthetic_seed` | `42` | Base seed for `replacement_mode='synthetic'` |
| `synthetic_per_document` | `True` | Seed each document's fakes from `synthetic_seed` + its text: fakes differ across documents, the same document always gets the same fakes (any worker count or order). `False` = 0.7: one Faker stream per thread, so batched documents could all get the same first fake |
| `positional_tags` | `('PER', 'LOC', 'ORG', 'MISC')` | Entity types to recognize |
| `ner_confidence_threshold` | `None` | Minimum confidence score (0..1). `None` = 0.85, or the GLiNER threshold in PII mode; an explicit value is always used as given |
| `ner_batch_size` | `8` | NER chunks per model forward pass, batched across the documents of a `process_batch` call (must be >= 1). `1` = 0.7.1 call pattern and exact scores, and less memory; which is faster depends on the CPU, so measure |
| `onnx_intra_op_threads` | `None` | ONNX Runtime threads per inference (`None` = ORT default: every physical core, ignoring CPU affinity). Set it to cap CPU use; fewer threads is usually slower |
| `onnx_inter_op_threads` | `None` | ONNX Runtime inter-op threads (`None` = ORT default). Only matters with ORT parallel execution mode, so `onnx_intra_op_threads` is the usual knob |
| `ner_device` | `None` | NER device: `None` = auto (CUDA when available), `'cpu'`, `'cuda'`. GLiClass moves to the GPU only with `'cuda'` |
| `cuda_device_id` | `0` | GPU index for ONNX, torch, GLiNER and GLiClass when on CUDA |
| `onnx_cuda_provider_options` | `None` | Extra `CUDAExecutionProvider` options, e.g. `{"gpu_mem_limit": 8 << 30}`; passed to ONNX Runtime unchecked |
| `ner_chunk_overlap` | `0` | Tokens of neighbouring text each NER window adds on both sides, for documents longer than one window (about 460 tokens with the default models). Finds entities that a chunk boundary would split (`John \| Smith`); costs about `2 * overlap / window` more inference and can change recall either way. `0` = 0.7.1 chunks. Must stay below a quarter of the window |
| `ner_models` | `None` | Language-keyed dict of ONNX model repo IDs |
| `torch_ner_models` | `None` | Language-keyed dict of PyTorch model repo IDs |
| `gliner_model` | `None` | GLiNER model ID (required for gliner/ensemble backends) |
| `gliner_variant` | `'gliner'` | `'gliner'` or `'gliner2'` |
| `gliner_labels` | `('person', 'organization', 'location')` | GLiNER entity labels |
| `gliner_label_map` | `None` | Maps GLiNER labels to NER tags |
| `gliner_threshold` | `None` | GLiNER confidence threshold (0..1). `None` = 0.4, or 0.3 in PII mode |
| `gliner_label_descriptions` | `None` | ZERONER-style: `{label: "description"}` for improved zero-shot accuracy |
| `json_key_cleaning` | `'pii'` | How `process_json` (and `sct-clean --json-input`, MCP `clean_json`, RAG chunk metadata) cleans dict keys: `'pii'` = regex PII steps only (no NER: models tag lone words such as `message` out of context), `'pii_ner'` = PII steps plus NER, `'full'` = the whole config (0.7), `'off'` = keys as written (can carry PII) |
| `fuzzy_date_score_cutoff` | `85` | Fuzzy matching threshold (0-100) for misspelled months |
| `custom_pipeline_steps` | `()` | Tuple of `(text: str) -> str` callables appended after all built-in steps |

**Language settings**:

| Field | Default | Description |
|-------|---------|-------------|
| `language` | `None` | Pin language (`'en'`, any of the 75 Lingua languages), restrict detection to a set (`('en','nl')`), or `None` for auto-detect. Accepts Lingua names, ISO 639-1, ISO 639-3 codes. |
| `detect_all_languages` | `True` | Auto-detect over all 75 Lingua languages. `False` = 0.7 detector over `supported_languages` only (EN/NL/DE/ES plus extras). A `language` tuple always restricts detection to its entries |
| `language_confidence_warning` | `0.5` | Adds `language_low_confidence` to `metadata['warnings']` when the detected language's confidence is below this (0..1). Warning only, never changes routing |
| `extra_languages` | `()` | Additional language names/codes for detection |
| `custom_stopwords` | `None` | `{LANG: frozenset({...})}` custom stopword sets |
| `custom_month_names` | `None` | `{LANG: ('Jan', 'Feb', ...)}` for date detection |

</details>

## Privacy Engine

Applications can apply versioned policy rules and inspect each decision.
This makes privacy behavior testable across services, datasets, and model upgrades.

```python
from sct import EntityRule, PrivacyPolicy, TextCleaner, TextCleanerConfig

policy = PrivacyPolicy(
    name="support-export",
    version="1",
    rules={
        "PERSON": EntityRule(action="redact", min_score=0.8),
        "ORGANISATION": EntityRule(action="keep"),
    },
    allowlist=frozenset({"Jane Public"}),
)

cleaner = TextCleaner(
    cfg=TextCleanerConfig(ner_mode="pii"),
    privacy_policy=policy,
    include_entities=True,
)
result = cleaner.process("Contact Maria Chen for help.")

print(result.metadata["findings"])
print(result.metadata["provenance"])
```

Finding offsets (`start`, `end`) and NER entries in the reversible map are
document-relative positions in the post-regex text: the text after the regex
stages replaced URLs, emails, phones and the like. They are not offsets into the
raw input.

Reversible output is pseudonymized personal data. Anyone with the token map can
restore the source values. Keep token maps outside logs and cleaned datasets.

Use a `TokenStore` implementation to keep maps behind opaque references:

```python
from sct import InMemoryTokenStore, TextCleaner, TextCleanerConfig

store = InMemoryTokenStore()
cleaner = TextCleaner(
    cfg=TextCleanerConfig(replacement_mode="reversible"),
    token_store=store,
)
result = cleaner.process("Maria lives in Utrecht.")
reference = result.metadata["anon_map_reference"]
```

`InMemoryTokenStore` supports tests and short sessions. Production systems
should implement `TokenStore` with encryption, access control, expiry, and audit logs.

### Boundary adapters

The base package includes JSON, DataFrame, retrieval, command-line, and streaming adapters:

```python
from sct import RAGChunk, TextCleaner, process_json, process_rag_chunks

cleaner = TextCleaner()
cleaned_json = process_json(cleaner, {"message": "Email a@example.com"})
# Keys get regex PII replacement only (json_key_cleaning='pii'); 'pii_ner', 'full' or 'off' to change.
cleaned_chunks = process_rag_chunks(
    cleaner,
    [RAGChunk(id="1", text="Call +1-202-555-0147", metadata={})],
)
```

```sh
printf 'Email maria@example.com' | sct-clean --no-ner
printf 'Email maria@example.com' | sct-clean --no-ner \
  --replacement-mode reversible --token-map tokens.json
pip install 'SqueakyCleanText[mcp]'
sct-mcp
```

`BufferedStreamingRedactor` holds a full document for cross-chunk NER safety.
`WindowedStreamingRedactor` provides bounded latency for contact tokens.

The OCR adapter accepts any `OCRProvider`. The included Tesseract provider also
needs a local Tesseract executable. Browser export writes an ONNX Runtime Web
manifest, but it does not copy model files or Python language detection.

Corpus owners can call `assess_reidentification_risk` to measure k-anonymity
for selected quasi-identifiers. Reports contain counts only, not source values.

### Framework integrations

```python
from datasets import Dataset
from sct.integrations.hf import clean_dataset

ds = Dataset.from_dict({"text": ["Mail jan@example.nl", "Call +31 20 123 4567"]})
ds = clean_dataset(ds, columns=["text"], num_proc=4)          # Hugging Face datasets
```

`sct.integrations.datatrove.SCTCleaner`, `sct.integrations.langchain.SCTDocumentTransformer`
and `sct.integrations.haystack.SCTDocumentCleaner` take the same options: `columns`,
`output="overwrite"` or `"new_column"` (`<col>_clean`), and `config` (default the `training`
preset; reversible mode is rejected). Document adapters write `metadata["sct"]`. Each needs
its extra. See [docs/INTEGRATIONS.md](docs/INTEGRATIONS.md); other engines (Ray, Spark,
Polars, DuckDB, dlt, Daft) in [docs/RECIPES.md](docs/RECIPES.md).

### Score columns and `sct-report`

```python
from sct import TextCleaner
from sct.score import score_record

cleaner = TextCleaner()
row = score_record(cleaner.process("Mail jan@example.nl"))   # sct_language, sct_pii_count, sct_pii_share, ...
```

```sh
sct-report data.jsonl --text-column text --preset training \
    --minhash --quasi-identifiers city,age_band --k 5 \
    --out report.json --card card.md --scores scores.jsonl
```

Score A (language, PII count, share and types, warnings) is always on. Score B is off by
default: a quality score on the original text and a PII-normalised MinHash of the cleaned
text (`[score]` extra). The report and data card hold counts, shares and hashes only.
See [docs/SCORE.md](docs/SCORE.md).

### MCP server

```sh
pip install 'SqueakyCleanText[mcp]'
SCT_MCP_PRESET=training SCT_MCP_ROOT=~/notes sct-mcp
```

Tools `clean_text` (`text` or `path`) and `clean_json` return `warnings`, a one-line
`summary`, the cleaned `text` / `value` and `signals`; `detail="full"` adds value-free
findings. `path` input is off unless `SCT_MCP_ROOT` is set (`SCT_MCP_MAX_BYTES`, default
1 MiB). `SCT_MCP_PRESET` (default `sct`) picks the config preset. See [docs/MCP.md](docs/MCP.md).

## PII Evaluation

Model changes need ground-truth privacy evaluation, not only backend agreement.
The evaluation harness reports exact entity metrics by label, language, domain,
and named slice.

```sh
python -m scripts.evaluate_pii \
  --dataset evaluation/pii_starter.jsonl \
  --predictions path/to/model-predictions.jsonl \
  --model model-name-and-revision \
  --output evaluation/results/model-name.json
```

The included starter corpus is synthetic and only checks the evaluation path.
See [`evaluation/README.md`](evaluation/README.md) for schemas and release rules.

Pinned candidates live in `sct.model_candidates`. Run one candidate with:

```sh
python -m scripts.benchmark_models \
  --dataset evaluation/pii_starter.jsonl \
  --candidate gliner2-pii \
  --output evaluation/results/gliner2-pii.json
```

The multilingual set `evaluation/multilingual_pii.jsonl` (220 synthetic records, en, nl, de,
es, zh, ja, th, ar, hi, bn, ta) scores the default ensemble at 0.87 recall / 0.92 precision
in 0.8.0, ahead of the GLiNER multilingual models; see
[docs/MULTILINGUAL_EVAL.md](docs/MULTILINGUAL_EVAL.md). The release benchmark
(`scripts/benchmark_release.py`, 30 documents in 9 languages) gives 0.98 exact-span F1 and
0 of 67 leaks. Throughput by hardware: [docs/PERFORMANCE.md](docs/PERFORMANCE.md).

Published third-party results are reference data only. `assess_candidate`
requires local recall, exact-span F1, latency, and ONNX parity before adoption.

## Architecture

SqueakyCleanText processes text through a configurable pipeline of sequential steps:

```
Input Text
  │
  ├─ Fix Unicode (ftfy)
  ├─ ASCII transliteration (unidecode)
  ├─ Emoji removal
  ├─ HTML replacement
  ├─ URL / Email / Phone replacement
  ├─ Date & Year replacement
  ├─ Number & Currency replacement
  ├─ Isolated letter/symbol removal
  ├─ Whitespace normalization
  │
  ├─ NER Processing (ONNX / Torch / GLiNER / Ensemble)
  │   ├─ Language detection (Lingua)
  │   ├─ Text chunking (token-bounded)
  │   ├─ Entity recognition (per-chunk)
  │   ├─ Ensemble voting (cross-model)
  │   └─ Entity anonymization (Presidio)
  │
  └─ Statistical Model Output
      ├─ Case folding
      ├─ Stopword removal
      └─ Punctuation removal

  ▼
(lm_text, stat_text, language)
```

Each step is toggled by a `TextCleanerConfig` field. The pipeline is built once at initialization; disabled steps are skipped entirely (zero overhead).

## What's New

**v0.8.0** (breaking changes, see [CHANGELOG.md](CHANGELOG.md))
- **Speed**: NER batched across the documents of `process_batch`; ONNX sessions run in parallel. 4.2x (short) and 3.2x (long documents) faster on 16 CPU cores; GPU device choice (`ner_device`, `cuda_device_id`).
- **All 75 Lingua languages** detected, with confidence and warnings; pin any language; bring your own model per language.
- **`training` preset**, `json_key_cleaning`, `check_decode_unicode_escapes`, `ner_chunk_overlap`.
- **`result.signals`**: value-free view for logging and data cards; provenance of every model run.
- **Integrations**: Hugging Face `datasets`, DataTrove, LangChain, Haystack; recipes for other engines.
- **Score columns, data cards, `sct-report`**, k-anonymity on named columns.
- **MCP server**: new output shape, `outputSchema`, annotations, file input under `SCT_MCP_ROOT`.
- **Multilingual PII eval** (11 languages) and a required Linux worker CI job (fork crash fixed).
- Python 3.12+ only.

**v0.7.0**
- **Privacy policies**: Apply versioned entity rules, allowlists, thresholds, and redaction actions.
- **Structured audit findings**: Inspect entity decisions, detector backends, and pinned model revisions.
- **Safer reversible processing**: Restore contact values and keep token maps behind an optional `TokenStore`.
- **Boundary adapters**: Clean JSON, DataFrame-like objects, RAG chunks, standard input, and MCP tool requests.
- **Privacy evaluation**: Score exact PII spans by label, language, domain, and custom slice.
- **Model adoption gates**: Compare pinned candidates using recall, F1, latency, and ONNX parity.
- **Extended media support**: Redact OCR regions, process bounded streams, and export browser deployment manifests.
- **Corpus risk indicators**: Measure k-anonymity for user-selected quasi-identifiers.
- **Supply-chain improvements**: Pin model revisions, publish through PyPI Trusted Publishing, and ship standard lockfiles.
- **Python 3.14**: Added CI coverage alongside Python 3.11 through 3.13.

Migration notes:
- `gliclass_onnx=True` now raises an error because the upstream package has no native ONNX loader.
- HTML processing now honors `replace_with_html` instead of silently stripping tags.
- Structured adapters reject reversible mode unless each token map can be retained safely.

**v0.6.0**
- **PII detection mode** (`ner_mode='pii'`): auto-configures GLiNER with 60+ PII entity labels (personal, financial, healthcare, identity, digital)
- **Synthetic replacement** (`replacement_mode='synthetic'`): Faker-generated realistic values instead of `<TAG>` placeholders, with per-document consistency
- **Reversible anonymization** (`replacement_mode='reversible'`): indexed placeholders (`<PERSON_0>`) with `AnonymizationMap` for round-trip deanonymization
- **Document classification** (`check_classify_document=True`): zero-shot GLiClass pre-classification before text processing
- **ProcessResult**: `process()` returns `ProcessResult` (backward-compatible 3-tuple) with `.metadata` for anonymization maps and classification results
- **GLiNER ONNX mode** (`gliner_onnx=True`): load GLiNER with pre-built ONNX weights from HuggingFace Hub (auto-set for PII + ONNX backend)
- **Bi-encoder support**: auto-detects ModernBERT and other bi-encoder GLiNER models, caches label embeddings, dynamic context windows (2048-8192 tokens)
- **Entity description labels**: ZERONER-style natural-language descriptions for improved zero-shot accuracy
- **Presidio GLiNER backend** (beta): opt-in `ner_backend='presidio_gliner'` for Presidio's context-aware recognition pipeline
- **ModernBERT ONNX export**: updated export script with ModernBERT support (English, 8192 token context)
- **Dynamic chunk sizing**: GLiNER chunk size adapts to model's actual context window instead of hardcoded 384

**v0.5.x**
- `aprocess_batch()`: async batch processing for FastAPI / aiohttp integrations
- `warmup(languages)`: pre-load NER models at startup to eliminate first-request latency
- `custom_pipeline_steps`: attach arbitrary `(text: str) -> str` callables after the built-in pipeline
- French, Portuguese, and Italian NER support via a shared multilingual ONNX session
- Improved NER sentence boundary detection with abbreviation guard

**v0.4.5**
- Frozen `TextCleanerConfig` dataclass: immutable, thread-safe, per-instance configuration
- ONNX-first NER inference: torch-free base install (~400 MB models vs ~7 GB)
- Thread-parallel batch processing via `ThreadPoolExecutor`
- Five NER backends: `onnx`, `torch`, `gliner`, `ensemble_onnx`, `ensemble_torch`
- GLiNER zero-shot NER for custom entity types (PRODUCT, EVENT, SKILL, etc.)
- Ensemble voting across backends for improved recall
- Lazy per-language model loading
- Multilingual date detection and fuzzy date matching
- Configurable emoji removal, bracket/brace content removal, and smart case folding
- `stop-words` replaces NLTK (50 KB bundled vs 30 MB download)
- PyTorch and Transformers moved to optional extras
- Migrated to `pyproject.toml` (PEP 517), Python 3.11-3.13, ruff linter

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an issue.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

The package took inspirations from the following repo:

- [clean-text](https://github.com/jfilter/clean-text)
