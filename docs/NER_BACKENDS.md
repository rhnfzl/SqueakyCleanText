# NER backends and models

## How the default runs

- The default backend is `onnx`: ONNX Runtime with no PyTorch or Transformers install
  (about 3-5x faster than PyTorch).
- Each language runs an ensemble of models (`ner_ensemble`, default below) and the
  ensemble combines their spans and averages their scores. English runs the English and multilingual models; Dutch,
  German and Spanish run their own model plus the English and multilingual models. Every
  other language runs the multilingual and English models (`ner_ensemble_default_keys`)
  and gets the `ner_fallback_models` warning, unless you bring a model for it with
  `ner_models`. That dict must name `ENGLISH` and `MULTILINGUAL` too, so start from the
  defaults: `ner_models={**DEFAULT_NER_MODELS, "HINDI": "..."}` (`from sct.config import DEFAULT_NER_MODELS`).
- The English and multilingual models load when `TextCleaner` is built. Other language models
  load on first use. `warmup()` loads them up front.
- Languages that point at the same model share one ONNX session (French, Portuguese
  and Italian use the multilingual model, which saves about 600 MB of RAM).
- Long documents are cut into token-bounded chunks (CJK and Arabic safe);
  `ner_chunk_overlap` adds overlapping context to each window.
- `ner_confidence_threshold` drops low-score entities; a privacy policy can keep, redact
  or raise the minimum score per entity type ([PRIVACY.md](PRIVACY.md)).
- CUDA works for the ONNX and PyTorch backends (`ner_device`, `cuda_device_id`; see
  [PERFORMANCE.md](PERFORMANCE.md)). The Apple GPU (`ner_device='mps'`, opt-in) works for the
  PyTorch backend and GLiNER in torch mode; ONNX stays on CPU.

## Backends

SqueakyCleanText supports six NER backends, selectable via the `ner_backend` config field:

| Backend | Description | Dependencies | Best for |
|---------|-------------|-------------|----------|
| `onnx` (default) | ONNX Runtime inference with quantized XLM-RoBERTa models | Base install | Production: fast, torch-free |
| `torch` | PyTorch/Transformers pipeline with full XLM-RoBERTa models | `[torch]` extra | Compatibility with existing PyTorch workflows |
| `gliner` | GLiNER zero-shot NER with custom entity labels | `[gliner]` or `[gliner2]` extra | Custom entity types, PII detection, bi-encoder models |
| `ensemble_onnx` | ONNX + GLiNER ensemble voting | `[gliner]` extra | Maximum recall with custom entities |
| `ensemble_torch` | Torch + GLiNER ensemble voting | `[torch,gliner]` extra | Maximum recall with PyTorch |
| `presidio_gliner` | Presidio + GLiNER recognizer (beta) | `[presidio]` extra | Context-aware NER via Presidio's pipeline |

## Default NER Models (ONNX)

| Language | Model |
|----------|-------|
| English | [`rhnfzl/xlm-roberta-large-conll03-english-onnx`](https://huggingface.co/rhnfzl/xlm-roberta-large-conll03-english-onnx) |
| Dutch | [`rhnfzl/xlm-roberta-large-conll02-dutch-onnx`](https://huggingface.co/rhnfzl/xlm-roberta-large-conll02-dutch-onnx) |
| German | [`rhnfzl/xlm-roberta-large-conll03-german-onnx`](https://huggingface.co/rhnfzl/xlm-roberta-large-conll03-german-onnx) |
| Spanish | [`rhnfzl/xlm-roberta-large-conll02-spanish-onnx`](https://huggingface.co/rhnfzl/xlm-roberta-large-conll02-spanish-onnx) |
| French / Portuguese / Italian | [`rhnfzl/wikineural-multilingual-ner-onnx`](https://huggingface.co/rhnfzl/wikineural-multilingual-ner-onnx) (shared session) |
| Multilingual (fallback) | [`rhnfzl/wikineural-multilingual-ner-onnx`](https://huggingface.co/rhnfzl/wikineural-multilingual-ner-onnx) |

## GLiNER Model Recommendations

| Model | Architecture | Context | Languages | Best for |
|-------|-------------|---------|-----------|----------|
| `knowledgator/gliner-bi-base-v2.0` | Bi-encoder (ModernBERT) | 2048 | Multi | General NER, long documents |
| `knowledgator/gliner-pii-base-v1.0` | Uni-encoder (DeBERTa) | 2048 | Multi | PII detection (60+ entity types) |
| `urchade/gliner_large-v2.1` | Uni-encoder (DeBERTa) | 512 | Multi | Legacy, high accuracy on short texts |
| `MatteoFasulo/ModernBERT-base-NER` | ModernBERT | 8192 | English | English-only, very long context |

> **GLiNER2 note**: `pip install squeakycleantext[gliner2]` installs the PyPI `gliner2` package, [Fastino AI's GLiNER2](https://github.com/fastino-ai/GLiNER2). Its API differs from Knowledgator's `gliner` package; SCT wraps both. GLiNER2 has no ONNX loader and always runs in torch.

## GLiNER Label Mapping

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

With `gliner_label_map=None`, the ensemble backends map `person`, `organization` and `location` to `PER`, `ORG` and `LOC`, so GLiNER's spans reach the vote and become `<PERSON>`, `<ORGANISATION>` and `<LOCATION>`. Your own map replaces this one entirely. The plain `gliner` backend has no default map, so those labels become `<PERSON>`, `<ORGANIZATION>` and `<LOCATION>`.

## GLiNER: Zero-Shot Custom NER

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

## Ensemble NER

Combine ONNX/Torch models with GLiNER for improved recall via ensemble voting:

```python
from sct import TextCleaner, TextCleanerConfig

cfg = TextCleanerConfig(
    ner_backend='ensemble_onnx',  # or 'ensemble_torch'
    gliner_model='urchade/gliner_large-v2.1',
    gliner_labels=('person', 'organization', 'location'),
)

cleaner = TextCleaner(cfg=cfg)
lm_text, stat_text, lang = cleaner.process("Angela Merkel visited the Bundestag in Berlin.")
```

## PII Detection Mode

Automatically configure GLiNER for PII detection with 58 PII labels (personal, financial, healthcare, identity, digital):

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

`gliner_onnx=True` loads a GLiNER model from its pre-built ONNX weights on the Hub. It works only with uni-encoder models. PII mode sets it for you when `ner_backend` is left at `'onnx'`.

**Alternative PII models** (pass as `gliner_model`):

| Model | Type | Size | Labels | F1 |
|-------|------|------|--------|-----|
| [`knowledgator/gliner-pii-base-v1.0`](https://huggingface.co/knowledgator/gliner-pii-base-v1.0) | Uni-encoder | 330MB (ONNX FP16) | 60+ | 80.99% |
| [`nvidia/gliner-PII`](https://huggingface.co/nvidia/gliner-PII) | Bi-encoder | 570MB | 55+ | — |
| [`gretelai/gretel-gliner-bi-base-v1.0`](https://huggingface.co/gretelai/gretel-gliner-bi-base-v1.0) | Bi-encoder | ~800MB | 40+ | 95% |
| [`urchade/gliner_multi_pii-v1`](https://huggingface.co/urchade/gliner_multi_pii-v1) | Multilingual | — | — | — |


## Bi-Encoder GLiNER Models

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

## Entity Description Labels (ZERONER-Style)

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
