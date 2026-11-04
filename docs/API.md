# API

## `TextCleaner`

`TextCleaner(cfg=None, *, privacy_policy=None, token_store=None, include_entities=False, metrics_callback=None)`

### `process(text: str) -> ProcessResult`

Cleans one text. It runs `process_batch([text])` and returns its one result. A
`ProcessResult` unpacks as a 3-tuple `(lm_text, stat_text, language)`:
  - Cleaned text formatted for language models.
  - Cleaned text formatted for statistical models (`None` if `check_statistical_model_processing` is `False`).
  - Detected language of the text. `None` when it is unknown, or when no step needs it: `check_detect_language`, `check_ner_process` and `check_remove_stopwords` all `False` and no `language` set. NER and stopword removal run detection even with `check_detect_language=False`.

### `process_batch(texts: List[str], batch_size: Optional[int] = None) -> List[ProcessResult]`

Processes multiple texts using thread-parallel execution. Each result follows the same format as `process()`.
`batch_size` is the maximum number of worker threads (default `os.cpu_count()`; it must be >= 1).
Empty or whitespace-only input returns `ProcessResult("", "", None)` with the `empty_input` warning,
and a non-string input raises `ValueError`. With NER on and `ner_batch_size > 1`, the steps before
NER run per document in threads, then one NER call batches the chunks of all documents.

### `aprocess_batch(texts: List[str], batch_size: Optional[int] = None) -> List[ProcessResult]`

Async version of `process_batch` for use with asyncio-based frameworks (FastAPI, aiohttp). Runs the batch in a thread-pool executor so it does not block the event loop:

```python
from sct import TextCleaner

cleaner = TextCleaner()

# In an async context (FastAPI route, aiohttp handler, etc.)
async def handler(texts):
    return await cleaner.aprocess_batch(texts)
```

### `warmup(languages: Optional[List[str]] = None) -> None`

Pre-loads NER models to avoid first-request latency. Call once during application startup:

```python
from sct import TextCleaner

cleaner = TextCleaner()
cleaner.warmup(['ENGLISH', 'DUTCH'])  # or warmup() for all supported languages
```

## `ProcessResult`

> **Note**: `ProcessResult` from `process()` unpacks as a 3-tuple (`lm_text, stat_text, language`) for backward compatibility, but also exposes `.metadata` for reversible maps and document classification. `metadata` is always a dict with value-free counts: `metadata['regex_counts']` (e.g. `{"EMAIL": {"count": 1, "chars": 17}}`, one entry per regex replacement type) and `metadata['entity_counts']` (NER findings per type, same shape). It also holds `language_confidence` (Lingua top-1 confidence, `1.0` when pinned, `None` when unknown), `language_source` (`pinned`, `detected`, `fallback` when a `language` tuple fell back to its first entry, or `none`) and `warnings`, a sorted list of codes: `language_unknown`, `language_fallback`, `language_low_confidence`, `ner_fallback_models` (the language has no own NER model and ran MULTILINGUAL + ENGLISH), `cuda_fallback` (CUDA was requested but a model ran on CPU), `mps_fallback` (the Apple GPU was requested but a torch or GLiNER model ran on CPU), `custom_pattern_timeout` (a `custom_patterns` regex timed out and was skipped, `custom_patterns_on_timeout='warn'`), `empty_input`. `metadata['provenance']` is `{"backend", "models"}`, one `{"key", "id", "revision", "provider"}` entry per NER model that runs for the document's language (`revision` is the resolved Hub commit for ONNX models, `provider` the ONNX execution provider or the torch device); `{"backend": None, "models": []}` when NER is off.

A NER model that cannot load raises `sct.ModelLoadError`, on the first text that needs it.

`result.signals` is the value-free view of that metadata, with the same keys for every input:
`language`, `language_confidence`, `language_source`, `chars`, `pii_count`, `pii_chars`,
`pii_share`, `pii_types`, `regex_counts`, `entity_counts`, `sensitive`, `warnings` (adds
`ner_off`) and `models`. It never holds entity text, so it is safe to log or store next to
cleaned data. Score columns, `sct-report`, the adapters and the MCP server read it.

## Batch processing

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
