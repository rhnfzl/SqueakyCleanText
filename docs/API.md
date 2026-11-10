# API

## `TextCleaner`

`TextCleaner(cfg=None, *, privacy_policy=None, token_store=None, include_entities=False, metrics_callback=None, detector_registry=None, unified_policy=None)`

Pass a `TextCleanerConfig` to change behavior. Omitting `cfg` uses its defaults; module-level configuration globals are no longer supported.

### `process(text: str, *, context: DetectionContext | None = None) -> ProcessResult`

Cleans one text. It runs `process_batch([text])` and returns its one result. A
`ProcessResult` unpacks as a 3-tuple `(lm_text, stat_text, language)`:
  - Cleaned text formatted for language models.
  - Cleaned text formatted for statistical models (`None` if `check_statistical_model_processing` is `False`).
  - Detected language of the text. `None` when it is unknown, or when no step needs it: `check_detect_language`, `check_ner_process` and `check_remove_stopwords` all `False` and no `language` set. NER and stopword removal run detection even with `check_detect_language=False`.

### `process_batch(texts, max_workers=None, *, contexts=None) -> list[ProcessResult]`

Processes multiple texts using thread-parallel execution. Each result follows the same format as `process()`.
`max_workers` is the maximum number of worker threads (default `os.cpu_count()`; it must be >= 1).
Use `max_workers` for document concurrency; the deprecated `batch_size` keyword is removed.
Empty or whitespace-only input returns `ProcessResult("", "", None)` with the `empty_input` warning,
and a non-string item or a non-empty plain `str` for `texts` raises `ValueError`. With NER on and `ner_batch_size > 1`, the steps before
NER run per document in threads, then one NER call batches the chunks of all documents.

With unified processing, `contexts` contains one `DetectionContext` per input, in input order.
An explicit context requires unified processing. The configuration and each context are immutable.

### `aprocess_batch(texts, max_workers=None, *, contexts=None) -> list[ProcessResult]`

Async version of `process_batch` for use with asyncio-based frameworks (FastAPI, aiohttp). Runs the batch in a thread-pool executor so it does not block the event loop:

```python
from sct import TextCleaner

cleaner = TextCleaner()

# In an async context (FastAPI route, aiohttp handler, etc.)
async def handler(texts):
    return await cleaner.aprocess_batch(texts)
```

### `warmup(languages: Optional[List[str]] = None) -> None`

Pre-loads legacy NER models to reduce first-request latency. Call once during application startup:

```python
from sct import TextCleaner

cleaner = TextCleaner()
cleaner.warmup(['ENGLISH', 'DUTCH'])  # or warmup() for all supported languages
```

Unified processing rejects `warmup()` with `ValueError`.
Its detector providers belong to each worker, so warming one thread cannot initialize providers created by later batch threads.
Prepare and validate the selected model artifacts before starting the application.
Unified workers still initialize their providers on first use; this method does not remove that startup cost.
Legacy model load failures produce a warning and remain possible on first use.

## `ProcessResult`

> **Note**: `ProcessResult` from `process()` unpacks as a 3-tuple (`lm_text, stat_text, language`) for backward compatibility, but also exposes `.metadata` for reversible maps and document classification. `metadata` is always a dict with value-free counts: `metadata['regex_counts']` (e.g. `{"EMAIL": {"count": 1, "chars": 17}}`, one entry per regex replacement type) and `metadata['entity_counts']` (NER findings per type whose value was concealed, same shape; entities a privacy policy kept, allowed or ignored are not counted). It also holds `language_confidence` (Lingua top-1 confidence, `1.0` when pinned, `None` when unknown), `language_source` (`pinned`, `detected`, `fallback` when a `language` tuple fell back to its first entry, or `none`) and `warnings`, a sorted list of codes: `language_unknown`, `language_fallback`, `language_low_confidence`, `ner_fallback_models` (the language has no own NER model and ran MULTILINGUAL + ENGLISH), `cuda_fallback` (CUDA was requested but a model ran on CPU), `mps_fallback` (the Apple GPU was requested but a torch or GLiNER model ran on CPU), `custom_pattern_timeout` (a `custom_patterns` regex timed out and was skipped, `custom_patterns_on_timeout='warn'`), `empty_input`. `metadata['provenance']` is `{"backend", "models"}`, one `{"key", "id", "revision", "provider"}` entry per NER model that runs for the document's language (`revision` is the resolved Hub commit for ONNX models, `provider` the ONNX execution provider or the torch device); `{"backend": None, "models": []}` when NER is off.

A NER model that cannot load raises `sct.ModelLoadError`, on the first text that needs it. GLiNER and Presidio backends raise it when the NER object is built, not on the first text; a missing extra for them stays `ImportError`.

`result.signals` is the value-free view of that metadata, with the same keys for every input:
`language`, `language_confidence`, `language_source`, `chars`, `pii_count`, `pii_chars`,
`pii_share`, `pii_types`, `regex_counts`, `entity_counts`, `sensitive`, `warnings` (adds
`ner_off`) and `models`. `sensitive` is `pii_count > 0`, so a replaced date, year, number,
currency or HTML tag alone is not sensitive. It never holds entity text, so it is safe to log or store next to
cleaned data. Score columns, `sct-report`, the adapters and the MCP server read it.

These count and offset descriptions apply to legacy processing.
Unified processing adds versioned metadata and signals described below.
Caller-chosen identifiers and model names can still identify private work; restrict access when required.

## Unified findings and caller context

Users can apply one policy to format matches and model findings before any replacement.
Set `pii_processing='unified'` and pass `unified_policy=UnifiedPrivacyPolicy(...)` to select this path.
`PrivacyPolicy` remains the separate legacy NER policy; do not pass it to the unified path.

`DetectionContext(document_id=None, path=(), declared_entity_types=(), schema_id=None)` carries caller-owned context.
It does not infer field meanings. Direct processing forwards it to context-capable detectors.
`FieldSchema(id, version, rules)` and `FieldRule(path, entity_types, mode='hint')` select explicit structured fields.
`mode='protect_value'` protects the whole selected scalar; `hint` requires a context-capable detector.
The JSON, DataFrame and RAG APIs accept `schema=` and `context=`; see [adapters](ADAPTERS.md).

`Detection` and `findings_v2` contain canonical Unicode code-point positions, source ranges, detector identity and decisions.
Positions are zero-based and end-exclusive. Source ranges refer to the supplied string, never PDF bytes.
`mapping_quality` is `exact`, `covering` or `unavailable`.
`counts_v2` separates detector hits from resolved replacements.
`coverage_v2` reports coverage bounds; uncertain coverage leaves `pii_chars` and `pii_share` unset.
`provenance_v2` records configured detector and policy identities.
`include_entities=True` explicitly adds sensitive `findings_v2_values`.
Reversible processing returns `anon_map`, or only `anon_map_reference` when a token store is supplied.
The three-value result unpacking contract stays unchanged.

Register external detectors through `DetectorRegistry.register(name, factory)` and select them with `DetectorSpec(name, options)`.
Nonempty `detector_specs` replaces the complete default detector set.
Factories and credentials belong to trusted application code. Requests cannot supply executable factory imports.
See [configuration](CONFIG.md) for network permission, source mapping and experimental routing.

## Document, corpus and token-store APIs

| API | Result and boundary |
|---|---|
| `extract_document(path, provider=...)` | Validated `ExtractedDocument`; source files remain unchanged |
| `clean_document(document, cleaner)` | Cleaned text and structural elements; unified placeholder or synthetic output only |
| `analyse_records(..., cleaner=..., score_config=..., selection=..., scorer=None)` | Streaming analyses with private output references; scores receive original text |
| `select_records(analyses, selection=...)` | Context-managed report; close it to release owned temporary text and shingles |
| `curate_resumable(source, output, retention='caller_managed', ...)` | Private native checkpoints, checksum validation and replay; no exactly-once guarantee |
| `EncryptedSQLiteTokenStore(path, scope=..., key_provider=..., ttl_seconds=..., ...)` | Encrypted reversible maps behind opaque references; application-owned authorization and keys |

Corpus APIs live in `sct.corpus`; the encrypted store lives in `sct.token_stores`.
Document APIs are also exported from `sct`.
See [documents](DOCUMENTS.md), [corpus integrations](INTEGRATIONS.md) and [token-store operations](TOKEN_STORES.md) for their complete restrictions.

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
results = cleaner.process_batch(texts, max_workers=2)

for lm_text, stat_text, lang in results:
    print(f"Language: {lang}")
    print(f"LM Format:    {lm_text}")
    print(f"Stat Format:  {stat_text}")
    print("-" * 40)
```
