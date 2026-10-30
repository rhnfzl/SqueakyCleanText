# Changelog

## 0.8.0 - 2026-10-30

This is a speed release. NER model calls run in batches, also across the documents of one
`process_batch` call, and ONNX models no longer queue on a lock. On the release benchmark
(`evaluation/results/comparison-0.7.1-0.8.0.md`) `process_batch` on 16 Linux cores is 4.2x
faster on short documents (3.69 to 15.55 docs/s) and 3.2x faster on long documents (0.65 to
2.08 docs/s). Single-document `process` is unchanged within noise (0.95x to 1.10x).
Exact-span TP/FP/FN, recall (0.98) and leaks (0 of 67) are identical to 0.7.1, and 29 of 30
default-config outputs are byte-identical; the other one is the documented chunk-boundary
placeholder merge below.

### Behaviour changes

- Python 3.11 is no longer supported; 0.8.0 needs Python 3.12 or newer. pip on Python 3.11 keeps resolving to 0.7.1.
- `lingua-language-detector` minimum raised from 2.1.0 to 2.2. Lingua 2.2 can detect the language of some texts differently from 2.1.1, which changes the language-specific NER model, stopwords and date handling chosen for those texts.
- ONNX NER models no longer share one lock per model: threads in `process_batch` now run ONNX inference at the same time (`InferenceSession.run` is thread-safe). Output is unchanged. Torch models and GLiNER still run one call at a time. With the ONNX Runtime default threads, a session uses every physical core of the machine and ignores `taskset`/CPU affinity; set `onnx_intra_op_threads` to cap CPU use. On the benchmark, fewer threads per session was slower, not faster, because the batched NER pass runs a few large calls rather than one per worker.
- Two language keys that use the same model repo with different revisions now load each revision. Before, the second key silently reused the first key's revision. The INT8 quantized cache directory now includes the revision (`<model>@<digest>`, a lowercase hash of the revision, so revisions that differ only in case do not share a directory on case-insensitive filesystems), so a quantized file from one revision is not reused for another.
- `ner_batch_size` (default 8) now batches model calls: NER chunks of one document, and of all documents in one `process_batch` call, go through each ONNX, torch or GLiNER model `ner_batch_size` chunks per forward pass (rows right-padded). `process_batch` runs the steps before NER per document in threads, then one batched NER pass over all documents (ONNX batches run on the same number of threads), then the steps after NER. Output text, languages and finding spans are unchanged; padded batches can move scores in the last digits (max 5e-6 on the release and long-document sets with the default models). `ner_batch_size=1` keeps the 0.7.1 call pattern and scores exactly. GLiNER2 still runs one chunk per call.
- Throughput with the default models on a 16-core CPU (`scripts/benchmark_release.py --batch 64`, docs/s, `ner_batch_size` 1 vs 8): short documents 14.8 vs 14.9; long documents (about 8 joined documents each) 2.66 vs 2.38 with cpu-count workers, 1.50 vs 1.64 with one worker. With many workers on CPU, threads already fill the cores, so batching mainly adds padding and memory (peak RSS on the long set about 6 GB vs 12 GB). The release benchmark on another machine (Threadripper, 16 cores) measured the opposite: `ner_batch_size` 8 vs 1 was 15.55 vs 10.31 docs/s (short) and 2.08 vs 1.82 (long) with 16 workers. Which setting is faster depends on the CPU; measure before changing it. `ner_batch_size=1` always uses less memory.
- With `ner_batch_size > 1` and NER on, `metrics_callback` `duration_ms` includes the time a document waits for the shared NER pass of its batch.
- NER now collects the spans of all chunks of a document in document coordinates, votes once and replaces once per document, instead of voting and replacing chunk by chunk. Chunks and model input are unchanged. Differences, only for documents longer than one chunk:
  - `replacement_mode='reversible'`: placeholder numbers (`<PERSON_0>`...) are assigned right to left over the whole document, as for short documents, so indices can differ. Deanonymisation still round-trips.
  - `replacement_mode='synthetic'`: Faker values are drawn in whole-document order, so the fake values can differ.
  - Placeholder mode: two entities of the same type separated only by spaces at a chunk boundary now merge into one placeholder (`<PERSON>` instead of `<PERSON> <PERSON>`), as they already did inside a chunk. If any chunk has a label outside PER/LOC/ORG/MISC, the whole document uses the plain right-to-left replacement that short documents with such labels use.
  - Privacy-policy output and findings are unchanged.

### New options

- `onnx_intra_op_threads` and `onnx_inter_op_threads` (default `None`, the ONNX Runtime default, same as 0.7.1). Legacy module globals `ONNX_INTRA_OP_THREADS` / `ONNX_INTER_OP_THREADS`. Not applied to GLiNER's own ONNX session (`gliner_onnx=True`).
- `ner_chunk_overlap` (default `0`, legacy module global `NER_CHUNK_OVERLAP`): each NER window adds this many tokens of the neighbouring text on both sides of its chunk (GLiNER-only: 2 characters per token), and a span is kept from the window whose chunk holds its start; a neighbouring window's span is kept too when no span of the owning window that passes the confidence threshold (and, with a privacy policy, is replaced or redacted by it) covers it. A span seen by several windows is one vote per model, with its best score and that score's label, so repeated context (even with a different label) never counts as extra models in ensemble voting. A document that fits one window is never split, so overlap only affects documents that are chunked. This finds entities that a chunk boundary would split (sentence, whitespace or hard cut). The default `0` keeps the 0.7.1 chunks; overlap changes the context the model sees in long documents, so detections can change either way, and inference grows by about `2 * overlap / window`. An entity longer than the overlap that starts at the very end of a chunk can still be cut. Must satisfy `2 * overlap < window // 2`; this is checked again for each document, so a model loaded later with a smaller limit raises `ValueError` instead of running with a negative chunk budget.

### Fixes

- A warning is logged when CUDA was requested but ONNX Runtime fell back to CPU (missing CUDA/cuDNN libraries or a driver that does not match the onnxruntime-gpu build). Before, this happened silently. New `docs/PERFORMANCE.md` gives CPU and GPU throughput and CUDA setup notes.
- NER windows now fit every tokenizer of the models a document's language runs, not only the one with the smallest limit. Models loaded for other languages do not change a document's windows. With the default models, the multilingual model (WordPiece) can need far more tokens than XLM-RoBERTa for the same text and silently truncated the rest of a window, so it never saw entities past that point. Such texts now get more, smaller windows; on the release set 2 of 30 documents (and 2 of 4 joined long documents) are affected. Default-config output of the release set is unchanged, but these documents run more model calls (single-run Mac benchmark of the release set: 6.9 to 5.4 docs/s).
- Overlapping detections (for example `John Paul Smith` and `Paul` from two ensemble models) are merged into one span before replacement in reversible and synthetic mode and when a custom label is present. Before, right-to-left replacement left part of the outer entity visible (`<PERSON_1> Smith`). Placeholder mode with standard labels already merged them through Presidio. In the release benchmark, reversible output of `zh-glued-001` changes from the garbled `<LOCATION_1>OCATION_0>` to `<LOCATION_0>`.
- ONNX models exported with a fixed batch dimension of 1 run one row per call instead of failing under `ner_batch_size > 1`.
- `replacement_mode='synthetic'`: the regex steps and NER now share one consistency cache per document under `ner_batch_size > 1`, so the same value replaced by both gets the same fake value (as with `ner_batch_size=1`).
- The INT8 quantized cache directory name is collision-free: revisions are hashed (`release/v1` and `release__v1` no longer share a directory), and repo ids that contain `__` are %-quoted. Plain `org/name` ids keep the 0.7.1 `org__name` directory.
- ONNX models with external weights (the default XLM-RoBERTa-large models) load from a huggingface-hub cache that stores each blob in its own directory (`hub/blobs/<xx>/<sha256>`, seen with huggingface-hub 1.32.0). onnxruntime refused them with "External data path escapes model directory"; 0.7.1 fails the same way. In that layout SCT now hard-links `model.onnx` and `model.onnx_data` into `~/.cache/sct_linked/<model>@<commit hash>` (under `cache_dir` when set; one directory per snapshot, so a branch that moves back to an already-cached commit loads that commit's weights) and loads them from there; if hard links are not possible (another filesystem) the files are copied once (about 2.2 GB per model) and a warning is logged; later loads, and other processes starting at the same time, reuse the copy while its size matches and its source blob is not newer. Other cache layouts load in place as before.
- Chunk planning ignores fixed padding set in a model's `tokenizer.json`. Such a tokenizer counted every text as its padded length, so NER windows were split down to single characters and entities were missed (0.7.1 had this for the primary tokenizer; 0.8.0 had spread it to every ensemble tokenizer). None of the default models set padding.
- `scripts/benchmark_release.py` imports on Windows (no `resource` module); peak RSS is reported as `null` there.

### Known limits

- In ensembles with GLiNER, windows are sized for the token-classification models, not for GLiNER's own word limit.
- Only `process_batch` is faster. `process` of one document is not batched across documents.
- GLiNER2 runs one chunk per call; torch models and GLiNER keep one lock per model, so they do not run in parallel across threads.
- `ner_chunk_overlap` (off by default) can still cut an entity longer than the overlap that starts at the very end of a chunk.

## 0.7.1 - 2026-10-30

This release fixes PII that 0.7.0 left in the output. On the release benchmark
(`evaluation/results/comparison-0.7.0-0.7.1.md`) exact-span recall goes from 0.62 to 0.98
and leaked PII values from 6 of 67 to 0.

### Behaviour changes

- More PII is found and replaced: glued and international phone numbers, IP addresses, and GLiNER findings between 0.3 and 0.85 in PII mode. Output text changes for inputs that contain them.
- IP addresses are now replaced with `<IP>` by default (`<IP_ADDRESS_n>` in reversible mode), including private ones.
- New config flags: `check_replace_ips` (default `True`), `replace_ips_public_only` (default `False`) and `replace_with_ips` (default `"<IP>"`). They are the last `TextCleanerConfig` fields, so positional arguments keep their 0.7.0 meaning.
- The browser manifest rejects replacement tokens that contain digits, or that a later enabled rule would rewrite (for example an HTML token holding an email); the browser has no protection for emitted tokens.
- `process_json` finding paths for dict keys use `["@key", <index>]` instead of the key text.

### Fixes

- Phone and year rules now find digits glued to letters in any script (for example `电话9876543210`, `वर्ष2023में`).
- The number rule now finds numbers glued to letters in scripts written without spaces (`金额5000元`). Digits inside Latin, Greek and Cyrillic words stay as they are (`GPT-4o`, `Qwen3-8B`, `H2O`). Set `replace_numbers_in_words=True` to replace those too (`GPT4` becomes `GPT<NUMBER>`).
- IPv4 and IPv6 addresses (public and private) are replaced with `<IP>` before the date, phone and year steps. Set `replace_ips_public_only=True` to keep private, loopback and reserved addresses, or `check_replace_ips=False` to turn the step off.
- Phones now run before years, so `+31 6 1234 2019` and `2001:4860:4860::8888` are no longer split by the year rule. The browser manifest uses the same order.
- IPv6 is matched in every valid compressed form, including IPv4 tails (`::abcd:192.0.2.1`, `1:2:3:4:5::192.0.2.1`), and invalid forms are not matched, also in the browser rule.
- Replacement tokens and synthetic values (`<EMAIL1>`, a synthetic IP or phone) and IPs kept by `replace_ips_public_only` are no longer rewritten by the later phone, year and number steps.
- Phone detection covers more international formats (`+34 612 345 678`, `+91 98765 43210`) and Indian mobiles (`98765 43210`, `098765-43210`).
- Oversized NER chunks are split again, also long Chinese, Japanese and Hindi text without spaces: to the tokenizer's token budget for ONNX/torch models, and to a cap of `2 * max_tokens` characters in GLiNER-only mode (a character estimate, not a token count; token-budget chunking is planned for 0.8.0). Text is split after `。！？` and `। ॥`, and reversible map offsets for NER entities are now document-relative. Long whitespace runs at a chunk cut no longer break chunk alignment. Chunk planning ignores truncation set in a model's `tokenizer.json`, and a slice that grows when re-tokenized is split again, so text past the model limit is no longer left out of model input.
- Languages the `stop-words` package does not support (for example `THAI`) no longer crash `TextCleaner` creation.
- PII mode: the final `ner_confidence_threshold` now follows `gliner_threshold` (0.3 by default) instead of a hidden 0.85, so GLiNER findings between 0.3 and 0.85 are no longer dropped. An explicit `ner_confidence_threshold` other than 0.85 still wins (0.85 equals the default and cannot be told apart), and `0.0` is no longer turned into 0.85.
- `process_json` and MCP `clean_json` no longer return original dict keys in finding paths: a key's own findings use `["@key", <index>]`.
- `is_sensitive_result` compares placeholder counts, so `<EMAIL> alice@example.com` (HTML step off) is reported as sensitive. Marker-like tags removed by the HTML step (`<EMAIL> Alice` with `replace_with_html=""`) no longer cancel a new placeholder.
- `TextCleanerConfig` can be pickled and deep-copied, and the copy keeps the exact state. `dataclasses.replace(cfg)` no longer grows the default 5 NER models to 8 and 4 supported languages to 7. A partial `ner_models` mapping gets the same defaults whatever its mapping type.

## 0.7.0 - 2026-09-19

SqueakyCleanText now supports privacy workflows beyond single-text cleanup.
Applications can apply policies, inspect decisions, and use consistent adapters.

### New features

- Added versioned privacy policies, structured findings, detector provenance, and metrics callbacks.
- Added external token storage for reversible anonymization maps.
- Added JSON, DataFrame, RAG, command-line, and MCP adapters.
- Added exact-span PII evaluation with language, domain, label, and custom slice reports.
- Added pinned model candidates, benchmark tooling, and strict model adoption gates.
- Added buffered streaming, OCR image redaction, browser manifests, and corpus risk indicators.

### Improvements

- Pinned default ONNX, PyTorch, GLiNER, and GLiClass model revisions.
- Preserved source offsets across normalized NER chunks.
- Made `process_batch(batch_size=...)` control document concurrency.
- Added Python 3.14 CI and an 85 percent coverage gate.
- Added optional backend CI, standard lockfiles, and PyPI Trusted Publishing.
- Updated reversible mode to restore emails, phone numbers, URLs, dates, numbers, HTML, and currencies.

### Fixes

- HTML replacement now honors its configured replacement token.
- Presidio now receives the detected document language.
- Default configuration no longer emits a deprecation warning.
- MCP findings no longer return original entity values.
- Overlapping policy findings now redact their complete combined span.

### Migration notes

- `gliclass_onnx=True` now raises an error because upstream GLiClass has no native ONNX loader.
- Structured adapters reject reversible mode when they cannot return each token map safely.
- Reversible fuzzy date replacement remains unsupported and now fails during configuration.

## 0.6.1

- Improved GLiNER ONNX fallback behavior and test infrastructure.

## 0.6.0

- Added PII mode, synthetic replacement, reversible anonymization, and document classification.
