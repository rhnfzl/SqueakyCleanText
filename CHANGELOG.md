# Changelog

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
