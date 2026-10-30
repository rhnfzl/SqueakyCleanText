# Changelog

## 0.7.1 - Unreleased

### Fixes

- Phone, number and year rules now find digits glued to letters in any script (for example `电话9876543210`, `वर्ष2023में`).
- IPv4 and IPv6 addresses (public and private) are replaced with `<IP>` before the date, phone and year steps. Set `replace_ips_public_only=True` to keep private, loopback and reserved addresses, or `check_replace_ips=False` to turn the step off.
- Phones now run before years, so `+31 6 1234 2019` and `2001:4860:4860::8888` are no longer split by the year rule. The browser manifest uses the same order.
- Phone detection covers more international formats (`+34 612 345 678`, `+91 98765 43210`) and Indian mobiles (`98765 43210`, `098765-43210`).
- Languages the `stop-words` package does not support (for example `THAI`) no longer crash `TextCleaner` creation.

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
