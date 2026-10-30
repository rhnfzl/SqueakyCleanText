# Changelog

## 0.7.1 - 2026-10-30

This release fixes PII that 0.7.0 left in the output. On the release benchmark
(`evaluation/results/comparison-0.7.0-0.7.1.md`) exact-span recall goes from 0.62 to 0.98
and leaked PII values from 6 of 67 to 0.

### Behaviour changes

- More PII is found and replaced: glued and international phone numbers, IP addresses, and GLiNER findings between 0.3 and 0.85 in PII mode. Output text changes for inputs that contain them.
- IP addresses are now replaced with `<IP>` by default (`<IP_ADDRESS_n>` in reversible mode), including private ones.
- New config flags: `check_replace_ips` (default `True`), `replace_ips_public_only` (default `False`) and `replace_with_ips` (default `"<IP>"`). They are the last `TextCleanerConfig` fields, so positional arguments keep their 0.7.0 meaning.
- The browser manifest rejects replacement tokens that contain digits (the browser has no protection for emitted tokens).
- `process_json` finding paths for dict keys use `["@key", <index>]` instead of the key text.

### Fixes

- Phone, number and year rules now find digits glued to letters in any script (for example `电话9876543210`, `वर्ष2023में`).
- IPv4 and IPv6 addresses (public and private) are replaced with `<IP>` before the date, phone and year steps. Set `replace_ips_public_only=True` to keep private, loopback and reserved addresses, or `check_replace_ips=False` to turn the step off.
- Phones now run before years, so `+31 6 1234 2019` and `2001:4860:4860::8888` are no longer split by the year rule. The browser manifest uses the same order.
- IPv6 is matched in every valid compressed form, including IPv4 tails (`::abcd:192.0.2.1`, `1:2:3:4:5::192.0.2.1`), and invalid forms are not matched, also in the browser rule.
- Replacement tokens and synthetic values (`<EMAIL1>`, a synthetic IP or phone) and IPs kept by `replace_ips_public_only` are no longer rewritten by the later phone, year and number steps.
- Phone detection covers more international formats (`+34 612 345 678`, `+91 98765 43210`) and Indian mobiles (`98765 43210`, `098765-43210`).
- NER chunks never exceed the model budget, also for long Chinese, Japanese and Hindi text without spaces. Text is split after `。！？` and `। ॥`, and reversible map offsets for NER entities are now document-relative. Long whitespace runs at a chunk cut no longer break chunk alignment.
- Languages the `stop-words` package does not support (for example `THAI`) no longer crash `TextCleaner` creation.
- PII mode: the final `ner_confidence_threshold` now follows `gliner_threshold` (0.3 by default) instead of a hidden 0.85, so GLiNER findings between 0.3 and 0.85 are no longer dropped. An explicit `ner_confidence_threshold` other than 0.85 still wins (0.85 equals the default and cannot be told apart), and `0.0` is no longer turned into 0.85.
- `process_json` and MCP `clean_json` no longer return original dict keys in finding paths: a key's own findings use `["@key", <index>]`.
- `is_sensitive_result` compares placeholder counts, so `<EMAIL> alice@example.com` (HTML step off) is reported as sensitive.
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
