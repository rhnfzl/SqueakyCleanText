# SCT adapters Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Execution method requires human selection; this document starts no workers.

**Goal:** Implement the adapters workstream with explicit compatibility and acceptance evidence.

**Architecture:** Reuse SCT processing, scoring and replacement mechanisms. Add opt-in contracts and narrow integrations around those mechanisms. Read the shared specification before implementing a task.

**Tech Stack:** Python 3.12+, SCT, pytest, Ruff, mypy, uv; optional provider versions require P0 qualification.

**Spec:** [Shared design specification](../../references/2026-11-07-sct-improvements-design.md)

## Global Constraints

- Baseline v0.8.3 / 8afef61c74ea738a2c904374776dea34d9c1c39e; re-audit after rebasing.
- Python >=3.12; test 3.12, 3.13 and 3.14. No new mandatory core dependencies.
- Existing default behavior, imports, tuple results and supported modes remain compatible.
- pii_processing defaults to legacy. Unified processing and adaptive routing require explicit opt-in.
- No hidden model downloads, raw values in default traces, built-in keyword exceptions or automatic input deletion.
- All offsets use zero-based, end-exclusive Unicode code points with an explicit text stage.
- Branch/worktree changes only. Each task is a separately reviewable change; never publish from this plan.
- No em dashes or AI coauthor trailers. Record deviations and changed interfaces before downstream work.

## Review Focus

- Unicode, normalization and opaque callbacks: exact, covering and unavailable mappings must remain distinct (C1/C4).
- Worker restart and mixed structured paths: context remains aligned and runtime providers remain process-owned (C3/C5).
- Interrupted corpus jobs and changed inputs: no silent reuse or missing output records (D3).
- Wrong scope, expired keys and tampered storage: fail without exposing reversible values (S1/S2).
- Combined features and error paths: default artifacts contain no raw sensitive values and legacy goldens pass (R1).

---

### E1: Add offline structure-preserving Docling extraction and cleaning

**Dependencies:** C1, C4, C5, P0. **Feature IDs:** 7.

**Files and responsibilities:**
- Create: `sct/documents.py`
- Create: `sct/integrations/docling.py`
- Create: `tests/test_docling_integration.py`
- Create: `tests/fixtures/documents/pii-layout.pdf`
- Create: `docs/DOCUMENTS.md`
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Modify: `pylock.toml`
- Modify: `docs/INTEGRATIONS.md`
- Test: `tests/test_docling_integration.py`

**Interfaces consumed and produced:**
- Produces DocumentExtractor, ExtractedDocument, ProcessedDocument; implements extract_document and clean_document.
- Consumes SourceMap and unified findings to map canonical extracted text back to element/page IDs.
Exact DTO fields, defaults and callable signatures are fixed in the shared spec. Test snippets below are assertion targets; construct fixtures in the named test files.

- [ ] **Step 1: Add failing tests in the listed test files.**

Name the primary test `test_e1_acceptance`; add separate focused tests for the distinct checks below.

- A synthetic multi-page fixture includes cross-element names, table values, OCR text and metadata PII. Detection uses joined mapped text, not blind per-line isolated calls.
- Only cleaned output text/elements and allowlisted structural provenance survive; no raw Docling objects, source text or original-element metadata is retained in output.
- Missing offline artifacts cause a clear error; no network/model download occurs unless explicitly enabled.
- Run real Docling conversion in the optional CI profile as well as fake-provider unit tests.
- Never claim original PDF/images/attachments are redacted; coordinates describe extracted text and layout provenance.

```python
assert secret not in processed_text_or_metadata
assert processed.element_ids == extracted.element_ids
assert no_network_calls_in_offline_mode
```

- [ ] **Step 2: Verify the missing behavior.**

Run `.venv/bin/python -m pytest tests/test_docling_integration.py -q`. New feature tests must fail because the specified API or behavior is missing. Dependency or fixture errors are setup failures, not evidence. P0 instead captures independent legacy expectations and checks that they pass before implementation.

- [ ] **Step 3: Implement the interfaces in the listed production files.**

Use the documented Docling 2.148.0 adapter contract validated in P0. Preserve element IDs and mapped separators; use SCT existing long-document windows. Unstructured is a later optional adapter, not a prerequisite.
Update listed callers, exports, documentation and optional dependency configuration in the same change. Implement only this task's contracts; preserve the defaults in Global Constraints.

- [ ] **Step 4: Verify the task and compatibility.**

Run `.venv/bin/python -m pytest tests/test_docling_integration.py -q` and `.venv/bin/python -m pytest tests/test_processing_compatibility.py -q`. Expected: all pass. Run touched-file Ruff and applicable project CI checks before committing. Real-provider tasks also need installed-profile smoke evidence, not mocks alone.

- [ ] **Step 5: Commit the reviewed task on its feature branch.**

```bash
git add sct/documents.py sct/integrations/docling.py tests/test_docling_integration.py tests/fixtures/documents/pii-layout.pdf docs/DOCUMENTS.md pyproject.toml uv.lock pylock.toml docs/INTEGRATIONS.md
git commit -m "feat: add offline structure-preserving docling extraction and cleaning"
```

Run the full matching CI checks before commit/push. Review the staged diff, omit unrelated edits, and open a separate PR for this task. Commit paths include new files only once created; do not stage another worker's changes.

### E2: Add an optional format-aware credential detector

**Dependencies:** C3, C4, P0. **Feature IDs:** 9.

**Files and responsibilities:**
- Create: `sct/detectors_datafog.py`
- Create: `tests/test_secret_detector.py`
- Create: `docs/SECRET_DETECTION.md`
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Modify: `pylock.toml`
- Modify: `docs/NER_BACKENDS.md`
- Test: `tests/test_secret_detector.py`

**Interfaces consumed and produced:**
- Produces DataFogSecretDetector implementing Detector with explicit rule-based capabilities.
- Calls datafog-core 1.0.0 scan(text, {"entities": [...]}) for the four declared credential categories; no ML session required.
Exact DTO fields, defaults and callable signatures are fixed in the shared spec. Test snippets below are assertion targets; construct fixtures in the named test files.

- [ ] **Step 1: Add failing tests in the listed test files.**

Name the primary test `test_e2_acceptance`; add separate focused tests for the distinct checks below.

- Synthetic API_KEY, BEARER_TOKEN, JWT and PRIVATE_KEY examples return correct Unicode code-point spans, including emoji prefixes.
- Vendor byte/UTF-16 ranges are not misread as Python positions; malformed ranges fail validation.
- Unknown selected types, missing extra and scanner errors fail explicitly without a successful regex-only disguise.
- Raw matched_text and vendor result objects do not reach public findings/logs. Near-miss code literals and false positives are evaluated.
- Existing GLiNER password/api_key/access_token labels remain usable; do not describe secret recognition as previously absent.

```python
assert detected.entity_type == "JWT"
assert input_text[detected.start:detected.end] == synthetic_token
assert synthetic_token not in public_metadata
```

- [ ] **Step 2: Verify the missing behavior.**

Run `.venv/bin/python -m pytest tests/test_secret_detector.py -q`. New feature tests must fail because the specified API or behavior is missing. Dependency or fixture errors are setup failures, not evidence. P0 instead captures independent legacy expectations and checks that they pass before implementation.

- [ ] **Step 3: Implement the interfaces in the listed production files.**

Convert rule-based findings into the canonical interface and drop raw values. Register explicitly through an application-owned factory. Do not load optional models or silently expand detector scope.
Update listed callers, exports, documentation and optional dependency configuration in the same change. Implement only this task's contracts; preserve the defaults in Global Constraints.

- [ ] **Step 4: Verify the task and compatibility.**

Run `.venv/bin/python -m pytest tests/test_secret_detector.py -q` and `.venv/bin/python -m pytest tests/test_processing_compatibility.py -q`. Expected: all pass. Run touched-file Ruff and applicable project CI checks before committing. Real-provider tasks also need installed-profile smoke evidence, not mocks alone.

- [ ] **Step 5: Commit the reviewed task on its feature branch.**

```bash
git add sct/detectors_datafog.py tests/test_secret_detector.py docs/SECRET_DETECTION.md pyproject.toml uv.lock pylock.toml docs/NER_BACKENDS.md
git commit -m "feat: add an optional format-aware credential detector"
```

Run the full matching CI checks before commit/push. Review the staged diff, omit unrelated edits, and open a separate PR for this task. Commit paths include new files only once created; do not stage another worker's changes.
