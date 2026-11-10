# SCT core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Execution method requires human selection; this document starts no workers.

**Goal:** Implement the core workstream with explicit compatibility and acceptance evidence.

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

### C1: Add canonical findings and honest coordinate maps

**Dependencies:** P0. **Feature IDs:** 1.

**Files and responsibilities:**
- Create: `sct/findings.py`
- Create: `sct/coordinates.py`
- Create: `tests/test_coordinates.py`
- Create: `tests/test_detections.py`
- Test: `tests/test_coordinates.py`
- Test: `tests/test_detections.py`

**Interfaces consumed and produced:**
- Produces Detection, DetectionContext, TextRange, Projection and SourceMap from the common specification.
- No public cleaner behavior changes in this PR.
Exact DTO fields, defaults and callable signatures are fixed in the shared spec. Test snippets below are assertion targets; construct fixtures in the named test files.

- [ ] **Step 1: Add failing tests in the listed test files.**

Name the primary test `test_c1_acceptance`; add separate focused tests for the distinct checks below.

- Reject reversed/out-of-bounds ranges, NaN/out-of-range scores and mixed coordinate spaces. None confidence remains None.
- Compose identity, insertion, deletion, many-to-one and one-to-many edits; classify exact/covering/unavailable correctly.
- For original "👋 A&amp;B" and decoded text "👋 A&B", B maps from [4,5) to [8,9); & maps conservatively to [3,8).
- An opaque reverse-string callback cannot produce claimed exact raw offsets. Findings/repr/JSON contain no matched values.

```python
assert projected_b.ranges == (TextRange(8, 9),)
assert projected_amp.mapping_quality == "covering"
assert opaque_projection.mapping_quality == "unavailable"
```

- [ ] **Step 2: Verify the missing behavior.**

Run `.venv/bin/python -m pytest tests/test_coordinates.py tests/test_detections.py -q`. New feature tests must fail because the specified API or behavior is missing. Dependency or fixture errors are setup failures, not evidence. P0 instead captures independent legacy expectations and checks that they pass before implementation.

- [ ] **Step 3: Implement the interfaces in the listed production files.**

Use immutable DTOs and explicit edit composition. Keep source text transient and out of default serializers. Never infer exact provenance from a generic diff after an arbitrary callable.
Update listed callers, exports, documentation and optional dependency configuration in the same change. Implement only this task's contracts; preserve the defaults in Global Constraints.

- [ ] **Step 4: Verify the task and compatibility.**

Run `.venv/bin/python -m pytest tests/test_coordinates.py tests/test_detections.py -q` and `.venv/bin/python -m pytest tests/test_processing_compatibility.py -q`. Expected: all pass. Run touched-file Ruff and applicable project CI checks before committing. Real-provider tasks also need installed-profile smoke evidence, not mocks alone.

- [ ] **Step 5: Commit the reviewed task on its feature branch.**

```bash
git add sct/findings.py sct/coordinates.py tests/test_coordinates.py tests/test_detections.py
git commit -m "feat: add canonical findings and honest coordinate maps"
```

Run the full matching CI checks before commit/push. Review the staged diff, omit unrelated edits, and open a separate PR for this task. Commit paths include new files only once created; do not stage another worker's changes.

### C2: Expose native regex and NER span analysis without replacement

**Dependencies:** C1. **Feature IDs:** 1, 2.

**Files and responsibilities:**
- Create: `sct/detectors_builtin.py`
- Create: `tests/test_builtin_detections.py`
- Modify: `sct/utils/contact.py`
- Modify: `sct/utils/datetime.py`
- Modify: `sct/utils/ner.py`
- Modify: `sct/utils/ner_spans.py`
- Modify: `sct/utils/normtext.py`
- Test: `tests/test_builtin_detections.py`
- Test: `tests/test_ner_windows.py`
- Test: `tests/test_ner_spans.py`

**Interfaces consumed and produced:**
- Produces BuiltinRegexDetector and BuiltinNERDetector implementing Detector.
- Add GeneralNER.analyze(text: str, *, language: str | None, context: DetectionContext) -> tuple[Detection, ...]; legacy rendering reuses extracted span finalization.
Exact DTO fields, defaults and callable signatures are fixed in the shared spec. Test snippets below are assertion targets; construct fixtures in the named test files.

- [ ] **Step 1: Add failing tests in the listed test files.**

Name the primary test `test_c2_acceptance`; add separate focused tests for the distinct checks below.

- Regex spans match the existing contact/date/number/custom-pattern behavior and validations. Do not copy independent regex tables.
- All six existing NER backends return canonical findings; model-window merges, votes and offsets remain stable.
- Use fake models for unit tests and one pinned real model per applicable optional profile for smoke tests.
- Legacy process outputs and include_entities metadata match P0 fixtures; no extra model call appears in legacy mode.

```python
assert regex_adapter.detect(text, language=None, context=ctx)[0].entity_type == "EMAIL"
assert text == original_text
assert legacy_cleaner.process(text) == golden_result
```

- [ ] **Step 2: Verify the missing behavior.**

Run `.venv/bin/python -m pytest tests/test_builtin_detections.py tests/test_ner_windows.py tests/test_ner_spans.py -q`. New feature tests must fail because the specified API or behavior is missing. Dependency or fixture errors are setup failures, not evidence. P0 instead captures independent legacy expectations and checks that they pass before implementation.

- [ ] **Step 3: Implement the interfaces in the listed production files.**

Separate native span discovery from rendering with small helpers in ner_spans.py. Preserve current validation, timeout, source-marker and NER voting rules.
Update listed callers, exports, documentation and optional dependency configuration in the same change. Implement only this task's contracts; preserve the defaults in Global Constraints.

- [ ] **Step 4: Verify the task and compatibility.**

Run `.venv/bin/python -m pytest tests/test_builtin_detections.py tests/test_ner_windows.py tests/test_ner_spans.py -q` and `.venv/bin/python -m pytest tests/test_processing_compatibility.py -q`. Expected: all pass. Run touched-file Ruff and applicable project CI checks before committing. Real-provider tasks also need installed-profile smoke evidence, not mocks alone.

- [ ] **Step 5: Commit the reviewed task on its feature branch.**

```bash
git add sct/detectors_builtin.py tests/test_builtin_detections.py sct/utils/contact.py sct/utils/datetime.py sct/utils/ner.py sct/utils/ner_spans.py sct/utils/normtext.py tests/test_ner_windows.py tests/test_ner_spans.py
git commit -m "feat: expose native regex and ner span analysis without replacement"
```

Run the full matching CI checks before commit/push. Review the staged diff, omit unrelated edits, and open a separate PR for this task. Commit paths include new files only once created; do not stage another worker's changes.

### C3: Add explicit detector registration and worker-owned factories

**Dependencies:** C2. **Feature IDs:** 2.

**Files and responsibilities:**
- Create: `sct/detectors.py`
- Create: `tests/test_detector_registry.py`
- Modify: `sct/config.py`
- Modify: `sct/sct.py`
- Modify: `sct/integrations/_runtime.py`
- Modify: `sct/__init__.py`
- Test: `tests/test_detector_registry.py`
- Test: `tests/test_integrations_runtime.py`
- Test: `tests/test_cleaner_lifecycle.py`

**Interfaces consumed and produced:**
- Produces DetectorRegistry, DetectorSpec, DetectorCapabilities and DetectorUnavailable.
- Consumes BuiltinRegexDetector/BuiltinNERDetector; supplies create(spec,cfg) and trusted registry_factory(cfg) for workers.
Exact DTO fields, defaults and callable signatures are fixed in the shared spec. Test snippets below are assertion targets; construct fixtures in the named test files.

- [ ] **Step 1: Add failing tests in the listed test files.**

Name the primary test `test_c3_acceptance`; add separate focused tests for the distinct checks below.

- Duplicate/unknown IDs and invalid data-only options fail before processing; untrusted config does not trigger imports.
- Registry freezes at cleaner construction; factories instantiate inside each process, never pickle sessions/keys.
- Threaded batches have stable order and no leaked per-document context. Remote/network-required detectors fail unless explicitly allowed.
- Round-trip DetectorSpec config through supported config serialization and legacy-global conventions.

```python
registry.register("example", factory)
with raises(ValueError): registry.register("example", factory)
assert worker_result.session_pid == worker_pid
```

- [ ] **Step 2: Verify the missing behavior.**

Run `.venv/bin/python -m pytest tests/test_detector_registry.py tests/test_integrations_runtime.py tests/test_cleaner_lifecycle.py -q`. New feature tests must fail because the specified API or behavior is missing. Dependency or fixture errors are setup failures, not evidence. P0 instead captures independent legacy expectations and checks that they pass before implementation.

- [ ] **Step 3: Implement the interfaces in the listed production files.**

Make registration application-owned. Built-in factories remain lazy; optional package imports occur only when selected. Add kwargs without replacing existing constructor arguments.
Update listed callers, exports, documentation and optional dependency configuration in the same change. Implement only this task's contracts; preserve the defaults in Global Constraints.

- [ ] **Step 4: Verify the task and compatibility.**

Run `.venv/bin/python -m pytest tests/test_detector_registry.py tests/test_integrations_runtime.py tests/test_cleaner_lifecycle.py -q` and `.venv/bin/python -m pytest tests/test_processing_compatibility.py -q`. Expected: all pass. Run touched-file Ruff and applicable project CI checks before committing. Real-provider tasks also need installed-profile smoke evidence, not mocks alone.

- [ ] **Step 5: Commit the reviewed task on its feature branch.**

```bash
git add sct/detectors.py tests/test_detector_registry.py sct/config.py sct/sct.py sct/integrations/_runtime.py sct/__init__.py tests/test_integrations_runtime.py tests/test_cleaner_lifecycle.py
git commit -m "feat: add explicit detector registration and worker-owned factories"
```

Run the full matching CI checks before commit/push. Review the staged diff, omit unrelated edits, and open a separate PR for this task. Commit paths include new files only once created; do not stage another worker's changes.

### C4: Apply one unified policy and renderer across detector families

**Dependencies:** C3. **Feature IDs:** 1.

**Files and responsibilities:**
- Create: `sct/policy_v2.py`
- Create: `sct/unified.py`
- Create: `tests/test_unified_processing.py`
- Modify: `sct/sct.py`
- Modify: `sct/config.py`
- Modify: `sct/privacy.py`
- Modify: `sct/utils/process_result.py`
- Modify: `sct/utils/anonymization_map.py`
- Modify: `sct/streaming.py`
- Modify: `sct/browser.py`
- Modify: `sct/__init__.py`
- Test: `tests/test_unified_processing.py`
- Test: `tests/test_privacy.py`
- Test: `tests/test_processing_compatibility.py`

**Interfaces consumed and produced:**
- Produces UnifiedPrivacyPolicy and HandlingRule; adds opt-in pii_processing and source_mapping config.
- process/process_batch use the shared unified path; metadata.findings_v2 is value-free. Existing PrivacyPolicy and post-regex metadata remain unchanged.
Exact DTO fields, defaults and callable signatures are fixed in the shared spec. Test snippets below are assertion targets; construct fixtures in the named test files.

- [ ] **Step 1: Add failing tests in the listed test files.**

Name the primary test `test_c4_acceptance`; add separate focused tests for the distinct checks below.

- Mixed EMAIL regex and PERSON NER findings receive the same per-type policy in all three replacement modes.
- keep affects the PII stage; redact dominates overlap and is deliberately irreversible; unknown scores are not dropped.
- Source marker strings, overlapping spans, custom maps and Unicode do not cause double masking or marker collisions.
- Best-effort mapping reports unavailable instead of fabricating raw positions; require_source rejects opaque transforms explicitly.
- Buffered streaming uses the same document path; windowed streaming and browser manifests reject unsupported opt-in combinations.
- Legacy defaults, presets, 3-tuple unpacking, public exports and existing reversible adapter refusal remain compatible.

```python
assert unified_process(text, mode="reversible").restored == normalized_original
assert "original" not in public_findings_json
assert legacy_process(text) == golden_result
```

- [ ] **Step 2: Verify the missing behavior.**

Run `.venv/bin/python -m pytest tests/test_unified_processing.py tests/test_privacy.py tests/test_processing_compatibility.py -q`. New feature tests must fail because the specified API or behavior is missing. Dependency or fixture errors are setup failures, not evidence. P0 instead captures independent legacy expectations and checks that they pass before implementation.

- [ ] **Step 3: Implement the interfaces in the listed production files.**

Collect detections before marker replacement and render once. Reuse existing replacement/map code. Explicitly version unified counts and metadata rather than changing legacy signal meanings.
Update listed callers, exports, documentation and optional dependency configuration in the same change. Implement only this task's contracts; preserve the defaults in Global Constraints.

- [ ] **Step 4: Verify the task and compatibility.**

Run `.venv/bin/python -m pytest tests/test_unified_processing.py tests/test_privacy.py tests/test_processing_compatibility.py -q` and `.venv/bin/python -m pytest tests/test_processing_compatibility.py -q`. Expected: all pass. Run touched-file Ruff and applicable project CI checks before committing. Real-provider tasks also need installed-profile smoke evidence, not mocks alone.

- [ ] **Step 5: Commit the reviewed task on its feature branch.**

```bash
git add sct/policy_v2.py sct/unified.py tests/test_unified_processing.py sct/sct.py sct/config.py sct/privacy.py sct/utils/process_result.py sct/utils/anonymization_map.py sct/streaming.py sct/browser.py sct/__init__.py tests/test_privacy.py tests/test_processing_compatibility.py
git commit -m "feat: apply one unified policy and renderer across detector families"
```

Run the full matching CI checks before commit/push. Review the staged diff, omit unrelated edits, and open a separate PR for this task. Commit paths include new files only once created; do not stage another worker's changes.

### C5: Carry caller schema through existing structured and framework adapters

**Dependencies:** C4. **Feature IDs:** 5.

**Files and responsibilities:**
- Create: `sct/schema.py`
- Create: `tests/test_schema_context.py`
- Modify: `sct/adapters.py`
- Modify: `sct/integrations/_runtime.py`
- Modify: `sct/integrations/hf.py`
- Modify: `sct/integrations/datatrove.py`
- Modify: `sct/integrations/langchain.py`
- Modify: `sct/integrations/haystack.py`
- Modify: `sct/mcp_server.py`
- Modify: `sct/cli.py`
- Modify: `sct/utils/ner.py`
- Test: `tests/test_schema_context.py`
- Test: `tests/test_integrations_frameworks.py`
- Test: `tests/test_integrations_hf.py`

**Interfaces consumed and produced:**
- Produces FieldSchema and FieldRule(path, entity_types, mode="hint"|"protect_value").
- Add schema/context keyword options to process_json/process_dataframe/process_rag_chunks and propagate the trusted registry factory through adapters.
Exact DTO fields, defaults and callable signatures are fixed in the shared spec. Test snippets below are assertion targets; construct fixtures in the named test files.

- [ ] **Step 1: Add failing tests in the listed test files.**

Name the primary test `test_c5_acceptance`; add separate focused tests for the distinct checks below.

- No built-in field-name guessing. Identical text receives identical default treatment regardless of key name.
- Caller-declared protect_value can explicitly mask a whole value; hints reach only a context-capable provider and are not silently ignored.
- JSON key cleaning, duplicate cleaned keys, tuples, non-string cells, metadata and input-copy semantics stay compatible.
- A Presidio-capable analysis path receives explicit context; ONNX-only providers reject unsupported required hints clearly.
- CLI/MCP expose data-only schema with an application whitelist, not arbitrary Python factory loading.
- HF/process workers recreate configured detectors; reversible mode remains refused on adapters that cannot return maps.

```python
assert default_process_json({"name": "May", "status": "May"}) == existing_result
assert protected_schema_result["name"] == "<PERSON>"
assert original_input == unchanged_copy
```

- [ ] **Step 2: Verify the missing behavior.**

Run `.venv/bin/python -m pytest tests/test_schema_context.py tests/test_integrations_frameworks.py tests/test_integrations_hf.py -q`. New feature tests must fail because the specified API or behavior is missing. Dependency or fixture errors are setup failures, not evidence. P0 instead captures independent legacy expectations and checks that they pass before implementation.

- [ ] **Step 3: Implement the interfaces in the listed production files.**

Reuse current traversal and runtime modules. Field paths are transient context, not automatically logged. Preserve cleaned structural metadata and identify unsupported schema/path combinations explicitly.
Update listed callers, exports, documentation and optional dependency configuration in the same change. Implement only this task's contracts; preserve the defaults in Global Constraints.

- [ ] **Step 4: Verify the task and compatibility.**

Run `.venv/bin/python -m pytest tests/test_schema_context.py tests/test_integrations_frameworks.py tests/test_integrations_hf.py -q` and `.venv/bin/python -m pytest tests/test_processing_compatibility.py -q`. Expected: all pass. Run touched-file Ruff and applicable project CI checks before committing. Real-provider tasks also need installed-profile smoke evidence, not mocks alone.

- [ ] **Step 5: Commit the reviewed task on its feature branch.**

```bash
git add sct/schema.py tests/test_schema_context.py sct/adapters.py sct/integrations/_runtime.py sct/integrations/hf.py sct/integrations/datatrove.py sct/integrations/langchain.py sct/integrations/haystack.py sct/mcp_server.py sct/cli.py sct/utils/ner.py tests/test_integrations_frameworks.py tests/test_integrations_hf.py
git commit -m "feat: carry caller schema through existing structured and framework adapters"
```

Run the full matching CI checks before commit/push. Review the staged diff, omit unrelated edits, and open a separate PR for this task. Commit paths include new files only once created; do not stage another worker's changes.
