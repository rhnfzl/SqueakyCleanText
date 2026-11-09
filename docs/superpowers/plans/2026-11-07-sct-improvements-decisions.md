# SCT decisions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Execution method requires human selection; this document starts no workers.

**Goal:** Implement the decisions workstream with explicit compatibility and acceptance evidence.

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

### L1: Add explicit low-confidence language abstention and fallback policy

**Dependencies:** P0. **Feature IDs:** 10.

**Files and responsibilities:**
- Create: `sct/language_policy.py`
- Create: `tests/test_language_policy.py`
- Modify: `sct/sct.py`
- Modify: `sct/config.py`
- Modify: `sct/signals.py`
- Modify: `sct/utils/process_result.py`
- Modify: `docs/LANGUAGES.md`
- Modify: `docs/CONFIG.md`
- Test: `tests/test_language_policy.py`
- Test: `tests/test_language_all.py`

**Interfaces consumed and produced:**
- Produces LanguageDecision and apply_language_policy(decision, policy) -> LanguageDecision.
- Adds opt-in decision thresholds/actions; defaults remain legacy and thresholds None.
Exact DTO fields, defaults and callable signatures are fixed in the shared spec. Test snippets below are assertion targets; construct fixtures in the named test files.

- [ ] **Step 1: Add failing tests in the listed test files.**

Name the primary test `test_l1_acceptance`; add separate focused tests for the distinct checks below.

- Pinned languages bypass thresholding. No-score/tie and restricted-tuple fallback remain unchanged by default.
- With min_confidence=0.8 and min_margin=0.1, confidence 0.79 abstains; 0.8 with margin 0.1 passes.
- Opt-in abstention records None language, source=abstained and language_abstained warning.
- Multilingual/error/explicit skip actions are tested; no language outside a configured restriction is invented.
- Measure known-language accuracy/coverage and unknown-language false answers before choosing any new default.

```python
assert abstained.language is None and abstained.source == "abstained"
assert boundary_case.language == "ENGLISH"
assert pinned.language == "DUTCH"
```

- [ ] **Step 2: Verify the missing behavior.**

Run `.venv/bin/python -m pytest tests/test_language_policy.py tests/test_language_all.py -q`. New feature tests must fail because the specified API or behavior is missing. Dependency or fixture errors are setup failures, not evidence. P0 instead captures independent legacy expectations and checks that they pass before implementation.

- [ ] **Step 3: Implement the interfaces in the listed production files.**

Keep raw confidence/margin evidence and make the decision layer pure. Distinguish synthetic-locale fallback from language detection. Never imply a threshold alone guarantees unknown-language detection.
Update listed callers, exports, documentation and optional dependency configuration in the same change. Implement only this task's contracts; preserve the defaults in Global Constraints.

- [ ] **Step 4: Verify the task and compatibility.**

Run `.venv/bin/python -m pytest tests/test_language_policy.py tests/test_language_all.py -q` and `.venv/bin/python -m pytest tests/test_processing_compatibility.py -q`. Expected: all pass. Run touched-file Ruff and applicable project CI checks before committing. Real-provider tasks also need installed-profile smoke evidence, not mocks alone.

- [ ] **Step 5: Commit the reviewed task on its feature branch.**

```bash
git add sct/language_policy.py tests/test_language_policy.py sct/sct.py sct/config.py sct/signals.py sct/utils/process_result.py docs/LANGUAGES.md docs/CONFIG.md tests/test_language_all.py
git commit -m "feat: add explicit low-confidence language abstention and fallback policy"
```

Run the full matching CI checks before commit/push. Review the staged diff, omit unrelated edits, and open a separate PR for this task. Commit paths include new files only once created; do not stage another worker's changes.

### L2: Implement opt-in detector routing with safe fallback and trace identity

**Dependencies:** C3, C4, L1. **Feature IDs:** 8.

**Files and responsibilities:**
- Create: `sct/routing.py`
- Create: `tests/test_detector_routing.py`
- Modify: `sct/unified.py`
- Modify: `sct/config.py`
- Modify: `sct/utils/process_result.py`
- Test: `tests/test_detector_routing.py`

**Interfaces consumed and produced:**
- Produces RoutingContext, DetectorRun, RoutingDecision and RoutingPolicy.plan.
- Provides always_run and experimental calibrated_cascade using explicit fast/full specs plus a calibration artifact.
Exact DTO fields, defaults and callable signatures are fixed in the shared spec. Test snippets below are assertion targets; construct fixtures in the named test files.

- [ ] **Step 1: Add failing tests in the listed test files.**

Name the primary test `test_l2_acceptance`; add separate focused tests for the distinct checks below.

- Unknown language, unsupported entity coverage/context, missing calibration and detector errors do not authorize an early exit.
- One regex email never proves names absent; mixed-entity fixture routes to the full detector.
- Routing cannot skip a requested capability that the fast path does not declare. All skip decisions carry a value-free reason and experimental policy identity.
- Always_run reproduces unified baseline results; incompatible model/revision/config identities reject the artifact.

```python
assert "full" in route(mixed_name_and_email).selected_ids
assert missing_calibration_uses_full
assert always_run_result == unified_baseline
```

- [ ] **Step 2: Verify the missing behavior.**

Run `.venv/bin/python -m pytest tests/test_detector_routing.py -q`. New feature tests must fail because the specified API or behavior is missing. Dependency or fixture errors are setup failures, not evidence. P0 instead captures independent legacy expectations and checks that they pass before implementation.

- [ ] **Step 3: Implement the interfaces in the listed production files.**

Keep routing separate from model implementation. Treat completion of configured stages as execution status, never complete privacy. Default to always_run and bound stages/calls.
Update listed callers, exports, documentation and optional dependency configuration in the same change. Implement only this task's contracts; preserve the defaults in Global Constraints.

- [ ] **Step 4: Verify the task and compatibility.**

Run `.venv/bin/python -m pytest tests/test_detector_routing.py -q` and `.venv/bin/python -m pytest tests/test_processing_compatibility.py -q`. Expected: all pass. Run touched-file Ruff and applicable project CI checks before committing. Real-provider tasks also need installed-profile smoke evidence, not mocks alone.

- [ ] **Step 5: Commit the reviewed task on its feature branch.**

```bash
git add sct/routing.py tests/test_detector_routing.py sct/unified.py sct/config.py sct/utils/process_result.py
git commit -m "feat: implement opt-in detector routing with safe fallback and trace identity"
```

Run the full matching CI checks before commit/push. Review the staged diff, omit unrelated edits, and open a separate PR for this task. Commit paths include new files only once created; do not stage another worker's changes.

### L3: Calibrate, benchmark and gate adaptive promotion

**Dependencies:** L2. **Feature IDs:** 8, 10.

**Files and responsibilities:**
- Create: `scripts/benchmark_detector_routing.py`
- Create: `evaluation/improvements/routing-protocol.md`
- Create: `tests/test_routing_calibration.py`
- Modify: `benchmarks/compare/PROTOCOL.md`
- Modify: `docs/PERFORMANCE.md`
- Test: `tests/test_routing_calibration.py`

**Interfaces consumed and produced:**
- Produces a versioned calibration JSON and held-out evaluation report, including failed candidates.
- Consumes the 12-candidate dev-only grid and promotion criteria fixed in the design spec.
Exact DTO fields, defaults and callable signatures are fixed in the shared spec. Test snippets below are assertion targets; construct fixtures in the named test files.

- [ ] **Step 1: Add failing tests in the listed test files.**

Name the primary test `test_l3_acceptance`; add separate focused tests for the distinct checks below.

- Model/config/input hashes separate development from the untouched test split; no test tuning or threshold search until a desired result appears.
- Every language/entity/domain slice reports recall, precision and document exposure; shared/full taxonomies are not pooled.
- Record common-runtime model calls, median/p95 latency and peak memory with warmup/repeats.
- Promotion uses the fixed exposure interval and efficiency gates. Inconclusive/failing candidates leave always_run as default and retain the negative result.

```python
assert fit_uses_dev_only and test_runs_once
assert failed_gate.default_policy == "always_run"
assert artifact.model_ids == evaluated_model_ids
```

- [ ] **Step 2: Verify the missing behavior.**

Run `.venv/bin/python -m pytest tests/test_routing_calibration.py -q`. New feature tests must fail because the specified API or behavior is missing. Dependency or fixture errors are setup failures, not evidence. P0 instead captures independent legacy expectations and checks that they pass before implementation.

- [ ] **Step 3: Implement the interfaces in the listed production files.**

Use the existing evaluators and paired resampling with pinned models and samples. No new library is needed to enumerate the finite grid. Keep calibration artifacts free of source documents or reversible maps.
Update listed callers, exports, documentation and optional dependency configuration in the same change. Implement only this task's contracts; preserve the defaults in Global Constraints.

- [ ] **Step 4: Verify the task and compatibility.**

Run `.venv/bin/python -m pytest tests/test_routing_calibration.py -q` and `.venv/bin/python -m pytest tests/test_processing_compatibility.py -q`. Expected: all pass. Run touched-file Ruff and applicable project CI checks before committing. Real-provider tasks also need installed-profile smoke evidence, not mocks alone.

- [ ] **Step 5: Commit the reviewed task on its feature branch.**

```bash
git add scripts/benchmark_detector_routing.py evaluation/improvements/routing-protocol.md tests/test_routing_calibration.py benchmarks/compare/PROTOCOL.md docs/PERFORMANCE.md
git commit -m "feat: calibrate, benchmark and gate adaptive promotion"
```

Run the full matching CI checks before commit/push. Review the staged diff, omit unrelated edits, and open a separate PR for this task. Commit paths include new files only once created; do not stage another worker's changes.
