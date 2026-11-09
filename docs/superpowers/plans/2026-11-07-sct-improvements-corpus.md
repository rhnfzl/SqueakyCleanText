# SCT corpus Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Execution method requires human selection; this document starts no workers.

**Goal:** Implement the corpus workstream with explicit compatibility and acceptance evidence.

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

### D1: Implement report-first quality selection and corpus duplicate decisions

**Dependencies:** P0. **Feature IDs:** 3.

**Files and responsibilities:**
- Create: `sct/corpus/__init__.py`
- Create: `sct/corpus/types.py`
- Create: `sct/corpus/selection.py`
- Create: `tests/test_corpus_selection.py`
- Modify: `sct/score.py`
- Test: `tests/test_corpus_selection.py`

**Interfaces consumed and produced:**
- Produces CorpusRecord, SelectionConfig, RecordAnalysis, SelectionReport; implements analyse_records and select_records.
- Consumes existing ScoreConfig/QualityScorer/MinHash semantics without changing score_record defaults.
Exact DTO fields, defaults and callable signatures are fixed in the shared spec. Test snippets below are assertion targets; construct fixtures in the named test files.

- [ ] **Step 1: Add failing tests in the listed test files.**

Name the primary test `test_d1_acceptance`; add separate focused tests for the distinct checks below.

- Default report mode retains every record. A non-None quality_min requires an available scorer.
- Exact matching and near matching produce real clusters and deterministic min_stable_id keepers, including transitive clusters and tied scores.
- Reject duplicate record IDs and incompatible signature metadata, including tokenizer/library/config identity.
- Two different sensitive records that normalize to the same placeholder text are flagged as a basis-dependent collision; apply mode requires an explicit basis choice.
- Missing/short signatures are not equated; audit rows contain stable IDs, scores and reasons, not raw text.

```python
assert report_mode.kept_ids == all_ids
assert apply_mode.kept_ids == expected_representatives
with raises(SignatureMismatch): compare(incompatible_a, incompatible_b)
```

- [ ] **Step 2: Verify the missing behavior.**

Run `.venv/bin/python -m pytest tests/test_corpus_selection.py -q`. New feature tests must fail because the specified API or behavior is missing. Dependency or fixture errors are setup failures, not evidence. P0 instead captures independent legacy expectations and checks that they pass before implementation.

- [ ] **Step 3: Implement the interfaces in the listed production files.**

Use existing datasketch support plus disk-backed candidate/decision state. Keep threshold decisions outside the ordinary cleaner and avoid holding the whole text corpus in memory.
Update listed callers, exports, documentation and optional dependency configuration in the same change. Implement only this task's contracts; preserve the defaults in Global Constraints.

- [ ] **Step 4: Verify the task and compatibility.**

Run `.venv/bin/python -m pytest tests/test_corpus_selection.py -q` and `.venv/bin/python -m pytest tests/test_processing_compatibility.py -q`. Expected: all pass. Run touched-file Ruff and applicable project CI checks before committing. Real-provider tasks also need installed-profile smoke evidence, not mocks alone.

- [ ] **Step 5: Commit the reviewed task on its feature branch.**

```bash
git add sct/corpus/__init__.py sct/corpus/types.py sct/corpus/selection.py tests/test_corpus_selection.py sct/score.py
git commit -m "feat: implement report-first quality selection and corpus duplicate decisions"
```

Run the full matching CI checks before commit/push. Review the staged diff, omit unrelated edits, and open a separate PR for this task. Commit paths include new files only once created; do not stage another worker's changes.

### D2: Compose a complete DataTrove curation recipe with private run identity

**Dependencies:** D1. **Feature IDs:** 3, 4.

**Files and responsibilities:**
- Create: `sct/corpus/manifest.py`
- Create: `sct/corpus/datatrove.py`
- Create: `tests/test_corpus_manifest.py`
- Create: `examples/corpus/clean_select_dedup.py`
- Modify: `sct/integrations/datatrove.py`
- Modify: `docs/INTEGRATIONS.md`
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Modify: `pylock.toml`
- Test: `tests/test_corpus_manifest.py`
- Test: `tests/test_linux_workers.py`

**Interfaces consumed and produced:**
- Produces RunIdentity and RunManifest(schema_version=1); fingerprint configuration and record output identities.
- Provides a DataTrove-compatible workflow through identification, clustering and actual duplicate removal on an explicitly declared basis.
Exact DTO fields, defaults and callable signatures are fixed in the shared spec. Test snippets below are assertion targets; construct fixtures in the named test files.

- [ ] **Step 1: Add failing tests in the listed test files.**

Name the primary test `test_d2_acceptance`; add separate focused tests for the distinct checks below.

- End-to-end fixture writes a new kept corpus plus stable decisions, with filtered/rejected counts reconciling to input IDs.
- No incompatible reuse of SCT signatures as DataTrove files; use one complete native workflow or an explicit tested conversion.
- Manifest excludes raw text, custom term values, keys, anon maps and raw SDK objects. Treat config digests and IDs as private identifying state.
- Input source remains unchanged; apply mode writes new outputs and report-only remains the default.
- Run the real DataTrove 0.10.1 profile on Linux with process-owned cleaner handles.

```python
assert kept_count + rejected_count == input_count
assert manifest["schema_version"] == 1
assert sensitive_literal not in serialized_manifest
```

- [ ] **Step 2: Verify the missing behavior.**

Run `.venv/bin/python -m pytest tests/test_corpus_manifest.py tests/test_linux_workers.py -q`. New feature tests must fail because the specified API or behavior is missing. Dependency or fixture errors are setup failures, not evidence. P0 instead captures independent legacy expectations and checks that they pass before implementation.

- [ ] **Step 3: Implement the interfaces in the listed production files.**

Prefer DataTrove execution rather than adding a second scheduler. Store deterministic shard references and secret-free operational metadata; no automatic raw rejected-document writer.
Update listed callers, exports, documentation and optional dependency configuration in the same change. Implement only this task's contracts; preserve the defaults in Global Constraints.

- [ ] **Step 4: Verify the task and compatibility.**

Run `.venv/bin/python -m pytest tests/test_corpus_manifest.py tests/test_linux_workers.py -q` and `.venv/bin/python -m pytest tests/test_processing_compatibility.py -q`. Expected: all pass. Run touched-file Ruff and applicable project CI checks before committing. Real-provider tasks also need installed-profile smoke evidence, not mocks alone.

- [ ] **Step 5: Commit the reviewed task on its feature branch.**

```bash
git add sct/corpus/manifest.py sct/corpus/datatrove.py tests/test_corpus_manifest.py examples/corpus/clean_select_dedup.py sct/integrations/datatrove.py docs/INTEGRATIONS.md pyproject.toml uv.lock pylock.toml tests/test_linux_workers.py
git commit -m "feat: compose a complete datatrove curation recipe with private run identity"
```

Run the full matching CI checks before commit/push. Review the staged diff, omit unrelated edits, and open a separate PR for this task. Commit paths include new files only once created; do not stage another worker's changes.

### D3: Add safe resumption, failure quarantine and reference decontamination

**Dependencies:** D2. **Feature IDs:** 3, 4.

**Files and responsibilities:**
- Create: `sct/corpus/runner.py`
- Create: `tests/test_corpus_resume.py`
- Create: `tests/test_corpus_decontamination.py`
- Modify: `examples/corpus/clean_select_dedup.py`
- Test: `tests/test_corpus_resume.py`
- Test: `tests/test_corpus_decontamination.py`

**Interfaces consumed and produced:**
- Implements resume_run(manifest_path: Path, *, expected: RunIdentity) -> ResumeState.
- Adds caller-supplied decontamination reference indexes and typed private failure/quarantine records.
Exact DTO fields, defaults and callable signatures are fixed in the shared spec. Test snippets below are assertion targets; construct fixtures in the named test files.

- [ ] **Step 1: Add failing tests in the listed test files.**

Name the primary test `test_d3_acceptance`; add separate focused tests for the distinct checks below.

- Crash after output write but before completion publication; restart has identical stable-ID outputs to uninterrupted execution, with no duplicate writes.
- Changed input partition, config, detector/model revision, seed or signature identity refuses reuse.
- Completion requires atomic output plus checksum; incomplete work replays and sinks are idempotent.
- Retries are bounded; failure ledger stores reason and stable ID, not raw exception values or document contents.
- Reference matching has report/apply behavior, measured false-positive fixtures and no built-in benchmark/keyword lists.

```python
assert sorted(resumed_ids) == sorted(uninterrupted_ids)
assert len(resumed_ids) == len(set(resumed_ids))
with raises(RunIdentityMismatch): resume_run(path, expected=changed)
```

- [ ] **Step 2: Verify the missing behavior.**

Run `.venv/bin/python -m pytest tests/test_corpus_resume.py tests/test_corpus_decontamination.py -q`. New feature tests must fail because the specified API or behavior is missing. Dependency or fixture errors are setup failures, not evidence. P0 instead captures independent legacy expectations and checks that they pass before implementation.

- [ ] **Step 3: Implement the interfaces in the listed production files.**

Use framework checkpoint boundaries plus an SCT identity guard. Never claim exactly-once semantics. Make retention of optional private samples explicit and separate from count-only quarantine.
Update listed callers, exports, documentation and optional dependency configuration in the same change. Implement only this task's contracts; preserve the defaults in Global Constraints.

- [ ] **Step 4: Verify the task and compatibility.**

Run `.venv/bin/python -m pytest tests/test_corpus_resume.py tests/test_corpus_decontamination.py -q` and `.venv/bin/python -m pytest tests/test_processing_compatibility.py -q`. Expected: all pass. Run touched-file Ruff and applicable project CI checks before committing. Real-provider tasks also need installed-profile smoke evidence, not mocks alone.

- [ ] **Step 5: Commit the reviewed task on its feature branch.**

```bash
git add sct/corpus/runner.py tests/test_corpus_resume.py tests/test_corpus_decontamination.py examples/corpus/clean_select_dedup.py
git commit -m "feat: add safe resumption, failure quarantine and reference decontamination"
```

Run the full matching CI checks before commit/push. Review the staged diff, omit unrelated edits, and open a separate PR for this task. Commit paths include new files only once created; do not stage another worker's changes.
