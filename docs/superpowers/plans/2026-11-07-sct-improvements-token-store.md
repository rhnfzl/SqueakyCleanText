# SCT token-store Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Execution method requires human selection; this document starts no workers.

**Goal:** Implement the token-store workstream with explicit compatibility and acceptance evidence.

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

### S1: Implement encrypted, scoped and expiring local token storage

**Dependencies:** P0. **Feature IDs:** 6.

**Files and responsibilities:**
- Create: `sct/token_stores/__init__.py`
- Create: `sct/token_stores/sqlite.py`
- Create: `tests/test_sqlite_token_store.py`
- Modify: `sct/privacy.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Modify: `pylock.toml`
- Modify: `sct/__init__.py`
- Test: `tests/test_sqlite_token_store.py`
- Test: `tests/test_privacy.py`

**Interfaces consumed and produced:**
- Produces EncryptedSQLiteTokenStore, KeyProvider, StoreEvent and TokenStoreError under the specified interface.
- Implements existing TokenStore.save/load/delete and existing AnonymizationMap.to_dict/from_dict.
Exact DTO fields, defaults and callable signatures are fixed in the shared spec. Test snippets below are assertion targets; construct fixtures in the named test files.

- [ ] **Step 1: Add failing tests in the listed test files.**

Name the primary test `test_s1_acceptance`; add separate focused tests for the distinct checks below.

- Round-trip across a process restart, preserving marker reservations and counters.
- Expired/missing/wrong-scope references return None. Tamper/key substitution and malformed authenticated payloads fail explicitly.
- Keys and raw values do not appear in database bytes, returned references, repr, exceptions or audit events.
- SQLite file/directory access and transaction handling are explicit; no plaintext fallback on encryption/store errors.

```python
assert restarted_store.load(ref).deanonymize(masked) == original
assert other_scope.load(ref) is None
assert expired_store.load(ref) is None
```

- [ ] **Step 2: Verify the missing behavior.**

Run `.venv/bin/python -m pytest tests/test_sqlite_token_store.py tests/test_privacy.py -q`. New feature tests must fail because the specified API or behavior is missing. Dependency or fixture errors are setup failures, not evidence. P0 instead captures independent legacy expectations and checks that they pass before implementation.

- [ ] **Step 3: Implement the interfaces in the listed production files.**

Use vetted Fernet authentication, random references and an encrypted envelope binding scope/reference/expiry. Inject clock/keys for deterministic tests. Do not change the default in-memory store.
Update listed callers, exports, documentation and optional dependency configuration in the same change. Implement only this task's contracts; preserve the defaults in Global Constraints.

- [ ] **Step 4: Verify the task and compatibility.**

Run `.venv/bin/python -m pytest tests/test_sqlite_token_store.py tests/test_privacy.py -q` and `.venv/bin/python -m pytest tests/test_processing_compatibility.py -q`. Expected: all pass. Run touched-file Ruff and applicable project CI checks before committing. Real-provider tasks also need installed-profile smoke evidence, not mocks alone.

- [ ] **Step 5: Commit the reviewed task on its feature branch.**

```bash
git add sct/token_stores/__init__.py sct/token_stores/sqlite.py tests/test_sqlite_token_store.py sct/privacy.py pyproject.toml uv.lock pylock.toml sct/__init__.py tests/test_privacy.py
git commit -m "feat: implement encrypted, scoped and expiring local token storage"
```

Run the full matching CI checks before commit/push. Review the staged diff, omit unrelated edits, and open a separate PR for this task. Commit paths include new files only once created; do not stage another worker's changes.

### S2: Prove token-store concurrency, rotation and documented operational limits

**Dependencies:** S1. **Feature IDs:** 6.

**Files and responsibilities:**
- Create: `tests/test_token_store_operations.py`
- Create: `docs/TOKEN_STORES.md`
- Modify: `sct/token_stores/sqlite.py`
- Modify: `README.md`
- Modify: `docs/PRIVACY.md`
- Modify: `docs/INSTALL.md`
- Test: `tests/test_token_store_operations.py`

**Interfaces consumed and produced:**
- Adds rotation/expiry/cleanup operations without changing save/load/delete compatibility.
- Documents trusted scope assignment, key ownership, audit and logical-deletion semantics.
Exact DTO fields, defaults and callable signatures are fixed in the shared spec. Test snippets below are assertion targets; construct fixtures in the named test files.

- [ ] **Step 1: Add failing tests in the listed test files.**

Name the primary test `test_s2_acceptance`; add separate focused tests for the distinct checks below.

- Concurrent threads/processes cannot collide references or observe partially committed maps; connections are not reused across fork.
- New saves use the current key ID, old maps remain readable through their retained key IDs, and explicit rotation is tested.
- Deletion/expiry does not claim to erase SQLite WAL, backups or storage media; key retirement consequences are explicit.
- An independent security-focused review must pass before labelling the reference integration ready for supported production use.

```python
assert concurrent_reference_count == saved_map_count
assert rotated_store.load(old_ref) == original_map
assert deleted_ref_not_loadable
```

- [ ] **Step 2: Verify the missing behavior.**

Run `.venv/bin/python -m pytest tests/test_token_store_operations.py -q`. New feature tests must fail because the specified API or behavior is missing. Dependency or fixture errors are setup failures, not evidence. P0 instead captures independent legacy expectations and checks that they pass before implementation.

- [ ] **Step 3: Implement the interfaces in the listed production files.**

Keep scope/key/provider access application-owned. Document backup, access-control and retention obligations instead of inventing a hosted auth service.
Update listed callers, exports, documentation and optional dependency configuration in the same change. Implement only this task's contracts; preserve the defaults in Global Constraints.

- [ ] **Step 4: Verify the task and compatibility.**

Run `.venv/bin/python -m pytest tests/test_token_store_operations.py -q` and `.venv/bin/python -m pytest tests/test_processing_compatibility.py -q`. Expected: all pass. Run touched-file Ruff and applicable project CI checks before committing. Real-provider tasks also need installed-profile smoke evidence, not mocks alone.

- [ ] **Step 5: Commit the reviewed task on its feature branch.**

```bash
git add tests/test_token_store_operations.py docs/TOKEN_STORES.md sct/token_stores/sqlite.py README.md docs/PRIVACY.md docs/INSTALL.md
git commit -m "feat: prove token-store concurrency, rotation and documented operational limits"
```

Run the full matching CI checks before commit/push. Review the staged diff, omit unrelated edits, and open a separate PR for this task. Commit paths include new files only once created; do not stage another worker's changes.
