# SCT ten improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Execution method requires human selection; this document starts no workers.

**Goal:** Implement all ten researched capabilities through 17 reviewable changes, preserving current defaults.

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

## Execution order and ownership

A task is a review boundary, not a promised release or a fixed effort estimate. Independent workstreams can progress after P0, but shared-file edits must merge serially. Rebase and verify each PR against the preceding accepted interfaces. Do not run multiple writers on config.py, sct.py, pyproject.toml or lockfiles.

1. P0: capture baseline, qualify optional profiles and preserve compatibility.
2. C1-C5: findings, detector analysis, registry, unified policy and field context.
3. D1-D3 and S1-S2: corpus processing and encrypted storage. They depend on P0, not each other.
4. E1-E2 and L1-L3: document/secret integrations, language abstention and experimental routing. Follow each task's exact dependencies.
5. R1: combined acceptance, documentation and staged release preparation. Publication requires a later release decision.

## Coverage and subplans

| Improvement | Owning tasks | Plan |
|---|---|---|
| 1. Finding, coordinate and policy contracts | C1, C2, C4 | [core](2026-11-07-sct-improvements-core.md) |
| 2. Public detector protocol and registry | C2, C3 | [core](2026-11-07-sct-improvements-core.md) |
| 3. Corpus deduplication and quality selection | D1, D2, D3 | [corpus](2026-11-07-sct-improvements-corpus.md) |
| 4. Safe resume and private lineage | D2, D3 | [corpus](2026-11-07-sct-improvements-corpus.md) |
| 5. Caller-declared field/schema context | C5 | [core](2026-11-07-sct-improvements-core.md) |
| 6. Encrypted token-store reference | S1, S2 | [token-store](2026-11-07-sct-improvements-token-store.md) |
| 7. Structure-preserving document extraction | E1 | [adapters](2026-11-07-sct-improvements-adapters.md) |
| 8. Adaptive detector escalation | L2, L3 | [decisions](2026-11-07-sct-improvements-decisions.md) |
| 9. Format-aware credential scanning | E2 | [adapters](2026-11-07-sct-improvements-adapters.md) |
| 10. Language abstention and downstream policy | L1, L3 | [decisions](2026-11-07-sct-improvements-decisions.md) |

## Acceptance and supported surfaces

The specification defines direct, batch, structured, framework, CLI/MCP, streaming and browser behavior. Unsupported combinations must fail explicitly. Document extraction protects extracted output, not the original PDF. Corpus deletion requires an explicit apply basis. Token-store expiry is not secure erasure.

All ten require their tests, documentation and real integration evidence. Adaptive routing can ship as experimental after implementation, but cannot claim a speed improvement or become the default without the predeclared evaluation gate.

## Verification commands for implementation

P0 must first create the project environment and qualify extras. These are execution-time checks; they were not run for this plan.

```bash
uv sync --locked --extra dev --extra test
.venv/bin/ruff check sct/ tests/ scripts/
uvx mypy==2.4.1 --python-executable .venv/bin/python
.venv/bin/python -m pytest tests/ --cov=sct --cov-fail-under=85
uv lock --check
uv export --format pylock.toml --all-extras --output-file pylock.toml --locked
git diff --exit-code pylock.toml
uv pip install --python .venv/bin/python -r docs/requirements.txt
NO_MKDOCS_2_WARNING=1 .venv/bin/python -m mkdocs build --strict
uv build
```

Match current CI versions and commands after rebasing. Check optional-profile locks/exported pylock.toml, Python 3.12/3.13/3.14, Linux worker start methods, each provider's actual module version/path, and real-provider smoke jobs. An unsupported dependency profile blocks that integration, not unrelated core work. Never replace project-runtime evidence with system Python behavior.

### P0: Qualify dependencies and freeze compatibility evidence

**Dependencies:** None. **Feature IDs:** Shared baseline.

**Files and responsibilities:**
- Create: `tests/test_processing_compatibility.py`
- Create: `tests/fixtures/processing_compatibility.json`
- Create: `evaluation/improvements/baseline-manifest.json`
- Modify: `.github/workflows/optional-backends.yml`
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Modify: `pylock.toml`
- Modify: `docs/NER_BACKENDS.md`
- Test: `tests/test_processing_compatibility.py`

**Interfaces consumed and produced:**
- Produces baseline-manifest.json with source/config/model/input identities and measured quality/runtime results.
- Optional profiles: secrets, documents, token-store, corpus; exact resolved locks must be recorded before each dependent PR.
Exact DTO fields, defaults and callable signatures are fixed in the shared spec. Test snippets below are assertion targets; construct fixtures in the named test files.

- [ ] **Step 1: Add failing tests in the listed test files.**

Name the primary test `test_p0_acceptance`; add separate focused tests for the distinct checks below.

- Capture independent expected outputs for default/training and placeholder/synthetic/reversible modes, blank input, source markers, custom steps and batch ordering.
- Install each proposed optional profile on Python 3.12/3.13/3.14 and verify versions plus module paths using that project interpreter; a registry classifier is not an install test.
- Measure current exact-span metrics, document exposure, median/p95 latency, model calls and peak memory on identical fixed samples before editing behavior.
- Correct the stale averaging sentence to reflect configurable max/mean voting, without changing the default.

```python
assert run_fixture("default_contacts") == golden["default_contacts"]
assert run_fixture("reversible_source_marker").restored == original
```

- [ ] **Step 2: Verify the missing behavior.**

Run `.venv/bin/python -m pytest tests/test_processing_compatibility.py -q`. New feature tests must fail because the specified API or behavior is missing. Dependency or fixture errors are setup failures, not evidence. P0 instead captures independent legacy expectations and checks that they pass before implementation.

- [ ] **Step 3: Implement the interfaces in the listed production files.**

Add model-free golden fixtures and an opt-in pinned-model baseline job. Preserve existing Linux worker CI and add path-filtered optional-provider smoke jobs. Do not install heavy extras in the base job.
Update listed callers, exports, documentation and optional dependency configuration in the same change. Implement only this task's contracts; preserve the defaults in Global Constraints.

- [ ] **Step 4: Verify the task and compatibility.**

Run `.venv/bin/python -m pytest tests/test_processing_compatibility.py -q` and `.venv/bin/python -m pytest tests/test_processing_compatibility.py -q`. Expected: all pass. Run touched-file Ruff and applicable project CI checks before committing. Real-provider tasks also need installed-profile smoke evidence, not mocks alone.

- [ ] **Step 5: Commit the reviewed task on its feature branch.**

```bash
git add tests/test_processing_compatibility.py tests/fixtures/processing_compatibility.json evaluation/improvements/baseline-manifest.json .github/workflows/optional-backends.yml pyproject.toml uv.lock pylock.toml docs/NER_BACKENDS.md
git commit -m "feat: qualify dependencies and freeze compatibility evidence"
```

Run the full matching CI checks before commit/push. Review the staged diff, omit unrelated edits, and open a separate PR for this task. Commit paths include new files only once created; do not stage another worker's changes.

### R1: Run cross-feature acceptance and prepare a reviewed staged release

**Dependencies:** C5, D3, S2, E1, E2, L3. **Feature IDs:** 1, 2, 3, 4, 5, 6, 7, 8, 9, 10.

**Files and responsibilities:**
- Create: `tests/test_improvements_end_to_end.py`
- Create: `evaluation/improvements/ACCEPTANCE.md`
- Modify: `README.md`
- Modify: `docs/API.md`
- Modify: `docs/CONFIG.md`
- Modify: `docs/PRIVACY.md`
- Modify: `docs/INTEGRATIONS.md`
- Modify: `docs/INSTALL.md`
- Modify: `llms.txt`
- Modify: `CHANGELOG.md`
- Test: `tests/test_improvements_end_to_end.py`

**Interfaces consumed and produced:**
- Produces a per-feature completion ledger with test/evaluation evidence, compatibility status and documented limits.
- No automatic release publication or default migration is included.
Exact DTO fields, defaults and callable signatures are fixed in the shared spec. Test snippets below are assertion targets; construct fixtures in the named test files.

- [ ] **Step 1: Add failing tests in the listed test files.**

Name the primary test `test_r1_acceptance`; add separate focused tests for the distinct checks below.

- Exercise extract -> context-aware unified processing -> encrypted references for direct processing -> corpus selection -> interrupted/resumed outputs. Adapters retaining reversible restrictions use non-reversible mode.
- Check no raw values survive default manifests, traces, document metadata, exceptions or public findings.
- Run supported Python and optional-provider CI, Linux fork/forkserver tests, packaging, locks, strict docs and current quality baseline comparisons.
- Each feature is complete only with its own acceptance evidence. Adaptive implementation can be delivered as experimental without claiming a successful optimization.
- Document every unsupported combination and rollback profile; existing default fixtures must still pass.

```python
assert all(feature.evidence for feature in acceptance.features)
assert legacy_golden_suite_passes
assert no_sensitive_values_in_default_artifacts
```

- [ ] **Step 2: Verify the missing behavior.**

Run `.venv/bin/python -m pytest tests/test_improvements_end_to_end.py -q`. New feature tests must fail because the specified API or behavior is missing. Dependency or fixture errors are setup failures, not evidence. P0 instead captures independent legacy expectations and checks that they pass before implementation.

- [ ] **Step 3: Implement the interfaces in the listed production files.**

Release in reviewed opt-in milestones. Update examples/exports/types and user-facing docs together. A later default switch needs a separate reviewed migration decision.
Update listed callers, exports, documentation and optional dependency configuration in the same change. Implement only this task's contracts; preserve the defaults in Global Constraints.

- [ ] **Step 4: Verify the task and compatibility.**

Run `.venv/bin/python -m pytest tests/test_improvements_end_to_end.py -q` and `.venv/bin/python -m pytest tests/test_processing_compatibility.py -q`. Expected: all pass. Run touched-file Ruff and applicable project CI checks before committing. Real-provider tasks also need installed-profile smoke evidence, not mocks alone.

- [ ] **Step 5: Commit the reviewed task on its feature branch.**

```bash
git add tests/test_improvements_end_to_end.py evaluation/improvements/ACCEPTANCE.md README.md docs/API.md docs/CONFIG.md docs/PRIVACY.md docs/INTEGRATIONS.md docs/INSTALL.md llms.txt CHANGELOG.md
git commit -m "feat: run cross-feature acceptance and prepare a reviewed staged release"
```

Run the full matching CI checks before commit/push. Review the staged diff, omit unrelated edits, and open a separate PR for this task. Commit paths include new files only once created; do not stage another worker's changes.

## Plan self-review and execution handoff

Coverage: all ten feature IDs map to tasks. Dependencies form an acyclic graph. Contracts fix coordinate units, unknown scores, schema identity, storage scope and deletion limits. The five Review Focus conditions each have owning tests. Production code and dependencies are unchanged by this planning work.

Review this plan and the human roadmap before implementation. Choose native execution or separate implementer/reviewer workers. A negative routing experiment is an acceptable recorded research result, not permission to report an unproven optimization. Feature completion still requires every applicable acceptance check.
