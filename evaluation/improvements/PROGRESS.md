# Ten-improvement implementation evidence

Original research baseline: `8afef61` (v0.8.3), branch `research/pii-routing-landscape`. The current integration base appears below.

## Current v0.9.0 integration status

The current base is published v0.9.0 at `882342fd08c03718d21d22278814cfbfe4b883f3`.
The release owner completed publication and fresh PyPI-install verification.
The port uses `feat/pipeline-improvements-v090` in a separate worktree. The original research worktree remains unchanged.
All 17 tasks and 79 requirements have current source and runtime qualification.
Each actual task PR head must pass GitHub checks before merge. No merge, release or default migration is authorized here.

The final isolated core profiles each pass 3,271 tests with 169 explicit skips and two visible warnings.
Coverage is `6695/7717 = 86.75651159777115%` on Python 3.12 and 3.13.
Python 3.14 reports `6478/7502 = 86.3503065849107%`. The threshold stays 85%, with report precision set to two.
All 17 exact task prefixes also pass full Linux Python 3.14 and static checks.
The 24 new boundary cases qualify core orchestration after adapter preflight. Real SDK formats retain separate evidence.
Isolated native integration profiles each pass 328 tests. The single missing-Presidio test has a separate unchanged-function qualification.
Independent whole-branch review and the combined final fix review approve the source.
Wheel, source distribution, installed-wheel checks, Ruff, pinned mypy and strict documentation checks pass.

Earlier concurrent coverage reports could combine data. Those reports are retained as superseded diagnostics, not final coverage proof.
Every final profile uses its own absolute `COVERAGE_FILE` directory.
The controlled v0.9.0 baseline matches Python 3.13.3 and all 44 dependency versions: `4453/4818 = 92.42424242424242%`.
The final feature delta is `-5.667730826471271` percentage points in that core profile.
Optional modules add unexecuted core statements; their native qualification has separate coverage scopes.

The v2 routing experiment passes its limited relative gates: expensive calls fall 8% and measured median latency falls 3.556%.
Both policies still expose 78 of 110 held-out documents and three of six mixed fixtures.
The default remains `always_run`. The result does not establish production privacy or general speed.
Saved observations pass independent reconstruction without inference replay. All historical v1 artifacts remain unchanged.

Current evidence: [final qualification](v090-final-qualification.json), [native profiles](v090-native-profiles-qualification.json),
[model/export proof](v090-model-qualification.json), [routing validation](routing-validation-v2.json) and [requirement ledger](acceptance-ledger.json).
The [test-only follow-up](v090-boundary-test-qualification.json) preserves source bindings and the additional SDK fixture-isolation checks.
The [previous final record](v090-pre-boundary-qualification.json) preserves qualification before that test-only addition.
The [port addendum](../../docs/superpowers/plans/2026-11-09-sct-improvements-v090-port.md) defines current API compatibility.
Explicit config and `max_workers` remain. Removed globals, model tuples, aliases and worker `batch_size` stay removed.
The current record supersedes historical pending statements below only within its stated source and runtime scopes.

## Historical implementation status before the v0.9.0 port

- P0 compatibility fixtures pass. Full baseline suite: 2718 passed, 64 skipped, 2 sandbox failures, coverage 91.62%. Both failed atomic-file tests pass when rerun outside the sandbox. All 12 Linux provider/Python profiles pass. P0 review fixes passed scoped re-review in `56d9e4f`: six actual model calls, source-byte identity verification and corrected provider CI filters.
- C1 committed as `eb7f42c`; 32 focused and compatibility tests pass, Ruff passes. Independent review passed. A later type-check failure was fixed in `29f9b27`; 32 tests, Ruff and targeted mypy now pass.
- S1 encrypted token storage is committed as `b276fa0`; 38 focused tests and Ruff pass. Independent review and targeted mypy pass. S2 operational proof is recorded below.
- C2 built-in span analysis has passed 323 final affected tests, real pinned-model CPU checks, full Ruff and mypy across 48 source files. Linux broad checks passed with 2742 tests, 64 skips and 92.18% coverage. A final 63-test run and real-model smoke cover the later score conversion. Independent review passed. Final integrated CI remains pending.
- C3 registry and worker factories passed review after two boundary fixes (`1299e1d`, `08e1ae3`). Final Linux checks: 2777 passed, 65 skipped, 91.95% coverage. Actual MCP/Haystack checks: 62 passed, 11 unrelated optional skips.
- C4 unified detection, policy and rendering passed review and committed as `bc4efd8`. All11 committed files match the reviewed snapshot. Final local full check:2829 passed,66 skipped,two sandbox-only failures which pass separately outside sandbox;92.17%coverage. Frozen source also passed171 focused tests each on Python3.12/3.13.
- C5 schema/context, framework forwarding and versioned adapter summaries passed independent review in `2c7b7ad`. All 19 committed files match the reviewed snapshot. Full local suite: 2924 passed, 25 skipped, two known sandbox failures. Both failures pass outside the sandbox. Coverage: 95.95%. Ruff and mypy pass. Default coverage summaries retain uncertainty for covering or unavailable source ranges.
- D1 corpus selection passed review in `fd3880e`. All four committed files match the corrected reviewed snapshot. Full local suite: 2956 passed, 25 skipped, two known sandbox stdout failures pass outside the sandbox. Coverage: 95.91%, compared with C5 95.95% (0.04 percentage points lower, above the 85% gate). Ruff and mypy pass. Ownership checks protect caller paths; report cleanup closes active readers. MinHash candidate recall remains probabilistic.
- D2 native DataTrove composition and run manifests passed review in `a689495`. All 13 committed files match the reviewed snapshot. Full local suite: 2985 passed, 26 skipped; two known sandbox stdout failures pass outside the sandbox. Coverage: 95.71%, down 0.20 percentage points from D1 same profile 95.91%, above 85%. Real Linux profile: 62 passed, one optional xxhash skip, three older model tests deselected. Python 3.12/3.13 each 66 passed, four Linux-only skips. Base optional-install checks and required installed-provider CI coverage pass. Native SHA1/English profile is qualified; broader model/provider acceptance remains R1. Active experimental presidio_gliner is explicitly unsupported in this recipe.
- D3 native checkpoint replay, private quarantine and caller reference matching passed review in `4ddb77a`. All nine committed files match the final reviewed snapshot. Full suite: 3013 passed, 26 skipped; two known sandbox stdout failures pass outside the sandbox. Coverage: 95.27%, down 0.44 percentage points from D2 same-profile 95.71%, above 85%. Final Linux native: 89 passed, one optional backend skip. Core-only: 23 passed, 66 optional skips. Python 3.12/3.13 each pass 94 targeted checks. Cached real ONNX interruption/resume passes on three artificial English examples. FIFO guards and immutable reference configuration close both Important review findings.
- S2 token-store operations and scoped source-security review passed in `7cecd66`. All eight committed changes and four security source references match reviewed hashes. Full suite: 3023 passed, 26 skipped; two known sandbox stdout failures pass outside the sandbox. Coverage: 95.31%, up 0.04 percentage points from D3 same-profile 95.27%. All ten SQLite ResourceWarnings are gone. No S2 warnings remain. Linux profile: 35 passed. Python 3.12/3.13 each pass 49 targeted checks without warnings. Keys, authorization, trusted POSIX filesystem and retention remain application-owned; this is not certification or a hosted service.
- E1 offline document extraction passed corrected independent review in `76f43a0`. All 17 committed paths match the approved snapshot. Native Python 3.12/3.13/3.14 each pass 122 tests; whole Docling module coverage is 88.70%. Full core: 3071 passed, 28 skipped, two known sandbox stdout failures pass outside sandbox. Coverage: 94.37%, down 0.94 percentage points from S2. English CPU extraction and cross-element replacement are qualified on a synthetic three-page fixture. Original binaries remain unredacted. Six pinned adapter furniture deprecations remain visible per native run.
- E2 optional credential-format detection passed independent review in `bfc6395`. All six committed paths match the frozen snapshot. Native Python 3.12/3.13/3.14 each pass 29 tests without warnings. Full suite: 3100 passed, 28 skipped, two known sandbox stdout failures pass outside sandbox. Coverage: 94.41%, up 0.04 percentage points from E1; new module 46/46 statements. Warning count stays 96. A six-snippet synthetic negative cohort flags one short Bearer header; inline Authorization headers can be missed. These scanner limits remain explicit.
- L1 opt-in language abstention passed corrected independent review in `dab6d4c`. All ten committed paths match final frozen runtime, metric and full-run hashes. Full suite: 3141 passed, 28 skipped, 96 existing warnings; two known sandbox stdout failures pass outside sandbox. Coverage: 6887/7290 = 94.4718793%, up 0.0667703 percentage points from E2 exact fractions. Native Python 3.12/3.13/3.14 each pass 148 targeted tests. Same 220 synthetic samples preserve legacy evidence. Thresholds 0.8/0.1 reduce known-language answers from 80 to 71 and outside-restriction answers from 140 to zero. This does not establish a production unknown-language guarantee; defaults stay legacy.
- L2 experimental routing passed corrected independent review in `b232507`. All eight committed paths match final reviewed/full/runtime hashes. Full suite: 3200 passed, 29 skipped, 96 existing warnings; two known sandbox stdout failures pass outside sandbox. Coverage: 7105/7515 = 94.5442448%, up 0.0723656 percentage points from L1 exact fractions. Native Python 3.12/3.13/3.14 each pass 202 tests with two recorded optional-provider skips. Actual handling policy, config and capability identities bind calibration; incomplete fast stages force full routing. Always-run output comparisons match all 24 cases apart from random map session IDs. No routing promotion claim is made.
- L3 saved-result validation completed on 2026-11-09 with explicit audit limits, recorded in `routing-validation-v1.json`. Frozen local checks are recorded in `l3-local-qualification.json`: 3229 passed, 30 skipped, 96 existing warnings; two known sandbox stdout failures pass separately outside the sandbox. R1 local preparation and frozen-patch quality review are complete, as recorded in `r1-local-qualification.json`. Final combined acceptance, whole-branch review and staged PR checks remain pending.

## Verified baseline commands

`.venv/bin/python -m pytest tests/ --cov=sct --cov-fail-under=85` ran before integrated processing changes. The P0 report retains the exact output and model-free/model-backed distinction.

Outside the filesystem sandbox:

```bash
.venv/bin/python -m pytest tests/test_atomic_file.py::test_dev_stdout_redirected_to_a_file_is_written_through tests/test_atomic_file.py::test_dev_stdout_piped_is_written_through -q
```

Result: 2 passed in 0.24s. No source changes were needed.

The supported Python/provider matrix, model quality comparisons, combined feature acceptance and independent final review are still required. This document is progress evidence, not a completion claim.

Fifteen implementation tasks have passed their review gates. This is not completion of the ten-capability programme. The remaining execution, adapter, corpus, language/routing and combined acceptance requirements stay active.

Server access recovered on a new bounded SSH probe. No VPN settings changed. Frozen C5 Linux/provider checks passed in `/home/rehan/sct-improvements-c5-2c7b7ad-20261107`: 258 tests on Python 3.14.4, including worker tests. All 19 source hashes match the reviewed commit. Intensive model evaluation remains pending.

Unified processing runs custom callbacks before all PII detection on the shared canonical text. Legacy processing runs callbacks between regex and NER stages. Final migration documentation and acceptance must cover callbacks that depend on rendered regex markers.

Frozen C5 actual framework checks passed on Python 3.12.15 and 3.13.12: 255 tests each, with three recorded warnings per run. `c5-python-qualification.json` records source hashes, runtime module paths and log hashes.

D1 selection plus frozen compatibility checks also passed on Python 3.12.15 and 3.13.12: 36 tests each. Runtime and source evidence: `d1-python-qualification.json`.

A frozen D2 cached ONNX smoke also passed offline on gpuserver: three artificial English examples, four protected gold entity strings, negative control unchanged, all three output IDs retained. `d2-cached-model-smoke.json` binds archive, runtime and model revision. This is not broad quality or final D3 replay evidence.

Pre-publication task decomposition now has separate foundation proof in `stack-foundation-qualification.json`. Fourteen private-index task trees reproduce the reviewed source without changing refs or the checkout. The rebuilt P0/C1/S1 foundation passes 2734 tests on Linux Python 3.14.4, with 64 skips and 92.16% coverage. Its 13 warnings include ten old test-connection warnings corrected later in S2. This older profile is not final integrated acceptance. Two accidentally tracked scratch reports will be omitted from published task trees while preserving local history.

R1 model preparation is recorded in `r1-model-preflight.json`. Python 3.14 CPU checks ran the real pinned GLiNER model and the trained FineWeb quality head with network connections blocked. GLiNER produced four canonical findings across two Torch calls; the quality scorer received original inputs and its three selection decisions matched actual scores. All three quality controls passed the predeclared threshold, with gibberish scoring highest. This is functional integration evidence, not quality accuracy. The trained ONNX quality export remains blocked by a missing exporter dependency. Broader runtime and combined acceptance remain pending.

The task-tree check now covers all 15 reviewed tasks through L2 at source commit `150242f`. The private-index reconstruction changes no refs, checkout files or real index. `stack-task-source-map.json` records all 15 task trees. The 12 trees after the reordered foundation match the original reviewed code, tests, scripts, CI and dependency files exactly. Remaining differences are the three final status surfaces, two root qualification files and two scratch reports excluded from publication. This proves source decomposition, not execution of CI on future PR heads.

On 2026-11-07, server access was unavailable at `192.168.15.152:22`; both configured aliases resolved to that endpoint. No calibration inference was repeated. R1 proceeded with independent local integration tests and migration documentation. Server access recovered on 2026-11-09.

R1 local preparation now passes its frozen-patch quality review without Critical, Important or new actionable Minor findings. Its full suite has 3236 passes, 31 skips and two known sandbox stdout failures; the two failures pass separately outside the sandbox. Coverage remains exactly 7105/7515 = 94.54424484364604%, a delta of 0.0 percentage points. The 106 warnings comprise 105 multiprocessing fork notices and one legacy configuration deprecation; the new native replay test adds ten fork notices. Ruff, pinned mypy, strict docs, locks, wheel/sdist builds and a clean core-only wheel installation/API smoke pass. The controller verified all 13 evidence hashes, all 15 current owned path hashes, exact coverage and the built wheel hash. Only the empirical acceptance append changed after the source gate; strict docs passed again.

Local preparation is complete, but R1 is not accepted. The trained exporter test remains skipped and the production exporter remains unchanged until a genuine remote failing test runs. Actual native provider/runtime and quality checks, final whole-branch review and actual PR-head CI remain required. On 2026-11-07, artifact-only retrieval attempt 32145 exited 255 with `Network is unreachable`, and the goal was marked blocked after repeated failures and completion of independent local work.

On 2026-11-09, SSH and saved-result retrieval passed. Remote verification matched all 70 source files, three input files, eight model files and the frozen runtime identity. Local validation checked development selection, consumption identity, all shared/full slice denominators and aggregate metrics, actual call totals and five promotion gates. Full-model calls fell from 750 to 690 (8%); median latency fell from 314.9782 to 313.5620 ms (0.4496%). Both configurations exposed PII in 74 of 110 synthetic documents. Default remains `always_run`. Raw predictions, individual latency samples and per-document exposure counters were not saved; independent rescoring of those observations and the paired interval is unavailable without replay. No replay occurred. This validates the saved report with audit limits and does not establish a production privacy or speed guarantee.
