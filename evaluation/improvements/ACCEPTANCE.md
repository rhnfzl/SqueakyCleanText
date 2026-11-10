# Combined acceptance evidence

## Current v0.9.0 port

The ten capabilities are ported onto released v0.9.0, commit `882342fd08c03718d21d22278814cfbfe4b883f3`.
The original research checkout and its historical evidence remain preserved.
All 17 tasks and 79 requirements have source and runtime qualification. PRs #68 through #84 passed their checks and merged into main.
Post-merge package and docs CI passed. [GPU qualification](v090-gpu-qualification.json) records inference, quality parity and dataset resume on one prepared RTX 4090 profile.

Current records:

- [Requirement ledger](acceptance-ledger.json) separates historical and current acceptance.
- [Model qualification](v090-model-qualification.json) records actual provider inference and strengthened trained-export parity across three Python versions.
- [Native profiles](v090-native-profiles-qualification.json) records document, secret, framework, corpus, storage and worker checks.
- [Controlled baseline](v090-baseline-qualification.json) compares identical Python and dependency profiles without rewriting the release owner's historical record.
- [Routing validation](routing-validation-v2.json) reconstructs saved observations without inference replay.
- [Current roadmap](../../docs/human-html/2026-11-07-plan-sct-improvements-roadmap.html) presents the human review status.

The v2 synthetic routing experiment passes its preset relative gates.
Expensive calls fall from 750 to 690; median latency falls from 395.409 ms to 381.348 ms.
Both policies still expose 78 of 110 held-out documents and three of six mixed fixtures.
The default remains `always_run`. These observations do not establish production privacy or general speed.

The independent whole-branch review found one cache identity defect and three smaller integration defects.
Their fixes pass focused checks and independent scoped review.
Two malformed-checkpoint failures found during acceptance now return controlled errors.
Python 3.14 passes 3,271 tests with exact coverage `6478/7502 = 86.3503065849107%`.
The threshold stays 85%. Two-decimal report precision prevents the previous rounded false pass.
All three final isolated core profiles pass 3,271 tests. Python 3.12 and 3.13 cover `6695/7717 = 86.75651159777115%`.
All 17 exact task prefixes pass full Linux Python 3.14 and static checks.
The [24 additional boundary cases](v090-boundary-test-qualification.json) qualify core orchestration after adapter preflight.
They also pass with SDKs installed and restore the native adapter module. Runtime package files remain unchanged.
Existing real native adapter profiles pass 328 tests per version and retain their original source identities.

The port retains all 124 existing configuration defaults and appends 13 opt-in fields.
It preserves the v0.9.0 API removals, explicit configuration, `max_workers` and shared processing helpers.
See the [port addendum](../../docs/superpowers/plans/2026-11-09-sct-improvements-v090-port.md) for the current compatibility ruling.

## Historical integration record before the v0.9.0 port

The remaining sections retain the original qualification record and its original source identities.
Pending statements below describe that historical snapshot. They are superseded only by explicitly scoped current records above.

The ten capabilities are implemented through reviewed component tasks, but combined release acceptance remains pending.
No release, default migration, push or pull request is authorized by this evidence record.
The canonical requirement ledger is [acceptance-ledger.json](acceptance-ledger.json).
It retains all 17 tasks and 79 requirements; this document explains their integration evidence and limits.

Baseline: `8afef61c74ea738a2c904374776dea34d9c1c39e` (0.8.3).
Implementation branch: `research/pii-routing-landscape`.
Evidence is specific to the source hashes, runtime paths and artifacts recorded in each qualification file.
Configured CI is not executed CI. Historical component checks are not the final integrated matrix.

## Capability ledger

| Feature | Component evidence | Compatibility and remaining acceptance |
|---|---|---|
| 1. Unified findings, policy and rendering | C1, C2, C4; `tests/test_unified_processing.py`, `c4-python-qualification.json` | Legacy remains default. Combined coordinates, policy output and map references pass locally. Final runtime/provider matrix remains pending. |
| 2. External detectors and worker factories | C2, C3; `tests/test_detector_registry.py`, `c3-schema-qualification.json` | Trusted factories and data-only specs; no request-selected imports. Actual inference is not yet qualified for every native backend. |
| 3. Quality and duplicate selection | D1-D3; `d1-python-qualification.json`, `r1-model-preflight.json` | Report-only default; explicit removal basis. Trained Torch scoring is functional, but gibberish scored highest. Trained ONNX export/parity remains pending. |
| 4. Native corpus recipes and replay | D2-D3; `d2-python-qualification.json`, `d3-python-qualification.json` | Local combined publication interruption/replay passes. Final Linux/runtime matrix remains pending; private staging is caller-owned. |
| 5. Caller schema and structured adapters | C5; `c5-python-qualification.json`, `tests/test_schema_context.py` | Whole-value protection and direct context pass combined checks. Frameworks retain reversible restrictions. |
| 6. Encrypted token storage | S1-S2; `s2-python-qualification.json`, `tests/test_token_store_operations.py` | Combined direct restoration and value-free audit checks pass. Keys, authorization, backups and retention remain application-owned. |
| 7. Document extraction and projection | E1; `e1-python-qualification.json`, `tests/test_docling_integration.py` | Actual three-Python English CPU PDF fixture is qualified. Combined local checks use a deterministic extractor, not another native OCR qualification. |
| 8. Adaptive detector routing | L2-L3; `l2-python-qualification.json`, `l3-local-qualification.json` | Implementation stays experimental. Saved remote held-out and consumption artifacts still need retrieval and independent validation. No optimization or promotion claim. |
| 9. Credential-format detection | E2; `e2-python-qualification.json`, `tests/test_secret_detector.py` | Actual three-Python four-format checks pass. One short Bearer false positive and an inline Authorization miss remain documented. |
| 10. Language uncertainty | L1-L3; `l1-python-qualification.json`, routing protocol | Legacy remains default. Paired synthetic language metrics exist; neither unknown-language accuracy nor routing promotion follows from them. |

Component review status and exact counts remain in [PROGRESS.md](PROGRESS.md).
Independent whole-branch review and final staged-head verification remain required for every feature.

## Combined checks

`tests/test_improvements_end_to_end.py` exercises these actual contracts:

- Extracted typed elements enter context-aware unified processing with canonical and original-input ranges.
- Direct reversible output returns an encrypted SQLite reference that restores the supplied text.
- Non-reversible document projection and JSON field schemas remove the selected synthetic values.
- Adapter and document reversible refusals remain explicit.
- Real datasketch selection retains one stable record, publishes checksums and validates the completed manifest.
- Real DataTrove publication is interrupted, resumed and compared byte-for-byte with an uninterrupted run.
- Legacy callbacks receive regex markers; unified callbacks receive original values before detection.
- Default findings, metrics, storage audit, document fields, decisions and manifests omit the checked matched values.
- Rendered extraction, document and unified-callback error traces suppress provider values and chained exception display.

The local focused run passes 11 tests, with one opt-in trained-export skip and ten existing multiprocessing fork warnings.
This includes seven combined cases and four frozen compatibility cases on Python 3.14.3.
The first combined run had an incorrect test expectation about findings order; keyed range assertions corrected the fixture.
No production defect or test-driven exporter fix is claimed by that correction.

`tests/test_quality_export.py` requires explicit pinned trained artifacts and the locked Torch/quantize profile.
It runs the actual export script offline, loads the exported ONNX scorer and compares Torch scores across three batch/sequence shapes.
Its tolerance is `rel=1e-4, abs=1e-5`. Artifact hashes and immutable revisions are checked before execution.
Local skip is not export evidence. The genuine failing exporter run and corrected trained parity check remain pending remote access.

## Privacy and unsupported combinations

Public value-free fields do not make an entire workflow anonymous.
Caller identifiers, hashes, model names and stable record IDs can identify private work.
Extracted input, source binaries, explicitly requested values and reversible maps remain sensitive.
Private corpus SQLite files can retain original text under `pre_pii`.
Exception internals and debugger locals can retain suppressed source errors; public trace suppression is not memory erasure.

Document projection accepts unified placeholder or synthetic output and requires proven source associations.
It has no context/schema argument; direct processing of `document.text` supports caller context without element projection.
Indexed/reversible documents, reversible framework adapters, unified windowed streaming and unified browser manifests remain unsupported.
Opaque callbacks cannot satisfy strict source mapping or document projection.
Schema hints require context-capable detectors; whole-value protection rejects conflicting policy or allow-term settings.
Network-declared detectors require explicit permission. Unknown or incomplete fast-stage capability requires full routing.

Native corpus recipes reject active `presidio_gliner`.
[r1-presidio-source-audit.json](r1-presidio-source-audit.json) verifies installed Presidio 2.2.362 and spaCy 3.8.16 sources inside the project venv.
The recognizer accepts model kwargs, but SCT does not pass `gliner_revision`.
The default analyzer can load/download an additional spaCy NLP model and enable other recognizers.
This is source evidence, not an actual Presidio inference qualification or evidence that SDK revision pinning is impossible.

SCT candidate selection confirms exact Jaccard similarity after probabilistic MinHash candidate retrieval.
Native DataTrove bucket matching is approximate and does not apply that same exact threshold.
No exactly-once, distributed execution, original-binary redaction or comprehensive credential-detection claim is made.
Scoped token-store security review is not a full-repository or infrastructure certification.

## Rollback and publication gates

For legacy processing, clear detector/routing specs, remove unified policy/context/schema arguments, and set `pii_processing='legacy'`.
Restore `language_policy='legacy'` and unset confidence/margin thresholds for legacy language selection.
For full execution with explicit unified groups, set `routing_policy='always_run'` and clear the calibration artifact.
For corpus retention, return selection to report mode and manage existing private state separately.
Rollback cannot restore removed source values; reprocess retained inputs.
Token deletion is logical and cannot erase existing journals, backups or media.

Core and docs PR filters now include `improvements/**`; push/deployment branches remain unchanged.
Packaging must carry that policy in the P0 foundation so every stacked task can receive core and docs checks.
Optional CI adds combined token-store/corpus checks and trained export/parity checks across Python 3.12, 3.13 and 3.14.
These workflow changes do not establish executed CI or stage-head acceptance.

The final gates still require:

1. Retrieve and validate the saved L3 held-out and consumption artifacts without rerunning or tuning the held-out experiment.
2. Run the real trained exporter red/green test and qualify Torch/ONNX numerical parity on the locked runtime.
3. Complete all supported Python/provider inference checks, Linux worker checks and final whole-suite checks on frozen source.
4. Complete lint, typing, locks, package build/install, strict documentation and current quality-baseline comparisons.
5. Obtain independent whole-branch review, prepare the reviewed staged trees and verify their actual CI heads.

The previous exact core baseline is `7105 / 7515 = 94.54424484364604%` coverage.
The latest frozen L3 local suite passed 3229 tests with 30 skips and 96 existing warnings, plus two sandbox-only failures.
Those two atomic stdout tests pass separately outside the sandbox; this is split evidence, not one green invocation.
The R1 final integrated run and exact coverage delta remain pending.

## Frozen local R1 gate result

The first full R1 local run finished with 3236 passed, 31 skipped, 106 warnings and the same two sandbox stdout failures.
Both named atomic stdout tests then passed outside the sandbox in 0.42 seconds.
This remains split evidence; the full invocation exited with those two failures.
Coverage is exactly `7105/7515 = 94.54424484364604%`, a change of `0.0` percentage points from the preceding baseline.
The 106 warnings comprise 105 multiprocessing fork notices and one legacy `ner_models_list` deprecation.
The new native replay acceptance case accounts for ten additional fork notices. None were suppressed.
All 15 frozen paths matched their recorded hashes through completion.

Full Ruff, pinned mypy across 63 source files, strict MkDocs, lock validation and wheel/sdist builds pass locally.
The all-extras lock export has identical payload bytes and parsed TOML; only its generated command comment names a different temporary destination.
A fresh core-only wheel installation passes isolated import, unified contact protection, field-schema and three-value unpacking checks on Python 3.14.3.
The new runtime modules are present in the wheel; scratch reports are absent from both wheel and sdist.
No Torch, Docling, DataTrove or GLiNER dependency was installed in that wheel environment.

These results close the local preparatory checks only.
The trained exporter test remains explicitly skipped locally, and the exporter source remains unchanged pending a genuine remote failing test.
All remote runtime/provider, L3 artifact validation, final integrated acceptance and review/publication gates above remain open.
