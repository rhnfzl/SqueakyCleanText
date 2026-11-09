# Current-source routing experiment, v0.9.0 version 2

Routing can avoid the second model when the first stage covers the requested entity types.
This experiment measures that tradeoff against the configured two-model baseline.
The default remains `always_run`, regardless of the result.

## Status and immutable boundaries

This is a preregistration, not an empirical result.
The implementation base is published v0.9.0, commit `882342fd08c03718d21d22278814cfbfe4b883f3`, plus the planned improvements.
The current routing configuration differs from the historical v1 configuration.
Historical calibration must continue to reject that mismatch.

The v1 inputs, manifest, calibrations, consumption marker and results remain unchanged.
The old server directory `/home/rehan/sct-improvements-l3-20261107` remains untouched.
Neither the v1 held-out cohort nor its observed outcomes can qualify this source.
Current code refuses v1 execution, including the CLI default.
Historical source identity refers to the original frozen archive, not this refactored driver.

The coordinator must clear final source and an independent protocol review before runtime freeze or inference.
Preparation permits input validation and synthetic audit tests only.
All heavy execution belongs in a new owned directory, proposed as `/home/rehan/sct-improvements-l3-v090-v2-20261109`.
The earlier L3 and comparison directories must not be reused.
After runtime freeze, do not alter models, source, inputs, thresholds, grid, selection or gates to obtain a preferred result.
A failed or incomplete experiment is a valid research delivery.

## Fixed models, policy and candidate grid

The fast neural model is `protectai/bert-base-NER-onnx`, revision `84d28ab2bc28401b4353b26033eb043c8c8a0739`.
The full model is `rhnfzl/wikineural-multilingual-ner-onnx`, revision `74191432b21c956b5b04b4778c0e92e569b1f80d`.
Both use native `GeneralNER`, CPU execution, and its unchanged 0.85 score threshold.
Existing email, phone and IP regex detectors form the rest of the fast stage.
Each neural stage shares one pipeline across its language aliases.
Existing `instrument_sessions` wraps each actual ONNX session once and counts `session.run` calls.
Wrapper invocations are not substituted for neural calls.

The fast neural declaration covers English PERSON, LOCATION and ORGANISATION.
The full declaration covers English, Dutch, German, Spanish, French, Italian, Portuguese, Polish and Russian.
Other measured languages are extrapolation, not declared neural coverage.
License identities remain unknown; this experiment neither distributes weights nor approves their production use.

The cleaner uses unified processing with optional text transforms disabled.
Only language detection and email, phone and IP format detection are enabled.
All offsets refer to unchanged input text in Python Unicode code points.
The application handling policy remains `routing-benchmark` version `1`, with default protection rules.
Application detector capabilities remain version `benchmark-1`.
Those interface versions do not identify an experiment or attest installed binaries.
Every calibration binds the current semantic configuration, handling policy and actual fast-stage capability digest.
The experiment manifest binds actual source, protocol, inputs, cache bytes and runtime separately.

The grid is exactly confidence `{0.90, 0.95, 0.99}` times margin `{0.10, 0.20}` times maximum characters `{256, 512}`.
These twelve candidates execute in that stable product order.
The configured baseline always executes both model stages.
This baseline is not SCT's default model profile.

## Data and contamination boundaries

Development reuses all 220 previously inspected records in `evaluation/multilingual_pii.jsonl`.
No development record is treated as a new untouched test example.

`routing-test-v2.jsonl` contains 110 newly authored synthetic records.
Each of eleven languages has five domain prefixes and two entity-value variants.
There are eleven new receipt-and-dispatch body templates, one per language.
Every row annotates PERSON, ORGANISATION, LOCATION, EMAIL, PHONE and IP exactly once.
The generation seed is `20261109`; it fixes the input order before any model outcomes are observed.
IDs, body templates and value pools are new.
Contacts use `example.test`, documentation IP ranges, and fictional UK drama phone numbers.
The input provenance stores the body templates, value pools, rendering rules and input hashes separately from metrics.

Five prefixes share each language's body template.
Therefore the bootstrap has eleven template clusters, not fifty-five independent domain clusters.
The cohort has no independent native-language expert review.
Names and places are illustrative, and the synthetic texts cannot establish production privacy or domain generalization.

Six newly authored mixed-entity regression fixtures are separate from selection.
They include a long document and mixed scripts.
Their entity values can overlap the v2 cohort, but their source text and IDs cannot.
They are known deterministic fixtures, not an independent estimate of generalization.

Pure validation precedes freezing: schema, code-point bounds, unique source and IDs, exact row counts, six types, and language/domain balance.
Exact source and ID overlap is rejected across development, v2 test, v2 mixed, v1 test and v1 mixed.
The v1 files are read only for this partition check, not inference or candidate selection.
No v2 model outcomes may be inspected before development selection.

## Runtime, timing and retained observations

The manifest records the qualified project interpreter, its binary hash, module paths, package versions and binary hashes.
It also records every cached model file, source file, protocol, provenance and input hash.
Scoring identity includes `benchmarks/compare/scorer.py`, `benchmarks/score.py` and `benchmarks/labels.py`.
No model download is allowed. Freeze resolves only existing local snapshots.

Use CPU affinity 2-3, two ONNX intra-op threads, and one inter-op thread.
Required environment: `OMP_NUM_THREADS=2`, `OPENBLAS_NUM_THREADS=2`, `TOKENIZERS_PARALLELISM=false`, `HF_HUB_OFFLINE=1`, `CUDA_VISIBLE_DEVICES=`.
The driver rejects a different platform, affinity or environment before model initialization.
Runtime and frozen file identities are checked before and after inference.
Verification derives complete source, input and actual cached-model inventories independently of the manifest.
It also checks every fixed experiment field, calibration contents and the required CPU/environment envelope.
Missing entries, contradictory declarations and changed bytes fail before phase consumption.

Model initialization occurs before timing.
Each policy warms up on the partition's first two documents, then runs three complete timed passes in input order.
The test phase is one evaluation with three preregistered timing repeats, not three selection rounds.
Predictions must match across all three passes.
There is no automatic inference retry or resume after a consumed phase fails.

Each completed repeat is saved atomically and synchronized to disk before scoring.
After the last inference in a phase, full source/input/cache/runtime verification must pass before any final summary is written.
The inference path then atomically saves a synchronized phase-completion receipt.
The receipt binds the manifest, exclusive consumption marker and hashes of every expected raw observation file.
Postflight failure or interruption before that receipt leaves the phase incomplete, even when all raw observations exist.
Scoring starts only after the receipt is durable, so later postprocessing failures can be recovered without inference.
Each document observation contains only its opaque ID, positional predictions (`label`, `start`, `end`), duration, actual model-call deltas and structural route/language metadata.
Each repeat also stores cumulative counter start/end values, peak RSS, experiment identity and repeat index.
Actual per-document deltas must reconcile with counter totals for every repeat.
These records retain every measured duration and prediction, including fallback and unsupported-language outcomes.
Metric and calibration artifacts contain no source text, entity values or reversible maps.
IDs, offsets and file hashes still require appropriate storage controls.

The reported median and nearest-rank p95 use all measured document durations, excluding warmups and initialization.
Actual call totals use all three passes without assuming equal repeat counts.
Language coverage is a descriptive first-pass ratio. Accuracy retains every document.
Fixed sequential order and host drift can affect small latency differences; no latency uncertainty interval is claimed.

Peak memory is Linux `getrusage(RUSAGE_SELF).ru_maxrss`, in KiB.
It is a cumulative process high-water mark with both models resident.
It does not measure policy-specific memory use or memory savings.

## Scoring, selection and promotion

Reuse the existing scorer for exact typed-span counts, masking exposure and paired bootstrap.
Report shared neural taxonomy and full annotated taxonomy separately.
Every overall, language, entity and domain row retains support, precision, recall and document exposure.
Empty denominators remain null. Unannotated predicted types are ignored.
All 110 held-out records, including unsupported languages and abstentions, remain in accuracy denominators.

Document exposure means at least one alphanumeric gold character survives the union of protected findings.
It is type-agnostic masking coverage; precision and recall use exact typed spans.
Entity exposure is checked even when another entity already exposes the same document.
Regressions on one document cannot be cancelled by improvements on another.

Development eligibility requires zero added exposed documents in every language, entity and domain slice, in both taxonomy views.
Choose the eligible candidate with lowest median latency; ties prefer fewer skipped stages, then stable candidate ID.
Retain all twelve candidate reports, including no-skip, negative and inconclusive observations.
If none is eligible, use `c99-m20-s256` only as the fixed negative comparison.
A negative comparison can never promote.

Before test consumption, reconstruct all development reports from raw observations and recompute eligibility and selection.
Reject any disagreement with the saved development report.
Create and synchronize an exclusive test-consumption marker before any test inference.
The marker binds candidate, development report and manifest hashes and survives interruptions.

Promotion requires all of these conditions:

1. The upper paired 95% added-exposure interval is at most zero in both shared and full views.
2. No additional annotated mention becomes exposed in the mixed fixtures.
3. Actual full-model calls are strictly fewer across all three held-out repeats.
4. Median held-out document latency is strictly lower.
5. Every held-out language, entity and domain slice has zero added exposure in both views.
6. Development produced an eligible selected candidate.

Intervals use 1,000 seeded paired template-group bootstrap draws and language-macro exposure, candidate minus baseline.
The bootstrap seed is `20261109`; the scorer's percentile and rounding behavior is frozen by source hash.
A zero-crossing interval is inconclusive if every deterministic and efficiency gate passes.
A positive lower bound or a failed deterministic/efficiency gate fails promotion.
Identical observations can produce `[0, 0]`; that does not establish zero population risk.
No result changes the default or permits retuning this experiment.

## Execution and audit commands

These commands require prior source clearance and the new owned server directory.
`python` below means the exact qualified project interpreter recorded in the manifest.
Apply the required environment and CPU affinity to every inference command.

```sh
python -m scripts.benchmark_detector_routing validate-inputs --experiment v090-v2
python -m scripts.benchmark_detector_routing freeze --experiment v090-v2
python -m scripts.benchmark_detector_routing dev --experiment v090-v2
python -m scripts.benchmark_detector_routing audit-dev --experiment v090-v2
python -m scripts.benchmark_detector_routing test --experiment v090-v2
python -m scripts.benchmark_detector_routing audit-test --experiment v090-v2
```

Audit commands instantiate no models and replay no inference.
They validate saved observations, recompute summaries and gates, and reconstruct a missing summary after postprocessing failure.
Audit requires a valid existing phase-completion receipt; it never creates postflight proof.
An existing summary must match exactly; audit never overwrites a conflicting report.
Audit validates fixed manifest/calibration declarations, complete current source/input inventories and required pinned-model file declarations.
Each pinned snapshot must declare its model, config, tokenizer and tokenizer-config files.
Audit does not access cached weights or compare current SDK/runtime identities, including on a copied source tree.
Runtime and full actual model-cache verification are recorded postflight provenance, not checks independently repeated by numeric audit.
The receipt is a local evidence record, not cryptographic remote attestation; it assumes trusted artifact storage.
If any required repeat or valid completion receipt is absent, audit fails and the consumed phase remains incomplete.
Do not remove consumption markers or rerun inference to repair an incomplete result.

Empirical documentation must retain negative and inconclusive results and state the synthetic, runtime and memory limits above.
This experiment does not qualify other neural backends, quality scorers, production privacy or real-world speed.
