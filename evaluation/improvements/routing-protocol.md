# Experimental routing measurement, version 1

Routing can avoid the second model when the first stage covers the requested entity types.
This experiment measures the privacy and speed tradeoff before any default change.
The default remains `always_run`, including when an experimental gate passes.

## Frozen scope

The development grid has exactly 12 candidates: confidence 0.90, 0.95, 0.99;
margin 0.10, 0.20; maximum size 256 or 512 Unicode code points.
The comparison baseline runs both configured stages.
No model threshold search, extra candidate, test tuning, or default migration is permitted.

The fast stage combines existing email, telephone and IP regex detection with
`protectai/bert-base-NER-onnx`, revision `84d28ab2bc28401b4353b26033eb043c8c8a0739`.
Its neural declaration covers English PERSON, LOCATION and ORGANISATION.
The full stage uses `rhnfzl/wikineural-multilingual-ner-onnx`,
revision `74191432b21c956b5b04b4778c0e92e569b1f80d`.
Its declared training languages are English, Dutch, German, Spanish, French,
Italian, Portuguese, Polish and Russian. Other measured languages are extrapolation.
These declarations describe model scope, not a measured privacy guarantee.
License fields remain unknown because cached snapshots contain no license document.
This experiment does not distribute model weights or approve their production use.

Both models run through native `GeneralNER`, with its unchanged score threshold 0.85.
Application wrappers have distinct capability and finding IDs and version `benchmark-1`.
Each model uses one shared pipeline across language aliases.
Existing `instrument_sessions` counts actual ONNX `session.run` invocations once per session.
Detector wrapper calls are not reported as neural inference calls.
The expensive-model gate counts the full-stage session calls, excluding warmups.

The cleaner uses unified processing with all optional text transforms disabled.
Only language detection and the three format detectors are enabled.
Canonical text therefore equals the input text, making gold code-point spans comparable.
The injected handling policy is `routing-benchmark` version `1`, with default protection rules.
Both configuration and calibration use `routing_config_digest(..., unified_policy=POLICY)`.
The fast capability digest binds the actual regex and model declarations.
The manifest separately binds actual source files, model cache bytes, runtime providers and input files.
Scoring identity includes `benchmarks/labels.py`, which normalizes gold spans and controls scoring denominators.
Capability identity does not attest installed SDK binaries.

## Input contamination audit

All 220 rows of `evaluation/multilingual_pii.jsonl` are development data.
L1 already measured their language outcomes with two threshold profiles.
They contain 11 languages, five domains, six annotated types and 30 code-switch rows.
The existing starter and release files are established evaluation inputs.
The comparison protocol has historical test sets, but their model outcomes were inspected in prior comparisons.
None is claimed as a newly untouched test set here.

`routing-test-v1.jsonl` is a new, independently authored synthetic cohort.
Its 110 rows use new template bodies, value pools and seed 20261107.
Each language has ten rows, with two value variants in each of five domain categories.
All six types are annotated in each row.
Five domain prefixes share one body template per language, so the resampling group is that language's template.
The source templates and value pools remain in `routing-input-provenance-v1.json`, separate from calibration and metric artifacts.
No test model outcomes may be inspected before development selection.
Pure span, partition and schema validation occurs before freezing.
Exact cross-partition source and ID overlap is rejected.
Language translations have not received independent native-speaker review.
The small synthetic cohort cannot establish production privacy or broad domain generalization.

Six deterministic mixed-entity fixtures are a separate regression gate.
They include names with email, organization, location and phone; a long document;
and mixed scripts. They are not used to select thresholds.

## Execution and measurement

Run only in the new L3 directory on gpuserver, using the qualified L1 virtual environment read-only.
The manifest records its actual executable, prefix, provider versions, paths and binary hashes.
Use CPU affinity 2-3, two ONNX intra-op threads and one inter-op thread.
Set `OMP_NUM_THREADS=2`, `OPENBLAS_NUM_THREADS=2`, `TOKENIZERS_PARALLELISM=false` and `HF_HUB_OFFLINE=1`.
Model snapshots must already exist. `freeze` requires local cache access and never downloads.

Run from the L3 source root:

```sh
python -m scripts.benchmark_detector_routing freeze
python -m scripts.benchmark_detector_routing dev
python -m scripts.benchmark_detector_routing test
```

The interpreter in these commands must be the manifest's qualified project interpreter.
Model initialization occurs before timing. Each policy warms up on two partition documents.
Each policy then executes three complete passes, in input order.
Accuracy uses the first pass; subsequent passes must produce identical findings.
The test stage is one evaluation with three predeclared timing repeats, not three tuning rounds.
An exclusive consumption file prevents automatic test replay after interruption.
Before creating that file, validate the development manifest, complete candidate grid, finite measurements,
slice eligibility and the recomputed selection or preregistered negative fallback.

Record median latency and nearest-rank p95 in milliseconds, across measured document calls.
Report total and per-document real inference calls, stage skips and routing reasons.
Language coverage uses every input document, including abstention and fallback outcomes.
Never calculate accuracy only on confidently detected languages.
CPU affinity and runtime are identical across policies; execution is sequential.
Fixed candidate order can carry time-dependent host effects, so small timing differences need replication.

Peak memory uses Linux `getrusage(RUSAGE_SELF).ru_maxrss`, in KiB.
This is a cumulative process high-water mark with both models resident.
Per-policy observations are not independent memory measurements or evidence of memory savings.

## Scoring and selection

Reuse `benchmarks.compare.scorer` for exact typed-span counts, masking exposure and paired bootstrap.
The shared neural taxonomy has PERSON, LOCATION and ORGANISATION.
The full annotated taxonomy additionally has EMAIL, PHONE and IP.
Report the views separately, with every declared language, entity and domain slice.
Ignore predictions of unannotated types instead of treating them as false positives.
Report support counts and null when a denominator is empty.
Report unsupported language declarations separately from observed performance.

Document exposure means at least one alphanumeric gold character survives the union of protected findings.
Precision and recall use exact typed spans; document exposure is type-agnostic masking coverage.
A development candidate is eligible only when it adds no exposed document on any declared slice.
Entity slices check exposure for that entity, even when another entity already exposes the document.
Improvements on one document cannot cancel regressions on another.
Among eligible candidates, choose the lowest measured median latency.
Break ties by fewer skipped stages, then stable candidate ID.
Preserve every candidate report, including failures and candidates with zero skips.

If no development candidate qualifies, evaluate `c99-m20-s256` as a fixed negative comparison.
This fallback is preregistered and is never presented as a selected or promoted candidate.
Evaluate baseline and the chosen comparison once on test, then the mixed fixtures.

Promotion requires all four original conditions:

1. The upper paired 95% interval for added document exposure is at most zero.
2. No additional gold mention becomes exposed in the deterministic mixed fixtures.
3. The expensive full model executes strictly fewer actual inference calls.
4. Measured median document latency is strictly lower.

An additional required acceptance guard rejects added exposure in every declared held-out language, entity and domain slice.
This guard preserves entity-specific failures when another entity already exposes the document.
It supplements the paired document interval and keeps shared/full reports separate.
The guard was clarified during independent review before held-out consumption, without changing inputs, models or the grid.

The paired interval uses 1,000 seeded template-group bootstrap draws and language-macro document exposure.
Candidate minus baseline defines the difference. Intervals remain visible when zero is included.
An interval crossing zero is inconclusive if the other gates pass.
A positive lower bound or a failed deterministic or efficiency gate is a failure.
A failed held-out slice guard is also a failure.
A no-eligible-development result always fails promotion.
The empirical bootstrap can degenerate to zero with identical observations; this does not prove zero population risk.
Retain negative and inconclusive results without generating another test set or retuning.

## Artifact and completion boundaries

Manifest, calibrations, development and held-out results are versioned JSON.
Calibration and metric artifacts contain no source documents, entity values or reversible maps.
Opaque document IDs and hashes still require appropriate storage controls.
Default behavior, legacy fields, imports and replacement modes remain unchanged.
This ONNX experiment does not qualify a non-ONNX NER backend or a neural quality scorer.
Those remain separate R1 acceptance work.
