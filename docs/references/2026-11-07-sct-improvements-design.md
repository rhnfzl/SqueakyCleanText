# SCT improvement programme: design specification

This specification turns all ten researched improvements into implementable workstreams. It is a proposed design, not a claim that the interfaces already exist. No implementation is authorised by creating this plan.

## Scope and compatibility

- Baseline: SCT v0.8.3, commit `8afef61c74ea738a2c904374776dea34d9c1c39e`. Rebase and re-audit changed files before execution.
- Python floor stays 3.12. Test supported 3.12, 3.13 and 3.14. Keep the base install free of new mandatory document, storage, corpus or secret-scanner dependencies.
- Keep existing public imports, 3-tuple ProcessResult unpacking, default outputs, replacement modes, presets, legacy globals, worker ownership and adapter restrictions.
- New unified PII processing is opt-in: `TextCleanerConfig.pii_processing='legacy'` by default; `'unified'` selects the new path. Do not silently reinterpret the existing PrivacyPolicy or DetectedEntity.
- Existing `PrivacyPolicy`, NER-only/post-regex findings and their explicit opt-in sensitive metadata remain compatible. New unified findings contain no entity text by default.
- Every new cleaner behavior has a config field, validation, serialization rules and the corresponding legacy-global treatment. External integration options remain explicit at their own entry points.
- Use current patterns and detectors. Never copy built-in word exceptions, field-name keyword tables or model-specific word lists. Caller-declared schema and terms are allowed.
- No default network detector, hidden model download, silent dependency fallback, raw-text tracing or automatic input deletion. Native and external model licenses are separate.
- Work on feature branches/worktrees. Each task is a reviewable PR. Existing main/develop branches receive no direct push. Do not publish releases or posts as part of implementation.

## Core contract, proposed names

All offsets are zero-based, end-exclusive Python Unicode code-point offsets, relative to an explicitly named text stage. Byte and UTF-16 adapters must convert and validate before returning a Detection.

| New type or interface | Exact meaning |
|---|---|
| `sct.findings.TextRange(start: int, end: int)` | Validate nonnegative start and end > start; bounds are checked against the actual stage text. |
| `Detection(entity_type: str, start: int, end: int, detector_id: str, score: float | None, coordinate_space: str, source_ranges: tuple[TextRange, ...] = (), mapping_quality: Literal['exact','covering','unavailable'] = 'unavailable')` | Frozen, value-free finding. Validate finite scores in [0,1]. None is unknown, not zero. Source ranges refer to the original string supplied to SCT, not PDF bytes. |
| `DetectionContext(document_id: str | None = None, path: tuple[Hashable, ...] = (), declared_entity_types: tuple[str, ...] = (), schema_id: str | None = None)` | Immutable caller context. Do not serialize paths or context automatically; paths can themselves contain PII. |
| `sct.coordinates.SourceMap.identity(text: str) -> SourceMap` | Initial mapping; retain text privately, with repr disabled. |
| `SourceMap.replace(start: int, end: int, replacement: str) -> SourceMap` | Return a new map after a proven edit. Compositions retain conservative source spans. |
| `SourceMap.project(start: int, end: int) -> Projection` | Projection has ranges and exact/covering/unavailable quality. Never invent an exact position after an opaque transform. |
| `DetectorCapabilities` | Frozen fields: id, version, entity_types, languages, coordinate_system='unicode_codepoint', supports_context, supports_windowed, network_access. Include model revision/license identity where relevant. |
| `Detector.detect(text: str, *, language: str | None, context: DetectionContext) -> tuple[Detection, ...]` | One canonical stage input; return findings without replacement. Capabilities are exposed as a read-only property. Missing optional dependencies raise DetectorUnavailable. |
| `DetectorSpec(name: str, options: Mapping[str, JSONValue])` | Immutable, serializable non-secret configuration. Runtime credentials are separate injected dependencies. |
| `DetectorRegistry.register(name: str, factory: Callable[[TextCleanerConfig, Mapping[str, JSONValue]], Detector]) -> None` | Explicit application-owned registration. Duplicate names fail. Freeze the registry before processing. No automatic execution of modules named by an untrusted request. |
| `DetectorRegistry.create(spec: DetectorSpec, cfg: TextCleanerConfig) -> Detector` | Validate the name/options and build worker-local instances. Registry identity is included in private run metadata. |
| `UnifiedPrivacyPolicy(name: str, version: str, rules: Mapping[str, HandlingRule])` | New policy, separate from existing PrivacyPolicy. |
| `HandlingRule(action: Literal['use_mode','keep','redact']='use_mode', min_score: float | None=None, unknown_score_action: Literal['protect','error']='protect')` | use_mode delegates to global placeholder/synthetic/reversible mode. keep suppresses PII replacement, not independent text normalization. redact is irreversible removal/replacement as documented, even in reversible mode. |
| `TextCleaner(..., detector_registry: DetectorRegistry | None = None, unified_policy: UnifiedPrivacyPolicy | None = None)` | Add keyword-only injected dependencies. A non-None unified_policy requires pii_processing='unified'. Existing constructor arguments remain intact. |
| `TextCleaner.process(text: str, *, context: DetectionContext | None = None) -> ProcessResult` | Existing behavior when context is omitted and legacy mode is selected. Context requiring the unified detector interface cannot be silently ignored. |
| `TextCleaner.process_batch(texts: list[str], max_workers: int | None=None, *, batch_size: int | None=None, contexts: Sequence[DetectionContext] | None=None) -> list[ProcessResult]` | Keep batch_size's deprecated alias behavior. Validate contexts length; order and context ownership match input rows. |

### Unified execution semantics

1. Preserve cleanup ordering and options. Built-in transforms record proven edits. Arbitrary custom string callables mark source mapping unavailable; they do not disable detection.
2. In unified mode, enabled regex PII operations collect typed spans instead of writing markers before NER. Reuse existing contact/date/number validation and custom-pattern rules; do not clone regex logic.
3. Add read-only span analysis to GeneralNER and reuse it from legacy rendering. Keep window merge, offsets, voting, word boundaries and Presidio behavior. Split helpers into existing ner_spans.py rather than growing ner.py.
4. All selected detectors analyse the same canonical pre-replacement stage. Preserve detector provenance; backend scores are not assumed calibrated across models.
5. Apply policy once. For overlapping protected findings, redact dominates use_mode; protection dominates a conflicting keep. Union source cover deterministically, then tie-break by span length, start, detector ID and entity type. Do not compare incomparable confidence scores to choose a winner.
6. Render once using the existing placeholder, synthetic and AnonymizationMap mechanisms. Reserve markers present in source; source text resembling a marker is not automatically trusted.
7. New `metadata['findings_v2']` contains only canonical identifiers, positions, mapping quality, action and reason. Sensitive values require the existing explicit sensitive-output contract, never a new default. Count resolved replacements separately from detector hits.
8. Existing legacy signals stay byte/structure-compatible. Define and version unified counts separately; never silently change an existing metric's denominator.

### Coordinate honesty

A many-to-one or one-to-many normalization can yield a covering range, not a bijection. An opaque transform yields unavailable mapping. Add `source_mapping='best_effort'|'require_source'`, default best_effort in unified mode. Strict source mapping rejects unsupported opaque transforms explicitly. Whole-pipeline raw-offset proof is not possible for arbitrary string callbacks without a mapping-aware interface.

### Runtime, workers and surfaces

Detector instances and model sessions are process-owned. Extend CleanerHandle with a trusted, top-level `registry_factory(cfg) -> DetectorRegistry`; recreate inside every worker. Never pickle live model sessions, keys or a global mutable registry. In-process process_batch shares only explicitly thread-safe providers or uses per-thread instances.

| Surface | Required support |
|---|---|
| process/process_batch | All core options; identical batch/single semantics and stable ordering. |
| JSON/DataFrame/RAG | Forward caller schema/path and factory context. Preserve key handling, duplicate-key rejection, selected columns and input-copy behavior. Keep the existing reversible-mode refusal. |
| HF/DataTrove/LangChain/Haystack | Preserve worker-local lifecycle. Serialize config and a trusted factory reference only. Server/client inputs cannot request arbitrary imports. |
| MCP/CLI | Expose supported data-only configuration. Registry selection is server/application-owned and whitelisted. Do not expose factory import paths as request arguments. |
| Buffered streaming | Run the same document pipeline when complete; retain current mode restrictions. |
| Windowed streaming | Keep documented bounded/contact-token scope. Reject incompatible whole-document, registry, context or mapping options explicitly. Full general NER streaming is outside these ten improvements. |
| Browser manifest | Keep its existing restricted feature set. Reject unsupported unified/custom/provider configurations rather than exporting an incomplete manifest. Browser parity is not promised. |
| OCR/document extraction | Coordinates refer to extracted text and element/page provenance. No claim of byte-level PDF coordinates or safe original binary redaction. |

## Corpus contracts

- `sct.corpus.types.CorpusRecord(id: str, text: str, metadata: Mapping[str, JSONValue])`: unique stable IDs; input text is sensitive and excluded from repr/logging by default.
- `SelectionConfig`: quality_min=None, dedup_mode='report', dedup_basis=None (report mode resolves None to 'pii_normalized'; apply mode rejects None), similarity_threshold=0.9, keeper='min_stable_id'. All values validated. Removal requires an explicitly chosen basis and mode='apply'; never overwrite inputs.
- `analyse_records(records: Iterable[CorpusRecord], *, cleaner: TextCleaner, score_config: ScoreConfig, selection: SelectionConfig) -> Iterator[RecordAnalysis]`: stream text through existing scoring; quality filtering requires a configured scorer when quality_min is non-None.
- `select_records(analyses: Iterable[RecordAnalysis], *, selection: SelectionConfig) -> SelectionReport`: external/disk-backed matching, deterministic keeper, stable decision IDs and reason codes. Dataset-scale memory must not depend on holding all source text.
- `RecordAnalysis` carries record identity, scores, versioned signature identity and a private output reference, not a retained copy of raw text.
- SCT datasketch MinHash signatures are not interchangeable with DataTrove's signature format. A DataTrove recipe either uses its complete native dedup workflow on a declared text basis, or explicitly adapts and verifies every parameter. Never reuse incompatible hashes.
- Normalization/PII replacement can collapse different records into one apparent duplicate. Report-only is the default; applying removal requires an explicit pre-PII or PII-normalized identity choice and documented retention policy.
- `RunManifest`: schema_version=1, run_id, baseline/package versions, input partition digests/stable IDs, config digest, model/policy/detector identities, random seed, output shard checksums, completion status and counts. No raw records, allowlist values, reversible maps or keys.
- `resume_run(manifest_path: Path, *, expected: RunIdentity) -> ResumeState`: mismatch is an error, not silent reuse. Publish completion only after atomic shard write plus checksum. Incomplete partitions replay; sinks must be idempotent. Do not promise exactly-once execution.
- Dedup shards, config digests and IDs can remain identifying. Store private run state with restrictive access; hashing is not anonymization. Optional private rejection samples require separate retention and explicit enablement.
- Decontamination uses caller-provided references and report-first decisions. Do not ship built-in benchmark lists, word exclusions or automatic destructive record deletion.

## Token-store reference

Implement `EncryptedSQLiteTokenStore(path: Path, *, scope: str, key_provider: KeyProvider, ttl_seconds: int, clock: Callable[[], float], audit: Callable[[StoreEvent], None] | None=None)` against the existing save/load/delete protocol. Scope is a trusted application identifier, not an arbitrary client argument.

`KeyProvider.current_key_id() -> str` and `KeyProvider.fernet_key(key_id: str) -> bytes` supply externally owned Fernet keys. Use cryptography Fernet, not a new cipher. Encrypt a versioned AnonymizationMap.to_dict envelope binding scope, opaque reference and expiry inside the authenticated payload. Use random opaque references, never plaintext PII/session payloads in references or logs.

SQLite uses short-lived connections, protected files, transactions and a tested concurrency mode. No key is persisted in the database. load returns None for missing, expired or wrong-scope references; malformed/tampered ciphertext raises TokenStoreError. Deletion is logical deletion, not secure erasure of backups/WAL/media. Document expiry, key rotation, process trust and backup responsibilities. This is a local reference integration, not a hosted authorization service or compliance certification.

## External adapters

- Candidate secrets provider: datafog-core 1.0.0. Restrict its rule-based scan to API_KEY, BEARER_TOKEN, JWT and PRIVATE_KEY. Convert code-point ranges to Detection, drop matched_text from audit DTOs and validate returned bounds. No model loading/network work on this path. Existing GLiNER PII labels remain supported.
- Candidate extraction provider: docling 2.148.0. Use only a documented, tested offline/local-artifact configuration by default. No automatic model fetch; missing artifacts fail with an actionable error. Optional downloads require explicit caller enablement and manifest recording.
- `extract_document(path: Path, *, provider: DocumentExtractor) -> ExtractedDocument`: typed elements, page/bounds/element IDs and canonical extracted-text map. Join text with recorded separators so entities can cross element boundaries; reuse SCT long-document chunking, then project results to elements.
- `clean_document(document: ExtractedDocument, cleaner: TextCleaner) -> ProcessedDocument`: returns cleaned text/elements and allowlisted structural provenance. Do not retain raw SDK objects, orig_elements or uncleaned metadata. Original source-file redaction remains outside this adapter.
- Optional dependency candidates: datafog-core==1.0.0, docling==2.148.0, datatrove==0.10.1. Cryptography 48.0.1 already appears in the baseline lockfile; qualify that version first rather than upgrading unrelated dependencies. Registry metadata is not proof of installation or runtime compatibility. P0 must resolve/install every optional profile on the supported matrix, verify actual module versions/paths using the project interpreter, and record exact constraints before merging that profile.

## Language and adaptive routing

`LanguageDecision` has language, confidence, margin, source and reason. `apply_language_policy(decision, policy) -> LanguageDecision` is pure. Policy mode defaults to legacy; new abstention thresholds default None. For opted-in abstention, confidence < min_confidence or margin < min_margin returns language=None and source='abstained'. Equality passes. Pinned languages bypass this policy. Existing no-score/tie/tuple fallback semantics remain unchanged in legacy mode.

`unknown_language_action='multilingual'|'error'|'skip_ner'` is explicit; skip_ner must record reduced execution, and error must not return a partially protected success. A restricted tuple can abstain only under the new policy; it must not invent a detected language outside the allowed set. Add language_abstained to opt-in warning/score metadata.

`RoutingPolicy.plan(*, context: RoutingContext, completed: tuple[DetectorRun, ...]) -> RoutingDecision` decides the next configured detector stage. RoutingContext holds language/quality/classification decisions, requested entity types and size. RoutingDecision contains selected IDs, skipped IDs, reasons and an experimental-policy identity, never a claim of complete privacy.

Provide always_run and experimental calibrated_cascade. A cascade requires explicit fast/full detector specs, a versioned calibration artifact and opt-in configuration. Fast-detector capabilities must cover every requested entity type before an early exit is eligible; one detected email is never proof that names are absent. Unknown language, errors, unsupported context, missing calibration or missing capability forces full routing or an explicit failure.

Calibration is finite and reproducible: dev-only grid over language confidence {0.90,0.95,0.99}, margin {0.10,0.20}, and text-size cutoff {256,512}; compare 12 candidates plus always_run on frozen samples and pinned detector models. Select by lowest measured latency among candidates with no observed added exposure on every declared validation slice; tie-break by fewer skipped stages, then a stable policy ID. Then evaluate once on the untouched test split. Record model calls, recall, precision, document exposure, language coverage, median/p95 latency and peak memory on identical hardware/runtime stacks.

Promotion requires an upper paired 95% interval for added document exposure <= 0, no regression in the deterministic mixed-entity fixtures, and a strictly lower expensive-model call count with a measured median latency improvement. Intervals must not be hidden if inconclusive. If no candidate passes, retain always_run and leave cascade experimental; deliver the implementation and negative result rather than retuning on test or promising an optimization.

## Completion definition

All ten workstreams require code, documented interfaces, compatibility checks, relevant real optional-provider smoke tests and recorded evaluation. Adaptive routing remains an implemented experimental feature unless its promotion gate passes. Do not report all ten complete while any required integration, failure path, source mapping, data-removal action or acceptance evidence is missing. No new default or major migration occurs without a separate reviewed release decision.

## Interface details locked by plan review

- `FieldRule(path: tuple[Hashable, ...], entity_types: tuple[str, ...], mode: Literal['hint','protect_value']='hint')` and `FieldSchema(id: str, version: str, rules: tuple[FieldRule, ...])` are frozen. Paths use exact caller-specified components, with no guessed field-name rules. hint passes declared types; protect_value protects the complete selected scalar. Conflicting duplicate rules fail validation.
- `UnifiedPrivacyPolicy.evaluate(detections: tuple[Detection, ...]) -> tuple[FindingDecision, ...]` assigns actions. `FindingDecision(detection: Detection, action: Literal['use_mode','keep','redact'], reason: str)` is frozen. Resolution and rendering belong to unified.py, using existing replacement implementations.
- `EncryptedSQLiteTokenStore.rotate(ref: str) -> bool` keeps the reference and original expiry, and transactionally re-encrypts one current-scope entry with the current key. Missing or expired entries return False. `purge_expired() -> int` deletes expired rows in the current scope. Neither operation claims secure erasure.
- Corpus selection uses datasketch candidate matching with the configured signature identity and disk-backed decisions. Confirm candidate similarity against the same declared basis before cluster membership. Persist only private signatures, IDs and decision state; stream source texts. DataTrove composition is a separate native workflow, not a conversion of SCT signatures.
- Shared review tests belong to C1/C4 (Unicode and opaque transforms), C3/C5 (worker ownership and context), D3 (crash/retry/mismatch), S1/S2 (scope/tampering/expiry), and R1 (sensitive-output leakage and default compatibility).
