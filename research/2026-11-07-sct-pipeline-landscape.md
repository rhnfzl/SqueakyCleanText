# SCT pipeline landscape and missing capabilities

Research date: 2026-11-07. SCT baseline: released v0.8.3, commit 8afef61c74ea738a2c904374776dea34d9c1c39e. The local origin/main reference matched this commit. Fifteen external repository snapshots were reviewed. The inventory covers sct/, scripts/, README and relevant docs. External facts are source-verified or documented; adoption proposals and inferred risks are labelled as such. No package was installed, example run, performance benchmark measured, or production code changed. Default-branch snapshots can be ahead of published releases.

## Answers

- Yes. Presidio, DataFog Python, DataFog Core, scrubadub and Gorget have overlapping PII orchestration or extension mechanisms. Configurable routing is not unique to SCT.
- The closest wider-pipeline references are Data-Juicer, NeMo Curator and DataTrove. They cover substantial curation and execution capabilities. This review did not establish an exact drop-in replacement for every SCT API, mode and integration; it also does not establish uniqueness.
- The strongest next work is a consistent finding/policy contract and a public detector interface, plus corpus-selection and durable-execution recipes around the existing DataTrove integration. Rebuilding an entire distributed curation engine inside SCT is not the recommended path.

## What routing means here

| Meaning | Definition | SCT baseline | Related designs |
|---|---|---|---|
| Backend selection | Choose ONNX, Torch, GLiNER or another configured path. | Native, six named choices; custom models are configurable. | Presidio, DataFog and Gorget overlap here. |
| Language-based dispatch | Choose models for a supplied or detected language. | Native automatic language detection plus configurable language-to-model routing. | Presidio dispatches using caller-supplied language and configured language recognizers; this is a different boundary. |
| Open detector registration | Add a span-producing detector without editing the core. | No generic public registry found; custom string transforms are not equivalent. | Presidio recognizers, scrubadub detectors and Gorget callable/entry-point plugins are direct references. |
| Adaptive escalation | Run more expensive detection only when a policy says coverage requires it. | No cost/quality adaptive cascade found. | DataFog TextService uses entity-count exits; public scan smart mode merges regex and requested NER. These are different API paths. |

Sources: [sct-config](https://github.com/rhnfzl/SqueakyCleanText/blob/8afef61c74ea738a2c904374776dea34d9c1c39e/sct/config.py#L239-L274), [sct-routing](https://github.com/rhnfzl/SqueakyCleanText/blob/8afef61c74ea738a2c904374776dea34d9c1c39e/sct/utils/ner.py#L315-L339), [presidio-registry](https://github.com/data-privacy-stack/presidio/blob/05b60e1a6805f3b2c0ad6cbe7526ccd226647971/presidio-analyzer/presidio_analyzer/recognizer_registry/recognizer_registry.py), [presidio-languages](https://github.com/data-privacy-stack/presidio/blob/05b60e1a6805f3b2c0ad6cbe7526ccd226647971/docs/analyzer/languages.md), [gorget-plugins](https://github.com/Perruer/gorget/blob/8c3bca6a46280f5c47b07a4c551bdf02f57fa018/gorget/plugins.py), [datafog-cascade](https://github.com/DataFog/datafog-python/blob/cf81370dfd7a667d05969f38b88ebad9e38c94c1/datafog/services/text_service.py#L188-L268), [datafog-public-smart](https://github.com/DataFog/datafog-python/blob/cf81370dfd7a667d05969f38b88ebad9e38c94c1/datafog/engine.py#L460-L524)

## Existing SCT capabilities to preserve

| Capability | What exists | Evidence |
|---|---|---|
| Model configuration and voting | Language-specific model keys, custom model IDs, optional backends and max/mean voting exist. | [sct-routing](https://github.com/rhnfzl/SqueakyCleanText/blob/8afef61c74ea738a2c904374776dea34d9c1c39e/sct/utils/ner.py#L315-L339) |
| Policies and reversible replacement | Policy and TokenStore support already exist, with important scope and storage limits. | [sct-tokenstore](https://github.com/rhnfzl/SqueakyCleanText/blob/8afef61c74ea738a2c904374776dea34d9c1c39e/sct/privacy.py#L14-L41) |
| Structured data | JSON recursion, DataFrame columns and RAG adapters exist; field-context detection is the narrower gap. | [sct-structured](https://github.com/rhnfzl/SqueakyCleanText/blob/8afef61c74ea738a2c904374776dea34d9c1c39e/sct/adapters.py#L92-L158) |
| Quality and near-duplicate signals | Quality scores and PII-normalised MinHash exist. Corpus selection and duplicate removal are separate work. | [sct-score](https://github.com/rhnfzl/SqueakyCleanText/blob/8afef61c74ea738a2c904374776dea34d9c1c39e/sct/score.py#L44-L190) |
| OCR | OCRProvider, Tesseract and image-region redaction exist. Full document extraction is a different capability. | [sct-ocr](https://github.com/rhnfzl/SqueakyCleanText/blob/8afef61c74ea738a2c904374776dea34d9c1c39e/sct/ocr.py#L20-L130) |
| Integration and execution | A native DataTrove adapter exists. Executors can provide the large-corpus runtime. | [sct-executor](https://github.com/rhnfzl/SqueakyCleanText/blob/8afef61c74ea738a2c904374776dea34d9c1c39e/sct/integrations/datatrove.py#L14-L52) |
| Classification and provenance | Optional document classification, model revisions, value-free counts and data cards exist. | [sct-classify](https://github.com/rhnfzl/SqueakyCleanText/blob/8afef61c74ea738a2c904374776dea34d9c1c39e/sct/config.py#L288-L295) |

Streaming, synthetic replacement, risk reporting, evaluation and additional framework adapters also exist. The baseline audit distinguishes their limits rather than presenting them as absent. Detailed file/line references appear in the HTML report and source manifest.

## Project catalogue

### Presidio (PII orchestration)

Closest reference for an open recognizer ecosystem.

Registry-based recognition, configurable NLP engines and language-specific models, decision explanations, and structured analysis are documented and implemented. Dictionary keys are passed as recognition context. The caller supplies the analysis language; this is not SCT-style automatic language detection.

**Inspiration:** A public recognizer contract, registration/configuration, schema context and inspectable decisions.

**Boundary:** SCT already has a Presidio/GLiNER backend. Use the registry design or optional adapter; replacing SCT with Presidio alone would leave the wider curation pipeline to other components. The original Microsoft repository redirects to Data Privacy Stack.

Code license: [MIT](https://github.com/data-privacy-stack/presidio/blob/05b60e1a6805f3b2c0ad6cbe7526ccd226647971/LICENSE). Snapshot `05b60e1a6805f3b2c0ad6cbe7526ccd226647971`.

[presidio-registry](https://github.com/data-privacy-stack/presidio/blob/05b60e1a6805f3b2c0ad6cbe7526ccd226647971/presidio-analyzer/presidio_analyzer/recognizer_registry/recognizer_registry.py), [presidio-fields](https://github.com/data-privacy-stack/presidio/blob/05b60e1a6805f3b2c0ad6cbe7526ccd226647971/presidio-analyzer/presidio_analyzer/batch_analyzer_engine.py#L112-L143), [presidio-languages](https://github.com/data-privacy-stack/presidio/blob/05b60e1a6805f3b2c0ad6cbe7526ccd226647971/docs/analyzer/languages.md)

### DataFog Python (PII orchestration)

Engine selection with distinct public scan and TextService routing paths.

The inspected dev snapshot offers regex, spaCy, GLiNER, auto/smart choices. Public scan smart mode merges regex with GLiNER when requested entity types need NER, falling back to spaCy if GLiNER is unavailable. The separate TextService smart path uses an entity-count cascade.

**Inspiration:** Make routing and fallback decisions explicit, observable and configurable.

**Boundary:** Only TextService returns early after any regex entity or two GLiNER entities. Public scan smart mode does not use that count-based early exit. Copying the TextService rule could miss other entity categories; this is a source-derived design risk, not an executed accuracy result. Missing public NER backends produce warnings and can leave regex-only results.

Code license: [MIT](https://github.com/DataFog/datafog-python/blob/cf81370dfd7a667d05969f38b88ebad9e38c94c1/LICENSE). Snapshot `cf81370dfd7a667d05969f38b88ebad9e38c94c1`.

[datafog-public-smart](https://github.com/DataFog/datafog-python/blob/cf81370dfd7a667d05969f38b88ebad9e38c94c1/datafog/engine.py#L460-L524), [datafog-cascade](https://github.com/DataFog/datafog-python/blob/cf81370dfd7a667d05969f38b88ebad9e38c94c1/datafog/services/text_service.py#L188-L268), [datafog-readme](https://github.com/DataFog/datafog-python/blob/cf81370dfd7a667d05969f38b88ebad9e38c94c1/README.md)

### DataFog Core (PII orchestration)

Strongest close reference for cross-runtime finding and transformation contracts.

Rust core with Python, Node and browser bindings. Explicit byte/code-point/UTF-16 ranges, detector provenance, validated supplied findings, per-entity transformations, and capability discovery are documented. Browser support differs for provider-backed transformations. Its documented detector catalogue also includes several credential formats.

**Inspiration:** Named coordinate systems, separate detection and transformation, per-type policy, explicit capabilities and application-owned token/key providers.

**Boundary:** This is a separate repository from DataFog Python. It is not a Go implementation. Do not import its built-in name-field aliases or word exceptions into SCT. Compare contracts and user-declared schema instead.

Code license: [MIT](https://github.com/DataFog/datafog-core/blob/177cb6b57e00caeef5c096f5c7a9ad2b8db60328/LICENSE). Snapshot `177cb6b57e00caeef5c096f5c7a9ad2b8db60328`.

[core-ranges](https://github.com/DataFog/datafog-core/blob/177cb6b57e00caeef5c096f5c7a9ad2b8db60328/docs/concepts/findings-and-ranges.mdx), [core-policy](https://github.com/DataFog/datafog-core/blob/177cb6b57e00caeef5c096f5c7a9ad2b8db60328/docs/concepts/privacy-transformations.mdx), [core-capabilities](https://github.com/DataFog/datafog-core/blob/177cb6b57e00caeef5c096f5c7a9ad2b8db60328/docs/reference/capabilities.mdx), [core-provider](https://github.com/DataFog/datafog-core/blob/177cb6b57e00caeef5c096f5c7a9ad2b8db60328/docs/integrations/fogvault.md)

### scrubadub (PII orchestration)

Small, comprehensible detector/postprocessor extension pattern.

Scrubber accepts configurable detector and postprocessor lists, exposes add/remove detector operations and applies locale compatibility rules.

**Inspiration:** A lightweight detector interface that can live in a separate package.

**Boundary:** Locale configuration is not automatic multilingual detection. The API snapshot is useful, but the repository reports a 2023 last push; that alone does not prove abandonment or present compatibility.

Code license: [Apache-2.0](https://github.com/LeapBeyond/scrubadub/blob/53772cbef417da290d25c95373031f786ab3b5c6/LICENSE). Snapshot `53772cbef417da290d25c95373031f786ab3b5c6`.

[scrubadub-detectors](https://github.com/LeapBeyond/scrubadub/blob/53772cbef417da290d25c95373031f786ab3b5c6/scrubadub/scrubbers.py)

### Gorget (PII orchestration)

A self-described LLM Guard continuation with explicit plugin contracts.

Source defines callable span detectors and classifier protocols, plus import-path and Python entry-point plugin loading. Anonymization and other input/output scanners are separate stages.

**Inspiration:** Plugins with a small span-returning contract, explicit model/license inventory, and scanner result composition.

**Boundary:** The repository is not archived in the snapshot. That does not establish production maturity. Its code is MIT, but the documented default PII model has CC-BY-NC-4.0 weights; code and model permissions require separate checks.

Code license: [MIT](https://github.com/Perruer/gorget/blob/8c3bca6a46280f5c47b07a4c551bdf02f57fa018/LICENSE). Snapshot `8c3bca6a46280f5c47b07a4c551bdf02f57fa018`.

[gorget-plugins](https://github.com/Perruer/gorget/blob/8c3bca6a46280f5c47b07a4c551bdf02f57fa018/gorget/plugins.py), [gorget-models](https://github.com/Perruer/gorget/blob/8c3bca6a46280f5c47b07a4c551bdf02f57fa018/docs/models.md)

### LLM Guard (PII orchestration)

Historical reference for input/output scanner composition.

The official repository is archived and its README states that the project and associated models are no longer actively maintained.

**Inspiration:** Study the scanner/vault interface as historical design, with source-specific license checks.

**Boundary:** Do not select it as the default new dependency on the basis of its past popularity or the older SCT benchmark. Current alternatives and model licenses need independent validation.

Code license: [MIT](https://github.com/protectai/llm-guard/blob/168c1034ffdb33837e7ae6fd6a16b80567c1be03/LICENSE). Snapshot `168c1034ffdb33837e7ae6fd6a16b80567c1be03`.

[llmguard-archive](https://github.com/protectai/llm-guard/blob/168c1034ffdb33837e7ae6fd6a16b80567c1be03/README.md)

### PII Vault (PII orchestration)

Prototype reference for a shared recognizer specification and cross-language fixtures.

The root README identifies implemented Rust/TypeScript paths and a Python placeholder. Recognizer JSON specifications and reversible tokens are part of the design.

**Inspiration:** Portable contract fixtures and versioned recognizer metadata.

**Boundary:** Not a ready Python replacement. SCT already has reversible replacement and a TokenStore protocol. The README describes FNV hashing for one strategy; do not treat that as cryptographic privacy protection or an audited production vault.

Code license: [MIT](https://github.com/Jiansen/pii-vault/blob/e5fc84c3848c017668ec0149080a10b421b37d51/LICENSE-MIT). Snapshot `e5fc84c3848c017668ec0149080a10b421b37d51`.

[vault-readme](https://github.com/Jiansen/pii-vault/blob/e5fc84c3848c017668ec0149080a10b421b37d51/README.md)

### DataTrove (Corpus curation)

Best immediate composition target because SCT already supplies an adapter.

Composable blocks, local/Slurm execution, completed-task resumption, filtering reasons, corpus MinHash discovery/removal and reference-index decontamination are present.

**Inspiration:** Recipes for curation around SCT; rejection accounting and job identity.

**Boundary:** Task completion markers do not checkpoint arbitrary intermediate model state. Its inspected built-in PII formatter is narrow email/IP handling, not evidence of the same multilingual PII routing.

Code license: [Apache-2.0](https://github.com/huggingface/datatrove/blob/384d81c011fda5cf54be3ba1d58967c21fafe25d/LICENSE). Snapshot `384d81c011fda5cf54be3ba1d58967c21fafe25d`.

[datatrove-resume](https://github.com/huggingface/datatrove/blob/384d81c011fda5cf54be3ba1d58967c21fafe25d/src/datatrove/executor/base.py), [datatrove-dedup](https://github.com/huggingface/datatrove/blob/384d81c011fda5cf54be3ba1d58967c21fafe25d/examples/minhash_deduplication.py), [datatrove-filter](https://github.com/huggingface/datatrove/blob/384d81c011fda5cf54be3ba1d58967c21fafe25d/src/datatrove/pipeline/filters/base_filter.py), [datatrove-decontam](https://github.com/huggingface/datatrove/blob/384d81c011fda5cf54be3ba1d58967c21fafe25d/src/datatrove/pipeline/decont/n_grams.py)

### Data-Juicer (Corpus curation)

Closest broad reference for composable curation, quality analysis and execution provenance.

Operator pipelines, quality statistics, actual deduplication and partitioned Ray execution with content validation/checkpoints are implemented.

**Inspiration:** Analyse first, then select; reuse statistics; validate source identity before resume.

**Boundary:** The tracer stores original and processed text. Copying that behavior into a privacy pipeline would retain sensitive material. Prefer value-free traces and private, explicit sample retention. No blanket reliability or speed claim is established.

Code license: [Apache-2.0](https://github.com/datajuicer/data-juicer/blob/65383e6f8f0dc0b3c74de8475b0f8abdab700f51/LICENSE). Snapshot `65383e6f8f0dc0b3c74de8475b0f8abdab700f51`.

[datajuicer-analysis](https://github.com/datajuicer/data-juicer/blob/65383e6f8f0dc0b3c74de8475b0f8abdab700f51/data_juicer/core/analyzer.py), [datajuicer-resume](https://github.com/datajuicer/data-juicer/blob/65383e6f8f0dc0b3c74de8475b0f8abdab700f51/data_juicer/core/executor/ray_executor_partitioned.py), [datajuicer-trace](https://github.com/datajuicer/data-juicer/blob/65383e6f8f0dc0b3c74de8475b0f8abdab700f51/data_juicer/core/tracer/tracer.py)

### NeMo Curator (Corpus curation)

Broad optional reference for large-corpus CPU/GPU curation.

Ray-based pipelines, exact/fuzzy/semantic deduplication and source-partition resumability are documented. Current PII material includes a GLiNER tutorial.

**Inspiration:** Separate duplicate identification from removal and state recovery guarantees precisely.

**Boundary:** Fuzzy workflow defaults to identification without removal. Resumption can replay incomplete partitions and requires idempotent sinks. Current GLiNER tutorial is not proof of a built-in SCT-equivalent router, and older PiiModifier imports may not match this source version.

Code license: [Apache-2.0](https://github.com/NVIDIA-NeMo/Curator/blob/db56fbae11b0672fbc7f822629948a786a14cfb4/LICENSE). Snapshot `db56fbae11b0672fbc7f822629948a786a14cfb4`.

[nemo-dedup](https://github.com/NVIDIA-NeMo/Curator/blob/db56fbae11b0672fbc7f822629948a786a14cfb4/nemo_curator/stages/deduplication/fuzzy/workflow.py), [nemo-resume](https://github.com/NVIDIA-NeMo/Curator/blob/db56fbae11b0672fbc7f822629948a786a14cfb4/fern/versions/v26.09/pages/reference/infrastructure/resumable-processing.mdx), [nemo-pii](https://github.com/NVIDIA-NeMo/Curator/blob/db56fbae11b0672fbc7f822629948a786a14cfb4/tutorials/text/gliner-pii-redaction/README.md)

### Dolma toolkit (Corpus curation)

Reference for stable document identity and reusable derived attributes.

Document/attribute sidecars, duplicate tagging, later mixing/removal and decontamination recipes separate analysis from corpus decisions.

**Inspiration:** Versioned sidecars and repeatable threshold/filter changes without rerunning detection.

**Boundary:** Bloom-filter duplicate matches can be false positives. Attribute tagging alone does not remove duplicates. Toolkit licensing does not establish the license of every corpus it processes.

Code license: [Apache-2.0](https://github.com/allenai/dolma/blob/669f534823b08d266a8fff01f8a1c916a5a56576/LICENSE). Snapshot `669f534823b08d266a8fff01f8a1c916a5a56576`.

[dolma-sidecars](https://github.com/allenai/dolma/blob/669f534823b08d266a8fff01f8a1c916a5a56576/docs/data-format.md), [dolma-dedup](https://github.com/allenai/dolma/blob/669f534823b08d266a8fff01f8a1c916a5a56576/docs/deduplication.md), [dolma-mix](https://github.com/allenai/dolma/blob/669f534823b08d266a8fff01f8a1c916a5a56576/docs/mixer.md)

### Ray Data (Extraction and execution)

Distributed executor, not a replacement for SCT privacy semantics.

Source includes job-level row checkpointing; CheckpointConfig is beta and CheckpointBackend is alpha. Stable preserved row IDs or restricted generated IDs are required.

**Inspiration:** Use an optional executor recipe for retry/resume and cluster execution.

**Boundary:** Do not confuse Ray Data row checkpoints with Ray Train model checkpoints. A recipe is not proof of tested native SCT distributed execution.

Code license: [Apache-2.0](https://github.com/ray-project/ray/blob/e29f035ad4d9a8163e7b0e0c42a05b49c81b6d4d/LICENSE). Snapshot `e29f035ad4d9a8163e7b0e0c42a05b49c81b6d4d`.

[ray-checkpoint](https://github.com/ray-project/ray/blob/e29f035ad4d9a8163e7b0e0c42a05b49c81b6d4d/python/ray/data/checkpoint/interfaces.py)

### CocoIndex (Extraction and execution)

Reference for incremental RAG data preparation and lineage.

Official core concepts describe a state-driven target, source/code changes and incremental recomputation of affected transformations.

**Inspiration:** Config/model/input identity and cache invalidation for repeated processing.

**Boundary:** This is orchestration to integrate or learn from, not a replacement PII detector. Its timing/cost marketing claims were not adopted. Persistent execution state has its own privacy and lifecycle requirements.

Code license: [Apache-2.0](https://github.com/cocoindex-io/cocoindex/blob/1c8709569fb85f4b23bce81b2e60b329fa86d6c5/LICENSE). Snapshot `1c8709569fb85f4b23bce81b2e60b329fa86d6c5`.

[cocoindex-core](https://cocoindex.io/docs/programming_guide/core_concepts), [cocoindex-readme](https://github.com/cocoindex-io/cocoindex/blob/1c8709569fb85f4b23bce81b2e60b329fa86d6c5/README.md)

### Docling (Extraction and execution)

Structure-preserving extraction upstream of SCT.

Document hierarchy, reading order, tables, OCR and page/item provenance are documented.

**Inspiration:** Extract typed document elements, retain IDs/coordinates, then pass their text through SCT.

**Boundary:** SCT already supports OCR providers. The opportunity is PDF/DOCX/layout extraction, not simply adding OCR. Cleaning extracted text does not prove the original binary file, metadata or embedded content has been redacted.

Code license: [MIT](https://github.com/docling-project/docling/blob/1e26b84c5f5690128ddcd65a20a5d7ffb9e80dcd/LICENSE). Snapshot `1e26b84c5f5690128ddcd65a20a5d7ffb9e80dcd`.

[docling-structure](https://github.com/docling-project/docling/blob/1e26b84c5f5690128ddcd65a20a5d7ffb9e80dcd/docs/concepts/docling_document.md)

### Unstructured (Extraction and execution)

Alternative document partitioning and structural chunking upstream of SCT.

Format partitioning and title-aware chunks can preserve original elements and metadata.

**Inspiration:** Preserve section boundaries and document identity while preparing RAG input.

**Boundary:** Retained metadata.orig_elements can include original text-bearing elements. Cleaning chunk text alone could leave source text in metadata; this is an inferred integration risk, not an observed SCT leak. Hosted-platform features must not be attributed automatically to the local library.

Code license: [Apache-2.0](https://github.com/Unstructured-IO/unstructured/blob/293d235cfcd5f5e89c3cc5ff85562005e182ae33/LICENSE.md). Snapshot `293d235cfcd5f5e89c3cc5ff85562005e182ae33`.

[unstructured-chunks](https://github.com/Unstructured-IO/unstructured/blob/293d235cfcd5f5e89c3cc5ff85562005e182ae33/unstructured/chunking/title.py)

## Ranked gap candidates

A gap means no native implementation was found in the stated baseline scope, or an existing implementation has the narrower boundary described. A caller/framework may already provide it.

### 1: Consistent finding, coordinate and policy contracts

**Current SCT:** Native privacy policy is NER-scoped and placeholder-only; finding offsets are post-regex.

**Proposal:** Design one detector-independent finding contract and transformation policy. Track coordinate spaces explicitly, including original-to-transformed mappings where needed. Preserve source/model provenance without raw values by default.

**Boundary:** Core contract work. Existing modes and APIs must retain behavior unless a caller explicitly chooses the new contract.

**Acceptance experiment, not achieved work:** Mixed regex/NER inputs, all replacement modes, Unicode/emoji, normalization, overlap and long-text slices. Verify raw-span correspondence and report exposure/precision/recall; do not use a higher average as the only gate.

SCT evidence: [sct-policy](https://github.com/rhnfzl/SqueakyCleanText/blob/8afef61c74ea738a2c904374776dea34d9c1c39e/sct/sct.py#L152-L179), [sct-findings](https://github.com/rhnfzl/SqueakyCleanText/blob/8afef61c74ea738a2c904374776dea34d9c1c39e/sct/privacy.py#L45-L49). Other designs: [core-ranges](https://github.com/DataFog/datafog-core/blob/177cb6b57e00caeef5c096f5c7a9ad2b8db60328/docs/concepts/findings-and-ranges.mdx), [core-policy](https://github.com/DataFog/datafog-core/blob/177cb6b57e00caeef5c096f5c7a9ad2b8db60328/docs/concepts/privacy-transformations.mdx), [presidio-analyzer](https://github.com/data-privacy-stack/presidio/blob/05b60e1a6805f3b2c0ad6cbe7526ccd226647971/presidio-analyzer/presidio_analyzer/analyzer_engine.py).

### 2: A public detector protocol and registry

**Current SCT:** Six named backend choices and configurable model IDs exist. No generic public span-detector registration protocol was found.

**Proposal:** Allow external packages to return validated canonical spans with declared languages, entity types, batching support, coordinate system, offline/network behavior, model revision and license metadata.

**Boundary:** Core extension point; keep optional adapters outside base dependencies. Custom string transforms already exist and do not supply equivalent structured findings.

**Acceptance experiment, not achieved work:** Wrap two different detector families, reject malformed/out-of-range spans, test missing optional dependencies and explicit failure behavior, and confirm existing configured backends still work.

SCT evidence: [sct-config](https://github.com/rhnfzl/SqueakyCleanText/blob/8afef61c74ea738a2c904374776dea34d9c1c39e/sct/config.py#L239-L274), [sct-routing](https://github.com/rhnfzl/SqueakyCleanText/blob/8afef61c74ea738a2c904374776dea34d9c1c39e/sct/utils/ner.py#L315-L339). Other designs: [presidio-registry](https://github.com/data-privacy-stack/presidio/blob/05b60e1a6805f3b2c0ad6cbe7526ccd226647971/presidio-analyzer/presidio_analyzer/recognizer_registry/recognizer_registry.py), [gorget-plugins](https://github.com/Perruer/gorget/blob/8c3bca6a46280f5c47b07a4c551bdf02f57fa018/gorget/plugins.py), [scrubadub-detectors](https://github.com/LeapBeyond/scrubadub/blob/53772cbef417da290d25c95373031f786ab3b5c6/scrubadub/scrubbers.py), [core-capabilities](https://github.com/DataFog/datafog-core/blob/177cb6b57e00caeef5c096f5c7a9ad2b8db60328/docs/reference/capabilities.mdx).

### 3: Corpus deduplication and quality-selection recipes

**Current SCT:** SCT emits quality scores and PII-normalised MinHash signatures. Matching/indexing, cluster keeper rules, duplicate removal and quality-based rejection are caller-owned in the audited scope.

**Proposal:** Complete the corpus workflow using DataTrove or another executor: analyse, match, choose representatives, filter and account for reasons. Add report-first decontamination against caller-supplied reference text if needed.

**Boundary:** Prefer recipes or an optional corpus package. Do not rebuild a scheduler or remove records silently in the ordinary cleaner.

**Acceptance experiment, not achieved work:** Check deterministic representatives, duplicate/false-positive rates, rejected counts, language slices and sample identities. Test the effects of deduplication before versus after PII normalization.

SCT evidence: [sct-score](https://github.com/rhnfzl/SqueakyCleanText/blob/8afef61c74ea738a2c904374776dea34d9c1c39e/sct/score.py#L44-L190), [sct-filter](https://github.com/rhnfzl/SqueakyCleanText/blob/8afef61c74ea738a2c904374776dea34d9c1c39e/docs/SCORE.md#L63-L92), [sct-executor](https://github.com/rhnfzl/SqueakyCleanText/blob/8afef61c74ea738a2c904374776dea34d9c1c39e/sct/integrations/datatrove.py#L14-L52). Other designs: [datatrove-dedup](https://github.com/huggingface/datatrove/blob/384d81c011fda5cf54be3ba1d58967c21fafe25d/examples/minhash_deduplication.py), [datatrove-filter](https://github.com/huggingface/datatrove/blob/384d81c011fda5cf54be3ba1d58967c21fafe25d/src/datatrove/pipeline/filters/base_filter.py), [datatrove-decontam](https://github.com/huggingface/datatrove/blob/384d81c011fda5cf54be3ba1d58967c21fafe25d/src/datatrove/pipeline/decont/n_grams.py), [dolma-mix](https://github.com/allenai/dolma/blob/669f534823b08d266a8fff01f8a1c916a5a56576/docs/mixer.md), [datajuicer-analysis](https://github.com/datajuicer/data-juicer/blob/65383e6f8f0dc0b3c74de8475b0f8abdab700f51/data_juicer/core/analyzer.py).

### 4: Durable job identity, safe resume and value-free lineage

**Current SCT:** Data cards, model revisions, value-free signals, HF caching and adapters exist. No native corpus run controller or complete persistent input/config/output lineage manifest was found.

**Proposal:** Provide an example run manifest linking input identities, config, model/policy versions and output shards. Let the selected executor own checkpoints, retry/quarantine and scheduling.

**Boundary:** Integration first. Treat cache/ledger contents as data with lifecycle controls.

**Acceptance experiment, not achieved work:** Interrupt a multi-shard run; resumed output must match an uninterrupted run by stable ID, without duplicate writes. Changed input/config/model identity must invalidate or refuse the old resume state.

SCT evidence: [sct-provenance](https://github.com/rhnfzl/SqueakyCleanText/blob/8afef61c74ea738a2c904374776dea34d9c1c39e/sct/score.py#L253-L284), [sct-executor](https://github.com/rhnfzl/SqueakyCleanText/blob/8afef61c74ea738a2c904374776dea34d9c1c39e/sct/integrations/datatrove.py#L14-L52). Other designs: [datatrove-resume](https://github.com/huggingface/datatrove/blob/384d81c011fda5cf54be3ba1d58967c21fafe25d/src/datatrove/executor/base.py), [datajuicer-resume](https://github.com/datajuicer/data-juicer/blob/65383e6f8f0dc0b3c74de8475b0f8abdab700f51/data_juicer/core/executor/ray_executor_partitioned.py), [nemo-resume](https://github.com/NVIDIA-NeMo/Curator/blob/db56fbae11b0672fbc7f822629948a786a14cfb4/fern/versions/v26.09/pages/reference/infrastructure/resumable-processing.mdx), [ray-checkpoint](https://github.com/ray-project/ray/blob/e29f035ad4d9a8163e7b0e0c42a05b49c81b6d4d/python/ray/data/checkpoint/interfaces.py), [cocoindex-core](https://cocoindex.io/docs/programming_guide/core_concepts).

### Next: Field/schema context in existing structured adapters

**Current SCT:** JSON, DataFrame and RAG adapters already exist, but leaf text is processed independently. Paths are provenance rather than detector context.

**Proposal:** Pass caller-declared schema, field path and relevant context through the detector contract. Assess model/context support without hardcoded field-name exception tables.

**Boundary:** Optional, explicit context. Presidio passes dictionary keys as recognition context; learn the interface rather than copy keyword lists.

**Acceptance experiment, not achieved work:** Compare ambiguous values with/without caller schema and measure both false positives and misses by field/language. Define how nested metadata is handled.

SCT evidence: [sct-structured](https://github.com/rhnfzl/SqueakyCleanText/blob/8afef61c74ea738a2c904374776dea34d9c1c39e/sct/adapters.py#L92-L158). Other designs: [presidio-fields](https://github.com/data-privacy-stack/presidio/blob/05b60e1a6805f3b2c0ad6cbe7526ccd226647971/presidio-analyzer/presidio_analyzer/batch_analyzer_engine.py#L112-L143), [presidio-structured](https://github.com/data-privacy-stack/presidio/blob/05b60e1a6805f3b2c0ad6cbe7526ccd226647971/docs/structured/index.md).

### Next: A production token-store reference integration

**Current SCT:** Reversible replacement and TokenStore are native; the included store is process-local memory. Durable security and lifecycle are caller-owned.

**Proposal:** Publish an optional provider integration with explicit scope/tenant isolation, authenticated access, encryption, expiry and deletion semantics.

**Boundary:** Do not invent a new crypto scheme or equate opaque tokens with anonymity. Native token APIs should remain usable without a hosted service.

**Acceptance experiment, not achieved work:** Exercise restarts, tenant separation, expiry, deletion, wrong-scope restoration and audit logs that omit original values.

SCT evidence: [sct-tokenstore](https://github.com/rhnfzl/SqueakyCleanText/blob/8afef61c74ea738a2c904374776dea34d9c1c39e/sct/privacy.py#L14-L41). Other designs: [core-provider](https://github.com/DataFog/datafog-core/blob/177cb6b57e00caeef5c096f5c7a9ad2b8db60328/docs/integrations/fogvault.md), [core-policy](https://github.com/DataFog/datafog-core/blob/177cb6b57e00caeef5c096f5c7a9ad2b8db60328/docs/concepts/privacy-transformations.mdx).

### Next: Structure-preserving document extraction

**Current SCT:** OCR providers exist; no full PDF/DOCX/table/layout extraction engine was found in the audited core.

**Proposal:** Add a documented upstream Docling or Unstructured composition, retaining page/element identity and establishing which text and metadata are cleaned.

**Boundary:** An adapter/recipe, not a replacement parser implementation. Binary-document redaction is a separate claim.

**Acceptance experiment, not achieved work:** Use a synthetic document with PII in text, tables and metadata. Verify extracted text and retained original elements; do not claim the source PDF is safe without a separate binary audit.

SCT evidence: [sct-ocr](https://github.com/rhnfzl/SqueakyCleanText/blob/8afef61c74ea738a2c904374776dea34d9c1c39e/sct/ocr.py#L20-L130), [sct-structured](https://github.com/rhnfzl/SqueakyCleanText/blob/8afef61c74ea738a2c904374776dea34d9c1c39e/sct/adapters.py#L92-L158). Other designs: [docling-structure](https://github.com/docling-project/docling/blob/1e26b84c5f5690128ddcd65a20a5d7ffb9e80dcd/docs/concepts/docling_document.md), [unstructured-chunks](https://github.com/Unstructured-IO/unstructured/blob/293d235cfcd5f5e89c3cc5ff85562005e182ae33/unstructured/chunking/title.py).

### Research only: Adaptive detector escalation

**Current SCT:** SCT has deterministic language routing, classification and confidence controls, but no cost/quality-driven backend cascade was found.

**Proposal:** Evaluate an opt-in escalation policy that preserves requested entity coverage, rather than stopping because one cheap detector found something.

**Boundary:** An experiment, not a default or a speed promise. DataFog TextService illustrates entity-count exits; its public smart scanner instead combines regex/NER. These paths must not be conflated.

**Acceptance experiment, not achieved work:** Use fixed mixed-entity and multilingual samples; compare against the always-run baseline. Record exposure, precision, recall, model calls, latency and memory by slice before considering a default.

SCT evidence: [sct-routing](https://github.com/rhnfzl/SqueakyCleanText/blob/8afef61c74ea738a2c904374776dea34d9c1c39e/sct/utils/ner.py#L315-L339), [sct-classify](https://github.com/rhnfzl/SqueakyCleanText/blob/8afef61c74ea738a2c904374776dea34d9c1c39e/sct/config.py#L288-L295). Other designs: [datafog-public-smart](https://github.com/DataFog/datafog-python/blob/cf81370dfd7a667d05969f38b88ebad9e38c94c1/datafog/engine.py#L460-L524), [datafog-cascade](https://github.com/DataFog/datafog-python/blob/cf81370dfd7a667d05969f38b88ebad9e38c94c1/datafog/services/text_service.py#L188-L268).

### Next: Dedicated format-aware credential scanning

**Current SCT:** GLiNER PII mode already includes password, api_key and access_token labels. No dedicated JWT/PEM parser, vendor-key pattern pack or entropy-based secret scanner was found in the inspected scope.

**Proposal:** Evaluate a separate, opt-in credential detector through the proposed span interface. Keep format-aware checks distinct from model recognition and from replacement generators.

**Boundary:** Optional detector integration. Learn from DataFog Core credential coverage and Gorget secret-scanner composition; do not assume all strings called tokens are secrets.

**Acceptance experiment, not achieved work:** Test synthetic valid/invalid credentials, code samples, false positives, partial masking and multilingual surroundings. Do not use live credentials. Record the selected detector/model license and offline behavior.

SCT evidence: [sct-secret-labels](https://github.com/rhnfzl/SqueakyCleanText/blob/8afef61c74ea738a2c904374776dea34d9c1c39e/sct/config.py#L121-L126). Other designs: [core-secrets](https://github.com/DataFog/datafog-core/blob/177cb6b57e00caeef5c096f5c7a9ad2b8db60328/README.md#detection), [gorget-plugins](https://github.com/Perruer/gorget/blob/8c3bca6a46280f5c47b07a4c551bdf02f57fa018/gorget/plugins.py).

### Research only: Low-confidence language abstention and downstream policy

**Current SCT:** SCT can return unknown on no-positive-score or top-score ties in unrestricted mode. Confidence/margin thresholds emit warnings rather than forcing abstention; restricted-language fallback has separate behavior.

**Proposal:** Evaluate an explicit unknown/abstain or review route for uncertain language decisions. Use existing confidence and margin signals, preserving caller-pinned/restricted-language semantics.

**Boundary:** Optional decision policy, not a claim that SCT lacks unknown-language support. DataTrove language filtering illustrates threshold-based decisions, but corpus rejection is different from choosing a cleaner fallback.

**Acceptance experiment, not achieved work:** Measure in-set accuracy/coverage and out-of-set false answers across held-out languages. Compare warning-only, abstain and explicit fallback modes, then measure their effect on downstream PII detection.

SCT evidence: [sct-language-decision](https://github.com/rhnfzl/SqueakyCleanText/blob/8afef61c74ea738a2c904374776dea34d9c1c39e/sct/sct.py#L394-L440). Other designs: [datatrove-language](https://github.com/huggingface/datatrove/blob/384d81c011fda5cf54be3ba1d58967c21fafe25d/src/datatrove/pipeline/filters/language_filter.py).

## Integration and adoption limits

Keep original values and reversible maps out of default traces, receipts and quarantine outputs. Data-Juicer tracing stores original/processed text; Unstructured can retain original text-bearing elements. Those are concrete integration risks, not observed SCT leaks. New detector plugins also need explicit network and failure contracts. Do not silently send source text to a remote detector or treat a missing optional model as complete protection.

SCT project rules forbid built-in keyword-based fixes. Learn interface, configuration, context and validation designs from other projects; do not copy name-field alias lists, stopword lists or per-word detection exceptions into SCT. Caller-supplied schema and terms remain distinct from built-in exceptions.

The snapshot code licenses are permissive: MIT or Apache-2.0. This does not license every model, dataset, native library or hosted service. Gorget’s documented default PII weights are separately CC-BY-NC-4.0, confirmed in the model publisher’s card. LLM Guard is archived. PII Vault’s root documentation marks Python as a placeholder. Presidio’s original Microsoft repository now resolves to Data Privacy Stack. These status checks affect adoption; they do not establish comparative accuracy.

[Model publisher license](https://huggingface.co/Isotonic/deberta-v3-base_finetuned_ai4privacy_v2/blob/main/README.md).

Do not turn this feature survey into an accuracy ranking. The previous SCT benchmark used specific versions and configurations; its results do not automatically apply to these current source snapshots or alternate model backends.

## Supporting evidence

- `2026-11-07-source-manifest.json`: repository SHAs, licenses, source URLs and inspected SCT scope.
- `2026-11-07-landscape-data.json`: the project and gap matrices used in the HTML report.
- `2026-11-07-pipeline-landscape-notes.md`: deeper curation/executor source notes.

## Remaining verification

Integration compatibility, detector quality, latency/memory, operational recovery, full license compatibility and source-document redaction remain untested. The listed experiments are proposed gates for implementation. This report is research, not an implemented roadmap.
