# Text-pipeline landscape: verified source notes

Research date: 2026-11-07. This note covers other projects, not an audit of missing SCT features. The parent research compares these findings with SCT's current implementation.

The closest broad processing references are DataTrove, Data-Juicer and NeMo Curator. Dolma contributes a useful separation of document text, derived attributes and selection. Ray Data is execution infrastructure. Docling and Unstructured are extraction partners upstream of cleaning and PII handling. None is established here as a complete SCT replacement.

## Evidence method

Tavily advanced searches located official repositories and documentation. Tavily Extract confirmed the relevant pages. Public GitHub API tree snapshots located implementation files; raw source and license files were then read at the exact commits below. No package was installed, no example executed, and no comparative performance claim was tested. Source inspection proves the presence and design of the named paths, not production reliability in a particular workload.

| Project | Source snapshot inspected | Code license |
|---|---|---|
| DataTrove | `huggingface/datatrove@384d81c011fda5cf54be3ba1d58967c21fafe25d` | Apache-2.0, read `LICENSE` |
| Data-Juicer | `datajuicer/data-juicer@65383e6f8f0dc0b3c74de8475b0f8abdab700f51` | Apache-2.0, read `LICENSE`; old `modelscope/data-juicer` URLs redirect here |
| NeMo Curator | `NVIDIA-NeMo/Curator@db56fbae11b0672fbc7f822629948a786a14cfb4` | Apache-2.0, read `LICENSE` |
| Dolma toolkit | `allenai/dolma@669f534823b08d266a8fff01f8a1c916a5a56576` | Apache-2.0, read `LICENSE`; this is the toolkit, not a claim about all dataset licenses |
| Ray | `ray-project/ray@e29f035ad4d9a8163e7b0e0c42a05b49c81b6d4d` | Apache-2.0, read root `LICENSE` |
| Docling | `docling-project/docling@1e26b84c5f5690128ddcd65a20a5d7ffb9e80dcd` | MIT, read `LICENSE`; model licenses remain separate |
| Unstructured | `Unstructured-IO/unstructured@293d235cfcd5f5e89c3cc5ff85562005e182ae33` | Apache-2.0, read `LICENSE.md`; hosted platform and optional dependencies are separate |

These are repository snapshots, not a claim that every inspected feature appears in each latest published package. Ray's rendered latest pages reported different release numbers during this research, so the pinned implementation is the precise reference.

## Project comparison

| Project | Processing and execution | Deduplication / contamination | Quality and provenance | Useful relationship to SCT |
|---|---|---|---|---|
| DataTrove | Generator-based composable blocks; local and Slurm executors, explicit distributed task assignment; completed-task markers on relaunch | Multi-stage MinHash signatures, buckets, clusters and removal; sentence deduplication; n-gram reference-index decontamination | Per-step filter reasons, optional rejected-document writer; executor config and per-task/global statistics | Prefer an adapter and corpus-stage recipes, rather than rebuilding the executor inside SCT |
| Data-Juicer | Operator pipelines; local and Ray execution; partitioned Ray executor with checkpoint strategies, event logs and resume validation | Local MinHash implementation clusters then filters to retained representatives; separate Ray implementations also perform filtering | Analyzer computes filter statistics and distributions; optional tracer stores before/after examples; partition identities include content hashes | Learn inspectable quality analysis and safe operation-level accounting; use the framework for scale where suitable |
| NeMo Curator | Pipeline stages on Ray, CPU and GPU paths depending on operation; source-partition completion checkpoints | Exact, fuzzy and semantic workflows; fuzzy identification defaults to `perform_removal=False`; explicit removal stage/workflow | Workflow timing/count metadata; stable document IDs connect identification to removal | Optional large-corpus backend or interoperable recipe, not a mandatory base dependency |
| Dolma | File-oriented tagging, deduplication, mixing and tokenization; library and CLI | Bloom-filter document/paragraph duplicate attributes; later mixer actually drops documents or spans; decontamination recipes | Original documents and derived sidecar attributes are separate, with stable `(source,id)` lineage | Learn reusable annotations, versioned identities, and independently repeatable filtering decisions |
| Ray Data | Generic distributed data execution, actor/task-backed transforms and batch pipelines | Not a turnkey privacy or text-curation policy in the inspected material | Job-level checkpoint configuration tracks completed rows; execution statistics and retry controls | Optional execution adapter, keeping SCT's processing semantics in SCT |
| Docling | Parses heterogeneous files into a structured document with OCR, layout, tables and reading order | Not evaluated here as a corpus deduplication framework | Structured document hierarchy, item bounding boxes and source-document provenance | Extract upstream, then clean/redact while preserving page and element identity |
| Unstructured | File-type dispatch to partitioners, typed document elements, structural chunking | Not evaluated here as a corpus deduplication framework | Title/section-aware chunks can retain original elements and metadata | Another upstream extraction/chunking adapter; avoid flattening away document structure early |

## Detailed evidence and boundaries

### 1. DataTrove

The [README](https://github.com/huggingface/datatrove/blob/384d81c011fda5cf54be3ba1d58967c21fafe25d/README.md) documents local execution, Slurm, filesystem abstraction, executor options and the logging-directory layout. The [base executor](https://github.com/huggingface/datatrove/blob/384d81c011fda5cf54be3ba1d58967c21fafe25d/src/datatrove/executor/base.py) checks completion markers, writes per-task statistics and marks a rank complete after the pipeline. This is task-level resume, not a claim that every intermediate record or model state is checkpointed. Logs/completions must not be reused for semantically different jobs.

The [MinHash example](https://github.com/huggingface/datatrove/blob/384d81c011fda5cf54be3ba1d58967c21fafe25d/examples/minhash_deduplication.py) demonstrates distinct stages through actual duplicate removal. The [sentence example](https://github.com/huggingface/datatrove/blob/384d81c011fda5cf54be3ba1d58967c21fafe25d/examples/sentence_deduplication.py) similarly separates signatures, duplicate discovery and filtering. The [n-gram module](https://github.com/huggingface/datatrove/blob/384d81c011fda5cf54be3ba1d58967c21fafe25d/src/datatrove/pipeline/decont/n_grams.py) is a concrete decontamination reference.

The [base filter](https://github.com/huggingface/datatrove/blob/384d81c011fda5cf54be3ba1d58967c21fafe25d/src/datatrove/pipeline/filters/base_filter.py) accepts a boolean or a `(False, reason)` result and can send rejected documents to a separate writer. Useful inspiration: count exclusions by reason and make rejected-record storage explicit. A rejection writer can retain sensitive raw text, so it is not a safe default for PII workflows.

The built-in [PII formatter](https://github.com/huggingface/datatrove/blob/384d81c011fda5cf54be3ba1d58967c21fafe25d/src/datatrove/pipeline/formatters/pii.py) replaces email and IP patterns, with validation and configurable replacements. That inspected module is not a multilingual NER router. This narrower PII scope makes composition with SCT plausible, rather than proving equivalent privacy handling. [License](https://github.com/huggingface/datatrove/blob/384d81c011fda5cf54be3ba1d58967c21fafe25d/LICENSE).

### 2. Data-Juicer

The [partitioned Ray executor](https://github.com/datajuicer/data-juicer/blob/65383e6f8f0dc0b3c74de8475b0f8abdab700f51/data_juicer/core/executor/ray_executor_partitioned.py) implements checkpoint management and partition validation. Its partition content hashing is order-sensitive but independent of Arrow batch boundaries, a useful design for safe resumption after partition reconstruction. The [checkpoint/event-log configuration](https://github.com/datajuicer/data-juicer/blob/65383e6f8f0dc0b3c74de8475b0f8abdab700f51/demos/partition_and_checkpoint/configs/partition-checkpoint-eventlog.yaml) names separate partition, checkpoint and event-log directories plus configurable checkpoint strategies. The shown demo has checkpointing disabled until explicitly enabled. Do not infer default crash recovery from the existence of checkpoint options.

The [analyzer](https://github.com/datajuicer/data-juicer/blob/65383e6f8f0dc0b3c74de8475b0f8abdab700f51/data_juicer/core/analyzer.py) computes operator statistics and aggregate/distribution analyses. Its exporter can export stats without dataset text, though original-data export is configurable. The [tracer](https://github.com/datajuicer/data-juicer/blob/65383e6f8f0dc0b3c74de8475b0f8abdab700f51/data_juicer/core/tracer/tracer.py) stores `original_text` and `processed_text` for selected changed samples. Copying that behavior unchanged into a PII pipeline could persist the exact data being protected. Prefer value-free traces by default, with private, explicitly enabled sample traces and retention controls.

The [local MinHash operator](https://github.com/datajuicer/data-juicer/blob/65383e6f8f0dc0b3c74de8475b0f8abdab700f51/data_juicer/ops/deduplicator/document_minhash_deduplicator.py) uses union-find clustering and filters to retained representatives; it does not stop at emitting signatures. [License](https://github.com/datajuicer/data-juicer/blob/65383e6f8f0dc0b3c74de8475b0f8abdab700f51/LICENSE).

### 3. NeMo Curator

The official [overview](https://docs.nvidia.com/nemo/curator/about) describes its Ray stage architecture and text pipeline. The [deduplication guide](https://docs.nvidia.com/nemo/curator/curate-text/process-data/deduplication) separates exact, fuzzy and semantic approaches. The [fuzzy workflow source](https://github.com/NVIDIA-NeMo/Curator/blob/db56fbae11b0672fbc7f822629948a786a14cfb4/nemo_curator/stages/deduplication/fuzzy/workflow.py) has `perform_removal=False` by default. It can therefore identify duplicate IDs without producing a fully filtered corpus. The [text removal workflow](https://github.com/NVIDIA-NeMo/Curator/blob/db56fbae11b0672fbc7f822629948a786a14cfb4/nemo_curator/stages/text/deduplication/removal_workflow.py) applies those IDs to input documents. Stable IDs and persisted ID-generator state matter across these stages.

The [versioned resumability documentation](https://github.com/NVIDIA-NeMo/Curator/blob/db56fbae11b0672fbc7f822629948a786a14cfb4/fern/versions/v26.09/pages/reference/infrastructure/resumable-processing.mdx) says completion checkpoints record source partitions, not serialized task payloads, model state or intermediate datasets. Incomplete sources replay. It explicitly describes at-least-once behavior and requires idempotent sinks or deterministic/atomic output. It also documents fixed actor naming and source-ID constraints. This is a useful reference for recovery boundaries, not a promise of exactly-once processing.

Current [GLiNER PII material](https://github.com/NVIDIA-NeMo/Curator/blob/db56fbae11b0672fbc7f822629948a786a14cfb4/tutorials/text/gliner-pii-redaction/README.md) is a GPU-dependent tutorial using a separate redactor class and external model. Do not conflate it with the obsolete 0.x `PiiModifier` API or with a verified built-in multi-backend PII router. [License](https://github.com/NVIDIA-NeMo/Curator/blob/db56fbae11b0672fbc7f822629948a786a14cfb4/LICENSE). Model and GPU dependency terms require their own checks before adoption.

### 4. Dolma toolkit

The [data-format specification](https://github.com/allenai/dolma/blob/669f534823b08d266a8fff01f8a1c916a5a56576/docs/data-format.md) separates source documents and derived attributes. Stable `(source,id)` keys allow tracing versions and recomputing annotations without duplicating original text. Sidecars must preserve the required row alignment with their input files.

The [deduplication guide](https://github.com/allenai/dolma/blob/669f534823b08d266a8fff01f8a1c916a5a56576/docs/deduplication.md) documents probabilistic Bloom-filter document/paragraph duplicate detection. Duplicate labels or spans are attributes; the [mixer](https://github.com/allenai/dolma/blob/669f534823b08d266a8fff01f8a1c916a5a56576/docs/mixer.md) performs subsequent document filtering and span deletion. Bloom filters can have false positives. The [versioned decontamination recipe](https://github.com/allenai/dolma/blob/669f534823b08d266a8fff01f8a1c916a5a56576/configs/dolma-v1_6/decontamination/README.md) shows building reference indexes, tagging overlap and mixing the result. It is a design reference, not a copy-ready recipe: its named data locations and corpus-specific parameters are not generic requirements. [License](https://github.com/allenai/dolma/blob/669f534823b08d266a8fff01f8a1c916a5a56576/LICENSE).

### 5. Ray Data

The official [execution configuration guide](https://docs.ray.io/en/latest/data/execution-configurations.html) describes job-level checkpointing that skips rows already completed after restart. In the inspected [checkpoint interfaces](https://github.com/ray-project/ray/blob/e29f035ad4d9a8163e7b0e0c42a05b49c81b6d4d/python/ray/data/checkpoint/interfaces.py), `CheckpointConfig` is beta and `CheckpointBackend` is alpha. Unique preserved row IDs are required, or restricted generated IDs for Parquet V2 reads. Checkpoint storage must be accessible to the cluster. These are Ray Data APIs, not Ray Train model checkpoints.

Use Ray as execution infrastructure behind an SCT adapter or recipe, rather than importing its scheduler into SCT's ordinary cleaner. There is no claim here that generic Ray Data supplies SCT's privacy policy, model routing or text-quality semantics. [License](https://github.com/ray-project/ray/blob/e29f035ad4d9a8163e7b0e0c42a05b49c81b6d4d/LICENSE).

### 6. Docling

The [README](https://github.com/docling-project/docling/blob/1e26b84c5f5690128ddcd65a20a5d7ffb9e80dcd/README.md) documents OCR, PDF layout, reading order, table structure and structured export. The [document model](https://github.com/docling-project/docling/blob/1e26b84c5f5690128ddcd65a20a5d7ffb9e80dcd/docs/concepts/docling_document.md) retains hierarchy and bounding boxes where available; the [converter](https://github.com/docling-project/docling/blob/1e26b84c5f5690128ddcd65a20a5d7ffb9e80dcd/docling/document_converter.py) routes formats through registered format options and pipelines.

Natural boundary: parse upstream, pass text-bearing items through SCT, preserve page/item IDs and bounding boxes, then export. Redacting only extracted text does not establish removal from original PDF images, attachments, document metadata or binary files. No such end-to-end redaction was tested here. [MIT license](https://github.com/docling-project/docling/blob/1e26b84c5f5690128ddcd65a20a5d7ffb9e80dcd/LICENSE); README explicitly separates individual model licenses.

### 7. Unstructured

The [README](https://github.com/Unstructured-IO/unstructured/blob/293d235cfcd5f5e89c3cc5ff85562005e182ae33/README.md) documents file-type dispatch through `partition` and per-format dependencies. The [title-aware chunker](https://github.com/Unstructured-IO/unstructured/blob/293d235cfcd5f5e89c3cc5ff85562005e182ae33/unstructured/chunking/title.py) starts chunks around section titles and metadata changes, with size/overlap controls and optional original-element preservation. `metadata.orig_elements` is normally retained, but very large elements have a documented exception. This is stronger structural context than plain fixed-character splitting.

A privacy integration must inspect retained original elements as well as cleaned chunk text. Since `orig_elements` preserves original text-bearing elements, cleaning only a chunk’s visible text can leave source content in metadata. This is an integration risk inferred from the documented retention behavior, not a demonstrated SCT leak.

Treat the open-source partitioning library separately from the hosted platform and ingestion services. Their connector catalog, operations, authentication and service assurances are not automatically capabilities of the locally installed Python library. [Apache-2.0 license](https://github.com/Unstructured-IO/unstructured/blob/293d235cfcd5f5e89c3cc5ff85562005e182ae33/LICENSE.md).

## Candidate inspirations, pending SCT baseline checks

1. **Durable job identity and resumption:** persist config/model/input identities, deterministic output shards and completed partitions. Test interrupted and changed-input restarts. Data-Juicer provides useful partition-validation detail; DataTrove and NeMo show simpler completion boundaries. Prefer an external executor integration where it already solves the requirement.
2. **Real corpus deduplication lifecycle:** distinguish signature generation, candidate matching, connected components/representative selection and removal. Add deterministic retained-record policy and audit counts only if SCT lacks the needed layer; otherwise strengthen recipes around existing support.
3. **Separate analysis from decisions:** reusable, versioned per-record signals can support threshold sweeps and filtering without rerunning detection. Dolma sidecars and Data-Juicer's analyzer are references. Preserve stable identities and versioned score schemas.
4. **Safe per-stage provenance:** counts, durations, rejection reasons, settings and model revisions by default. Raw before/after samples must remain an explicit private diagnostic feature with retention limits. Data-Juicer tracing is useful precisely because its privacy consequences are visible in code.
5. **Benchmark/train decontamination:** user-supplied reference indexes, explicit matching units, measured false positives and report-first operation. Do not add built-in word lists or corpus-specific exclusions. DataTrove and Dolma provide concrete reference designs.
6. **Structure-preserving extraction recipes:** integrate with Docling or Unstructured, retain page/section/element identities, and define where document-level PII detection happens relative to chunking. Text cleaning and redacting the original binary document must remain separate claims.

These are design directions, not confirmed SCT gaps or promises of improved downstream model quality. No performance, accuracy, integration compatibility or reliability benchmark was run. Optional model and native dependency licenses need a separate adoption check.
