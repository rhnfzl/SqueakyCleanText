# Integrations

SCT ships adapters for four frameworks. Each needs its extra:

| Framework | Install | Entry point |
|---|---|---|
| Hugging Face `datasets` | `pip install "squeakycleantext[datasets]"` | `sct.integrations.hf.clean_dataset` |
| DataTrove | `pip install "squeakycleantext[datatrove]"` | `sct.integrations.datatrove.SCTCleaner` |
| LangChain | `pip install "squeakycleantext[langchain]"` | `sct.integrations.langchain.SCTDocumentTransformer` |
| Haystack | `pip install "squeakycleantext[haystack]"` | `sct.integrations.haystack.SCTDocumentCleaner` |

Ray, Spark, Polars, DuckDB, dlt and Daft: see [RECIPES.md](RECIPES.md).

## Shared options

Every adapter takes the same three options.

- `columns`: what to clean. Default is the main text field (`text`, `page_content` or `content`). For document frameworks any other name is a metadata key; missing keys and non-string cells pass through.
- `output`: `"overwrite"` (default) replaces the text. `"new_column"` writes `<col>_clean` and keeps the raw text next to it. Only use it when you need both, and drop the raw column before you publish.
- `config`: a `TextCleanerConfig`, a preset name, or `None`. `None` means the `"training"` preset: PII placeholders only (email, URL, phone, IP, NER); numbers, dates, HTML, case, punctuation, line breaks, indentation and code escapes stay. Use `"sct"` for the library defaults or `TextCleanerConfig.from_preset("training", ...)` to change single flags. Reversible mode is rejected.

Columns you do not list are left as they are. If they can hold personal data, clean them too or drop them.

Document adapters (DataTrove, LangChain, Haystack) also write `metadata["sct"]`: the public signals (language, PII counts and types, warnings, model provenance) merged over all listed columns, so PII in any of them counts. Values only, never entity text.

## Hugging Face datasets

```python
from datasets import load_dataset
from sct.integrations.hf import clean_dataset

if __name__ == "__main__":
    ds = load_dataset("json", data_files="raw.jsonl", split="train")
    ds = clean_dataset(ds, columns=["text"], num_proc=4)
    ds.to_parquet("clean.parquet")
```

`score=True` adds the `sct_*` score columns (`sct.score.SCORE_A_COLUMNS`) for the first column. Extra keyword arguments go to `Dataset.map`. Works for `Dataset` and `IterableDataset`. Results are cached by the dataset fingerprint, the config and the SCT version.

For `Dataset` calls with a custom `registry_factory`, automatic cache reuse is disabled and each output receives a fresh fingerprint.
This also prevents later dataset transforms from reusing results from an older custom detector.
The policy overrides `load_from_cache_file=True`, including with a fixed `cache_file_name`.
To enable reuse, supply an explicit nonempty `new_fingerprint` string.
That caller-owned identity must cover the input dataset, cleaner configuration, map options, factory, provider implementation and model artifacts.
Change it whenever any of those inputs or behaviors change. Factory module names and declared detector versions alone are insufficient.
`new_fingerprint=None` keeps automatic custom-registry cache bypass enabled.
Calls without a custom registry retain their existing automatic cache behavior.

## DataTrove

```python
from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers import JsonlWriter
from sct.integrations.datatrove import SCTCleaner

if __name__ == "__main__":
    LocalPipelineExecutor(
        pipeline=[
            JsonlReader("raw/"),
            # language and quality filters first, SCT after them
            SCTCleaner(),
            JsonlWriter("clean/"),
        ],
        tasks=8,
        workers=4,
    ).run()
```

The step counts `pii`, `pii_<TYPE>` and `sensitive` (documents with `pii_count > 0`) in the DataTrove stats.

## LangChain

```python
from sct.integrations.langchain import SCTDocumentTransformer

docs = SCTDocumentTransformer().transform_documents(docs)
```

Returns new documents; `atransform_documents` works too.

## Haystack

```python
from haystack import Pipeline
from sct.integrations.haystack import SCTDocumentCleaner

pipe = Pipeline()
pipe.add_component("sct", SCTDocumentCleaner())
```

Returns new documents; their ids are recomputed from the cleaned content.

## Workers and memory

- Each adapter owns its cleaner. It is built on first use inside the worker and rebuilt when the process ID changes, so a fork never reuses the parent's model sessions. Two adapters with different configs never share a cleaner.
- Plan for 5 to 7 GB of RAM per worker with the default NER ensemble. `workers x 6 GB` must fit in memory; lower `num_proc` / `workers` before anything else.
- An NER model without a pinned revision (`ner_model_revisions`) makes one Hub request per worker to resolve `main` when the parent never loaded it before the workers started. Pin the revision, load the cleaner once in the parent, or set `HF_HUB_OFFLINE=1` once the models are cached. The default ensemble is pinned and makes none.
- Guard multi-process code with `if __name__ == "__main__":` (macOS and Windows start workers with spawn and re-import your script).
- Run SCT after language and quality filters: NER is the slow step, so do not spend it on documents you drop anyway.

### Native corpus curation

`pip install 'SqueakyCleanText[corpus]'` enables a separate native DataTrove 0.10.1 recipe.
`examples/corpus/clean_select_dedup.py input.jsonl new-run` writes a new cleaned corpus and private decisions.
Input JSONL rows require unique string `id` and string `text` fields. Other input fields are not copied.
The example disables model-based NER explicitly. Regex protection and configured cleanup still run.

```python
from pathlib import Path
from sct.config import TextCleanerConfig
from sct.corpus import SelectionConfig
from sct.corpus.datatrove import curate_datatrove

manifest = curate_datatrove(
    Path('input.jsonl'), Path('new-run'),
    config=TextCleanerConfig(check_ner_process=False),
    retention='caller_managed',
    selection=SelectionConfig(dedup_mode='apply', dedup_basis='pii_normalized'),
)
assert manifest.counts['kept'] + manifest.counts['rejected'] == manifest.counts['input']
```

Report mode retains all records by default. Applying removal requires an explicit `pre_pii` or
`pii_normalized` basis. Normalized placeholders can collapse different sensitive records.
Quality rejection requires an explicit scorer, threshold and versioned `component_identities`.
Schema, detector registry factory and detection context use the existing process-owned cleaner handle.
Custom runtime components require caller-owned version identities. Unsupported non-JSON configuration fails before processing.
Named cleaner presets can request models. Callers must provide their required local model artifacts.

The recipe runs native signature, bucket, cluster and duplicate-filter stages through `LocalPipelineExecutor`.
Native bucket matching is approximate. It does not verify D1's exact Jaccard threshold.
`native_config=MinhashConfig(...)` controls native shingles, bucket count, hashes, normalization and seed.
Changing `SelectionConfig.similarity_threshold` from its compatibility default raises an error.
The recipe default explicitly uses DataTrove's SHA1 hash implementation. Caller hash algorithms are never substituted.
DataTrove's default xxhash fails its text probe in the qualified Python 3.14 installation.
SCT/datasketch hashes never become DataTrove signature files.

Eligible rows enter native stages in stable-ID order. Quality failures cannot become duplicate keepers.
The pinned native cluster file format maps ordinal pairs to stable IDs.
A validated native removal file selects each cluster's minimum stable ID before the actual native filter runs.
Data tasks use one deterministic shard. `workers` controls framework processes, including parallel bucket tasks.
This recipe stages texts on disk; native clustering still holds its union graph in memory.

The new output directory must not exist. Inputs remain unchanged.
`kept.jsonl` contains cleaned text, while `decisions.jsonl` contains IDs, scores and decisions without text.
`manifest.json` publishes completion only after atomic shard renames and checksum verification.
Its schema version is 1. It records baseline/package versions, input digest and stable IDs,
configuration digest, native signature identity, caller component identities, seed, counts and output checksums.
The manifest excludes raw configuration values and reversible maps.

The directory is private (0700), and finished files use 0600 permissions.
Private SQLite staging retains source-basis text. Signatures, IDs and configuration digests can identify records.
Hashing does not anonymize them. `retention='caller_managed'` explicitly assigns storage protection,
backup handling and eventual deletion to the caller. No rejected-document text writer is enabled.
Incomplete runs remain private for inspection. This API does not implement resume or retries.

Model-enabled native recipes require 40-character immutable Hub commit revisions for every active SCT-owned model route.
The manifest records those exact model repository/revision pairs. Disabled model fields do not load models.
Pre-cache the pinned artifacts outside the recipe, then start Python with `HF_HUB_OFFLINE=1`.
The recipe checks the imported Hub runtime flag and never changes global download settings.
Missing cached artifacts fail privately. Cached model execution needs separate provider qualification.
Active `presidio_gliner` is unsupported here because its default NLP engine can load separate unpinned models.
This restriction does not change the legacy adapter. Trusted external registries remain application-owned.
Their registry and quality identities must describe immutable artifacts and all behavior-changing configuration.
