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
