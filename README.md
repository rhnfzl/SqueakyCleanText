# SqueakyCleanText

[![PyPI](https://img.shields.io/pypi/v/squeakycleantext.svg)](https://pypi.org/project/squeakycleantext/)
[![Python package](https://github.com/rhnfzl/SqueakyCleanText/actions/workflows/python-package.yml/badge.svg)](https://github.com/rhnfzl/SqueakyCleanText/actions/workflows/python-package.yml)
[![Python Versions](https://img.shields.io/badge/Python-3.12%20|%203.13%20|%203.14-blue)](https://pypi.org/project/squeakycleantext/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/rhnfzl/SqueakyCleanText/blob/main/LICENSE)

**Clean, privacy-safe text for LLM training, fine-tuning and RAG, in 75 languages.**

SqueakyCleanText (SCT) is a text processing pipeline for datasets. Raw text from the web, tickets, chats or documents goes in. Clean text comes out, with personal data replaced and a value-free record of what was found. Run it on one string, a batch, a Hugging Face dataset or a DataTrove pipeline, on CPU or GPU.

Documentation: <https://rhnfzl.github.io/SqueakyCleanText/>

## What it does

One call runs these stages. Each one is a config field with a safe default, so you can switch it off.

1. **Detect the language.** One of 75 languages, with a confidence score. Pin it or limit it when you know.
2. **Fix broken text.** Repairs mojibake and HTML entities (ftfy), folds styled characters, replaces or drops HTML tags, and tidies whitespace.
3. **Find personal data.** Regex rules for emails, phones, URLs, IPs, IBAN and card numbers, dates and numbers, then an ensemble of multilingual NER models for names, places and organisations.
4. **Replace it.** With tags (`<PERSON>`), reversible placeholders (`<PERSON_0>` plus a map you store apart), or realistic fake values.
5. **Report on it.** Every result has value-free `signals` (counts, types, shares, warnings, model ids). Turn them into score columns and a data card with `sct-report`.
6. **Scale it.** `process_batch` runs documents in threads and batches the NER calls. NER runs on CPU (ONNX Runtime, no PyTorch needed) or an NVIDIA GPU.

SCT finds most personal data, not all of it. Check its output before you publish sensitive data.

<!-- --8<-- [start:quickstart] -->
## Install

```sh
pip install squeakycleantext
```

Python 3.12 or newer. Add extras for what you use, for example `pip install "squeakycleantext[datasets,gpu]"`:

| Extra | Why you need it |
|---|---|
| `datasets`, `datatrove` | Clean Hugging Face datasets or DataTrove pipelines |
| `langchain`, `haystack` | Clean documents in RAG pipelines |
| `gpu` | Run NER on an NVIDIA GPU |
| `synthetic` | Replace personal data with fake values |
| `gliner` | Your own entity labels, and `ner_mode="pii"` (58 PII labels) |
| `mcp` | Run SCT as an MCP server for AI assistants |

All extras: [docs/INSTALL.md](https://github.com/rhnfzl/SqueakyCleanText/blob/main/docs/INSTALL.md).

## Quickstart

```python
from sct import TextCleaner

cleaner = TextCleaner()
result = cleaner.process("<p>Hi, Iâ€™m John Smith. Mail me at john.smith@example.com or call +31 6 1234 5678.</p>")
print(result.lm_text)
# -> <HTML>Hi, I'm <PERSON>. Mail me at <EMAIL> or call <PHONE>.<HTML>
print(result.language, result.signals["pii_types"])
# -> ENGLISH {'EMAIL': 1, 'PERSON': 1, 'PHONE': 1}
```

The broken `â€™` became an apostrophe, the HTML tags became `<HTML>` (set `replace_with_html=""` to drop them), and the name, email and phone became tags. The first run downloads the NER models.

**For training and RAG data, use the `training` preset.** It only replaces personal data and repairs unicode. Numbers, dates, HTML, case, line breaks and code stay as written:

```python
from sct import TextCleaner, preset_config

cleaner = TextCleaner(cfg=preset_config("training"))
for r in cleaner.process_batch([
    "Order 4711 from <b>Anna Berg</b> (anna@example.se) on 2024-05-01.",
    "Bel +31 20 123 4567 voor Jan de Vries.",
]):
    print(r.language, r.lm_text)
# -> ENGLISH Order 4711 from <b><PERSON></b> (<EMAIL>) on 2024-05-01.
# -> DUTCH Bel <PHONE> voor <PERSON>.
```
<!-- --8<-- [end:quickstart] -->

## Use it in your data pipeline

**Hugging Face `datasets`** (`datasets` extra). `score=True` adds the `sct_*` score columns:

```python
from datasets import Dataset
from sct.integrations.hf import clean_dataset

if __name__ == "__main__":
    ds = Dataset.from_dict({"text": ["Mail jan@example.nl", "Call Maria Chen on +44 20 7946 0958."]})
    ds = clean_dataset(ds, columns=["text"], score=True)
    print(ds[0]["text"], ds[0]["sct_pii_types"])
    # -> Mail <EMAIL> ['EMAIL']
```

**DataTrove** (`datatrove` extra). Put SCT after your language and quality filters, so the NER step only sees documents you keep:

```python
from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers import JsonlWriter
from sct.integrations.datatrove import SCTCleaner

if __name__ == "__main__":
    LocalPipelineExecutor(
        pipeline=[JsonlReader("raw/"), SCTCleaner(), JsonlWriter("clean/")],
        tasks=4,
        workers=2,
    ).run()
```

**Everything else:**

| Where | Usage | Docs |
|---|---|---|
| LangChain | `SCTDocumentTransformer().transform_documents(docs)` | [INTEGRATIONS](https://github.com/rhnfzl/SqueakyCleanText/blob/main/docs/INTEGRATIONS.md) |
| Haystack | `pipe.add_component("sct", SCTDocumentCleaner())` | [INTEGRATIONS](https://github.com/rhnfzl/SqueakyCleanText/blob/main/docs/INTEGRATIONS.md) |
| Ray, Spark, Polars, DuckDB, dlt, Daft | `clean_columns(handle, batch)` | [RECIPES](https://github.com/rhnfzl/SqueakyCleanText/blob/main/docs/RECIPES.md) |
| JSON, DataFrames, RAG chunks, streams | `process_json(cleaner, value)`, `WindowedStreamingRedactor(cleaner)` | [ADAPTERS](https://github.com/rhnfzl/SqueakyCleanText/blob/main/docs/ADAPTERS.md) |
| Command line | `printf 'Email maria@example.com' \| sct-clean --no-ner` | [ADAPTERS](https://github.com/rhnfzl/SqueakyCleanText/blob/main/docs/ADAPTERS.md) |
| Data card for a file | `sct-report data.jsonl --text-column text --out report.json --card card.md` | [SCORE](https://github.com/rhnfzl/SqueakyCleanText/blob/main/docs/SCORE.md) |
| AI assistants | `sct-mcp` (MCP server, `mcp` extra) | [MCP](https://github.com/rhnfzl/SqueakyCleanText/blob/main/docs/MCP.md) |

The framework adapters use the `training` preset unless you pass a config, build one cleaner per worker process, and reject reversible mode. Plan for 5 to 7 GB of RAM per worker with the default NER ensemble.

## How it works

<picture>
  <img src="https://raw.githubusercontent.com/rhnfzl/SqueakyCleanText/main/resources/sct_flow.png" alt="SqueakyCleanText pipeline: entry points, language detection, fixes, regex PII, NER, replacement, outputs and metadata" width="900">
</picture>

<details>
<summary>The diagram as text</summary>

- Entry points: `process()`, `process_batch()`, the `sct-clean` CLI, the `sct-mcp` server, the Hugging Face `datasets`, DataTrove, LangChain and Haystack adapters, the JSON, DataFrame and RAG adapters, and the streaming redactors. All of them pass text to the same pipeline.
- 1 Detect language (Lingua, 75 languages, on the raw input) -> 2 Fix the text -> 3 Regex PII (counted in `regex_counts`) -> 4 NER (chunk windows for long text) -> 5 Replace (`<TAG>`, synthetic fakes, or reversible `<PERSON_0>` with a map) -> 6 Output (`lm_text`, `stat_text`, `language`) and `metadata` / `signals`.
- Across all stages: `process_batch` batches NER across documents; NER runs on CPU or CUDA; the `training` preset turns stages on and off.

</details>

Every step is a `TextCleanerConfig` field ([docs/CONFIG.md](https://github.com/rhnfzl/SqueakyCleanText/blob/main/docs/CONFIG.md)). The diagram source is [resources/sct_flow.excalidraw](https://github.com/rhnfzl/SqueakyCleanText/blob/main/resources/sct_flow.excalidraw).

<!-- --8<-- [start:results] -->
## What you get back

`process` returns a `ProcessResult`:

- `lm_text`: the cleaned text, sentences intact, for LLMs, RAG and training.
- `stat_text`: lower-cased, without stopwords and punctuation, for classic ML models.
- `language`: the detected language.
- `signals`: a value-free view with the same keys for every input: language and confidence, `pii_count`, `pii_share`, `pii_types`, `sensitive`, warnings and the NER models that ran. Safe to log or store next to the text.
- `metadata`: the same counts, plus raw values only when you ask for them (the reversible map, or `findings` with `include_entities=True`). Keep those out of logs.

```python
from sct import TextCleaner, TextCleanerConfig
from sct.score import score_record

cleaner = TextCleaner(cfg=TextCleanerConfig(check_ner_process=False))  # regex only, for speed
row = score_record(cleaner.process("Mail jan@example.nl or call +31 20 123 4567."))
print(row["sct_pii_count"], row["sct_pii_types"], round(row["sct_pii_share"], 2))
# -> 2 ['EMAIL', 'PHONE'] 0.66
```

`score_record` gives flat `sct_*` columns for filtering and data cards. Full field list: [docs/API.md](https://github.com/rhnfzl/SqueakyCleanText/blob/main/docs/API.md), score columns: [docs/SCORE.md](https://github.com/rhnfzl/SqueakyCleanText/blob/main/docs/SCORE.md).

## PII modes

| Mode | Output | Use it for |
|---|---|---|
| `placeholder` (default) | `<PERSON> lives in <LOCATION>.` | Training and RAG data |
| `reversible` | `<PERSON_0> lives in <LOCATION_0>.` plus a map | Sending text to an LLM and restoring the answer |
| `synthetic` (`synthetic` extra) | Fake values, for example `Lena Vos lives in Delft.` (the same document always gets the same fakes) | Data that must still read naturally |

```python
from sct import InMemoryTokenStore, TextCleaner, TextCleanerConfig

store = InMemoryTokenStore()
cleaner = TextCleaner(cfg=TextCleanerConfig(replacement_mode="reversible"), token_store=store)
result = cleaner.process("Maria lives in Utrecht.")
print(result.lm_text)  # -> <PERSON_0> lives in <LOCATION_0>.
print(store.load(result.metadata["anon_map_reference"]).deanonymize(result.lm_text))  # -> Maria lives in Utrecht.
```

A `PrivacyPolicy` decides per entity type what to keep or redact and leaves a versioned audit trail. For more PII types (IDs, bank numbers, health data), use the GLiNER PII model with `ner_mode="pii"` (`gliner` extra). Details: [docs/REPLACEMENT_MODES.md](https://github.com/rhnfzl/SqueakyCleanText/blob/main/docs/REPLACEMENT_MODES.md), [docs/PRIVACY.md](https://github.com/rhnfzl/SqueakyCleanText/blob/main/docs/PRIVACY.md), [docs/NER_BACKENDS.md](https://github.com/rhnfzl/SqueakyCleanText/blob/main/docs/NER_BACKENDS.md).
<!-- --8<-- [end:results] -->

## Speed and hardware

Default config (full multilingual NER ensemble), release benchmark documents, SCT 0.8.0, AMD Threadripper PRO 7985WX and RTX 4090, median of 3 runs. 0.8.1 runs at the same speed.

| Workload | Hardware | Docs/s |
|---|---|---|
| `process`, one document at a time | 16 CPU cores | 5.0 |
| `process_batch`, short documents, 16 workers | 16 CPU cores | 15.5 |
| `process_batch`, short documents, 16 workers | 64 CPU cores | 29.2 |
| `process_batch`, short documents, full ensemble | RTX 4090 | 130 |
| `process_batch`, short documents, English only (2 models) | RTX 4090 | 646 |

NER is the slow stage. Turn it off (`check_ner_process=False`) for regex-only cleaning, or limit `language=` to load fewer models. Hardware advice and GPU setup: [docs/PERFORMANCE.md](https://github.com/rhnfzl/SqueakyCleanText/blob/main/docs/PERFORMANCE.md).

## Quality

Measured on open datasets with [benchmarks/](https://github.com/rhnfzl/SqueakyCleanText/tree/main/benchmarks); scores per release in `benchmarks/results/`, the 0.8.1 run in [comparison-0.8.0-unreleased.md](https://github.com/rhnfzl/SqueakyCleanText/blob/main/benchmarks/results/comparison-0.8.0-unreleased.md).

- **Cleaning** (FineWeb-2 with injected noise, default config): 100% of HTML, 99.6% of mojibake and 100% of URLs removed.
- **Language detection** (WiLI, 0.8.0 run): 95.1% accuracy over 7,200 documents.
- **PII** (12 open PII datasets, about 30 languages, default config): leaked values fell on every dataset in 0.8.1, for example 18.1% to 8.2% on REDACT and 42.4% to 11.7% on long court documents (TAB).
- **Release benchmark** (30 documents, 9 languages): exact-span F1 0.99, 0 of 67 PII values leaked (0.8.0).
- **Multilingual PII eval** (220 records, 11 languages): 0.93 recall, 0.95 precision ([docs/MULTILINGUAL_EVAL.md](https://github.com/rhnfzl/SqueakyCleanText/blob/main/docs/MULTILINGUAL_EVAL.md)).

Test your own data or model with the evaluation harness ([docs/EVALUATION.md](https://github.com/rhnfzl/SqueakyCleanText/blob/main/docs/EVALUATION.md)).

## Languages

Language detection covers 75 languages. Full NER models cover English, Dutch, German and Spanish. The other languages use the multilingual WikiNEural model with the English model, or a model you bring with `ner_models`. Full list: [docs/LANGUAGES.md](https://github.com/rhnfzl/SqueakyCleanText/blob/main/docs/LANGUAGES.md).

<!-- --8<-- [start:limits] -->
## Limits and design principles

- **Safe defaults, every choice configurable.** Each behaviour is a config field; nothing is forced.
- **No guarantee that all PII is found.** Models miss entities, mostly organisations and non-Latin scripts. Review output before you publish it.
- **Reversible maps are personal data.** Anyone with the map can restore the values. Store maps apart from the cleaned data, with access control.
- **Offsets are post-regex.** Finding offsets point into the text after the regex step, not into the raw input.
- **Synthetic data is not anonymisation.** Fake values keep the data useful; they do not make it anonymous in a legal sense.
- **Free text, not tables.** For tables with known PII columns, drop or hash those columns first.

<details>
<summary>Known limits of 0.8.1 (details in <a href="https://github.com/rhnfzl/SqueakyCleanText/blob/main/CHANGELOG.md">CHANGELOG.md</a>)</summary>

- Dates: only full dates become one `<DATE>`. `1990年1月`, `1月15日`, Japanese era years and `1990.1.15.` do not; the year and number steps still change them (`1990年1月` becomes `<YEAR>年<NUMBER>月`). Thai month names need `custom_month_names`.
- In ensembles with GLiNER, windows are sized for the token-classification models, not for GLiNER's own word limit. Torch models and GLiNER keep one lock per model, so they do not run in parallel across threads.
- Only `process_batch` is faster; `process` of one document is not batched across documents.
- `ner_chunk_overlap` (off by default) can still cut an entity longer than the overlap that starts at the very end of a chunk.
- `06-12345678` becomes `06-<PHONE>`. Adversarial strings such as `+000000000(000)0` can leave `(000)0` after phone replacement.
- ORGANISATION is the weakest NER label (0.75 recall in the multilingual eval); Bengali and Tamil name spans can be a few characters off.
- A URL path that switches from Latin letters straight into a script written without spaces is replaced only up to the switch (`https://x.org/a東京secret` becomes `<URL>東京secret`).
- Score B's quality head was tested with a small ONNX graph only; GLiNER PII mode was not run end to end for this release.
- An unpinned NER model makes one Hub request per worker when the parent never loaded it before forking; pin `ner_model_revisions` or set `HF_HUB_OFFLINE=1`.
- `WindowedStreamingRedactor` raises `BufferError` when the buffer passes `max_buffer_chars` (default 4096) with no safe whitespace split; raise the limit or use `BufferedStreamingRedactor`.

</details>
<!-- --8<-- [end:limits] -->

## Documentation

| Topic | Where |
|---|---|
| Install, extras, speed and hardware | [docs/INSTALL.md](https://github.com/rhnfzl/SqueakyCleanText/blob/main/docs/INSTALL.md), [docs/PERFORMANCE.md](https://github.com/rhnfzl/SqueakyCleanText/blob/main/docs/PERFORMANCE.md) |
| Every config field, presets, language options | [docs/CONFIG.md](https://github.com/rhnfzl/SqueakyCleanText/blob/main/docs/CONFIG.md) |
| `TextCleaner`, `ProcessResult`, `signals` | [docs/API.md](https://github.com/rhnfzl/SqueakyCleanText/blob/main/docs/API.md) |
| Data pipelines, other engines, MCP server | [docs/INTEGRATIONS.md](https://github.com/rhnfzl/SqueakyCleanText/blob/main/docs/INTEGRATIONS.md), [docs/RECIPES.md](https://github.com/rhnfzl/SqueakyCleanText/blob/main/docs/RECIPES.md), [docs/MCP.md](https://github.com/rhnfzl/SqueakyCleanText/blob/main/docs/MCP.md) |
| JSON, DataFrame, RAG, CLI, streaming, OCR | [docs/ADAPTERS.md](https://github.com/rhnfzl/SqueakyCleanText/blob/main/docs/ADAPTERS.md) |
| Score columns, data cards, `sct-report` | [docs/SCORE.md](https://github.com/rhnfzl/SqueakyCleanText/blob/main/docs/SCORE.md) |
| NER backends, GLiNER, PII mode | [docs/NER_BACKENDS.md](https://github.com/rhnfzl/SqueakyCleanText/blob/main/docs/NER_BACKENDS.md) |
| Replacement modes, privacy policies, `TokenStore` | [docs/REPLACEMENT_MODES.md](https://github.com/rhnfzl/SqueakyCleanText/blob/main/docs/REPLACEMENT_MODES.md), [docs/PRIVACY.md](https://github.com/rhnfzl/SqueakyCleanText/blob/main/docs/PRIVACY.md) |
| Evaluation | [docs/EVALUATION.md](https://github.com/rhnfzl/SqueakyCleanText/blob/main/docs/EVALUATION.md), [docs/MULTILINGUAL_EVAL.md](https://github.com/rhnfzl/SqueakyCleanText/blob/main/docs/MULTILINGUAL_EVAL.md) |
| Languages and adding a language | [docs/LANGUAGES.md](https://github.com/rhnfzl/SqueakyCleanText/blob/main/docs/LANGUAGES.md), [docs/ADDING_LANGUAGES.md](https://github.com/rhnfzl/SqueakyCleanText/blob/main/docs/ADDING_LANGUAGES.md) |
| The whole API in one file, for AI coding assistants | [llms.txt](https://github.com/rhnfzl/SqueakyCleanText/blob/main/llms.txt) |
| Release notes | [CHANGELOG.md](https://github.com/rhnfzl/SqueakyCleanText/blob/main/CHANGELOG.md) |

## What's new in 0.8.1

Fewer leaks, same speed: leaked PII drops on all 12 open benchmark datasets, bank and card numbers get their own `<IBAN>` and `<CARD>` tags, and JSON and longer bracketed text survive cleanup. Some defaults changed; each has a setting to switch it back ([CHANGELOG.md](https://github.com/rhnfzl/SqueakyCleanText/blob/main/CHANGELOG.md)).

## Contributing

Issues and pull requests are welcome. Read [AGENTS.md](https://github.com/rhnfzl/SqueakyCleanText/blob/main/AGENTS.md) first: model mistakes are fixed with model-driven rules, never with word lists, and every behaviour stays configurable.

## Citation

```bibtex
@software{fazal_squeakycleantext,
  author = {Fazal, Rehan},
  title = {SqueakyCleanText},
  url = {https://github.com/rhnfzl/SqueakyCleanText},
  version = {0.8.1},
  year = {2026}
}
```

## License and acknowledgements

MIT. See [LICENSE](https://github.com/rhnfzl/SqueakyCleanText/blob/main/LICENSE). The package took inspiration from [clean-text](https://github.com/jfilter/clean-text). It builds on [ftfy](https://github.com/rspeer/python-ftfy), [Lingua](https://github.com/pemistahl/lingua-py), [ONNX Runtime](https://onnxruntime.ai/) and the NER models listed in [docs/NER_BACKENDS.md](https://github.com/rhnfzl/SqueakyCleanText/blob/main/docs/NER_BACKENDS.md).
