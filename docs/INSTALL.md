# Install

```sh
pip install SqueakyCleanText
```

SqueakyCleanText needs Python 3.12 or newer.

The base install uses **ONNX Runtime** for NER inference - no PyTorch or Transformers required.

## Optional extras

| Extra | Command | What it adds |
|-------|---------|--------------|
| GPU | `pip install SqueakyCleanText[gpu]` | CUDA-accelerated ONNX inference; needs the extra install steps in [docs/PERFORMANCE.md](PERFORMANCE.md), which also covers speed by hardware and [older drivers](PERFORMANCE.md#older-drivers) |
| Quantize | `pip install SqueakyCleanText[quantize]` | `onnx`, needed by `load_onnx_ner_model(quantize=True)` |
| Fuzzy dates | `pip install SqueakyCleanText[fuzzy]` | Fuzzy month name matching ([rapidfuzz](https://github.com/rapidfuzz/RapidFuzz)) |
| PyTorch NER | `pip install SqueakyCleanText[torch]` | PyTorch/Transformers NER backend |
| GLiNER | `pip install SqueakyCleanText[gliner]` | [GLiNER](https://github.com/urchade/GLiNER) zero-shot NER |
| GLiNER2 | `pip install SqueakyCleanText[gliner2]` | [GLiNER2](https://github.com/fastino-ai/GLiNER2) (Fastino AI) backend |
| Synthetic | `pip install SqueakyCleanText[synthetic]` | Faker-based synthetic replacement (realistic fake values instead of `<TAG>` tokens) |
| Presidio | `pip install SqueakyCleanText[presidio]` | Presidio-analyzer and GLiNER for `presidio_gliner` backend |
| Classify | `pip install SqueakyCleanText[classify]` | GLiClass document-level pre-classification |
| MCP | `pip install SqueakyCleanText[mcp]` | FastMCP server with text and JSON cleaning tools |
| Datasets | `pip install SqueakyCleanText[datasets]` | Hugging Face `datasets` adapter ([docs/INTEGRATIONS.md](INTEGRATIONS.md)) |
| DataTrove | `pip install SqueakyCleanText[datatrove]` | DataTrove pipeline step |
| LangChain | `pip install SqueakyCleanText[langchain]` | LangChain document transformer |
| Haystack | `pip install SqueakyCleanText[haystack]` | Haystack document cleaner component |
| Score | `pip install SqueakyCleanText[score]` | MinHash signatures for score columns and `sct-report` ([docs/SCORE.md](SCORE.md)) |
| Parquet | `pip install SqueakyCleanText[parquet]` | `.parquet` input for `sct-report` ([pyarrow](https://arrow.apache.org/docs/python/)) |
| OCR | `pip install SqueakyCleanText[ocr]` | Pillow image redaction and Tesseract OCR adapter |
| All NER | `pip install SqueakyCleanText[all-ner]` | PyTorch, GLiNER and GLiNER2 backends combined (`presidio_gliner` needs `[presidio]` as well) |
| Token store | `pip install SqueakyCleanText[token-store]` | Authenticated encryption for [local reversible maps](TOKEN_STORES.md); application-owned keys required |
| Corpus | `pip install 'SqueakyCleanText[corpus]'` | DataTrove 0.10.1, signatures, selection and private resumable runs |
| Documents | `pip install 'SqueakyCleanText[documents]'` | Docling 2.148.0 PDF extraction with separately prepared local artifacts |
| Secrets | `pip install 'SqueakyCleanText[secrets]'` | datafog-core 1.0.0 offline credential-format detection |
| Development | `pip install SqueakyCleanText[dev]` | Testing and linting tools |
| Test | `pip install SqueakyCleanText[test]` | Coverage tools for the test suite |

You can combine extras: `pip install SqueakyCleanText[gpu,fuzzy,gliner]`. In zsh, quote the
name: `pip install "SqueakyCleanText[gpu]"`.

## Opt-in capability setup

Installing an extra does not enable its behavior or establish model quality.
Unified processing, field schemas, language abstention and routing need explicit configuration.
Document extraction requires separately prepared, verified assets; see [document setup](DOCUMENTS.md).
Encrypted storage requires application-owned keys, scope and retention; see [token stores](TOKEN_STORES.md).
Native corpus jobs require a private output directory and explicit caller-managed retention; see [integrations](INTEGRATIONS.md).

Python 3.12, 3.13 and 3.14 are the supported verification matrix.
Provider qualification is specific to the installed version, model artifacts, runtime and tested operation.
An ONNX smoke test does not qualify Torch, GLiNER, GLiNER2, classification or Presidio inference.
The current combined acceptance record is maintained in `evaluation/improvements/ACCEPTANCE.md` in the source checkout.

The quality export script needs the `torch` and `quantize` extras at export time.
Runtime quality inference uses ONNX and tokenizer files, without loading the trained pickle.
The trained Torch scorer has functional integration evidence; trained ONNX export/parity remains pending in the current acceptance record.
The synthetic controls do not establish effective quality filtering.
