# Install

```sh
pip install SqueakyCleanText
```

SqueakyCleanText needs Python 3.12 or newer.

The base install uses **ONNX Runtime** for NER inference - no PyTorch or Transformers required.

## Optional extras

| Extra | Command | What it adds |
|-------|---------|--------------|
| GPU | `pip install SqueakyCleanText[gpu]` | CUDA-accelerated ONNX inference (see [docs/PERFORMANCE.md](PERFORMANCE.md) for CUDA setup and speed by hardware) |
| Fuzzy dates | `pip install SqueakyCleanText[fuzzy]` | Fuzzy month name matching ([rapidfuzz](https://github.com/rapidfuzz/RapidFuzz)) |
| PyTorch NER | `pip install SqueakyCleanText[torch]` | PyTorch/Transformers NER backend |
| GLiNER | `pip install SqueakyCleanText[gliner]` | [GLiNER](https://github.com/urchade/GLiNER) zero-shot NER |
| GLiNER2 | `pip install SqueakyCleanText[gliner2]` | [GLiNER2](https://github.com/Knowledgator/GLiNER) (knowledgator) backend |
| Synthetic | `pip install SqueakyCleanText[synthetic]` | Faker-based synthetic replacement (realistic fake values instead of `<TAG>` tokens) |
| Presidio | `pip install SqueakyCleanText[presidio]` | Presidio-analyzer for `presidio_gliner` backend |
| Classify | `pip install SqueakyCleanText[classify]` | GLiClass document-level pre-classification |
| MCP | `pip install SqueakyCleanText[mcp]` | FastMCP server with text and JSON cleaning tools |
| Datasets | `pip install SqueakyCleanText[datasets]` | Hugging Face `datasets` adapter ([docs/INTEGRATIONS.md](INTEGRATIONS.md)) |
| DataTrove | `pip install SqueakyCleanText[datatrove]` | DataTrove pipeline step |
| LangChain | `pip install SqueakyCleanText[langchain]` | LangChain document transformer |
| Haystack | `pip install SqueakyCleanText[haystack]` | Haystack document cleaner component |
| Score | `pip install SqueakyCleanText[score]` | MinHash signatures for score columns and `sct-report` ([docs/SCORE.md](SCORE.md)) |
| OCR | `pip install SqueakyCleanText[ocr]` | Pillow image redaction and Tesseract OCR adapter |
| All NER | `pip install SqueakyCleanText[all-ner]` | All NER backends combined |
| Development | `pip install SqueakyCleanText[dev]` | Testing and linting tools |
| Test | `pip install SqueakyCleanText[test]` | Coverage tools for the test suite |

You can combine extras: `pip install SqueakyCleanText[gpu,fuzzy,gliner]`. In zsh, quote the
name: `pip install "SqueakyCleanText[gpu]"`.
