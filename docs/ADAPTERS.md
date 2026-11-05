# Boundary adapters, CLI, streaming, OCR and browser export

Framework adapters (Hugging Face `datasets`, DataTrove, LangChain, Haystack) are in
[INTEGRATIONS.md](INTEGRATIONS.md); the MCP server is in [MCP.md](MCP.md).

The base package includes JSON, DataFrame, retrieval, command-line, and streaming adapters:

```python
from sct import RAGChunk, TextCleaner, process_json, process_rag_chunks

cleaner = TextCleaner()
cleaned_json = process_json(cleaner, {"message": "Email a@example.com"})
# Keys get regex PII replacement only (json_key_cleaning='pii'); 'pii_ner', 'full' or 'off' to change.
cleaned_chunks = process_rag_chunks(
    cleaner,
    [RAGChunk(id="1", text="Call +1-202-555-0147", metadata={})],
)
```

`process_json` also walks tuples and any `Mapping`, and returns them in JSON shape (lists and dicts).
A tuple key stays a tuple, with the strings inside it cleaned as keys. `json.dumps` rejects any tuple key.

`findings` on these results stay empty unless the cleaner has `include_entities=True` or a privacy
policy. `sensitive` and `signals` are always filled: `StructuredResult` and `DataFrameResult` cover every
processed string, and `ProcessedRAGChunk` covers text and metadata (`findings` and `metadata["sct"]` cover
the text only, `metadata_findings` the metadata). `process_rag_chunks` writes `metadata["sensitive"]` and
`metadata["sct"]`, replacing any caller keys with those names.

```sh
printf 'Email maria@example.com' | sct-clean --no-ner
printf 'München 日本語' | sct-clean --no-ner --to-ascii   # 0.7 folding: Munchen Ri Ben Yu
printf 'Email maria@example.com' | sct-clean --no-ner \
  --replacement-mode reversible --token-map tokens.json
pip install 'SqueakyCleanText[mcp]'
sct-mcp
```

Both streaming redactors refuse a reversible cleaner (`ValueError`): a stream cannot hand back a token map, so process each text directly and retain its map.
`BufferedStreamingRedactor` holds a full document for cross-chunk NER safety.
`WindowedStreamingRedactor` provides bounded latency for contact tokens. It refuses `custom_allow_terms` (`ValueError`): whether an allowed value is kept depends on the whole document.
When its buffer passes `max_buffer_chars` (default 4096) and has no safe whitespace split, it raises
`BufferError`. Raise `max_buffer_chars` or use `BufferedStreamingRedactor` for such input.
It never splits a window inside an email, phone, URL, IP (or a date or number when those steps are on): it reads the
cleaner's own `TextCleaner.normalise` view of the buffer, so its split points follow every pre-regex step.

The OCR adapter accepts any `OCRProvider`. The included Tesseract provider also
needs a local Tesseract executable. Browser export writes an ONNX Runtime Web
manifest, but it does not copy model files or Python language detection.
