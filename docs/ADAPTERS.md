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
policy. `sensitive` (`pii_count > 0`) and `signals` are always filled: `StructuredResult` and `DataFrameResult` cover every
processed string, and `ProcessedRAGChunk` covers text and metadata (`findings` and `metadata["sct"]` cover
the text only, `metadata_findings` the metadata). `process_rag_chunks` writes `metadata["sensitive"]` and
`metadata["sct"]`, replacing any caller keys with those names.

## Command line

`sct-clean` reads a UTF-8 file or standard input. Omit the file name or pass `-` to read from a pipe.

```sh
printf 'Email maria@example.com' | sct-clean --no-ner
sct-clean input.txt --preset training --processing unified --no-ner --signals
printf '{"contact":"maria@example.com"}' | sct-clean --json-input --no-ner
```

`--signals` returns JSON with `text` (or `value` for JSON input), `signals`, and `summary`.
Use `--no-ner` for format-based masking without a name model.
Use `--preset training` to preserve formatting and ordinary dates and numbers, or `--to-ascii` to transliterate.
See `sct-clean --help` for all flags.

Version 0.13.1 adds these CLI features.

- Pass `--processing unified --allow-network-detectors` to run model detection in unified mode.
  This permits downloads if model files are missing. You must pass the flag even when the model files are cached.
- For a plain-text file, `signals.chars` counts Unicode code points before normalization. A CRLF pair counts as two characters.

For reversible text output, pass `--replacement-mode reversible --token-map tokens.json` on POSIX.
The CLI writes the map with mode `0600`. Store it apart from the cleaned text.
See [replacement modes](REPLACEMENT_MODES.md) for other options.

## Streaming

Both streaming redactors refuse a reversible cleaner (`ValueError`): a stream cannot hand back a token map, so process each text directly and retain its map.
`BufferedStreamingRedactor` holds a full document for cross-chunk NER safety.
`WindowedStreamingRedactor` provides bounded latency for contact tokens. It refuses `custom_allow_terms` (`ValueError`): whether an allowed value is kept depends on the whole document.
When its buffer passes `max_buffer_chars` (default 4096) and has no safe whitespace split, it raises
`BufferError`. Raise `max_buffer_chars` or use `BufferedStreamingRedactor` for such input.
It never splits a window inside an email, phone, URL, IP (or a date or number when those steps are on): it reads the
cleaner's own `TextCleaner.normalise` view of the buffer, so its split points follow every pre-regex step.

Both redactors stay open if processing raises during `finish()`. Retry `finish()` after resolving the error.
After a windowed processing error, unprocessed text stays buffered and previously processed output stays queued.
When processing raises inside `feed(chunk)`, the stream has already accepted the chunk.
Retry with `feed("")` or call `finish()`. Do not resend the same chunk, because that adds it again.
Retries can repeat side effects in custom steps or callbacks; stream recovery does not undo those effects.
Calls to `feed()` or `finish()` while another stream operation runs raise `RuntimeError`. This includes calls from callbacks.
A successful `finish()` closes the stream. Later `feed()` and `finish()` calls raise `RuntimeError`.

## OCR and browser export

The OCR adapter accepts any `OCRProvider`. The included Tesseract provider also
needs a local Tesseract executable. Browser export writes an ONNX Runtime Web
manifest, but it does not copy model files or Python language detection.
