# Chunking

Clean a whole document before you split it for search, training, or export. Choose smaller model windows separately when a PII detector needs them.

## Get chunks from cleaned text

```python
from sct import TextCleaner, TextCleanerConfig, chunk_document

cleaner = TextCleaner(TextCleanerConfig(check_ner_process=False))
result = chunk_document(cleaner, "Contact alice@example.com for details.", max_size=24)
for chunk in result.chunks:
    print(chunk.range.start, chunk.range.end, chunk.text)
```

Offsets are half-open Unicode character offsets in the cleaned text. `result.chunker`
records the selected provider. For `ExtractedDocument` input, chunks can also report
the IDs of the cleaned elements they cover. Pass a unified cleaner for extracted
documents, as with [document cleaning](DOCUMENTS.md).

By default, the output preset prefers extracted element boundaries, Markdown headings
when you pass `document_format="markdown"`, and plain text boundaries otherwise.
The Markdown strategy uses the markup retained by the cleaner. Configure cleaning
to preserve the markup and newlines you want it to use.

Set `TextCleanerConfig(output_chunking="text", output_chunk_max_size=512)` for a
pipeline default. The call can override both with `chunking=` and `max_size=`.
Size is in characters by default. Pass `count=` with the downstream model's token
counter to use its units. SCT keeps built-in placeholders and configured markers
together and reports an error if a protected span cannot fit the budget.
Synthetic chunk output requires unified processing, which supplies exact rendered ranges for synthetic values.
Chunk output does not support indexed or reversible modes. Use direct text processing when you need a reversible map.
See the [chunking API reference](reference/chunking.md) for call signatures and result fields.

## Select windows for PII detectors

Unified processing accepts `detector_chunking` as the shared default and
`DetectorSpec(..., chunking=...)` as a per-detector override. With neither,
each detector uses its native behavior. An explicit `"native"` override also
keeps that behavior when a shared default applies to other detectors.

```python
from sct import ChunkingPolicy, ChunkingRule, DetectorSpec, TextCleanerConfig

config = TextCleanerConfig(
    pii_processing="unified",
    allow_network_detectors=True,
    detector_specs=(
        DetectorSpec("builtin_regex", chunking="native"),
        DetectorSpec("builtin_ner", chunking=ChunkingPolicy(
            "auto",
            rules=(ChunkingRule("semchunk", min_chars=4000),),
            candidates=("native",),
        )),
    ),
)
```

`auto` takes the first compatible matching rule, then the first compatible
candidate, then native behavior. Rules can match the SCT language ID, a caller
format, and inclusive character limits. For ordinary `process()` calls, supply a
format with `DetectionContext(document_format="markdown")`. Rules are optional.
SCT supplies an output preset. Applications configure their own PII rules.

Built-in ONNX and Torch NER use the model tokenizer and the existing overlap
setting. Regex, GLiNER-only, and Presidio GLiNER use their native behavior.
External detector adapters can opt in by declaring `supports_windowed=True`
and implementing `window_limits(language)`, which returns
`DetectorWindowLimits(max_size, count, overlap_chars=0)`. SCT checks offsets and
budgets, projects findings back to the document, and applies privacy policy once.
If an externally windowed detector runs in a calibrated fast stage, SCT also runs
the full stage because the calibration does not qualify the new windowing.

## Providers and custom adapters

| Name | Stages | Boundary strategy |
| --- | --- | --- |
| `text` | Detection, output | Paragraphs, lines, whitespace, and bounded cuts |
| `markdown` | Output | Markdown headings, then text boundaries |
| `elements` | Output | Cleaned element boundaries, then text boundaries |
| `semchunk` | Detection, output | Recursive boundaries sized with the stage counter |
| `chonkie` | Output | Chonkie recursive boundary hints |

Install `pip install 'SqueakyCleanText[chunking]'` to use Semchunk and Chonkie.
`auto` skips optional providers that are unavailable. A pinned provider must
be available and compatible with its stage.

For another package, register a callable with `ChunkerRegistry.register()`
and pass the registry as `TextCleaner(..., chunker_registry=registry)`.
The callable receives a `ChunkRequest` and returns ordered `TextRange` values.
Return ranges into the supplied text. Account for all input, including separators.
Output ranges can overlap. Detector ranges cover the input once before SCT
adds the detector's window context. Use `request.count` and `request.max_size`
to check the size of each slice.

Chunkers run inside the host application. Register only trusted code.
Providers declared with `network_access=True` require
`allow_network_chunkers=True` and can receive sensitive data during detection.

`process_rag_chunks()` still cleans chunks that you already have. Streaming
redactors use their existing buffer behavior.
