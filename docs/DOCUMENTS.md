# Extract and clean documents

SCT cleans extracted PDF text while keeping element IDs, page boxes and table cell spans.
One cleaning pass can protect a name that crosses elements, pages or table cells.

This adapter does not redact the original PDF, images, metadata or attachments.
Keep those files private. The input file remains unchanged.
Layout boxes describe extracted provenance, not safe binary redaction coordinates.
OCR and detection can miss sensitive text. Review outputs before sharing them.

## Install and prepare local assets

Install `pip install 'SqueakyCleanText[documents]'`.
The adapter requires Docling 2.148.0 and an explicit local artifact directory.
Normal extraction does not download models or call remote services.
Missing or changed assets fail before converter construction.

Run setup separately with explicit download permission:

```python
from pathlib import Path
from sct.integrations.docling import prepare_docling_artifacts

prepare_docling_artifacts(Path('private-document-models'), allow_download=True)
```

Setup requires a new directory. It downloads pinned Heron layout and accurate TableFormer assets.
It also downloads the pinned provider's English RapidOCR ONNX detector, classifier and recognizer assets.
The provider resolves any required OCR dictionary before setup and extraction.
`sct-artifacts.json` records model revisions, OCR source URLs and SHA256 file digests.
Extraction validates this inventory before loading the pipeline.
Protect the directory against untrusted changes. File hashes do not authenticate an untrusted manifest.
Model licenses remain separate from SCT's license.

The qualified provider uses two CPU threads, English OCR and table structure extraction.
It disables remote services, external plugins and optional enrichments.
This version accepts PDFs only. Other document providers can implement the protocol below.

## Clean an extracted document

Configure and cache the selected cleaner's own detector models separately before running this example.
The `documents` extra supplies extraction, not the cleaner's optional NER dependencies.
The example grants detector permission with `allow_network_detectors=True`, which native NER requires even with cached artifacts.
Prepare and validate cleaner models separately when runtime network access must be blocked.

```python
from pathlib import Path
from sct import TextCleaner, TextCleanerConfig
from sct.documents import clean_document, extract_document
from sct.integrations.docling import DoclingExtractor

provider = DoclingExtractor(artifacts_path=Path('private-document-models'))
document = extract_document(Path('private-input.pdf'), provider=provider)
cleaner = TextCleaner(TextCleanerConfig.from_preset(
    'training', pii_processing='unified', allow_network_detectors=True,
))
processed = clean_document(document, cleaner)
print(processed.text)
```

Disabling `check_ner_process` leaves configured format detectors active but does not protect names by itself.

Unified processing requires explicit opt-in. Legacy cleaner defaults stay unchanged.
Indexed and reversible modes fail because this output cannot return their required maps.
Placeholder and synthetic modes support cross-element replacements with proven source associations.
Bounded separator normalization also supports cross-element placeholders and synthetic replacements.
Opaque callbacks and ambiguous cross-element normalization fail instead of guessing provenance.
Disable unsupported transforms or use the ordinary whole-text API when layout projection is unnecessary.

## Protocol and output

`DocumentExtractor.extract(path: Path) -> ExtractedDocument` is the provider protocol.
`extract_document(path, *, provider)` validates that result and hides provider exception values.
`clean_document(document, cleaner) -> ProcessedDocument` runs the existing unified whole-document pipeline.
The cleaner retains its existing long-document detection windows.

`DocumentElement` contains `id`, `text`, `kind`, `page`, `bbox`, `cell`, `table` and `separator`.
IDs are stable structural values such as `e0`, never source names or metadata.
`kind` is `text` or `table_cell`. Pages start at one.
Boxes use top-left page coordinates `(left, top, right, bottom)` in extracted layout units.
Missing cell boxes stay absent. `table` identifies the table within this document.
Structural tuples and scalar fields must use built-in primitive types. Subclasses with hidden state are rejected.
Cleaned elements are fresh allowlisted objects.
Cell spans are zero-based, end-exclusive `(row_start, row_end, column_start, column_end)`.

`ExtractedDocument.elements` is an immutable tuple.
Its canonical `text` joins elements with each recorded whitespace separator, excluding the final separator.
`source_ranges` records each element's zero-based, end-exclusive Unicode code-point range in that string.
These ranges are neither byte positions nor PDF coordinates.
Docling text items with several provenance spans become separate elements after validating every span.
Visible headers and footers are included. Elements use deterministic page, top and left layout order.
This physical order can differ from logical reading order in complex multi-column layouts.
Unmapped non-whitespace text and ambiguous multi-page tables fail explicitly.
Source metadata, filenames, titles, authors, images and raw SDK objects are not copied.

`ProcessedDocument` contains only cleaned `text` and cleaned `elements`.
Its `element_ids` property exposes stable IDs.
A cross-element placeholder appears once at the first covered element.
Later covered text becomes empty, and consumed separators disappear.
Joining cleaned elements and their remaining separators reconstructs the cleaned document exactly.
An unavailable or ambiguous source projection raises an error.
No raw text, source maps, findings with values or original SDK objects remain in the returned object.
Default representations omit extracted and cleaned text.

The checked synthetic PDF contains three pages, a table, raster-only OCR text and private metadata markers.
Required document-profile tests perform actual PDF conversion with network connections denied.
They verify extraction, cleaned output, table/page provenance and unchanged source bytes.
This fixture proves those paths run. It does not measure population OCR or PII recall.
