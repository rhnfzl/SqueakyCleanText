"""Document projection protects one canonical input without retaining source objects."""
from dataclasses import asdict, replace
import importlib.util
import json
from pathlib import Path

import pytest

from sct import TextCleaner, TextCleanerConfig
from sct.detectors import DetectorCapabilities, DetectorRegistry, DetectorSpec
from sct.findings import Detection


def api():
    assert importlib.util.find_spec('sct.documents') is not None, 'document API is missing'
    from sct.documents import DocumentElement, ExtractedDocument, clean_document, extract_document
    return DocumentElement, ExtractedDocument, clean_document, extract_document


def cleaner(**overrides):
    return TextCleaner(TextCleanerConfig.from_preset('training', pii_processing='unified',
        check_ner_process=False, check_detect_language=False, **overrides))


@pytest.mark.parametrize('mode', ['placeholder', 'synthetic'])
@pytest.mark.parametrize('normalize', [False, True])
def test_e1_acceptance(mode, normalize):
    Element, Document, clean, extract = api()
    seen = []
    class Names:
        capabilities = DetectorCapabilities('fixture', entity_types=('PERSON',))
        def detect(self, text, *, language, context):
            seen.append(text)
            start = text.index('Alice')
            return (Detection('PERSON', start, start + len('Alice\nExample'), 'fixture', .9, 'canonical'),)
    registry = DetectorRegistry()
    registry.register('fixture', lambda *_: Names())
    elements = (Element('e0', '😀 Alice', page=1), Element('e1', 'Example', page=2),
                Element('e2', 'table@example.com', kind='table_cell', page=2, cell=(0, 1, 1, 2)))
    class Provider:
        def extract(self, path):
            return Document(elements)
    source = extract(Path('unused.pdf'), provider=Provider())
    config = replace(cleaner().cfg, replacement_mode=mode, check_normalize_whitespace=normalize,
        detector_specs=(DetectorSpec('builtin_regex'), DetectorSpec('fixture')))
    processed = clean(source, TextCleaner(config, detector_registry=registry))
    assert seen == [source.text.replace('\n', ' ') if normalize else source.text]
    if mode == 'placeholder':
        assert processed.text == ('😀 <PERSON> <EMAIL>' if normalize else '😀 <PERSON>\n<EMAIL>')
    assert processed.element_ids == ('e0', 'e1', 'e2')
    assert processed.elements[1].text == ''
    assert processed.elements[0].text.startswith('😀 ')
    assert processed.elements[2].text and processed.elements[2].text != processed.text
    assert ''.join(e.text + e.separator for e in processed.elements[:-1]) + processed.elements[-1].text == processed.text
    assert processed.elements[2].cell == (0, 1, 1, 2)
    assert processed.elements[0].page == 1 and processed.elements[1].page == 2
    output = json.dumps(asdict(processed))
    assert all(secret not in output for secret in ('Alice', 'Example', 'table@example.com'))
    assert set(asdict(processed)) == {'text', 'elements'}
    assert source.elements == elements
    assert 'Alice' not in repr(source) and 'Alice' not in repr(elements[0])


def test_separators_empty_elements_and_unicode_ranges():
    Element, Document, clean, _ = api()
    source = Document((Element('e0', 'é😀', separator='\t'), Element('e1', ''), Element('e2', 'unchanged')))
    assert source.text == 'é😀\t\nunchanged'
    assert source.source_ranges == ((0, 2), (3, 3), (4, 13))
    result = clean(source, cleaner())
    assert result.text == source.text
    assert [e.text for e in result.elements] == ['é😀', '', 'unchanged']


@pytest.mark.parametrize('kwargs', [dict(id='private@example.com', text='x'), dict(id='e0', text=1),
    dict(id='e0', text='x', page=0), dict(id='e0', text='x', bbox=(0, 0, float('nan'), 2)),
    dict(id='e0', text='x', bbox=(3, 0, 2, 1)), dict(id='e0', text='x', cell=(0, 0, 1, 1)),
    dict(id='e0', text='x', separator='secret')])
def test_invalid_structure_is_rejected_without_values(kwargs):
    Element, _, _, _ = api()
    with pytest.raises(ValueError) as error:
        Element(**kwargs)
    assert 'private@example.com' not in str(error.value)


def test_duplicate_ids_and_untyped_extraction_fail_privately():
    Element, Document, _, extract = api()
    with pytest.raises(ValueError, match='unique'):
        Document((Element('e0', 'x'), Element('e0', 'y')))
    class Provider:
        def extract(self, path):
            raise RuntimeError('secret source filename')
    with pytest.raises(ValueError) as error:
        extract(Path('sensitive-filename.pdf'), provider=Provider())
    assert 'secret' not in str(error.value) and 'sensitive' not in str(error.value)


@pytest.mark.parametrize('mode', ['indexed', 'reversible'])
def test_map_modes_are_rejected_before_processing(mode):
    Element, Document, clean, _ = api()
    with pytest.raises(ValueError, match='mode'):
        clean(Document((Element('e0', 'a@example.com'),)), cleaner(replacement_mode=mode))


def test_unified_opt_in_is_required():
    Element, Document, clean, _ = api()
    with pytest.raises(ValueError, match='unified'):
        clean(Document((Element('e0', 'a@example.com'),)), TextCleaner(replace(cleaner().cfg, pii_processing='legacy')))


def test_unavailable_and_ambiguous_covering_maps_fail():
    Element, Document, clean, _ = api()
    document = Document((Element('e0', 'first😀  part'), Element('e1', 'second')))
    with pytest.raises(ValueError, match='projection'):
        clean(document, cleaner(custom_pipeline_steps=(str.lower,)))
    with pytest.raises(ValueError, match='projection'):
        clean(document, cleaner(check_remove_emoji=True))


def test_covering_within_one_element_is_supported():
    Element, Document, clean, _ = api()
    result = clean(Document((Element('e0', 'é😀'),)), cleaner(check_to_ascii_unicode=True))
    assert result.text == result.elements[0].text
    assert 'é' not in result.text


def test_missing_artifacts_fail_before_loading_provider(tmp_path):
    api()
    from sct.integrations.docling import DoclingExtractor
    with pytest.raises(ValueError, match='artifacts') as error:
        DoclingExtractor(artifacts_path=tmp_path).extract(Path('secret@example.com.pdf'))
    assert 'secret@example.com' not in str(error.value)


def test_artifact_setup_requires_explicit_permission(tmp_path):
    from sct.integrations import docling
    assert hasattr(docling, 'prepare_docling_artifacts'), 'explicit artifact setup is missing'
    with pytest.raises(ValueError, match='allow_download'):
        docling.prepare_docling_artifacts(tmp_path / 'new')
    assert not (tmp_path / 'new').exists()


def test_native_docling_pdf_offline(tmp_path, monkeypatch, caplog):
    import importlib.metadata
    import os
    import socket
    import sys
    required = os.environ.get('SCT_OPTIONAL_PROFILE') == 'documents'
    if importlib.util.find_spec('docling') is None:
        if required:
            pytest.fail('The documents profile requires installed Docling')
        pytest.skip('Install the documents extra for native PDF qualification')
    assets = os.environ.get('SCT_DOCLING_ARTIFACTS')
    if not assets:
        if required:
            pytest.fail('The documents profile requires explicit SCT_DOCLING_ARTIFACTS')
        pytest.skip('Select verified local PDF artifacts explicitly')
    from sct.integrations.docling import DoclingExtractor
    import docling.document_converter
    assert importlib.metadata.version('docling') == '2.148.0'
    assert Path(docling.document_converter.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    network_attempts = []
    def denied(*args, **kwargs):
        network_attempts.append(True)
        raise AssertionError('Network access is forbidden during offline PDF conversion')
    monkeypatch.setattr(socket, 'create_connection', denied)
    monkeypatch.setattr(socket.socket, 'connect', denied)
    fixture = Path(__file__).parent / 'fixtures/documents/pii-layout.pdf'
    source = tmp_path / 'private-source@example.com.pdf'
    source.write_bytes(fixture.read_bytes())
    original = source.read_bytes()
    _, _, clean, extract = api()
    document = extract(source, provider=DoclingExtractor(artifacts_path=Path(assets)))
    assert 'Sample' in document.text
    assert 'scan@example.com' in document.text
    assert 'table@example.com' in document.text
    assert {e.page for e in document.elements} == {1, 2, 3}
    cells = [e for e in document.elements if e.kind == 'table_cell']
    assert cells and all(e.cell and e.bbox for e in cells)
    assert 'author@example.com' not in document.text
    seen = []
    class Names:
        capabilities = DetectorCapabilities('fixture_names', entity_types=('PERSON',))
        def detect(self, text, *, language, context):
            seen.append(text)
            hits = []
            for value in ('Alice\nExample', 'Mira\nSample', 'Noor Raster'):
                start = text.index(value)
                hits.append(Detection('PERSON', start, start + len(value), 'fixture_names', .9, 'canonical'))
            return tuple(hits)
    registry = DetectorRegistry()
    registry.register('fixture_names', lambda *_: Names())
    config = replace(cleaner().cfg, detector_specs=(DetectorSpec('builtin_regex'), DetectorSpec('fixture_names')))
    result = clean(document, TextCleaner(config, detector_registry=registry))
    assert seen == [document.text]
    serialized = json.dumps(asdict(result))
    for marker in ('Alice', 'Example', 'Mira', 'Sample', 'Noor Raster', 'scan@example.com', 'table@example.com',
                   'author@example.com', 'private-title@example.com', 'private-source@example.com'):
        assert marker not in serialized
        assert marker not in caplog.text
    assert result.element_ids == tuple(e.id for e in document.elements)
    assert source.read_bytes() == original
    from io import BytesIO
    from docling.datamodel.base_models import DocumentStream, InputFormat
    from docling.datamodel.accelerator_options import AcceleratorOptions
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
    control_options = PdfPipelineOptions(artifacts_path=Path(assets), do_ocr=False, do_table_structure=False,
        accelerator_options=AcceleratorOptions(device='cpu', num_threads=2))
    control = DocumentConverter(allowed_formats=[InputFormat.PDF], format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=control_options)}).convert(
            DocumentStream(name='control.pdf', stream=BytesIO(original)))
    assert 'scan@example.com' not in control.document.export_to_markdown()
    malformed = tmp_path / 'malformed-private@example.com.pdf'
    malformed.write_bytes(b'Not a PDF: confidential-fixture-value')
    with pytest.raises(ValueError) as error:
        extract(malformed, provider=DoclingExtractor(artifacts_path=Path(assets)))
    assert 'confidential-fixture-value' not in str(error.value) + caplog.text
    assert 'malformed-private@example.com' not in str(error.value) + caplog.text
    tampered = tmp_path / 'tampered-artifacts'
    tampered.mkdir()
    for directory in Path(assets).iterdir():
        if directory.is_dir():
            (tampered / directory.name).symlink_to(directory.resolve(), target_is_directory=True)
    manifest = json.loads((Path(assets) / 'sct-artifacts.json').read_text())
    manifest['files'][next(iter(manifest['files']))] = '0' * 64
    (tampered / 'sct-artifacts.json').write_text(json.dumps(manifest))
    with monkeypatch.context() as patch:
        patch.setattr(docling.document_converter, 'DocumentConverter', denied)
        with pytest.raises(ValueError, match='artifacts'):
            DoclingExtractor(artifacts_path=tampered).extract(source)
    assert not network_attempts
    print(json.dumps({'docling': importlib.metadata.version('docling'),
                      'source': docling.document_converter.__file__, 'pages': 3,
                      'elements': len(result.elements), 'table_cells': len(cells)}))


def test_table_identity_is_structural():
    Element, Document, clean, _ = api()
    cell = Element('e0', 'a@example.com', kind='table_cell', cell=(0, 1, 0, 1), table=2)
    assert clean(Document((cell,)), cleaner()).elements[0].table == 2


def test_ambiguous_covering_is_not_proven_by_a_finding():
    Element, Document, clean, _ = api()
    document = Document((Element('e0', '😀 intro a@example.com'), Element('e1', 'tail')))
    with pytest.raises(ValueError, match='projection'):
        clean(document, cleaner(check_remove_emoji=True))


@pytest.mark.parametrize('mode', ['placeholder', 'synthetic'])
def test_cross_cell_name_uses_one_shared_render(mode):
    Element, Document, clean, _ = api()
    class Names:
        capabilities = DetectorCapabilities('cells', entity_types=('PERSON',))
        def detect(self, text, *, language, context):
            assert text == 'Alex\tCase'
            return (Detection('PERSON', 0, len(text), 'cells', .9, 'canonical'),)
    registry = DetectorRegistry()
    registry.register('cells', lambda *_: Names())
    document = Document((Element('e0', 'Alex', kind='table_cell', page=1, table=0,
                                 cell=(0, 1, 0, 1), separator='\t'),
                         Element('e1', 'Case', kind='table_cell', page=1, table=0, cell=(0, 1, 1, 2))))
    config = replace(cleaner().cfg, detector_specs=(DetectorSpec('cells'),), replacement_mode=mode)
    result = clean(document, TextCleaner(config, detector_registry=registry))
    assert result.elements[0].text == result.text
    assert result.elements[1].text == ''
    assert result.elements[0].separator == ''
    assert 'Alex' not in json.dumps(asdict(result)) and 'Case' not in json.dumps(asdict(result))
    assert result.elements[0].cell == (0, 1, 0, 1)
    assert result.elements[1].cell == (0, 1, 1, 2)


def test_empty_document_and_whitespace_output_retain_only_structure():
    Element, Document, clean, _ = api()
    assert clean(Document(()), cleaner()).elements == ()
    result = clean(Document((Element('e0', '  '), Element('e1', ''))), cleaner())
    assert result.text == ''
    assert [e.text for e in result.elements] == ['', '']


def test_cleaner_errors_do_not_expose_values():
    Element, Document, clean, _ = api()
    def failure(text):
        raise ValueError(text)
    with pytest.raises(ValueError) as error:
        clean(Document((Element('e0', 'private@example.com'),)), cleaner(custom_pipeline_steps=(failure,)))
    assert 'private@example.com' not in str(error.value)


@pytest.mark.parametrize('missing', [False, True])
def test_setup_rejects_unqualified_sdk_without_creating_files(tmp_path, monkeypatch, missing):
    import importlib.metadata
    from sct.integrations.docling import prepare_docling_artifacts
    def version(name):
        if missing:
            raise importlib.metadata.PackageNotFoundError(name)
        return '0.0.0'
    monkeypatch.setattr(importlib.metadata, 'version', version)
    with pytest.raises(ValueError, match='documents extra'):
        prepare_docling_artifacts(tmp_path / 'new', allow_download=True)
    assert not (tmp_path / 'new').exists()


def test_native_sdk_invalid_provenance_and_missing_cell_bounds():
    import os
    if importlib.util.find_spec('docling') is None:
        if os.environ.get('SCT_OPTIONAL_PROFILE') == 'documents':
            pytest.fail('The documents profile requires installed Docling')
        pytest.skip('Install the documents extra for native SDK validation')
    from docling_core.types.doc import (
        BoundingBox, CoordOrigin, DocItemLabel, DoclingDocument, ProvenanceItem, Size, TableCell, TableData,
    )
    from sct.integrations.docling import _extract_elements
    box = BoundingBox(l=0, t=0, r=100, b=100, coord_origin=CoordOrigin.TOPLEFT)
    def document():
        doc = DoclingDocument(name='private metadata fixture')
        doc.add_page(1, Size(width=612, height=792))
        return doc
    def provenance(start, end):
        return ProvenanceItem(page_no=1, bbox=box, charspan=(start, end))
    doc = document()
    doc.add_text(label=DocItemLabel.TEXT, text='private@example.com')
    with pytest.raises(ValueError, match='provenance'):
        _extract_elements(doc)
    for start, end in ((0, 40), (0, 2)):
        doc = document()
        doc.add_text(label=DocItemLabel.TEXT, text='private@example.com', prov=provenance(start, end))
        with pytest.raises(ValueError, match='provenance') as error:
            _extract_elements(doc)
        assert 'private@example.com' not in str(error.value)
    doc = document()
    table = doc.add_table(data=TableData(num_rows=1, num_cols=1, table_cells=[TableCell(
        text='private@example.com', start_row_offset_idx=0, end_row_offset_idx=1,
        start_col_offset_idx=0, end_col_offset_idx=1)]), prov=provenance(0, 0))
    extracted = _extract_elements(doc)
    assert extracted.elements[0].bbox is None
    assert extracted.elements[0].cell == (0, 1, 0, 1)
    table.prov.append(provenance(0, 0))
    with pytest.raises(ValueError, match='provenance'):
        _extract_elements(doc)


@pytest.mark.parametrize('field', ['bbox', 'cell'])
def test_structural_tuple_subclasses_cannot_retain_source_objects(field):
    Element, Document, clean, _ = api()
    class SourceTuple(tuple):
        pass
    value = SourceTuple((0, 1, 1, 2))
    value.source_text = 'private@example.com'
    value.raw_sdk = object()
    kwargs = {'kind': 'table_cell', 'cell': (0, 1, 0, 1), 'bbox': (0, 0, 20, 20)}
    kwargs[field] = value
    try:
        element = Element('e0', 'private@example.com', **kwargs)
    except ValueError:
        return
    processed = clean(Document((element,)), cleaner())
    structural = getattr(processed.elements[0], field)
    assert type(structural) is tuple
    assert structural is not value
    assert not hasattr(structural, 'source_text')
    assert not hasattr(structural, 'raw_sdk')


def test_processed_elements_are_fresh_allowlisted_objects():
    Element, Document, clean, _ = api()
    element = Element('e0', 'private@example.com', bbox=(0, 0, 20, 20))
    raw_sdk = object()
    object.__setattr__(element, 'source_text', 'private@example.com')
    object.__setattr__(element, 'raw_sdk', raw_sdk)
    processed = clean(Document((element,)), cleaner())
    assert processed.elements[0] is not element
    assert not hasattr(processed.elements[0], 'raw_sdk')
    assert not hasattr(processed.elements[0], 'source_text')
    assert type(processed.elements) is tuple
    assert type(processed.elements[0].bbox) is tuple
    assert all(value is not raw_sdk for value in vars(processed.elements[0]).values())


@pytest.mark.parametrize('mode', ['placeholder', 'synthetic'])
def test_unicode_cross_element_name_with_disjoint_covering_ranges(mode):
    Element, Document, clean, _ = api()
    class Names:
        capabilities = DetectorCapabilities('split_cover', entity_types=('PERSON',))
        def detect(self, text, *, language, context):
            assert text == '😀 Alice Example'
            return (Detection('PERSON', 2, len(text), 'split_cover', .9, 'canonical'),)
    registry = DetectorRegistry()
    registry.register('split_cover', lambda *_: Names())
    document = Document((Element('e0', '😀 Alice', page=1), Element('e1', '<b>Example</b>', page=2)))
    config = replace(cleaner().cfg, detector_specs=(DetectorSpec('split_cover'),), replacement_mode=mode,
                     check_replace_html=True, replace_with_html='', check_normalize_whitespace=True)
    result = clean(document, TextCleaner(config, detector_registry=registry))
    assert result.text.startswith('😀 ')
    assert result.elements[1].text == '' and result.elements[0].text == result.text
    assert 'Alice' not in result.text and 'Example' not in result.text


def test_empty_opaque_string_subclass_cannot_retain_raw_source():
    Element, Document, clean, _ = api()
    class SourceString(str):
        pass
    empty = SourceString('')
    empty.source_text = 'private@example.com'
    empty.raw_sdk = object()
    try:
        processed = clean(Document((Element('e0', 'private@example.com'),)),
                          cleaner(custom_pipeline_steps=(lambda _: empty,)))
    except ValueError:
        return
    assert type(processed.text) is str
    assert not hasattr(processed.text, 'source_text')
    assert not hasattr(processed.text, 'raw_sdk')


@pytest.mark.parametrize('field', ['id', 'text', 'kind', 'separator', 'page', 'table', 'bbox', 'cell'])
def test_structural_scalar_subclasses_are_rejected(field):
    Element, _, _, _ = api()
    class SourceString(str):
        pass
    class SourceInt(int):
        pass
    class SourceFloat(float):
        pass
    values = {'id': SourceString('e0'), 'text': SourceString('private@example.com'),
              'kind': SourceString('table_cell'), 'separator': SourceString('\n'),
              'page': SourceInt(1), 'table': SourceInt(0),
              'bbox': (SourceFloat(0), 0, 20, 20), 'cell': (SourceInt(0), 1, 0, 1)}
    kwargs = {'id': 'e0', 'text': 'private@example.com', 'kind': 'table_cell'}
    kwargs[field] = values[field]
    with pytest.raises(ValueError):
        Element(**kwargs)


def test_element_and_root_container_subclasses_are_rejected():
    Element, Document, _, _ = api()
    class SourceElements(tuple):
        pass
    class SourceElement(Element):
        pass
    elements = SourceElements((Element('e0', 'private@example.com'),))
    elements.raw_sdk = object()
    with pytest.raises(ValueError):
        Document(elements)
    with pytest.raises(ValueError):
        Document((SourceElement('e0', 'private@example.com'),))
