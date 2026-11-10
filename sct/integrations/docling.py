"""Extract PDFs using verified local Docling assets without implicit downloads."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib.metadata
import json
from io import BytesIO
from pathlib import Path
from typing import Any

from sct.documents import DocumentElement, ExtractedDocument

_LAYOUT = ('docling-project/docling-layout-heron', '8f39ad3c0b4c58e9c2d2c84a38465abf757272d8')
_TABLE = ('docling-project/docling-models', 'fc0f2d45e2218ea24bce5045f58a389aed16dc23')
_LAYOUT_FILES = ('config.json', 'preprocessor_config.json', 'model.safetensors')
_TABLE_FILES = ('model_artifacts/tableformer/accurate/tm_config.json',
                'model_artifacts/tableformer/accurate/tableformer_accurate.safetensors')


def _digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _sdk() -> None:
    try:
        if importlib.metadata.version('docling') != '2.148.0':
            raise ValueError
    except (importlib.metadata.PackageNotFoundError, ValueError):
        raise ValueError('Install the documents extra with Docling 2.148.0') from None


def _rapid_files(path: Path) -> dict[Path, str]:
    # The pinned provider resolves every OCR model and any external dictionary before pipeline setup.
    from docling.models.stages.ocr.rapid_ocr_model import (
        _backend_to_engine_type, _rapidocr_artifacts, _resolve_rapidocr,
    )
    resolved = _resolve_rapidocr('en', 'onnxruntime')
    artifacts = _rapidocr_artifacts(path / 'RapidOcr', _backend_to_engine_type('onnxruntime'),
                                   resolved.ppocr_version, resolved.rapidocr_code, model_size='small')
    return {dest: url for artifact in artifacts.values() for dest, url in artifact.files.items()}


def prepare_docling_artifacts(path: Path, *, allow_download: bool = False) -> None:
    """Explicitly download pinned PDF assets into a new directory and record their identities."""
    if allow_download is not True:
        raise ValueError('Artifact setup requires allow_download=True')
    _sdk()
    try:
        from huggingface_hub import snapshot_download
        from docling.models.stages.ocr.rapid_ocr_model import RapidOcrModel

        path.mkdir(mode=0o700, parents=True, exist_ok=False)
        for (repo, revision), files in ((_LAYOUT, _LAYOUT_FILES), (_TABLE, _TABLE_FILES)):
            snapshot_download(repo, revision=revision, allow_patterns=list(files),
                              local_dir=path / repo.replace('/', '--'))
        RapidOcrModel.download_models('onnxruntime', local_dir=path / 'RapidOcr', lang='en')
        required = _required_files(path)
        manifest = {'schema_version': 1, 'docling': '2.148.0',
                    'models': dict((_LAYOUT, _TABLE)),
                    'ocr_sources': {str(p.relative_to(path)): url for p, url in _rapid_files(path).items()},
                    'files': {str(p.relative_to(path)): _digest(p) for p in required}}
        (path / 'sct-artifacts.json').write_text(json.dumps(manifest, sort_keys=True), encoding='utf-8')
    except Exception:
        raise ValueError('Artifact setup failed; inspect the private setup directory and use a new directory') from None


def _required_files(path: Path) -> tuple[Path, ...]:
    return (tuple(path / _LAYOUT[0].replace('/', '--') / name for name in _LAYOUT_FILES)
            + tuple(path / _TABLE[0].replace('/', '--') / name for name in _TABLE_FILES)
            + tuple(_rapid_files(path)))


def _validate_artifacts(path: Path) -> None:
    # Fail before importing the pipeline if the explicit local artifact manifest is missing.
    try:
        manifest = json.loads((path / 'sct-artifacts.json').read_text(encoding='utf-8'))
        _sdk()
        required = _required_files(path)
        expected = {str(p.relative_to(path)) for p in required}
        if (manifest['schema_version'] != 1 or manifest['docling'] != '2.148.0'
                or manifest['models'] != dict((_LAYOUT, _TABLE)) or set(manifest['files']) != expected
                or manifest['ocr_sources'] != {str(p.relative_to(path)): url for p, url in _rapid_files(path).items()}
                or any(not p.is_file() or _digest(p) != manifest['files'][str(p.relative_to(path))] for p in required)):
            raise ValueError
    except Exception:
        raise ValueError('Local artifacts are missing or invalid; '
                         'run prepare_docling_artifacts with explicit download permission') from None


class DoclingExtractor:
    """PDF extraction with pinned CPU layout, table structure and local English RapidOCR."""

    def __init__(self, *, artifacts_path: Path):
        self.artifacts_path = Path(artifacts_path)

    def extract(self, path: Path) -> ExtractedDocument:
        _validate_artifacts(self.artifacts_path)
        try:
            from docling.datamodel.accelerator_options import AcceleratorOptions
            from docling.datamodel.base_models import ConversionStatus, DocumentStream, InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
            from docling.document_converter import DocumentConverter, PdfFormatOption

            options = PdfPipelineOptions(artifacts_path=self.artifacts_path,
                accelerator_options=AcceleratorOptions(device='cpu', num_threads=2),
                enable_remote_services=False, allow_external_plugins=False,
                do_ocr=True, ocr_options=RapidOcrOptions(lang=['en'], backend='onnxruntime'),
                do_table_structure=True, do_code_enrichment=False, do_formula_enrichment=False,
                do_picture_classification=False, do_picture_description=False, do_chart_extraction=False,
                generate_page_images=False, generate_picture_images=False, generate_parsed_pages=False)
            converter = DocumentConverter(allowed_formats=[InputFormat.PDF],
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)})
            result = converter.convert(DocumentStream(name='input.pdf', stream=BytesIO(path.read_bytes())),
                                       raises_on_error=False)
            if result.status != ConversionStatus.SUCCESS:
                raise ValueError
            return _extract_elements(result.document)
        except Exception:
            raise ValueError('PDF conversion failed; check the source and verified local artifacts') from None


def _extract_elements(document: Any) -> ExtractedDocument:
    from docling_core.types.doc import ContentLayer, TableItem, TextItem

    elements: list[DocumentElement] = []
    table_number = 0
    from itertools import chain
    items = chain(document.iterate_items(included_content_layers=set(ContentLayer), traverse_pictures=True),
                  document.iterate_items(root=document.furniture, included_content_layers=set(ContentLayer),
                                         traverse_pictures=True))
    for item, _ in items:
        if not isinstance(item, (TextItem, TableItem)):
            continue
        if not item.prov:
            raise ValueError('Unsupported document element provenance')
        if isinstance(item, TextItem):
            previous_end = 0
            for provenance in item.prov:
                start, end = provenance.charspan
                if (type(start) is not int or type(end) is not int or start < previous_end
                        or end <= start or end > len(item.text)
                        or any(not char.isspace() for char in item.text[previous_end:start])):
                    raise ValueError('Unsupported text provenance ranges')
                page = provenance.page_no
                box = provenance.bbox.to_top_left_origin(document.pages[page].size.height)
                elements.append(DocumentElement(f'e{len(elements)}', item.text[start:end],
                    page=page, bbox=(box.l, box.t, box.r, box.b)))
                previous_end = end
            if any(not char.isspace() for char in item.text[previous_end:]):
                raise ValueError('Unsupported text provenance ranges')
            continue
        if len(item.prov) != 1:
            raise ValueError('Unsupported table provenance')
        provenance = item.prov[0]
        page = provenance.page_no
        box = provenance.bbox.to_top_left_origin(document.pages[page].size.height)
        if isinstance(item, TableItem):
            for cell in item.data.table_cells:
                cell_box = cell.bbox
                if cell_box is not None:
                    cell_box = cell_box.to_top_left_origin(document.pages[page].size.height)
                    cell_bounds = (cell_box.l, cell_box.t, cell_box.r, cell_box.b)
                else:
                    cell_bounds = None
                elements.append(DocumentElement(f'e{len(elements)}', cell.text, kind='table_cell',
                    page=page, bbox=cell_bounds, table=table_number,
                    cell=(cell.start_row_offset_idx, cell.end_row_offset_idx,
                          cell.start_col_offset_idx, cell.end_col_offset_idx), separator='\t'))
            if elements:
                elements[-1] = replace(elements[-1], separator='\n')
            table_number += 1
    # Split page-spanning logical items before restoring physical page/layout order, including furniture.
    elements.sort(key=lambda e: (e.page or 0, e.bbox[1] if e.bbox else 0, e.bbox[0] if e.bbox else 0))
    return ExtractedDocument(tuple(replace(e, id=f'e{i}') for i, e in enumerate(elements)))
