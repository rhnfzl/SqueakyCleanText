import sys
from types import SimpleNamespace

import pytest

from sct import TextCleaner, TextCleanerConfig
from sct.ocr import (
    OCRSpan,
    TesseractOCRProvider,
    redact_image,
    select_redaction_regions,
)


def test_ocr_region_selection_marks_only_sensitive_text():
    cleaner = TextCleaner(cfg=TextCleanerConfig(
        check_ner_process=False,
        check_detect_language=False,
        check_statistical_model_processing=False,
    ))
    spans = (
        OCRSpan("maria@example.com", (0, 0, 100, 20), 0.99),
        OCRSpan("Invoice", (0, 30, 60, 50), 0.98),
        OCRSpan("Café", (0, 60, 60, 80), 0.97),
    )

    regions = select_redaction_regions(cleaner, spans)

    assert regions == (spans[0],)


def test_redact_image_covers_sensitive_regions_with_fake_pillow(monkeypatch):
    rectangles = []
    drawing = SimpleNamespace(
        rectangle=lambda bbox, *, fill: rectangles.append((bbox, fill))
    )
    monkeypatch.setitem(
        sys.modules,
        "PIL",
        SimpleNamespace(
            ImageDraw=SimpleNamespace(Draw=lambda image: drawing),
        ),
    )
    output = object()
    image = SimpleNamespace(copy=lambda: output)
    provider = SimpleNamespace(extract=lambda _image: (
        OCRSpan("maria@example.com", (1, 2, 30, 12), 0.99),
        OCRSpan("Invoice", (1, 20, 30, 30), 0.98),
    ))
    cleaner = TextCleaner(cfg=TextCleanerConfig(
        check_ner_process=False,
        check_detect_language=False,
        check_statistical_model_processing=False,
    ))

    result = redact_image(cleaner, image, provider, fill="white")

    assert result.image is output
    assert result.regions == (provider.extract(image)[0],)
    assert rectangles == [((1, 2, 30, 12), "white")]


def test_tesseract_provider_groups_words_into_lines(monkeypatch):
    data = {
        "text": ["Maria", "Chen", "", "Utrecht"],
        "left": [10, 45, 0, 15],
        "top": [20, 20, 0, 50],
        "width": [30, 35, 0, 40],
        "height": [12, 12, 0, 10],
        "conf": [90, 80, 0, 100],
        "page_num": [1, 1, 1, 1],
        "block_num": [1, 1, 1, 1],
        "par_num": [1, 1, 1, 1],
        "line_num": [1, 1, 1, 2],
    }
    fake_tesseract = SimpleNamespace(
        Output=SimpleNamespace(DICT="dict"),
        image_to_data=lambda image, *, lang, output_type: data,
    )
    monkeypatch.setitem(sys.modules, "pytesseract", fake_tesseract)

    spans = TesseractOCRProvider(language="nld").extract(object())

    assert spans[0].text == "Maria Chen"
    assert spans[0].bbox == (10, 20, 80, 32)
    assert spans[0].confidence == pytest.approx(0.85)
    assert spans[1] == OCRSpan("Utrecht", (15, 50, 55, 60), 1.0)
