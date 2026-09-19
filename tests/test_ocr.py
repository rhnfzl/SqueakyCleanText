from sct import TextCleaner, TextCleanerConfig
from sct.ocr import OCRSpan, select_redaction_regions


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
