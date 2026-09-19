"""Optional OCR and image redaction interfaces."""

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from sct.adapters import is_sensitive_result
from sct.sct import TextCleaner


@dataclass(frozen=True)
class OCRSpan:
    """OCR text and its image bounding box."""

    text: str
    bbox: tuple[int, int, int, int]
    confidence: float


class OCRProvider(Protocol):
    """Boundary for OCR engines."""

    def extract(self, image: Any) -> Sequence[OCRSpan]:
        """Return text spans with image coordinates."""


@dataclass(frozen=True)
class ImageRedactionResult:
    """Redacted image and the regions that were covered."""

    image: Any
    regions: tuple[OCRSpan, ...]


def select_redaction_regions(
    cleaner: TextCleaner,
    spans: Sequence[OCRSpan],
) -> tuple[OCRSpan, ...]:
    """Select OCR spans that contain privacy replacements or findings."""
    selected = []
    for span in spans:
        result = cleaner.process(span.text)
        if is_sensitive_result(cleaner, span.text, result):
            selected.append(span)
    return tuple(selected)


def redact_image(
    cleaner: TextCleaner,
    image: Any,
    provider: OCRProvider,
    *,
    fill: str = "black",
) -> ImageRedactionResult:
    """Cover sensitive OCR regions on a copied Pillow image."""
    try:
        from PIL import ImageDraw
    except ImportError as exc:
        raise ImportError(
            "Pillow is required for image redaction. "
            "Install with: pip install squeakycleantext[ocr]"
        ) from exc

    output = image.copy()
    regions = select_redaction_regions(cleaner, provider.extract(image))
    drawing = ImageDraw.Draw(output)
    for region in regions:
        drawing.rectangle(region.bbox, fill=fill)
    return ImageRedactionResult(output, regions)


class TesseractOCRProvider:
    """OCR provider backed by pytesseract and a local Tesseract binary."""

    def __init__(self, language: str = "eng") -> None:
        self.language = language

    def extract(self, image: Any) -> tuple[OCRSpan, ...]:
        try:
            import pytesseract
        except ImportError as exc:
            raise ImportError(
                "pytesseract is required for Tesseract OCR. "
                "Install with: pip install squeakycleantext[ocr]"
            ) from exc

        data = pytesseract.image_to_data(
            image,
            lang=self.language,
            output_type=pytesseract.Output.DICT,
        )
        lines = {}
        for index, text in enumerate(data["text"]):
            text = text.strip()
            if not text:
                continue
            left = int(data["left"][index])
            top = int(data["top"][index])
            width = int(data["width"][index])
            height = int(data["height"][index])
            line_key = tuple(
                data.get(field, [index] * len(data["text"]))[index]
                for field in ("page_num", "block_num", "par_num", "line_num")
            )
            lines.setdefault(line_key, []).append((
                text,
                (left, top, left + width, top + height),
                max(0.0, float(data["conf"][index]) / 100),
            ))

        spans = []
        for words in lines.values():
            boxes = [word[1] for word in words]
            spans.append(OCRSpan(
                text=" ".join(word[0] for word in words),
                bbox=(
                    min(box[0] for box in boxes),
                    min(box[1] for box in boxes),
                    max(box[2] for box in boxes),
                    max(box[3] for box in boxes),
                ),
                confidence=sum(word[2] for word in words) / len(words),
            ))
        return tuple(spans)
