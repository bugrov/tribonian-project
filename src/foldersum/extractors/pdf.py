"""PDF files: the text layer first, OCR of rendered pages for pages that have none."""

import math
from pathlib import Path

import pypdfium2
from PIL import Image
from pypdf import PdfReader

from foldersum.errors import ExtractionError, OcrError
from foldersum.langdetect_utils import detect_language
from foldersum.models import ExtractedDoc, SourceFile
from foldersum.ocr import MAX_OCR_PIXELS, OcrFn, ocr_image

_MIN_TEXT_CHARS_PER_PAGE = 20
_RENDER_SCALE = 2.0


def _read_text_layer(source: SourceFile) -> list[str]:
    """Return the stripped text of every page."""
    try:
        reader = PdfReader(source.path)
        if reader.is_encrypted and not reader.decrypt(""):
            raise ExtractionError(f"{source.path.name} is password-protected")
        return [(page.extract_text() or "").strip() for page in reader.pages]
    except ExtractionError:
        raise
    except Exception as exc:  # pypdf raises many unrelated types on malformed files
        raise ExtractionError(f"Cannot read PDF {source.path.name}: {exc}") from exc


class PdfExtractor:
    def __init__(self, ocr: OcrFn = ocr_image, max_ocr_pages: int = 50) -> None:
        self._ocr = ocr
        self._max_ocr_pages = max_ocr_pages

    def extract(self, source: SourceFile) -> ExtractedDoc:
        texts = _read_text_layer(source)
        empty_pages = [i for i, text in enumerate(texts) if len(text) < _MIN_TEXT_CHARS_PER_PAGE]
        warnings: tuple[str, ...] = ()
        if empty_pages:
            warnings = self._ocr_empty_pages(source.path, texts, empty_pages)
        text = "\n\n".join(texts).strip()
        return ExtractedDoc(
            source=source,
            text=text,
            lang=detect_language(text),
            pages=len(texts),
            warnings=warnings,
        )

    def _ocr_empty_pages(
        self, path: Path, texts: list[str], empty_pages: list[int]
    ) -> tuple[str, ...]:
        """OCR the pages without a usable text layer, filling ``texts`` in place."""
        targets = empty_pages[: self._max_ocr_pages]
        try:
            self._ocr_into(path, texts, targets)
        except OcrError as exc:
            if not any(texts):
                raise
            return (f"OCR unavailable, {len(empty_pages)} page(s) left without text: {exc}",)
        warnings: tuple[str, ...] = (f"OCR used for {len(targets)} of {len(texts)} pages",)
        skipped = len(empty_pages) - len(targets)
        if skipped:
            warnings += (f"OCR limit reached: {skipped} page(s) left without text",)
        return warnings

    def _ocr_into(self, path: Path, texts: list[str], targets: list[int]) -> None:
        try:
            document = pypdfium2.PdfDocument(path)
        except pypdfium2.PdfiumError as exc:
            raise ExtractionError(f"Cannot render PDF {path.name}: {exc}") from exc
        try:
            for index in targets:
                texts[index] = self._ocr(self._render(document, index)).strip()
        except pypdfium2.PdfiumError as exc:
            raise ExtractionError(f"Cannot render PDF {path.name}: {exc}") from exc
        finally:
            document.close()

    @staticmethod
    def _render(document: pypdfium2.PdfDocument, index: int) -> Image.Image:
        """Render a page, shrinking the scale for huge pages so memory stays bounded."""
        page = document[index]
        try:
            width, height = page.get_size()
            scale = min(_RENDER_SCALE, math.sqrt(MAX_OCR_PIXELS / max(width * height, 1)))
            image: Image.Image = page.render(scale=scale).to_pil()
            return image
        finally:
            page.close()
