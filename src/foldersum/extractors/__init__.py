"""Content extractors, one per file kind."""

from foldersum.errors import UnsupportedFileError
from foldersum.extractors.base import Extractor
from foldersum.extractors.image import ImageExtractor
from foldersum.extractors.pdf import PdfExtractor
from foldersum.extractors.text import TextExtractor
from foldersum.models import ExtractedDoc, FileKind, SourceFile
from foldersum.ocr import OcrFn, ocr_image


def extract_document(source: SourceFile, ocr: OcrFn = ocr_image) -> ExtractedDoc:
    """Extract text from ``source`` with the extractor for its kind."""
    extractors: dict[FileKind, Extractor] = {
        FileKind.PDF: PdfExtractor(ocr),
        FileKind.IMAGE: ImageExtractor(ocr),
        FileKind.TEXT: TextExtractor(),
    }
    extractor = extractors.get(source.kind)
    if extractor is None:
        raise UnsupportedFileError(f"Unsupported file type ({source.kind}): {source.path.name}")
    return extractor.extract(source)
