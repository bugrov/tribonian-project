"""Exceptions raised while extracting content from files."""


class ExtractionError(RuntimeError):
    """A file could not be turned into text."""


class UnsupportedFileError(ExtractionError):
    """No extractor exists for this kind of file."""


class OcrError(ExtractionError):
    """OCR failed or Tesseract (or a language pack) is not available."""
