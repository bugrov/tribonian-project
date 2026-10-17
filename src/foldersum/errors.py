"""Exceptions raised by foldersum."""


class ExtractionError(RuntimeError):
    """A file could not be turned into text."""


class UnsupportedFileError(ExtractionError):
    """No extractor exists for this kind of file."""


class OcrError(ExtractionError):
    """OCR failed or Tesseract (or a language pack) is not available."""


class LlmError(RuntimeError):
    """The language model backend is unreachable or returned an unusable answer.

    Fatal for the whole run: connection, credentials, missing model, rate limits, timeouts.
    """


class LlmContentError(LlmError):
    """The model could not process this particular input (refusal, empty answer, context
    overflow). Only the file concerned fails; the run continues with the other files."""
