"""Plain text files."""

import codecs

from foldersum.errors import ExtractionError
from foldersum.langdetect_utils import detect_language
from foldersum.models import ExtractedDoc, SourceFile

_MAX_TEXT_BYTES = 10 * 1024 * 1024  # a larger file is read only up to this size

_BOM_ENCODINGS = (
    (codecs.BOM_UTF32_LE, "utf-32"),
    (codecs.BOM_UTF32_BE, "utf-32"),
    (codecs.BOM_UTF16_LE, "utf-16"),
    (codecs.BOM_UTF16_BE, "utf-16"),
)


def _decode(data: bytes, complete: bool = True) -> str:
    """Decode a text file; ``complete=False`` means ``data`` was cut at an arbitrary byte, so an
    incomplete character at the end is dropped instead of making the whole text look non-UTF-8."""
    for bom, encoding in _BOM_ENCODINGS:
        if data.startswith(bom):
            return data.decode(encoding, errors="replace")
    try:
        return codecs.getincrementaldecoder("utf-8-sig")().decode(data, final=complete)
    except UnicodeDecodeError:
        return data.decode("cp1251", errors="replace")


class TextExtractor:
    def extract(self, source: SourceFile) -> ExtractedDoc:
        try:
            with source.path.open("rb") as handle:
                data = handle.read(_MAX_TEXT_BYTES + 1)
        except OSError as exc:
            raise ExtractionError(f"Cannot read {source.path.name}: {exc}") from exc
        truncated = len(data) > _MAX_TEXT_BYTES
        text = _decode(data[:_MAX_TEXT_BYTES], complete=not truncated).strip()
        warnings: tuple[str, ...] = ()
        if truncated:
            warnings = (f"text truncated to the first {_MAX_TEXT_BYTES // 2**20} MB",)
        return ExtractedDoc(source=source, text=text, lang=detect_language(text), warnings=warnings)
