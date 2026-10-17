"""Plain text files."""

import codecs

from foldersum.errors import ExtractionError
from foldersum.langdetect_utils import detect_language
from foldersum.models import ExtractedDoc, SourceFile

_BOM_ENCODINGS = (
    (codecs.BOM_UTF32_LE, "utf-32"),
    (codecs.BOM_UTF32_BE, "utf-32"),
    (codecs.BOM_UTF16_LE, "utf-16"),
    (codecs.BOM_UTF16_BE, "utf-16"),
)


def _decode(data: bytes) -> str:
    for bom, encoding in _BOM_ENCODINGS:
        if data.startswith(bom):
            return data.decode(encoding, errors="replace")
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("cp1251", errors="replace")


class TextExtractor:
    def extract(self, source: SourceFile) -> ExtractedDoc:
        try:
            text = _decode(source.path.read_bytes()).strip()
        except OSError as exc:
            raise ExtractionError(f"Cannot read {source.path.name}: {exc}") from exc
        return ExtractedDoc(source=source, text=text, lang=detect_language(text))
