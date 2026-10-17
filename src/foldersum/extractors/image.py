"""Images (JPG, PNG, WebP, ...) via OCR."""

from PIL import Image, ImageOps

from foldersum.errors import ExtractionError
from foldersum.langdetect_utils import detect_language
from foldersum.models import ExtractedDoc, SourceFile
from foldersum.ocr import OcrFn, ocr_image


def _flatten(image: Image.Image) -> Image.Image:
    """Grayscale copy; transparency is composited onto white so dark text stays visible."""
    if image.mode in ("RGBA", "LA", "PA") or "transparency" in image.info:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, "white")
        return Image.alpha_composite(background, rgba).convert("L")
    return image.convert("L")


class ImageExtractor:
    def __init__(self, ocr: OcrFn = ocr_image) -> None:
        self._ocr = ocr

    def extract(self, source: SourceFile) -> ExtractedDoc:
        try:
            with Image.open(source.path) as raw:
                image = _flatten(ImageOps.exif_transpose(raw))
        except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
            raise ExtractionError(f"Cannot open image {source.path.name}: {exc}") from exc
        text = self._ocr(image).strip()
        warnings = () if text else ("no text recognized in the image",)
        return ExtractedDoc(source=source, text=text, lang=detect_language(text), warnings=warnings)
