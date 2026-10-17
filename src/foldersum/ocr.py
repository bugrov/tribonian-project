"""OCR through Tesseract."""

import math
import os
import shutil
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from PIL import Image

from foldersum.errors import OcrError

DEFAULT_LANGUAGES = "rus+eng"
_TARGET_SIDE = 2400
MAX_OCR_PIXELS = 25_000_000
_OCR_TIMEOUT_SECONDS = 180
_WINDOWS_TESSERACT = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")

OcrFn = Callable[[Image.Image], str]


def _locate_tesseract() -> None:
    """Find Tesseract and its language data when the environment is not set up for them.

    Falls back to the default Windows install location, and to ``~/tessdata`` for language
    packs (a user-writable place, since ``Program Files`` needs admin rights).
    """
    if shutil.which("tesseract") is None and _WINDOWS_TESSERACT.is_file():
        pytesseract.pytesseract.tesseract_cmd = str(_WINDOWS_TESSERACT)
    user_tessdata = Path.home() / "tessdata"
    if "TESSDATA_PREFIX" not in os.environ and (user_tessdata / "rus.traineddata").is_file():
        os.environ["TESSDATA_PREFIX"] = str(user_tessdata)


def prepare_for_ocr(image: Image.Image) -> Image.Image:
    """Grayscale, resize to a size Tesseract likes and flatten uneven lighting.

    Dividing by a heavily blurred copy removes shadows and gradients that photos of pages have;
    on the sample photos it raised exact word recall from ~78% to ~93% (see CLAUDE.md).
    """
    gray = np.asarray(image.convert("L"))
    pixels = gray.shape[0] * gray.shape[1]
    if pixels > MAX_OCR_PIXELS:
        factor = math.sqrt(MAX_OCR_PIXELS / pixels)
    else:
        factor = max(1.0, _TARGET_SIDE / max(gray.shape))
    if factor != 1.0:
        gray = cv2.resize(
            gray,
            None,
            fx=factor,
            fy=factor,
            interpolation=cv2.INTER_AREA if factor < 1 else cv2.INTER_CUBIC,
        )
    kernel = (max(gray.shape) // 20) | 1
    background = cv2.GaussianBlur(gray, (kernel, kernel), 0)
    flat = cv2.divide(gray, background, scale=255)
    cv2.normalize(flat, flat, 0, 255, cv2.NORM_MINMAX)
    return Image.fromarray(flat)


def ocr_image(image: Image.Image, languages: str = DEFAULT_LANGUAGES) -> str:
    """Recognize text in ``image``; raises OcrError if Tesseract or a language pack is missing."""
    _locate_tesseract()
    try:
        return str(
            pytesseract.image_to_string(
                prepare_for_ocr(image), lang=languages, timeout=_OCR_TIMEOUT_SECONDS
            )
        )
    except pytesseract.TesseractNotFoundError as exc:
        raise OcrError("Tesseract OCR is not installed or not on PATH") from exc
    except pytesseract.TesseractError as exc:
        raise OcrError(
            f"Tesseract failed; are the '{languages}' language packs installed?"
        ) from exc
    except RuntimeError as exc:  # pytesseract signals a timeout with a bare RuntimeError
        raise OcrError(f"Tesseract timed out after {_OCR_TIMEOUT_SECONDS}s") from exc
