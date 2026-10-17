import os
import shutil
from pathlib import Path

import numpy as np
import pytesseract
import pytest
from PIL import Image, ImageDraw, ImageFont

from foldersum import ocr
from foldersum.errors import OcrError
from foldersum.ocr import ocr_image, prepare_for_ocr


def test_ocr_image_returns_recognized_text(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, str] = {}

    def fake(_: Image.Image, lang: str, timeout: int) -> str:
        seen["lang"] = lang
        seen["timeout"] = str(timeout)
        return "recognized"

    monkeypatch.setattr(pytesseract, "image_to_string", fake)

    assert ocr_image(Image.new("L", (10, 10))) == "recognized"
    assert seen["lang"] == "rus+eng"


def test_missing_tesseract_raises_ocr_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake(*_: object, **__: object) -> str:
        raise pytesseract.TesseractNotFoundError

    monkeypatch.setattr(pytesseract, "image_to_string", fake)

    with pytest.raises(OcrError, match="not installed"):
        ocr_image(Image.new("L", (10, 10)))


def test_tesseract_failure_raises_ocr_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake(*_: object, **__: object) -> str:
        raise pytesseract.TesseractError(1, "bad language")

    monkeypatch.setattr(pytesseract, "image_to_string", fake)

    with pytest.raises(OcrError, match="language packs"):
        ocr_image(Image.new("L", (10, 10)))


def test_windows_default_location_used_when_not_on_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_exe = tmp_path / "tesseract.exe"
    fake_exe.write_bytes(b"")
    monkeypatch.setattr(ocr, "_WINDOWS_TESSERACT", fake_exe)
    monkeypatch.setattr(shutil, "which", lambda _: None)
    monkeypatch.setattr(pytesseract.pytesseract, "tesseract_cmd", "tesseract")

    ocr._locate_tesseract()

    assert pytesseract.pytesseract.tesseract_cmd == str(fake_exe)


def test_path_tesseract_is_left_alone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_exe = tmp_path / "tesseract.exe"
    fake_exe.write_bytes(b"")
    monkeypatch.setattr(ocr, "_WINDOWS_TESSERACT", fake_exe)
    monkeypatch.setattr(shutil, "which", lambda _: "/usr/bin/tesseract")
    monkeypatch.setattr(pytesseract.pytesseract, "tesseract_cmd", "tesseract")

    ocr._locate_tesseract()

    assert pytesseract.pytesseract.tesseract_cmd == "tesseract"


def _tesseract_ready() -> bool:
    ocr._locate_tesseract()
    try:
        return "eng" in pytesseract.get_languages()
    except (pytesseract.TesseractNotFoundError, pytesseract.TesseractError):
        return False


@pytest.mark.skipif(not _tesseract_ready(), reason="Tesseract with 'eng' is not available")
def test_real_tesseract_reads_english_text() -> None:
    image = Image.new("L", (900, 120), "white")
    ImageDraw.Draw(image).text(
        (20, 30), "Hello world", fill="black", font=ImageFont.load_default(60)
    )

    assert "Hello" in ocr_image(image, languages="eng")


def test_user_tessdata_dir_is_used_when_prefix_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "tessdata").mkdir()
    (tmp_path / "tessdata" / "rus.traineddata").write_bytes(b"")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("TESSDATA_PREFIX", raising=False)

    ocr._locate_tesseract()

    assert os.environ["TESSDATA_PREFIX"] == str(tmp_path / "tessdata")
    monkeypatch.delenv("TESSDATA_PREFIX")


def test_existing_tessdata_prefix_is_kept(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "tessdata").mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("TESSDATA_PREFIX", "custom")

    ocr._locate_tesseract()

    assert os.environ["TESSDATA_PREFIX"] == "custom"


def test_prepare_upscales_small_images_and_keeps_large_ones() -> None:
    small = prepare_for_ocr(Image.new("RGB", (450, 800), "white"))
    large = prepare_for_ocr(Image.new("RGB", (3000, 1000), "white"))

    assert small.size == (1350, 2400)
    assert small.mode == "L"
    assert large.size == (3000, 1000)


def test_prepare_flattens_uneven_lighting() -> None:
    width, height = 600, 300
    gradient = np.tile(np.linspace(90, 250, width, dtype=np.uint8), (height, 1))
    assert gradient[:, 50:150].mean() < gradient[:, -150:-50].mean() - 50

    flat = np.asarray(prepare_for_ocr(Image.fromarray(gradient))).astype(int)

    assert abs(flat[:, 50:150].mean() - flat[:, -150:-50].mean()) < 10


def test_user_tessdata_without_russian_pack_is_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "tessdata").mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("TESSDATA_PREFIX", raising=False)

    ocr._locate_tesseract()

    assert "TESSDATA_PREFIX" not in os.environ


def test_tesseract_timeout_raises_ocr_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake(*_: object, **__: object) -> str:
        raise RuntimeError("Tesseract process timeout")

    monkeypatch.setattr(pytesseract, "image_to_string", fake)

    with pytest.raises(OcrError, match="timed out"):
        ocr_image(Image.new("L", (10, 10)))


def test_prepare_downscales_huge_images(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ocr, "MAX_OCR_PIXELS", 10_000)

    prepared = prepare_for_ocr(Image.new("L", (400, 400)))

    assert prepared.width * prepared.height <= 10_000
