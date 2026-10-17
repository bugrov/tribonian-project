from pathlib import Path

import pypdfium2
import pytest
from PIL import Image
from pypdf import PdfReader, PdfWriter

from foldersum.errors import ExtractionError, OcrError, UnsupportedFileError
from foldersum.extractors import extract_document
from foldersum.extractors.image import ImageExtractor
from foldersum.extractors.pdf import PdfExtractor
from foldersum.extractors.text import TextExtractor
from foldersum.models import FileKind
from tests.conftest import PdfFactory, ScannedPdfFactory, SourceFactory

RU_TEXT = "Гражданский кодекс Российской Федерации, часть первая"
EN_TEXT = "The Civil Code of the Russian Federation, part one"


def never_ocr(_: Image.Image) -> str:
    raise AssertionError("OCR must not run")


# --- text ---------------------------------------------------------------------------------


def test_text_utf8_russian(tmp_path: Path, make_source: SourceFactory) -> None:
    path = tmp_path / "a.txt"
    path.write_text(RU_TEXT, encoding="utf-8")

    doc = TextExtractor().extract(make_source(path, FileKind.TEXT))

    assert doc.text == RU_TEXT
    assert doc.lang == "ru"


def test_text_cp1251_fallback(tmp_path: Path, make_source: SourceFactory) -> None:
    path = tmp_path / "a.txt"
    path.write_bytes(RU_TEXT.encode("cp1251"))

    doc = TextExtractor().extract(make_source(path, FileKind.TEXT))

    assert doc.text == RU_TEXT


def test_text_missing_file_raises(tmp_path: Path, make_source: SourceFactory) -> None:
    with pytest.raises(ExtractionError):
        TextExtractor().extract(make_source(tmp_path / "gone.txt", FileKind.TEXT))


# --- images -------------------------------------------------------------------------------


@pytest.mark.parametrize("suffix", [".jpg", ".png", ".webp"])
def test_image_runs_ocr_and_detects_language(
    tmp_path: Path, make_source: SourceFactory, suffix: str
) -> None:
    path = tmp_path / f"scan{suffix}"
    Image.new("RGB", (40, 30), "white").save(path)
    seen: list[tuple[int, int]] = []

    def fake_ocr(image: Image.Image) -> str:
        seen.append(image.size)
        return f"  {EN_TEXT}\n"

    doc = ImageExtractor(fake_ocr).extract(make_source(path, FileKind.IMAGE))

    assert seen == [(40, 30)]
    assert doc.text == EN_TEXT
    assert doc.lang == "en"
    assert doc.warnings == ()


def test_image_without_text_has_warning(tmp_path: Path, make_source: SourceFactory) -> None:
    path = tmp_path / "blank.png"
    Image.new("RGB", (10, 10), "white").save(path)

    doc = ImageExtractor(lambda _: "  ").extract(make_source(path, FileKind.IMAGE))

    assert doc.text == ""
    assert doc.lang == "unknown"
    assert doc.warnings == ("no text recognized in the image",)


def test_broken_image_raises(tmp_path: Path, make_source: SourceFactory) -> None:
    path = tmp_path / "broken.jpg"
    path.write_bytes(b"\xff\xd8\xff not really a jpeg")

    with pytest.raises(ExtractionError, match="Cannot open image"):
        ImageExtractor(never_ocr).extract(make_source(path, FileKind.IMAGE))


# --- pdf ----------------------------------------------------------------------------------


def test_pdf_text_layer_is_used_without_ocr(
    tmp_path: Path, make_source: SourceFactory, make_pdf: PdfFactory
) -> None:
    path = make_pdf(tmp_path / "a.pdf", [[EN_TEXT], ["Second page of the document"]])

    doc = PdfExtractor(never_ocr).extract(make_source(path, FileKind.PDF))

    assert EN_TEXT in doc.text
    assert "Second page" in doc.text
    assert doc.pages == 2
    assert doc.lang == "en"
    assert doc.warnings == ()


def test_scanned_pdf_falls_back_to_ocr(
    tmp_path: Path,
    make_source: SourceFactory,
    make_scanned_pdf: ScannedPdfFactory,
) -> None:
    path = make_scanned_pdf(tmp_path / "scan.pdf", 2)
    calls = 0

    def fake_ocr(_: Image.Image) -> str:
        nonlocal calls
        calls += 1
        return f"{RU_TEXT} {calls}"

    doc = PdfExtractor(fake_ocr).extract(make_source(path, FileKind.PDF))

    assert calls == 2
    assert doc.text == f"{RU_TEXT} 1\n\n{RU_TEXT} 2"
    assert doc.lang == "ru"
    assert doc.pages == 2
    assert doc.warnings == ("OCR used for 2 of 2 pages",)


def test_scanned_pdf_ocr_page_limit(
    tmp_path: Path,
    make_source: SourceFactory,
    make_scanned_pdf: ScannedPdfFactory,
) -> None:
    path = make_scanned_pdf(tmp_path / "scan.pdf", 3)
    calls = 0

    def fake_ocr(_: Image.Image) -> str:
        nonlocal calls
        calls += 1
        return EN_TEXT

    doc = PdfExtractor(fake_ocr, max_ocr_pages=1).extract(make_source(path, FileKind.PDF))

    assert calls == 1
    assert doc.warnings == (
        "OCR used for 1 of 3 pages",
        "OCR limit reached: 2 page(s) left without text",
    )


def test_scanned_pdf_without_tesseract_raises_ocr_error(
    tmp_path: Path,
    make_source: SourceFactory,
    make_scanned_pdf: ScannedPdfFactory,
) -> None:
    def broken_ocr(_: Image.Image) -> str:
        raise OcrError("no tesseract")

    path = make_scanned_pdf(tmp_path / "scan.pdf", 1)

    with pytest.raises(OcrError):
        PdfExtractor(broken_ocr).extract(make_source(path, FileKind.PDF))


def test_corrupt_pdf_raises(tmp_path: Path, make_source: SourceFactory) -> None:
    path = tmp_path / "bad.pdf"
    path.write_bytes(b"%PDF-1.4 this is not a pdf")

    with pytest.raises(ExtractionError, match="Cannot read PDF"):
        PdfExtractor(never_ocr).extract(make_source(path, FileKind.PDF))


def test_password_protected_pdf_raises(
    tmp_path: Path, make_source: SourceFactory, make_pdf: PdfFactory
) -> None:
    plain = make_pdf(tmp_path / "plain.pdf", [[EN_TEXT]])
    writer = PdfWriter(clone_from=PdfReader(plain))
    writer.encrypt("secret", algorithm="RC4-128")
    locked = tmp_path / "locked.pdf"
    with locked.open("wb") as handle:
        writer.write(handle)

    with pytest.raises(ExtractionError, match="password-protected"):
        PdfExtractor(never_ocr).extract(make_source(locked, FileKind.PDF))


def test_pdfium_cannot_open_pdf_raises(
    tmp_path: Path,
    make_source: SourceFactory,
    make_pdf: PdfFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failing(*_: object, **__: object) -> None:
        raise pypdfium2.PdfiumError("cannot open")

    monkeypatch.setattr(pypdfium2, "PdfDocument", failing)
    path = make_pdf(tmp_path / "short.pdf", [["hi"]])

    with pytest.raises(ExtractionError, match="Cannot render PDF"):
        PdfExtractor(never_ocr).extract(make_source(path, FileKind.PDF))


# --- dispatcher ---------------------------------------------------------------------------


def test_extract_document_routes_by_kind(tmp_path: Path, make_source: SourceFactory) -> None:
    path = tmp_path / "a.txt"
    path.write_text(EN_TEXT, encoding="utf-8")

    assert extract_document(make_source(path, FileKind.TEXT)).text == EN_TEXT


@pytest.mark.parametrize("kind", [FileKind.OFFICE, FileKind.UNKNOWN])
def test_unsupported_kinds_raise(
    tmp_path: Path, make_source: SourceFactory, kind: FileKind
) -> None:
    path = tmp_path / "a.bin"
    path.write_bytes(b"x")

    with pytest.raises(UnsupportedFileError):
        extract_document(make_source(path, kind))


def test_mixed_pdf_ocrs_only_pages_without_text(
    tmp_path: Path, make_source: SourceFactory, make_pdf: PdfFactory
) -> None:
    path = make_pdf(tmp_path / "mixed.pdf", [[EN_TEXT], [""], ["Third page has plenty of text"]])

    doc = PdfExtractor(lambda _: "RECOGNIZED PAGE TWO").extract(make_source(path, FileKind.PDF))

    assert doc.text.split("\n\n")[1] == "RECOGNIZED PAGE TWO"
    assert EN_TEXT in doc.text
    assert doc.warnings == ("OCR used for 1 of 3 pages",)


def test_ocr_failure_keeps_existing_text_layer(
    tmp_path: Path, make_source: SourceFactory, make_pdf: PdfFactory
) -> None:
    path = make_pdf(tmp_path / "mixed.pdf", [[EN_TEXT], [""]])

    def broken_ocr(_: Image.Image) -> str:
        raise OcrError("no tesseract")

    doc = PdfExtractor(broken_ocr).extract(make_source(path, FileKind.PDF))

    assert EN_TEXT in doc.text
    assert doc.warnings == ("OCR unavailable, 1 page(s) left without text: no tesseract",)


def test_render_scale_is_clamped_for_huge_pages(
    tmp_path: Path,
    make_source: SourceFactory,
    make_scanned_pdf: ScannedPdfFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("foldersum.extractors.pdf.MAX_OCR_PIXELS", 10_000)
    sizes: list[tuple[int, int]] = []

    def fake_ocr(image: Image.Image) -> str:
        sizes.append(image.size)
        return EN_TEXT

    path = make_scanned_pdf(tmp_path / "scan.pdf", 1)
    PdfExtractor(fake_ocr).extract(make_source(path, FileKind.PDF))

    assert sizes[0][0] * sizes[0][1] <= 10_000


def test_render_error_raises_extraction_error(
    tmp_path: Path,
    make_source: SourceFactory,
    make_scanned_pdf: ScannedPdfFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failing_render(*_: object) -> Image.Image:
        raise pypdfium2.PdfiumError("render failed")

    monkeypatch.setattr(PdfExtractor, "_render", staticmethod(failing_render))
    path = make_scanned_pdf(tmp_path / "scan.pdf", 1)

    with pytest.raises(ExtractionError, match="Cannot render PDF"):
        PdfExtractor(never_ocr).extract(make_source(path, FileKind.PDF))


def test_text_utf16_with_bom(tmp_path: Path, make_source: SourceFactory) -> None:
    path = tmp_path / "notepad.txt"
    path.write_bytes(RU_TEXT.encode("utf-16"))

    doc = TextExtractor().extract(make_source(path, FileKind.TEXT))

    assert doc.text == RU_TEXT
    assert doc.lang == "ru"


def test_transparent_png_is_composited_on_white(tmp_path: Path, make_source: SourceFactory) -> None:
    path = tmp_path / "shot.png"
    Image.new("RGBA", (20, 20), (0, 0, 0, 0)).save(path)
    seen: list[int] = []

    def fake_ocr(image: Image.Image) -> str:
        seen.append(image.getpixel((0, 0)))  # type: ignore[arg-type]
        return EN_TEXT

    ImageExtractor(fake_ocr).extract(make_source(path, FileKind.IMAGE))

    assert seen == [255]
