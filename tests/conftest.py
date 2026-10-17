from collections.abc import Callable
from pathlib import Path

import pytest
from PIL import Image

from foldersum.models import FileKind, SourceFile

PdfFactory = Callable[[Path, list[list[str]]], Path]
ScannedPdfFactory = Callable[[Path, int], Path]
SourceFactory = Callable[[Path, FileKind], SourceFile]


def _build_pdf(pages: list[list[str]]) -> bytes:
    """Build a small valid PDF with a Helvetica text layer, one inner list of lines per page."""
    font_id = 3
    objects: dict[int, bytes] = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        font_id: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    }
    kids: list[int] = []
    next_id = 4
    for lines in pages:
        page_id, content_id = next_id, next_id + 1
        next_id += 2
        kids.append(page_id)
        body = "BT /F1 12 Tf 72 720 Td 16 TL " + " T* ".join(f"({line}) Tj" for line in lines)
        stream = (body + " ET").encode("latin-1")
        objects[page_id] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {content_id} 0 R >>"
        ).encode()
        objects[content_id] = b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream)
    kid_refs = " ".join(f"{kid} 0 R" for kid in kids)
    objects[2] = f"<< /Type /Pages /Kids [{kid_refs}] /Count {len(kids)} >>".encode()

    out = bytearray(b"%PDF-1.4\n")
    offsets: dict[int, int] = {}
    for number in sorted(objects):
        offsets[number] = len(out)
        out += b"%d 0 obj\n%s\nendobj\n" % (number, objects[number])
    xref_at = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for number in sorted(objects):
        out += b"%010d 00000 n \n" % offsets[number]
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref_at,
    )
    return bytes(out)


@pytest.fixture
def make_pdf() -> PdfFactory:
    def factory(path: Path, pages: list[list[str]]) -> Path:
        path.write_bytes(_build_pdf(pages))
        return path

    return factory


@pytest.fixture
def make_scanned_pdf() -> ScannedPdfFactory:
    """A PDF made of page images only, i.e. with no text layer."""

    def factory(path: Path, page_count: int) -> Path:
        pages = [Image.new("RGB", (200, 200), "white") for _ in range(page_count)]
        pages[0].save(path, "PDF", save_all=True, append_images=pages[1:])
        return path

    return factory


@pytest.fixture
def make_source() -> SourceFactory:
    def factory(path: Path, kind: FileKind) -> SourceFile:
        size = path.stat().st_size if path.exists() else 0
        return SourceFile(path=path, kind=kind, size=size)

    return factory
