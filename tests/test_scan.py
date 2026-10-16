from pathlib import Path

import pytest

from foldersum.models import FileKind
from foldersum.scan import detect_kind, scan_folder

PDF_HEAD = b"%PDF-1.7\n"
JPEG_HEAD = b"\xff\xd8\xff\xe0" + b"\x00" * 8
PNG_HEAD = b"\x89PNG\r\n\x1a\n" + b"\x00" * 4
WEBP_HEAD = b"RIFF\x00\x00\x00\x00WEBPVP8 "


@pytest.mark.parametrize(
    ("name", "content", "expected"),
    [
        ("a.pdf", PDF_HEAD, FileKind.PDF),
        ("a.jpg", JPEG_HEAD, FileKind.IMAGE),
        ("a.png", PNG_HEAD, FileKind.IMAGE),
        ("a.webp", WEBP_HEAD, FileKind.IMAGE),
        ("a.gif", b"GIF89a" + b"\x00" * 6, FileKind.IMAGE),
        ("a.txt", b"hello", FileKind.TEXT),
        ("a.docx", b"PK\x03\x04", FileKind.OFFICE),
        ("a.bin", b"\x00\x01\x02", FileKind.UNKNOWN),
    ],
)
def test_detect_kind(tmp_path: Path, name: str, content: bytes, expected: FileKind) -> None:
    path = tmp_path / name
    path.write_bytes(content)
    assert detect_kind(path) is expected


def test_signature_wins_over_wrong_extension(tmp_path: Path) -> None:
    path = tmp_path / "scan.txt"
    path.write_bytes(PDF_HEAD)
    assert detect_kind(path) is FileKind.PDF


def test_extension_used_when_signature_unknown(tmp_path: Path) -> None:
    path = tmp_path / "UPPER.MD"
    path.write_bytes(b"# title")
    assert detect_kind(path) is FileKind.TEXT


def test_empty_file_is_classified_by_extension(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv"
    path.write_bytes(b"")
    assert detect_kind(path) is FileKind.TEXT


def test_scan_folder_is_recursive_sorted_and_skips_hidden(tmp_path: Path) -> None:
    (tmp_path / "b.pdf").write_bytes(PDF_HEAD)
    (tmp_path / "a.txt").write_bytes(b"x")
    nested = tmp_path / "sub"
    nested.mkdir()
    (nested / "c.jpg").write_bytes(JPEG_HEAD)
    hidden = tmp_path / ".hidden"
    hidden.mkdir()
    (hidden / "d.txt").write_bytes(b"x")
    (tmp_path / ".secret.txt").write_bytes(b"x")

    files = scan_folder(tmp_path)

    assert [f.path.relative_to(tmp_path).as_posix() for f in files] == [
        "a.txt",
        "b.pdf",
        "sub/c.jpg",
    ]
    assert [f.kind for f in files] == [FileKind.TEXT, FileKind.PDF, FileKind.IMAGE]
    assert files[0].size == 1


def test_scan_empty_folder(tmp_path: Path) -> None:
    assert scan_folder(tmp_path) == []


def test_scan_missing_folder_raises(tmp_path: Path) -> None:
    with pytest.raises(NotADirectoryError):
        scan_folder(tmp_path / "nope")


def test_scan_skips_symlinks_and_directories_named_like_files(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("x")
    root = tmp_path / "root"
    root.mkdir()
    (root / "folder.pdf").mkdir()
    try:
        (root / "link.txt").symlink_to(outside / "secret.txt")
    except OSError:
        pytest.skip("symlinks not permitted on this system")

    assert scan_folder(root) == []


def test_unreadable_file_is_reported_as_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "a.pdf").write_bytes(PDF_HEAD)

    def deny(_: Path) -> FileKind:
        raise PermissionError("denied")

    monkeypatch.setattr("foldersum.scan.detect_kind", deny)

    files = scan_folder(tmp_path)

    assert [(f.kind, f.size) for f in files] == [(FileKind.UNKNOWN, 0)]
