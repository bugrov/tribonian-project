"""Walk a folder and classify the files in it."""

from pathlib import Path

from foldersum.models import FileKind, SourceFile

_EXTENSIONS: dict[str, FileKind] = {
    ".pdf": FileKind.PDF,
    ".jpg": FileKind.IMAGE,
    ".jpeg": FileKind.IMAGE,
    ".png": FileKind.IMAGE,
    ".webp": FileKind.IMAGE,
    ".bmp": FileKind.IMAGE,
    ".gif": FileKind.IMAGE,
    ".tif": FileKind.IMAGE,
    ".tiff": FileKind.IMAGE,
    ".txt": FileKind.TEXT,
    ".md": FileKind.TEXT,
    ".csv": FileKind.TEXT,
    ".docx": FileKind.OFFICE,
    ".xlsx": FileKind.OFFICE,
    ".pptx": FileKind.OFFICE,
}

_SIGNATURE_BYTES = 12


def _kind_from_signature(head: bytes) -> FileKind:
    """Classify by magic bytes; files from Drive often have wrong or missing extensions.

    BMP ("BM") is deliberately left to the extension check: two bytes are too weak a signature
    and would misclassify text files that start with "BM".
    """
    if head.startswith(b"%PDF"):
        return FileKind.PDF
    if head.startswith((b"\xff\xd8\xff", b"\x89PNG\r\n\x1a\n", b"GIF87a", b"GIF89a")):
        return FileKind.IMAGE
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return FileKind.IMAGE
    return FileKind.UNKNOWN


def detect_kind(path: Path) -> FileKind:
    """Return the file kind, trusting the content signature over the extension."""
    with path.open("rb") as handle:
        by_signature = _kind_from_signature(handle.read(_SIGNATURE_BYTES))
    if by_signature is not FileKind.UNKNOWN:
        return by_signature
    return _EXTENSIONS.get(path.suffix.lower(), FileKind.UNKNOWN)


def _describe(path: Path) -> SourceFile:
    """Build a SourceFile; an unreadable file becomes UNKNOWN instead of aborting the scan."""
    try:
        return SourceFile(path=path, kind=detect_kind(path), size=path.stat().st_size)
    except OSError:
        # Keep the kind the extension suggests, so reading it fails later with the real OS error
        # instead of the file being reported as an unsupported type.
        kind = _EXTENSIONS.get(path.suffix.lower(), FileKind.UNKNOWN)
        return SourceFile(path=path, kind=kind, size=0)


def scan_folder(root: Path) -> list[SourceFile]:
    """Recursively list files under ``root`` in a stable order.

    Hidden entries and symlinks are skipped so the scan never reads outside ``root``.
    """
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")
    return [
        _describe(path)
        for path in sorted(root.rglob("*"))
        if not any(part.startswith(".") for part in path.relative_to(root).parts)
        and not path.is_symlink()
        and path.is_file()
    ]
