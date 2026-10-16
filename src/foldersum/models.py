"""Shared data models."""

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class FileKind(StrEnum):
    """Coarse file category that decides which extractor handles the file."""

    PDF = "pdf"
    IMAGE = "image"
    TEXT = "text"
    OFFICE = "office"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class SourceFile:
    """A file found in the scanned folder."""

    path: Path
    kind: FileKind
    size: int
