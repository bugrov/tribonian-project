"""Shared data models."""

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal


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


Lang = Literal["ru", "en", "unknown"]


@dataclass(frozen=True)
class ExtractedDoc:
    """Text pulled out of a source file, with the detected document language."""

    source: SourceFile
    text: str
    lang: Lang
    pages: int | None = None
    warnings: tuple[str, ...] = ()
