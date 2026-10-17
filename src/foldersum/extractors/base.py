"""Extractor interface."""

from typing import Protocol

from foldersum.models import ExtractedDoc, SourceFile


class Extractor(Protocol):
    """Turns one source file into text."""

    def extract(self, source: SourceFile) -> ExtractedDoc: ...
