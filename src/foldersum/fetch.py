"""Resolve the user-supplied source (local path or Google Drive folder link) to a local folder."""

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import gdown

_DRIVE_FOLDER_RE = re.compile(r"^https?://drive\.google\.com/.*?/folders/([\w-]+)")
_DEFAULT_DOWNLOAD_ROOT = Path("data")

Downloader = Callable[..., Any]


class FetchError(RuntimeError):
    """Raised when the source cannot be turned into a local folder."""


def drive_folder_id(source: str) -> str | None:
    """Return the Drive folder id if ``source`` is a Drive folder link, else None."""
    match = _DRIVE_FOLDER_RE.match(source.strip())
    return match.group(1) if match else None


def resolve_source(
    source: str,
    download_root: Path = _DEFAULT_DOWNLOAD_ROOT,
    downloader: Downloader = gdown.download_folder,
) -> Path:
    """Return a local folder for ``source``: the path itself, or a downloaded Drive folder."""
    folder_id = drive_folder_id(source)
    if folder_id is None:
        path = Path(source)
        if not path.is_dir():
            raise FetchError(f"Not a folder and not a Google Drive folder link: {source}")
        return path

    target = download_root / folder_id
    target.mkdir(parents=True, exist_ok=True)
    try:
        downloaded = downloader(id=folder_id, output=str(target), quiet=True)
    except Exception as exc:  # gdown raises requests/gdown-specific errors; surface one type
        raise FetchError(f"Could not download Drive folder {folder_id}: {exc}") from exc
    if not downloaded:
        raise FetchError(
            f"Drive folder {folder_id} returned no files (empty or not shared by link)"
        )
    root = target.resolve()
    for item in downloaded:
        if not Path(str(item)).resolve().is_relative_to(root):
            raise FetchError(f"Downloaded file escapes the target folder: {item}")
    return target
