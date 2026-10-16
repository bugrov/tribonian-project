from pathlib import Path
from typing import Any

import pytest

from foldersum.fetch import FetchError, drive_folder_id, resolve_source


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("https://drive.google.com/drive/folders/abc_DEF-123?usp=sharing", "abc_DEF-123"),
        ("https://drive.google.com/drive/u/0/folders/xyz789", "xyz789"),
        ("  https://drive.google.com/drive/folders/abc  ", "abc"),
        ("C:/docs/folder", None),
        ("https://example.com/folders/abc", None),
    ],
)
def test_drive_folder_id(source: str, expected: str | None) -> None:
    assert drive_folder_id(source) == expected


def test_local_folder_returned_as_is(tmp_path: Path) -> None:
    assert resolve_source(str(tmp_path)) == tmp_path


def test_missing_local_folder_raises(tmp_path: Path) -> None:
    with pytest.raises(FetchError):
        resolve_source(str(tmp_path / "missing"))


def test_drive_link_downloads_into_folder_id_subdir(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []

    def fake_downloader(**kwargs: Any) -> list[str]:
        calls.append(kwargs)
        return [str(tmp_path / "abc123" / "file.txt")]

    url = "https://drive.google.com/drive/folders/abc123?usp=sharing"
    result = resolve_source(url, download_root=tmp_path, downloader=fake_downloader)

    assert result == tmp_path / "abc123"
    assert result.is_dir()
    assert calls == [{"id": "abc123", "output": str(tmp_path / "abc123"), "quiet": True}]


def test_drive_download_failure_raises(tmp_path: Path) -> None:
    def failing_downloader(**_: Any) -> None:
        return None

    with pytest.raises(FetchError):
        resolve_source(
            "https://drive.google.com/drive/folders/abc123",
            download_root=tmp_path,
            downloader=failing_downloader,
        )


def test_drive_download_exception_is_wrapped(tmp_path: Path) -> None:
    def raising_downloader(**_: Any) -> list[str]:
        raise ConnectionError("boom")

    with pytest.raises(FetchError, match="boom"):
        resolve_source(
            "https://drive.google.com/drive/folders/abc123",
            download_root=tmp_path,
            downloader=raising_downloader,
        )


def test_drive_empty_result_raises(tmp_path: Path) -> None:
    with pytest.raises(FetchError, match="no files"):
        resolve_source(
            "https://drive.google.com/drive/folders/abc123",
            download_root=tmp_path,
            downloader=lambda **_: [],
        )


def test_drive_download_outside_target_raises(tmp_path: Path) -> None:
    outside = str(tmp_path / "elsewhere.txt")
    with pytest.raises(FetchError, match="escapes"):
        resolve_source(
            "https://drive.google.com/drive/folders/abc123",
            download_root=tmp_path,
            downloader=lambda **_: [outside],
        )


def test_local_file_instead_of_folder_raises(tmp_path: Path) -> None:
    file = tmp_path / "a.txt"
    file.write_text("x")
    with pytest.raises(FetchError):
        resolve_source(str(file))
