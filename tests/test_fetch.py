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


def test_successful_download_is_marked_as_ours(tmp_path: Path) -> None:
    def downloader(**kwargs: Any) -> list[str]:
        (Path(kwargs["output"]) / "a.txt").write_text("x")
        return [str(Path(kwargs["output"]) / "a.txt")]

    result = resolve_source(
        "https://drive.google.com/drive/folders/abc123",
        download_root=tmp_path,
        downloader=downloader,
    )

    assert (result / ".foldersum-download").is_file()


def test_redownload_replaces_our_previous_download(tmp_path: Path) -> None:
    link = "https://drive.google.com/drive/folders/abc123"
    contents = iter(["old.txt", "new.txt"])

    def downloader(**kwargs: Any) -> list[str]:
        name = Path(kwargs["output"]) / next(contents)
        name.write_text("x")
        return [str(name)]

    resolve_source(link, download_root=tmp_path, downloader=downloader)
    result = resolve_source(link, download_root=tmp_path, downloader=downloader)

    assert sorted(p.name for p in result.iterdir() if not p.name.startswith(".")) == ["new.txt"]


def test_existing_folder_not_made_by_us_is_never_wiped(tmp_path: Path) -> None:
    foreign = tmp_path / "abc123"
    foreign.mkdir()
    (foreign / "precious.txt").write_text("keep me")

    with pytest.raises(FetchError, match="not created by foldersum"):
        resolve_source(
            "https://drive.google.com/drive/folders/abc123",
            download_root=tmp_path,
            downloader=lambda **_: ["never called"],
        )

    assert (foreign / "precious.txt").read_text() == "keep me"


def test_existing_empty_folder_is_reused(tmp_path: Path) -> None:
    (tmp_path / "abc123").mkdir()

    def downloader(**kwargs: Any) -> list[str]:
        name = Path(kwargs["output"]) / "a.txt"
        name.write_text("x")
        return [str(name)]

    result = resolve_source(
        "https://drive.google.com/drive/folders/abc123",
        download_root=tmp_path,
        downloader=downloader,
    )

    assert (result / "a.txt").is_file()


def test_target_that_is_a_file_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "abc123").write_text("i am a file")

    with pytest.raises(FetchError, match="not a folder"):
        resolve_source(
            "https://drive.google.com/drive/folders/abc123",
            download_root=tmp_path,
            downloader=lambda **_: [],
        )


def test_failed_download_leaves_no_partial_folder(tmp_path: Path) -> None:
    def failing(**kwargs: Any) -> list[str]:
        (Path(kwargs["output"]) / "half.txt").write_text("x")
        raise ConnectionError("dropped")

    with pytest.raises(FetchError):
        resolve_source(
            "https://drive.google.com/drive/folders/abc123",
            download_root=tmp_path,
            downloader=failing,
        )

    assert not (tmp_path / "abc123").exists()


def test_escaping_download_is_cleaned_up(tmp_path: Path) -> None:
    outside = tmp_path / "elsewhere.txt"

    with pytest.raises(FetchError, match="escapes"):
        resolve_source(
            "https://drive.google.com/drive/folders/abc123",
            download_root=tmp_path,
            downloader=lambda **_: [str(outside)],
        )

    assert not (tmp_path / "abc123").exists()


def test_interrupted_download_does_not_lock_the_folder(tmp_path: Path) -> None:
    link = "https://drive.google.com/drive/folders/abc123"

    def interrupted(**kwargs: Any) -> list[str]:
        (Path(kwargs["output"]) / "half.txt").write_text("partial")
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        resolve_source(link, download_root=tmp_path, downloader=interrupted)

    def complete(**kwargs: Any) -> list[str]:
        name = Path(kwargs["output"]) / "full.txt"
        name.write_text("done")
        return [str(name)]

    result = resolve_source(link, download_root=tmp_path, downloader=complete)

    assert sorted(p.name for p in result.iterdir() if not p.name.startswith(".")) == ["full.txt"]


def test_os_error_while_preparing_the_download_folder_is_a_fetch_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    previous = tmp_path / "abc123"
    previous.mkdir()
    (previous / ".foldersum-download").write_text("x")

    def locked(_: object) -> None:
        raise PermissionError("file in use")

    monkeypatch.setattr("foldersum.fetch.shutil.rmtree", locked)

    with pytest.raises(FetchError, match="file in use"):
        resolve_source(
            "https://drive.google.com/drive/folders/abc123",
            download_root=tmp_path,
            downloader=lambda **_: ["unused"],
        )
