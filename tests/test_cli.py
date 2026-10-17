import json
import runpy
import sys
from pathlib import Path
from typing import ClassVar

import pytest

from foldersum import cli
from foldersum.errors import LlmError


class FakeClient:
    created: ClassVar[list["FakeClient"]] = []
    fail = False

    def __init__(self, model: str, host: str | None = None) -> None:
        self.model = model
        self.host = host
        self.prompts: list[str] = []
        FakeClient.created.append(self)

    def generate(self, system: str, prompt: str, max_tokens: int) -> str:
        if FakeClient.fail:
            raise LlmError("ollama is down")
        self.prompts.append(prompt)
        return f"SUMMARY{len(self.prompts)}"


@pytest.fixture(autouse=True)
def fake_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeClient.created = []
    FakeClient.fail = False
    monkeypatch.setattr(cli, "OllamaClient", FakeClient)


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    root = tmp_path / "docs"
    root.mkdir()
    (root / "a.txt").write_text("Первый документ о гражданских правах.", encoding="utf-8")
    (root / "b.txt").write_text("Second document about city planning.", encoding="utf-8")
    return root


def test_default_run_prints_markdown_overall(
    folder: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = cli.main([str(folder), "--quiet"])

    out = capsys.readouterr()
    assert code == cli.EXIT_OK
    assert out.out.startswith("# Саммари папки")
    assert "SUMMARY3" in out.out
    assert "## Файлы" not in out.out
    assert out.err == ""


def test_per_file_adds_file_sections(folder: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cli.main([str(folder), "--per-file", "--quiet"])

    out = capsys.readouterr().out
    assert "### a.txt" in out
    assert "### b.txt" in out


def test_json_format(folder: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cli.main([str(folder), "--format", "json", "--per-file", "--quiet"])

    data = json.loads(capsys.readouterr().out)
    assert data["overall"] == "SUMMARY3"
    assert [f["path"] for f in data["files"]] == ["a.txt", "b.txt"]


def test_output_file_is_written_instead_of_stdout(
    folder: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = tmp_path / "result.md"

    code = cli.main([str(folder), "--output", str(target), "--quiet"])

    assert code == cli.EXIT_OK
    assert capsys.readouterr().out == ""
    assert target.read_text(encoding="utf-8").startswith("# Саммари папки")


def test_progress_goes_to_stderr_unless_quiet(
    folder: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cli.main([str(folder)])

    err = capsys.readouterr().err
    assert "2 file(s)" in err
    assert "[1/2] a.txt" in err


def test_model_host_and_language_options(folder: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cli.main(
        [str(folder), "--model", "qwen2.5:3b", "--host", "http://h:1", "--lang", "en", "--quiet"]
    )

    client = FakeClient.created[0]
    assert (client.model, client.host) == ("qwen2.5:3b", "http://h:1")
    assert capsys.readouterr().out.startswith("# Folder summary")


def test_missing_source_is_an_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main([str(tmp_path / "nope")])

    assert code == cli.EXIT_SOURCE_ERROR
    assert capsys.readouterr().err.startswith("error:")


def test_empty_folder_is_an_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main([str(tmp_path)])

    assert code == cli.EXIT_NOTHING_TO_SUMMARIZE
    assert "no files found" in capsys.readouterr().err


def test_llm_failure_exits_with_message(folder: Path, capsys: pytest.CaptureFixture[str]) -> None:
    FakeClient.fail = True

    code = cli.main([str(folder), "--quiet"])

    assert code == cli.EXIT_LLM_ERROR
    assert "ollama is down" in capsys.readouterr().err


def test_drive_link_is_resolved_into_the_download_dir(
    folder: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, object] = {}

    def fake_resolve(source: str, download_root: Path) -> Path:
        seen["source"] = source
        seen["root"] = download_root
        return folder

    monkeypatch.setattr(cli, "resolve_source", fake_resolve)
    link = "https://drive.google.com/drive/folders/abc"

    code = cli.main([link, "--download-dir", str(tmp_path / "dl"), "--quiet"])

    assert code == cli.EXIT_OK
    assert seen == {"source": link, "root": tmp_path / "dl"}


@pytest.mark.parametrize("value", ["0", "-5"])
def test_chunk_chars_must_be_positive(value: str) -> None:
    with pytest.raises(SystemExit) as info:
        cli.main(["somewhere", "--chunk-chars", value])

    assert info.value.code == 2


def test_valid_chunk_chars_is_accepted(folder: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main([str(folder), "--chunk-chars", "50", "--quiet"])

    assert code == cli.EXIT_OK


def test_python_dash_m_entry_point(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["foldersum", str(tmp_path / "missing")])

    with pytest.raises(SystemExit) as info:
        runpy.run_module("foldersum", run_name="__main__")

    assert info.value.code == cli.EXIT_SOURCE_ERROR


def test_output_inside_the_scanned_folder_is_refused_before_summarizing(
    folder: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = cli.main([str(folder), "--output", str(folder / "result.md"), "--quiet"])

    assert code == cli.EXIT_OUTPUT_ERROR
    assert "must not be inside" in capsys.readouterr().err
    assert FakeClient.created == []
    assert not (folder / "result.md").exists()


def test_output_overwriting_a_source_file_is_refused(folder: Path) -> None:
    code = cli.main([str(folder), "--output", str(folder / "a.txt"), "--quiet"])

    assert code == cli.EXIT_OUTPUT_ERROR
    assert (folder / "a.txt").read_text(encoding="utf-8").startswith("Первый")


def test_output_directory_must_exist(
    folder: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = cli.main([str(folder), "--output", str(tmp_path / "nope" / "x.md"), "--quiet"])

    assert code == cli.EXIT_OUTPUT_ERROR
    assert "does not exist" in capsys.readouterr().err


def test_output_must_not_be_a_directory(
    folder: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = cli.main([str(folder), "--output", str(tmp_path), "--quiet"])

    assert code == cli.EXIT_OUTPUT_ERROR
    assert "is a directory" in capsys.readouterr().err


def test_write_failure_still_prints_the_result(
    folder: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def deny(*_: object, **__: object) -> int:
        raise PermissionError("read-only")

    monkeypatch.setattr(Path, "write_text", deny)

    code = cli.main([str(folder), "--output", str(tmp_path / "out.md"), "--quiet"])

    out = capsys.readouterr()
    assert code == cli.EXIT_OUTPUT_ERROR
    assert "read-only" in out.err
    assert out.out.startswith("# Саммари папки")


def test_all_files_failed_reports_and_exits_nonzero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    (root / "blob.bin").write_bytes(bytes([0, 1, 2]))

    code = cli.main([str(root), "--quiet"])

    out = capsys.readouterr()
    assert code == cli.EXIT_NOTHING_TO_SUMMARIZE
    assert "blob.bin" in out.out
    assert "no readable text" in out.err
    assert FakeClient.created[0].prompts == []


def test_only_blank_files_exit_nonzero(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    (root / "empty.txt").write_text("  ", encoding="utf-8")

    code = cli.main([str(root), "--quiet"])

    assert code == cli.EXIT_NOTHING_TO_SUMMARIZE
    assert "no readable text" in capsys.readouterr().err


def test_partial_failure_still_succeeds(folder: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (folder / "blob.bin").write_bytes(bytes([0, 1, 2]))

    code = cli.main([str(folder), "--quiet"])

    assert code == cli.EXIT_OK
    assert "blob.bin" in capsys.readouterr().out


def test_stdout_receives_utf8_bytes(
    folder: Path, capfdbinary: pytest.CaptureFixture[bytes]
) -> None:
    cli.main([str(folder), "--quiet"])

    assert "Саммари папки".encode() in capfdbinary.readouterr().out
