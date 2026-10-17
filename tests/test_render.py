import json
from pathlib import Path

from foldersum.models import FileKind, SourceFile
from foldersum.pipeline import FileResult, FolderSummary
from foldersum.render import render_json, render_markdown

ROOT = Path("docs")


def result(
    name: str,
    summary: str | None = "Резюме файла.",
    error: str | None = None,
    kind: FileKind = FileKind.TEXT,
    warnings: tuple[str, ...] = (),
) -> FileResult:
    return FileResult(
        source=SourceFile(path=ROOT / name, kind=kind, size=10),
        summary=summary,
        lang=None if error else "ru",
        warnings=warnings,
        error=error,
    )


def sample() -> FolderSummary:
    return FolderSummary(
        overall="Общее саммари.",
        files=(
            result("a.txt"),
            result("sub/b.pdf", kind=FileKind.PDF, warnings=("OCR used for 1 of 2 pages",)),
            result("bad.pdf", summary=None, error="broken pdf"),
        ),
    )


def test_markdown_without_per_file_has_overall_stats_and_failures_only() -> None:
    text = render_markdown(sample(), ROOT, per_file=False)

    assert text.startswith("# Саммари папки\n\nОбщее саммари.\n")
    assert "Файлов: 3, обработано: 2, с ошибками: 1." in text
    assert "## Не удалось обработать" in text
    assert "- bad.pdf: broken pdf" in text
    assert "## Файлы" not in text
    assert "Резюме файла." not in text


def test_markdown_per_file_lists_successful_files_with_metadata() -> None:
    text = render_markdown(sample(), ROOT, per_file=True)

    assert "## Файлы" in text
    assert "### a.txt" in text
    assert "### sub/b.pdf" in text
    assert "### bad.pdf" not in text
    assert "*тип: pdf, язык: ru, примечания: OCR used for 1 of 2 pages*" in text
    assert text.count("Резюме файла.") == 2


def test_markdown_in_english() -> None:
    text = render_markdown(sample(), ROOT, per_file=True, out_lang="en")

    assert text.startswith("# Folder summary")
    assert "Files: 3, processed: 2, failed: 1." in text
    assert "## Could not be processed" in text
    assert "## Files" in text


def test_markdown_without_failures_has_no_failure_section() -> None:
    summary = FolderSummary(overall="Ok.", files=(result("a.txt"),))

    text = render_markdown(summary, ROOT, per_file=False)

    assert "Не удалось" not in text
    assert text.endswith("\n")


def test_json_overall_only() -> None:
    data = json.loads(render_json(sample(), ROOT, per_file=False))

    assert data["overall"] == "Общее саммари."
    assert data["files_total"] == 3
    assert data["files_failed"] == 1
    assert data["errors"] == [{"path": "bad.pdf", "error": "broken pdf"}]
    assert "files" not in data


def test_json_per_file_keeps_cyrillic_unescaped() -> None:
    raw = render_json(sample(), ROOT, per_file=True)
    data = json.loads(raw)

    assert "Общее саммари." in raw
    assert [f["path"] for f in data["files"]] == ["a.txt", "sub/b.pdf", "bad.pdf"]
    first = data["files"][0]
    assert first["kind"] == "text"
    assert first["lang"] == "ru"
    assert first["summary"] == "Резюме файла."
    assert first["error"] is None
    assert data["files"][2]["error"] == "broken pdf"


def test_path_outside_root_falls_back_to_the_full_path() -> None:
    outside = FileResult(
        source=SourceFile(path=Path("elsewhere/x.txt"), kind=FileKind.TEXT, size=1),
        summary="s",
    )
    summary = FolderSummary(overall="o", files=(outside,))

    data = json.loads(render_json(summary, ROOT, per_file=True))

    assert data["files"][0]["path"] == "elsewhere/x.txt"


def test_markdown_file_names_and_errors_cannot_break_the_structure() -> None:
    nasty = FolderSummary(
        overall="o",
        files=(
            result("evil" + chr(10) + "# injected.txt"),
            result("bad.pdf", summary=None, error="line one" + chr(10) + "## fake heading"),
        ),
    )

    text = render_markdown(nasty, ROOT, per_file=True)

    assert chr(10) + "# injected" not in text
    assert chr(10) + "## fake heading" not in text
    assert "### evil # injected.txt" in text
