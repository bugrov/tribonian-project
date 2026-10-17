"""Turn a FolderSummary into Markdown or JSON."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from foldersum.pipeline import FileResult, FolderSummary
from foldersum.prompts import OutLang


@dataclass(frozen=True)
class _Labels:
    title: str
    stats: str
    failed: str
    files: str
    kind: str
    language: str
    warnings: str
    error: str


_LABELS: dict[OutLang, _Labels] = {
    "ru": _Labels(
        title="Саммари папки",
        stats="Файлов: {total}, обработано: {ok}, с ошибками: {failed}.",
        failed="Не удалось обработать",
        files="Файлы",
        kind="тип",
        language="язык",
        warnings="примечания",
        error="ошибка",
    ),
    "en": _Labels(
        title="Folder summary",
        stats="Files: {total}, processed: {ok}, failed: {failed}.",
        failed="Could not be processed",
        files="Files",
        kind="type",
        language="language",
        warnings="notes",
        error="error",
    ),
}


def _relative(result: FileResult, root: Path) -> str:
    try:
        return result.source.path.relative_to(root).as_posix()
    except ValueError:
        return result.source.path.as_posix()


def _oneline(text: str | None) -> str:
    """Collapse line breaks so a file name or error cannot break the Markdown structure."""
    return " ".join((text or "").split())


def render_markdown(
    summary: FolderSummary, root: Path, per_file: bool, out_lang: OutLang = "ru"
) -> str:
    """Overall summary, a short stats line, failures, and optionally one section per file."""
    labels = _LABELS[out_lang]
    failed = [r for r in summary.files if r.error]
    lines = [
        f"# {labels.title}",
        "",
        summary.overall,
        "",
        labels.stats.format(
            total=len(summary.files), ok=len(summary.files) - len(failed), failed=len(failed)
        ),
    ]
    if failed:
        lines += ["", f"## {labels.failed}", ""]
        lines += [f"- {_oneline(_relative(r, root))}: {_oneline(r.error)}" for r in failed]
    if per_file:
        lines += ["", f"## {labels.files}"]
        for result in (r for r in summary.files if not r.error):
            lines += ["", f"### {_oneline(_relative(result, root))}", ""]
            meta = [f"{labels.kind}: {result.source.kind.value}"]
            if result.lang:
                meta.append(f"{labels.language}: {result.lang}")
            if result.warnings:
                meta.append(f"{labels.warnings}: {'; '.join(result.warnings)}")
            lines += [f"*{', '.join(meta)}*", "", result.summary or ""]
    return "\n".join(lines).rstrip() + "\n"


def _file_json(result: FileResult, root: Path) -> dict[str, Any]:
    return {
        "path": _relative(result, root),
        "kind": result.source.kind.value,
        "size": result.source.size,
        "lang": result.lang,
        "summary": result.summary,
        "warnings": list(result.warnings),
        "error": result.error,
    }


def render_json(summary: FolderSummary, root: Path, per_file: bool) -> str:
    """JSON with the overall summary; ``files`` is included only with ``per_file``."""
    failed = [r for r in summary.files if r.error]
    data: dict[str, Any] = {
        "overall": summary.overall,
        "files_total": len(summary.files),
        "files_failed": len(failed),
        "errors": [{"path": _relative(r, root), "error": r.error} for r in failed],
    }
    if per_file:
        data["files"] = [_file_json(r, root) for r in summary.files]
    return json.dumps(data, ensure_ascii=False, indent=2) + "\n"
