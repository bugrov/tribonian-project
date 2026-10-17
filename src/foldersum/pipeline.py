"""Map-reduce summarization: files -> per-file summaries -> one folder summary."""

from collections.abc import Callable
from dataclasses import dataclass

from foldersum.chunking import split_text
from foldersum.errors import ExtractionError, LlmContentError, UnsupportedFileError
from foldersum.extractors import extract_document
from foldersum.models import ExtractedDoc, Lang, SourceFile
from foldersum.prompts import OutLang, clean_name, clean_text, prompts_for
from foldersum.summarizers.base import LlmClient

ProgressFn = Callable[[str], None]
ExtractFn = Callable[[SourceFile], ExtractedDoc]

# Worst case about 1 token per char (OCR noise, digits): 4000 + prompt + answer fit an 8k context.
DEFAULT_CHUNK_CHARS = 4000
_PART_TOKENS = 600
_FINAL_TOKENS = 800


@dataclass(frozen=True)
class FileResult:
    source: SourceFile
    summary: str | None
    lang: Lang | None = None
    warnings: tuple[str, ...] = ()
    error: str | None = None
    empty: bool = False  # read fine, but contained no text, so nothing was summarized
    skipped: bool = False  # unsupported file type: not a failure, just not handled


@dataclass(frozen=True)
class FolderSummary:
    overall: str
    files: tuple[FileResult, ...]


def _ignore(_: str) -> None:
    return None


class Summarizer:
    def __init__(
        self,
        client: LlmClient,
        out_lang: OutLang = "ru",
        chunk_chars: int = DEFAULT_CHUNK_CHARS,
        extract: ExtractFn = extract_document,
        progress: ProgressFn = _ignore,
    ) -> None:
        self._client = client
        self._prompts = prompts_for(out_lang)
        self._chunk_chars = chunk_chars
        self._extract = extract
        self._progress = progress

    def summarize_folder(self, sources: list[SourceFile]) -> FolderSummary:
        """Summarize every file, then combine the file summaries into one folder summary."""
        results: list[FileResult] = []
        for number, source in enumerate(sources, 1):
            self._progress(f"[{number}/{len(sources)}] {source.path.name}")
            results.append(self._summarize_file(source))
        summaries = [(r.source.path.name, r.summary) for r in results if r.summary and not r.empty]
        try:
            overall = self._overall(summaries)
        except LlmContentError as exc:  # keep the per-file results even if this one step fails
            overall = self._prompts.overall_failed.format(reason=exc)
        return FolderSummary(overall=overall, files=tuple(results))

    def summarize_document(self, doc: ExtractedDoc) -> str:
        """Summarize one document, splitting it into chunks when it is too long for the model."""
        name = clean_name(doc.source.path.name)
        hint = self._prompts.doc_lang[doc.lang]
        cleaned = (clean_text(chunk) for chunk in split_text(doc.text, self._chunk_chars))
        chunks = [chunk for chunk in cleaned if chunk.strip()]
        if not chunks:
            return self._prompts.no_text
        if len(chunks) == 1:
            prompt = self._prompts.single.format(name=name, hint=hint, text=chunks[0])
            return self._ask(prompt, _FINAL_TOKENS)
        partials = []
        for part, chunk in enumerate(chunks, 1):
            self._progress(f"    part {part}/{len(chunks)}")
            prompt = self._prompts.chunk.format(
                name=name, hint=hint, part=part, parts=len(chunks), text=chunk
            )
            partials.append(self._ask(prompt, _PART_TOKENS))
        return self._reduce(
            partials, lambda text: self._prompts.combine.format(name=name, text=text)
        )

    def _summarize_file(self, source: SourceFile) -> FileResult:
        """Summarize one file. Problems with this file alone (unreadable, unsupported, refused
        by the model) are recorded in the result; only run-wide model failures propagate."""
        try:
            doc = self._extract(source)
        except UnsupportedFileError as exc:
            return FileResult(source=source, summary=None, error=str(exc), skipped=True)
        except ExtractionError as exc:
            return FileResult(source=source, summary=None, error=str(exc))
        except Exception as exc:  # whatever a parser or OCR raises must not abort the whole run
            return FileResult(source=source, summary=None, error=f"{type(exc).__name__}: {exc}")
        if not doc.text.strip():
            return FileResult(
                source=source,
                summary=self._prompts.no_text,
                lang=doc.lang,
                warnings=doc.warnings,
                empty=True,
            )
        try:
            summary = self.summarize_document(doc)
        except LlmContentError as exc:
            return FileResult(
                source=source, summary=None, lang=doc.lang, warnings=doc.warnings, error=str(exc)
            )
        return FileResult(source=source, summary=summary, lang=doc.lang, warnings=doc.warnings)

    def _overall(self, summaries: list[tuple[str, str]]) -> str:
        if not summaries:
            return self._prompts.no_text
        if len(summaries) == 1:
            return summaries[0][1]
        texts = [f"### {clean_name(name)}\n{summary}" for name, summary in summaries]
        return self._reduce(texts, lambda text: self._prompts.folder.format(text=text))

    def _reduce(self, pieces: list[str], make_prompt: Callable[[str], str]) -> str:
        """Merge ``pieces`` into one answer, in several rounds if they do not fit one prompt."""
        while len("\n\n".join(pieces)) > self._chunk_chars and len(pieces) > 1:
            pieces = [self._merge_batch(batch, make_prompt) for batch in self._batches(pieces)]
        return self._ask(make_prompt(clean_text("\n\n".join(pieces))), _FINAL_TOKENS)

    def _merge_batch(self, batch: list[str], make_prompt: Callable[[str], str]) -> str:
        """Summarize a batch into one intermediate piece; a lone piece is passed through."""
        if len(batch) == 1:
            return batch[0]
        return self._ask(make_prompt(clean_text("\n\n".join(batch))), _PART_TOKENS)

    def _batches(self, pieces: list[str]) -> list[list[str]]:
        """Group pieces so each group fits in a prompt; every group holds at least two pieces."""
        batches: list[list[str]] = []
        current: list[str] = []
        size = 0
        for piece in pieces:
            if len(current) >= 2 and size + len(piece) > self._chunk_chars:
                batches.append(current)
                current, size = [], 0
            current.append(piece)
            size += len(piece) + 2
        if current:
            batches.append(current)
        return batches

    def _ask(self, prompt: str, max_tokens: int) -> str:
        return self._client.generate(self._prompts.system, prompt, max_tokens)
