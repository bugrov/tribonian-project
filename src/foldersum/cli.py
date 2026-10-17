"""Command line interface: ``python -m foldersum <folder|Drive link> [options]``."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from foldersum.errors import LlmError
from foldersum.fetch import FetchError, resolve_source
from foldersum.pipeline import DEFAULT_CHUNK_CHARS, Summarizer
from foldersum.prompts import OutLang
from foldersum.render import render_json, render_markdown
from foldersum.scan import scan_folder
from foldersum.summarizers.base import LlmClient
from foldersum.summarizers.claude import DEFAULT_CLAUDE_MODEL, ClaudeClient
from foldersum.summarizers.ollama import DEFAULT_MODEL, OllamaClient

# Claude's context is large enough to take big pieces, so documents rarely need map-reduce.
CLAUDE_CHUNK_CHARS = 100_000

EXIT_OK = 0
EXIT_NOTHING_TO_SUMMARIZE = 1  # no files, or none of them yielded any text
EXIT_SOURCE_ERROR = 2
EXIT_LLM_ERROR = 3
EXIT_OUTPUT_ERROR = 4


def _output_problem(output: Path | None, root: Path) -> str | None:
    """Why ``--output`` cannot be used, checked before the (long) summarization starts."""
    if output is None:
        return None
    target = output.resolve()
    if target.is_dir():
        return f"--output is a directory: {output}"
    if not target.parent.is_dir():
        return f"--output directory does not exist: {target.parent}"
    if target.is_relative_to(root.resolve()):
        return f"--output must not be inside the summarized folder: {output}"
    return None


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _build_client(args: argparse.Namespace) -> tuple[LlmClient, str, int]:
    """The model client for the chosen backend, its model name and default chunk size."""
    if args.backend == "claude":
        if args.host:
            print("warning: --host is ignored by the claude backend", file=sys.stderr)
        model = args.model or DEFAULT_CLAUDE_MODEL
        return ClaudeClient(model=model), model, CLAUDE_CHUNK_CHARS
    model = args.model or DEFAULT_MODEL
    return OllamaClient(model=model, host=args.host), model, DEFAULT_CHUNK_CHARS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="foldersum",
        description="Summarize the contents of a folder of documents (PDF, images, text).",
    )
    parser.add_argument("source", help="local folder or a public Google Drive folder link")
    parser.add_argument(
        "--per-file", action="store_true", help="also print a summary for every file"
    )
    parser.add_argument(
        "--backend",
        choices=["ollama", "claude"],
        default="ollama",
        help=(
            "language model backend: local Ollama (default) or the Claude API, which needs "
            "ANTHROPIC_API_KEY and sends document text to Anthropic"
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help=f"model name (default: {DEFAULT_MODEL} for ollama, {DEFAULT_CLAUDE_MODEL} for claude)",
    )
    parser.add_argument(
        "--host", default=None, help="Ollama server URL, ollama backend only (default: OLLAMA_HOST)"
    )
    parser.add_argument(
        "--lang", choices=["ru", "en"], default="ru", help="summary language (default: ru)"
    )
    parser.add_argument("--format", choices=["md", "json"], default="md", dest="fmt")
    parser.add_argument("--output", type=Path, default=None, help="write the result to a file")
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=Path("data"),
        help="where Drive folders are downloaded (default: %(default)s)",
    )
    parser.add_argument(
        "--chunk-chars",
        type=_positive_int,
        default=None,
        help=(
            f"max characters of text per model request (default: {DEFAULT_CHUNK_CHARS} for "
            f"ollama, {CLAUDE_CHUNK_CHARS} for claude)"
        ),
    )
    parser.add_argument("--quiet", action="store_true", help="no progress messages on stderr")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_lang: OutLang = args.lang

    def say(message: str) -> None:
        if not args.quiet:
            print(message, file=sys.stderr, flush=True)

    try:
        root = resolve_source(args.source, download_root=args.download_dir)
    except FetchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_SOURCE_ERROR
    sources = scan_folder(root)
    if not sources:
        print(f"error: no files found in {root}", file=sys.stderr)
        return EXIT_NOTHING_TO_SUMMARIZE
    problem = _output_problem(args.output, root)
    if problem:
        print(f"error: {problem}", file=sys.stderr)
        return EXIT_OUTPUT_ERROR

    client, model, default_chunk_chars = _build_client(args)
    chunk_chars = args.chunk_chars or default_chunk_chars
    summarizer = Summarizer(client, out_lang=out_lang, chunk_chars=chunk_chars, progress=say)
    say(f"{len(sources)} file(s) in {root}, {args.backend} backend, model {model}")
    try:
        summary = summarizer.summarize_folder(sources)
    except LlmError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_LLM_ERROR

    if args.fmt == "json":
        text = render_json(summary, root, args.per_file)
    else:
        text = render_markdown(summary, root, args.per_file, out_lang)
    exit_code = EXIT_OK
    written = False
    if args.output:
        try:
            args.output.write_text(text, encoding="utf-8", newline="\n")
            say(f"written to {args.output}")
            written = True
        except OSError as exc:
            print(
                f"error: cannot write {args.output}: {exc}; printing the result instead",
                file=sys.stderr,
            )
            exit_code = EXIT_OUTPUT_ERROR
    if not written:
        sys.stdout.buffer.write(text.encode("utf-8"))
        sys.stdout.flush()
    if exit_code == EXIT_OK and not any(r.summary and not r.empty for r in summary.files):
        print("error: no readable text in any file", file=sys.stderr)
        exit_code = EXIT_NOTHING_TO_SUMMARIZE
    return exit_code
