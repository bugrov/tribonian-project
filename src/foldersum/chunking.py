"""Split long texts into pieces that fit the model's context window."""

import re

_SENTENCE_END = re.compile(r"(?<=[.!?;])\s+")


def _split_long_line(line: str, max_chars: int) -> list[str]:
    """Split one oversized line at sentence ends, then at spaces, then hard."""
    pieces: list[str] = []
    current = ""
    for sentence in _SENTENCE_END.split(line):
        while len(sentence) > max_chars:
            cut = sentence.rfind(" ", 0, max_chars)
            if cut <= 0:
                cut = max_chars
            head, sentence = sentence[:cut], sentence[cut:].lstrip()
            if current:
                pieces.append(current)
                current = ""
            pieces.append(head)
        if current and len(current) + 1 + len(sentence) > max_chars:
            pieces.append(current)
            current = ""
        current = f"{current} {sentence}".strip() if current else sentence
    if current:
        pieces.append(current)
    return pieces


def split_text(text: str, max_chars: int) -> list[str]:
    """Pack whole lines into chunks of at most ``max_chars`` characters.

    Lines are kept intact when possible so that paragraphs and sentences are not cut in the
    middle; a single line longer than the limit is split at sentence ends or spaces.
    """
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for line in text.strip().splitlines():
        parts = [line] if len(line) <= max_chars else _split_long_line(line, max_chars)
        for part in parts:
            if current and size + 1 + len(part) > max_chars:
                chunks.append("\n".join(current))
                current, size = [], 0
            current.append(part)
            size += len(part) + (1 if len(current) > 1 else 0)
    if current:
        chunks.append("\n".join(current))
    return chunks
