"""Language model backend interface and the answer handling shared by the backends."""

import re
from collections.abc import Callable
from typing import Protocol

from foldersum.errors import LlmError

_SENTENCE_END = re.compile(r"[.!?…»)](?=\s|$)")


class LlmClient(Protocol):
    """Anything that can answer a prompt: a local Ollama model, the Claude API, ..."""

    def generate(self, system: str, prompt: str, max_tokens: int) -> str:
        """Return the model's answer; raises LlmError on failure."""
        ...


def drop_unfinished_sentence(text: str) -> str:
    """An answer cut off by the token limit ends mid-sentence: keep whole sentences only."""
    ends = [m.end() for m in _SENTENCE_END.finditer(text)]
    if not ends or ends[-1] < len(text) // 2:
        return text
    return text[: ends[-1]]


def answer_with_retry(ask: Callable[[int, bool], tuple[str, bool]], max_tokens: int) -> str:
    """Run ``ask(max_tokens, first_attempt)`` and handle an answer cut off by the token limit.

    ``ask`` returns ``(answer, truncated)``; the answer may be empty when truncated (a model that
    used its whole budget thinking). A truncated answer is retried once with twice the limit; only
    if that is still truncated, or the retry fails, is the unfinished last sentence dropped. If
    nothing usable remains, LlmError is raised.
    """
    answer, truncated = ask(max_tokens, True)
    if not truncated:
        return answer
    try:
        retry, retry_truncated = ask(max_tokens * 2, False)
    except LlmError:
        if not answer:
            raise
        return drop_unfinished_sentence(answer)
    result = drop_unfinished_sentence(retry) if retry_truncated else retry
    if not result:
        raise LlmError("The model produced no answer within the token limit")
    return result
