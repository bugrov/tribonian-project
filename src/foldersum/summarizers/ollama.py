"""Local language model through Ollama."""

import re
from typing import Any, Protocol

import httpx
import ollama

from foldersum.errors import LlmError

DEFAULT_MODEL = "qwen2.5:7b"

_SENTENCE_END = re.compile(r"[.!?…»)](?=\s|$)")


def _drop_unfinished_sentence(text: str) -> str:
    """An answer cut off by the token limit ends mid-sentence: keep whole sentences only."""
    ends = [m.end() for m in _SENTENCE_END.finditer(text)]
    if not ends or ends[-1] < len(text) // 2:
        return text
    return text[: ends[-1]]


class ChatApi(Protocol):
    """The part of ``ollama.Client`` that this module uses."""

    def chat(
        self, *, model: str, messages: list[dict[str, str]], options: dict[str, Any]
    ) -> Any: ...


class OllamaClient:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str | None = None,
        num_ctx: int = 8192,
        temperature: float = 0.2,
        timeout: float = 900.0,
        client: ChatApi | None = None,
    ) -> None:
        self._model = model
        self._num_ctx = num_ctx
        self._temperature = temperature
        self._timeout = timeout
        self._client: ChatApi = client or ollama.Client(host=host, timeout=timeout)

    def generate(self, system: str, prompt: str, max_tokens: int) -> str:
        """Answer the prompt; an answer cut off by the token limit is retried once with twice
        the limit, and only if it is still cut off (or the retry fails) is the unfinished last
        sentence dropped."""
        answer, truncated = self._chat(system, prompt, max_tokens, check_context=True)
        if not truncated:
            return answer
        try:
            retry, retry_truncated = self._chat(system, prompt, max_tokens * 2, check_context=False)
        except LlmError:
            return _drop_unfinished_sentence(answer)
        return _drop_unfinished_sentence(retry) if retry_truncated else retry

    def _chat(
        self, system: str, prompt: str, max_tokens: int, check_context: bool
    ) -> tuple[str, bool]:
        options = {
            "num_ctx": self._num_ctx,
            "temperature": self._temperature,
            "num_predict": max_tokens,
        }
        messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        try:
            response = self._client.chat(model=self._model, messages=messages, options=options)
        except ollama.ResponseError as exc:
            hint = f" (run `ollama pull {self._model}`)" if exc.status_code == 404 else ""
            raise LlmError(f"Ollama error: {exc.error}{hint}") from exc
        except httpx.TimeoutException as exc:
            raise LlmError(f"Ollama did not answer within {self._timeout:.0f}s") from exc
        except (ConnectionError, httpx.HTTPError, ollama.RequestError) as exc:
            raise LlmError(f"Cannot reach Ollama: {exc}") from exc
        used = getattr(response, "prompt_eval_count", None)
        if check_context and used is not None and used + max_tokens > self._num_ctx:
            raise LlmError(
                f"Prompt ({used} tokens) does not fit the {self._num_ctx}-token context; "
                "lower the chunk size or raise num_ctx"
            )
        content = response.message.content
        if not content or not content.strip():
            raise LlmError(f"Model {self._model} returned an empty answer")
        return str(content).strip(), getattr(response, "done_reason", None) == "length"
