"""Claude through the Anthropic API.

Note for users: with this backend the text of your documents is sent to Anthropic.
"""

from typing import Any, Protocol, cast

import anthropic

from foldersum.errors import LlmError
from foldersum.summarizers.base import answer_with_retry

DEFAULT_CLAUDE_MODEL = "claude-opus-5-5"

# Thinking tokens count against max_tokens on these models, so the answer budget requested by the
# pipeline gets this much headroom on top, otherwise a short summary could be eaten by thinking.
_THINKING_HEADROOM = 8000
_FALLBACK_BETA = "server-side-fallback-2026-07-01"
# Models that accept the server-side `fallbacks: "default"` parameter (re-runs a safety-declined
# request on another model instead of returning the refusal).
_FALLBACK_MODELS = frozenset(
    {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}
)
_NO_CREDENTIALS_HINT = "set ANTHROPIC_API_KEY (or run `ant auth login`)"


class MessagesApi(Protocol):
    """The part of ``anthropic.Anthropic().beta.messages`` that this module uses."""

    def create(self, **kwargs: Any) -> Any: ...


class ClaudeClient:
    def __init__(
        self,
        model: str = DEFAULT_CLAUDE_MODEL,
        api_key: str | None = None,
        timeout: float = 600.0,
        effort: str = "low",
        messages: MessagesApi | None = None,
    ) -> None:
        self._model = model
        self._effort = effort
        # The SDK's overloaded create() is wider than the one call shape used here.
        self._messages: MessagesApi = messages or cast(
            MessagesApi, anthropic.Anthropic(api_key=api_key, timeout=timeout).beta.messages
        )

    def generate(self, system: str, prompt: str, max_tokens: int) -> str:
        return answer_with_retry(
            lambda tokens, _first: self._ask(system, prompt, tokens), max_tokens
        )

    def _ask(self, system: str, prompt: str, max_tokens: int) -> tuple[str, bool]:
        # No temperature/top_p (rejected by current models) and no `thinking` (always on).
        request: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens + _THINKING_HEADROOM,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
            "output_config": {"effort": self._effort},
        }
        if self._model in _FALLBACK_MODELS:
            request["betas"] = [_FALLBACK_BETA]
            request["fallbacks"] = "default"
        response = self._call(request)
        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None) or "unspecified"
            raise LlmError(f"Claude declined to process this document (category: {category})")
        text = "".join(block.text for block in response.content if block.type == "text").strip()
        truncated = response.stop_reason == "max_tokens"
        if not text and not truncated:
            raise LlmError(f"Model {self._model} returned an empty answer")
        return text, truncated

    def _call(self, request: dict[str, Any]) -> Any:
        try:
            return self._messages.create(**request)
        except anthropic.AuthenticationError as exc:
            raise LlmError(f"Claude API rejected the credentials: {_NO_CREDENTIALS_HINT}") from exc
        except anthropic.NotFoundError as exc:
            raise LlmError(f"Claude model not found: {self._model} ({exc.message})") from exc
        except anthropic.RateLimitError as exc:
            raise LlmError("Claude API rate limit reached; try again later") from exc
        except anthropic.APITimeoutError as exc:
            raise LlmError("Claude API request timed out") from exc
        except anthropic.APIConnectionError as exc:
            raise LlmError(f"Cannot reach the Claude API: {exc}") from exc
        except anthropic.APIStatusError as exc:
            raise LlmError(f"Claude API error {exc.status_code}: {exc.message}") from exc
        except TypeError as exc:
            if "authentication" not in str(exc).lower():
                raise
            raise LlmError(f"No Claude API credentials found: {_NO_CREDENTIALS_HINT}") from exc
