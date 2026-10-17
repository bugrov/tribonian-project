from types import SimpleNamespace
from typing import Any

import anthropic
import httpx2
import pytest

from foldersum.errors import LlmContentError, LlmError
from foldersum.summarizers.claude import DEFAULT_CLAUDE_MODEL, ClaudeClient


def text_response(
    *texts: str, stop_reason: str = "end_turn", stop_details: Any = None
) -> SimpleNamespace:
    blocks = [SimpleNamespace(type="text", text=t) for t in texts]
    return SimpleNamespace(content=blocks, stop_reason=stop_reason, stop_details=stop_details)


class FakeMessages:
    """Returns queued responses (or raises queued exceptions), recording every request."""

    def __init__(self, *outcomes: SimpleNamespace | Exception) -> None:
        self.outcomes = list(outcomes)
        self.requests: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def request() -> httpx2.Request:
    return httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


def api_error(cls: type[anthropic.APIStatusError], status: int, message: str) -> Exception:
    response = httpx2.Response(status, request=request())
    return cls(message, response=response, body=None)


def test_request_shape_for_the_default_model() -> None:
    fake = FakeMessages(text_response("  Summary.  "))

    answer = ClaudeClient(messages=fake).generate("system text", "prompt text", 800)

    assert answer == "Summary."
    sent = fake.requests[0]
    assert sent["model"] == DEFAULT_CLAUDE_MODEL == "claude-opus-5-5"
    assert sent["system"] == "system text"
    assert sent["messages"] == [{"role": "user", "content": "prompt text"}]
    assert sent["output_config"] == {"effort": "low"}
    assert sent["betas"] == ["server-side-fallback-2026-07-01"]
    assert sent["fallbacks"] == "default"


def test_request_leaves_room_for_thinking_and_sends_no_forbidden_parameters() -> None:
    fake = FakeMessages(text_response("ok"))

    ClaudeClient(messages=fake).generate("s", "p", 800)

    sent = fake.requests[0]
    assert sent["max_tokens"] > 800 + 4000
    assert sent["max_tokens"] <= 16000
    for rejected in ("temperature", "top_p", "top_k", "thinking", "tool_choice"):
        assert rejected not in sent


def test_models_without_server_side_fallback_do_not_send_it() -> None:
    fake = FakeMessages(text_response("ok"))

    ClaudeClient(model="claude-haiku-5-5", messages=fake).generate("s", "p", 100)

    sent = fake.requests[0]
    assert sent["model"] == "claude-haiku-5-5"
    assert "betas" not in sent
    assert "fallbacks" not in sent


def test_effort_is_configurable() -> None:
    fake = FakeMessages(text_response("ok"))

    ClaudeClient(effort="medium", messages=fake).generate("s", "p", 100)

    assert fake.requests[0]["output_config"] == {"effort": "medium"}


def test_only_text_blocks_are_joined() -> None:
    response = text_response("First part. ", "Second part.")
    response.content.insert(0, SimpleNamespace(type="thinking", thinking="hmm"))

    answer = ClaudeClient(messages=FakeMessages(response)).generate("s", "p", 100)

    assert answer == "First part. Second part."


def test_refusal_is_an_error_with_its_category() -> None:
    refusal = text_response(
        "", stop_reason="refusal", stop_details=SimpleNamespace(category="cyber")
    )

    with pytest.raises(LlmError, match=r"declined.*cyber"):
        ClaudeClient(messages=FakeMessages(refusal)).generate("s", "p", 100)


def test_refusal_without_details_is_still_an_error() -> None:
    refusal = SimpleNamespace(content=[], stop_reason="refusal")

    with pytest.raises(LlmError, match="unspecified"):
        ClaudeClient(messages=FakeMessages(refusal)).generate("s", "p", 100)


def test_empty_answer_is_an_error() -> None:
    with pytest.raises(LlmError, match="empty answer"):
        ClaudeClient(messages=FakeMessages(text_response("  "))).generate("s", "p", 100)


def test_truncated_answer_is_retried_with_a_bigger_limit() -> None:
    fake = FakeMessages(
        text_response("First. Second sen", stop_reason="max_tokens"),
        text_response("Full answer.", stop_reason="end_turn"),
    )

    answer = ClaudeClient(messages=fake).generate("s", "p", 500)

    assert answer == "Full answer."
    assert fake.requests[1]["max_tokens"] - fake.requests[0]["max_tokens"] == 500


def test_still_truncated_answer_keeps_whole_sentences() -> None:
    cut = text_response("First sentence. Second sentence. Third unfin", stop_reason="max_tokens")
    fake = FakeMessages(cut, cut)

    assert ClaudeClient(messages=fake).generate("s", "p", 50) == "First sentence. Second sentence."


def test_failed_retry_falls_back_to_the_first_trimmed_answer() -> None:
    fake = FakeMessages(
        text_response("One sentence. Another sentence. Cut of", stop_reason="max_tokens"),
        anthropic.APIConnectionError(request=request()),
    )

    assert ClaudeClient(messages=fake).generate("s", "p", 50) == "One sentence. Another sentence."


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (api_error(anthropic.AuthenticationError, 401, "bad key"), "rejected the credentials"),
        (api_error(anthropic.NotFoundError, 404, "no such model"), "model not found"),
        (api_error(anthropic.RateLimitError, 429, "slow down"), "rate limit"),
        (anthropic.APITimeoutError(request=request()), "timed out"),
        (anthropic.APIConnectionError(request=request()), "Cannot reach"),
        (api_error(anthropic.BadRequestError, 400, "credit balance too low"), "error 400"),
        (api_error(anthropic.InternalServerError, 500, "oops"), "error 500"),
    ],
)
def test_api_errors_become_llm_errors(error: Exception, expected: str) -> None:
    with pytest.raises(LlmError, match=expected):
        ClaudeClient(messages=FakeMessages(error)).generate("s", "p", 10)


def test_missing_credentials_get_a_helpful_message() -> None:
    error = TypeError("Could not resolve authentication method. Expected api_key to be set.")

    with pytest.raises(LlmError, match="ANTHROPIC_API_KEY"):
        ClaudeClient(messages=FakeMessages(error)).generate("s", "p", 10)


def test_unrelated_type_errors_are_not_swallowed() -> None:
    with pytest.raises(TypeError, match="boom"):
        ClaudeClient(messages=FakeMessages(TypeError("boom"))).generate("s", "p", 10)


def test_real_client_is_built_without_contacting_the_api() -> None:
    assert ClaudeClient(api_key="sk-ant-test") is not None


def test_thinking_only_truncation_is_retried_with_a_bigger_limit() -> None:
    thinking_only = SimpleNamespace(
        content=[SimpleNamespace(type="thinking", thinking="long")], stop_reason="max_tokens"
    )
    fake = FakeMessages(thinking_only, text_response("Real answer."))

    answer = ClaudeClient(messages=fake).generate("s", "p", 300)

    assert answer == "Real answer."
    assert fake.requests[1]["max_tokens"] > fake.requests[0]["max_tokens"]


def test_thinking_only_truncation_twice_is_an_error() -> None:
    thinking_only = SimpleNamespace(content=[], stop_reason="max_tokens")
    fake = FakeMessages(thinking_only, thinking_only)

    with pytest.raises(LlmError, match="no answer within the token limit"):
        ClaudeClient(messages=fake).generate("s", "p", 300)


def test_other_sdk_errors_become_llm_errors() -> None:
    error = anthropic.APIError("weird response", request=request(), body=None)

    with pytest.raises(LlmError, match="weird response"):
        ClaudeClient(messages=FakeMessages(error)).generate("s", "p", 10)


def test_refusal_and_empty_answer_only_fail_the_file_but_credentials_fail_the_run() -> None:
    refusal = text_response("", stop_reason="refusal")
    with pytest.raises(LlmContentError):
        ClaudeClient(messages=FakeMessages(refusal)).generate("s", "p", 10)
    with pytest.raises(LlmContentError):
        ClaudeClient(messages=FakeMessages(text_response(""))).generate("s", "p", 10)

    fatal = api_error(anthropic.AuthenticationError, 401, "bad key")
    with pytest.raises(LlmError) as info:
        ClaudeClient(messages=FakeMessages(fatal)).generate("s", "p", 10)
    assert not isinstance(info.value, LlmContentError)
