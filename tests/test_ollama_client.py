import re
from types import SimpleNamespace
from typing import Any

import httpx
import ollama
import pytest

from foldersum.errors import LlmContentError, LlmError
from foldersum.summarizers.ollama import OllamaClient


class FakeChat:
    def __init__(
        self,
        content: str | None = "answer",
        error: Exception | None = None,
        **extra: Any,
    ) -> None:
        self.content = content
        self.error = error
        self.extra = extra
        self.calls: list[dict[str, Any]] = []

    def chat(self, *, model: str, messages: list[dict[str, str]], options: dict[str, Any]) -> Any:
        self.calls.append({"model": model, "messages": messages, "options": options})
        if self.error:
            raise self.error
        return SimpleNamespace(message=SimpleNamespace(content=self.content), **self.extra)


def test_generate_sends_system_and_prompt_and_options() -> None:
    fake = FakeChat("  an answer \n")
    client = OllamaClient(model="m1", num_ctx=4096, client=fake)

    assert client.generate("sys", "prompt", 300) == "an answer"

    call = fake.calls[0]
    assert call["model"] == "m1"
    assert call["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "prompt"},
    ]
    assert call["options"] == {"num_ctx": 4096, "temperature": 0.2, "num_predict": 300}


def test_temperature_is_configurable() -> None:
    fake = FakeChat()

    OllamaClient(temperature=0.0, client=fake).generate("s", "p", 10)

    assert fake.calls[0]["options"]["temperature"] == 0.0


def test_missing_model_hints_to_pull() -> None:
    fake = FakeChat(error=ollama.ResponseError("model not found", 404))

    with pytest.raises(LlmError, match=re.escape("ollama pull qwen2.5:7b")):
        OllamaClient(client=fake).generate("s", "p", 10)


def test_other_response_error_is_wrapped() -> None:
    fake = FakeChat(error=ollama.ResponseError("server exploded", 500))

    with pytest.raises(LlmError, match="server exploded") as info:
        OllamaClient(client=fake).generate("s", "p", 10)

    assert "ollama pull" not in str(info.value)


def test_connection_error_is_wrapped() -> None:
    fake = FakeChat(error=ConnectionError("refused"))

    with pytest.raises(LlmError, match="Cannot reach Ollama"):
        OllamaClient(client=fake).generate("s", "p", 10)


@pytest.mark.parametrize("content", [None, "", "   "])
def test_empty_answer_is_an_error(content: str | None) -> None:
    with pytest.raises(LlmError, match="empty answer"):
        OllamaClient(client=FakeChat(content)).generate("s", "p", 10)


def test_default_client_is_created_without_connecting() -> None:
    client = OllamaClient(host="http://127.0.0.1:1")

    with pytest.raises(LlmError, match="Cannot reach Ollama"):
        client.generate("s", "p", 10)


def test_timeout_is_reported_as_timeout_not_unreachable() -> None:
    fake = FakeChat(error=httpx.ReadTimeout("slow"))

    with pytest.raises(LlmError, match="did not answer within 900s"):
        OllamaClient(client=fake).generate("s", "p", 10)


def test_prompt_that_overflows_the_context_is_an_error() -> None:
    fake = FakeChat(prompt_eval_count=7900)

    with pytest.raises(LlmError, match="does not fit"):
        OllamaClient(num_ctx=8192, client=fake).generate("s", "p", 800)


class SequenceChat:
    """Returns the queued (content, done_reason) answers one per call and records num_predict."""

    def __init__(self, *answers: tuple[str, str]) -> None:
        self.answers = list(answers)
        self.limits: list[int] = []

    def chat(self, *, model: str, messages: list[dict[str, str]], options: dict[str, Any]) -> Any:
        self.limits.append(options["num_predict"])
        content, reason = self.answers.pop(0)
        return SimpleNamespace(message=SimpleNamespace(content=content), done_reason=reason)


def test_truncated_answer_is_retried_with_a_bigger_limit() -> None:
    fake = SequenceChat(("First sentence. Second sen", "length"), ("Full answer here.", "stop"))

    assert OllamaClient(client=fake).generate("s", "p", 100) == "Full answer here."
    assert fake.limits == [100, 200]


def test_still_truncated_answer_keeps_whole_sentences() -> None:
    cut = "First sentence. Second sentence. Third unfin"
    fake = SequenceChat((cut, "length"), (cut, "length"))

    assert OllamaClient(client=fake).generate("s", "p", 10) == "First sentence. Second sentence."
    assert fake.limits == [10, 20]


@pytest.mark.parametrize(
    "text",
    ["no sentence end here at all", "Short. and then a very long unfinished tail without end"],
)
def test_cut_answer_without_a_good_sentence_end_is_left_alone(text: str) -> None:
    fake = SequenceChat((text, "length"), (text, "length"))

    assert OllamaClient(client=fake).generate("s", "p", 10) == text


def test_complete_answer_is_not_trimmed() -> None:
    fake = FakeChat("Whole answer. Trailing words", done_reason="stop")

    assert OllamaClient(client=fake).generate("s", "p", 10) == "Whole answer. Trailing words"


class ContextChat:
    """Reports a prompt of 6700 tokens; the first answer is cut off, the retry may fail."""

    def __init__(self, retry_error: Exception | None = None) -> None:
        self.retry_error = retry_error
        self.calls = 0

    def chat(self, *, model: str, messages: list[dict[str, str]], options: dict[str, Any]) -> Any:
        self.calls += 1
        if self.calls == 2 and self.retry_error:
            raise self.retry_error
        content = "One sentence. Another sentence. Cut of" if self.calls == 1 else "Complete."
        reason = "length" if self.calls == 1 else "stop"
        return SimpleNamespace(
            message=SimpleNamespace(content=content), done_reason=reason, prompt_eval_count=6700
        )


def test_retry_is_not_blocked_by_the_context_check_when_the_prompt_fits() -> None:
    fake = ContextChat()

    answer = OllamaClient(num_ctx=8192, client=fake).generate("s", "p", 800)

    assert answer == "Complete."
    assert fake.calls == 2


def test_failed_retry_falls_back_to_the_first_trimmed_answer() -> None:
    fake = ContextChat(retry_error=ConnectionError("dropped"))

    answer = OllamaClient(num_ctx=8192, client=fake).generate("s", "p", 800)

    assert answer == "One sentence. Another sentence."


def test_content_problems_are_content_errors_but_connection_problems_are_not() -> None:
    with pytest.raises(LlmContentError):
        OllamaClient(client=FakeChat("")).generate("s", "p", 10)
    with pytest.raises(LlmContentError):
        OllamaClient(num_ctx=8192, client=FakeChat(prompt_eval_count=7900)).generate("s", "p", 800)

    with pytest.raises(LlmError) as info:
        OllamaClient(client=FakeChat(error=ConnectionError("refused"))).generate("s", "p", 10)
    assert not isinstance(info.value, LlmContentError)
