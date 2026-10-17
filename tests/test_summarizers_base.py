import pytest

from foldersum.errors import LlmError
from foldersum.summarizers.base import answer_with_retry, drop_unfinished_sentence


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("One. Two. Thre", "One. Two."),
        ("Один. Два. Тр", "Один. Два."),
        ("no sentence end at all", "no sentence end at all"),
        (
            "Short. and then a very long unfinished tail without any end",
            "Short. and then a very long unfinished tail without any end",
        ),
        ("Complete.", "Complete."),
    ],
)
def test_drop_unfinished_sentence(text: str, expected: str) -> None:
    assert drop_unfinished_sentence(text) == expected


def test_complete_answer_is_returned_as_is() -> None:
    calls: list[tuple[int, bool]] = []

    def ask(tokens: int, first: bool) -> tuple[str, bool]:
        calls.append((tokens, first))
        return "Done.", False

    assert answer_with_retry(ask, 100) == "Done."
    assert calls == [(100, True)]


def test_truncated_answer_is_retried_once_with_double_the_limit() -> None:
    calls: list[tuple[int, bool]] = []

    def ask(tokens: int, first: bool) -> tuple[str, bool]:
        calls.append((tokens, first))
        return ("Cut. Off", True) if first else ("Whole answer.", False)

    assert answer_with_retry(ask, 100) == "Whole answer."
    assert calls == [(100, True), (200, False)]


def test_retry_failure_keeps_the_first_answer_trimmed() -> None:
    def ask(tokens: int, first: bool) -> tuple[str, bool]:
        if not first:
            raise LlmError("down")
        return "Kept sentence. Dropped par", True

    assert answer_with_retry(ask, 100) == "Kept sentence."


def test_first_attempt_failure_propagates() -> None:
    def ask(tokens: int, first: bool) -> tuple[str, bool]:
        raise LlmError("down")

    with pytest.raises(LlmError):
        answer_with_retry(ask, 100)


def test_empty_truncated_answer_is_retried_and_the_retry_is_used() -> None:
    def ask(tokens: int, first: bool) -> tuple[str, bool]:
        return ("", True) if first else ("Real answer.", False)

    assert answer_with_retry(ask, 100) == "Real answer."


def test_empty_truncated_answer_twice_is_an_error() -> None:
    def ask(tokens: int, first: bool) -> tuple[str, bool]:
        return "", True

    with pytest.raises(LlmError, match="no answer within the token limit"):
        answer_with_retry(ask, 100)


def test_empty_truncated_answer_with_a_failing_retry_is_an_error() -> None:
    def ask(tokens: int, first: bool) -> tuple[str, bool]:
        if first:
            return "", True
        raise LlmError("down")

    with pytest.raises(LlmError, match="down"):
        answer_with_retry(ask, 100)
