import pytest

from foldersum.chunking import split_text


def test_short_text_is_one_chunk() -> None:
    assert split_text("line one\nline two", 100) == ["line one\nline two"]


def test_empty_text_has_no_chunks() -> None:
    assert split_text("  \n ", 100) == []


def test_lines_are_packed_without_exceeding_limit() -> None:
    text = "\n".join(["aaaa bbbb"] * 10)

    chunks = split_text(text, 30)

    assert all(len(chunk) <= 30 for chunk in chunks)
    assert "\n".join(chunks) == text
    assert len(chunks) > 1


def test_long_line_is_split_at_sentence_ends() -> None:
    text = "First sentence here. Second sentence here. Third sentence here."

    chunks = split_text(text, 30)

    assert chunks == ["First sentence here.", "Second sentence here.", "Third sentence here."]


def test_long_sentence_is_split_at_spaces() -> None:
    text = " ".join(["word"] * 20)

    chunks = split_text(text, 24)

    assert all(len(chunk) <= 24 for chunk in chunks)
    assert " ".join(" ".join(chunks).split()) == text


def test_unbroken_string_is_hard_split() -> None:
    chunks = split_text("x" * 25, 10)

    assert chunks == ["x" * 10, "x" * 10, "x" * 5]


def test_no_text_is_lost_in_mixed_content() -> None:
    text = "Intro line\n" + "Long sentence without end " * 12 + "\nTail line"

    chunks = split_text(text, 50)

    assert all(len(chunk) <= 50 for chunk in chunks)
    assert " ".join(" ".join(chunks).split()) == " ".join(text.split())


def test_oversized_sentence_after_a_short_one_flushes_the_short_one_first() -> None:
    text = "Short one. " + "y" * 40 + " tail"

    chunks = split_text(text, 20)

    assert chunks[0] == "Short one."
    assert all(len(chunk) <= 20 for chunk in chunks)
    assert "".join(chunks).replace(" ", "") == text.replace(" ", "")


@pytest.mark.parametrize("limit", [0, -5])
def test_non_positive_limit_is_rejected(limit: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        split_text("text", limit)
