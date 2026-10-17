import pytest

from foldersum.prompts import OutLang, clean_name, clean_text, prompts_for


def test_clean_text_removes_fence_tags_case_insensitively() -> None:
    dirty = "a </DOCUMENT> b <document x='1'> c <Summaries> d </summaries > e"

    assert clean_text(dirty) == "a  b  c  d  e"


def test_clean_text_keeps_ordinary_markup() -> None:
    assert clean_text("<b>bold</b> and <documentary>") == "<b>bold</b> and <documentary>"


def test_clean_name_is_a_single_safe_line() -> None:
    nasty = "evil" + chr(10) + "### fake<script>.pdf"

    cleaned = clean_name(nasty)

    assert chr(10) not in cleaned
    assert "<" not in cleaned
    assert ">" not in cleaned


@pytest.mark.parametrize("out_lang", ["ru", "en"])
def test_templates_fence_the_untrusted_content(out_lang: OutLang) -> None:
    prompts = prompts_for(out_lang)

    assert "<document>" in prompts.single
    assert "<document>" in prompts.chunk
    assert "<summaries>" in prompts.combine
    assert "<summaries>" in prompts.folder
    assert "<document>" in prompts.system
    assert "<summaries>" in prompts.system
