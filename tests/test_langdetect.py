import pytest

from foldersum.langdetect_utils import detect_language
from foldersum.models import Lang


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Гражданский кодекс Российской Федерации, часть первая", "ru"),
        ("The Civil Code of the Russian Federation, part one", "en"),
        ("Договор поставки (supply agreement) между сторонами заключён", "ru"),
        ("The contract (договор) is concluded between the parties hereto", "en"),
        ("Да", "unknown"),
        ("", "unknown"),
        ("12345 67890 !!! ???", "unknown"),
    ],
)
def test_detect_language(text: str, expected: Lang) -> None:
    assert detect_language(text) == expected
