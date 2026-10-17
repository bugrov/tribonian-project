"""Detect whether a text is Russian or English by its alphabet."""

from foldersum.models import Lang

_MIN_LETTERS = 20
_RUSSIAN_SHARE = 0.5


def detect_language(text: str) -> Lang:
    """Return "ru" or "en" by the share of Cyrillic letters, or "unknown" for too little text.

    The tool only deals with Russian and English, for which the alphabet is a reliable and
    dependency-free signal that also holds up on noisy OCR output.
    """
    cyrillic = sum(1 for ch in text if "Ѐ" <= ch <= "ӿ")
    latin = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    total = cyrillic + latin
    if total < _MIN_LETTERS:
        return "unknown"
    return "ru" if cyrillic / total >= _RUSSIAN_SHARE else "en"
