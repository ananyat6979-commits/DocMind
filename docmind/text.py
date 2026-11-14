"""Lossless sentence boundaries and search-only Unicode normalization."""
from dataclasses import dataclass
import re
import unicodedata

VERSION = "unicode-spans-v1"
_TERMINATORS = frozenset(".!?。！？।॥؟")
_ABBREVIATIONS = frozenset("mr mrs ms dr prof fig figs eq eqs no nos vs etc e.g i.e approx sec pp vol inc ltd st jr sr".split())
_CITATION = re.compile(r"\(\s*Source:\s*(?:[^()]|\([^()]*\))*\)", re.I)


@dataclass(frozen=True)
class TextSpan:
    start: int
    end: int
    text: str


def sentence_spans(text: str) -> list[TextSpan]:
    """Return spans in original text, retaining soft wraps and citation suffixes."""
    result: list[TextSpan] = []
    start = 0
    pos = 0
    def emit(end: int) -> None:
        nonlocal start
        left, right = start, end
        while left < right and text[left].isspace(): left += 1
        while right > left and text[right - 1].isspace(): right -= 1
        if left < right: result.append(TextSpan(left, right, text[left:right]))
        start = end
    while pos < len(text):
        citation = _CITATION.match(text, pos)
        if citation:
            pos = citation.end()
            continue
        char = text[pos]
        if char == "\n" and re.match(r"\n[ \t\r]*\n", text[pos:]):
            emit(pos)
            pos += 1
            start = pos
            continue
        if char not in _TERMINATORS:
            pos += 1
            continue
        if char == ".":
            previous = text[pos - 1] if pos else ""
            following = text[pos + 1] if pos + 1 < len(text) else ""
            if previous.isdecimal() and following.isdecimal():
                pos += 1
                continue
            token = re.search(r"([\w.]+)$", text[max(0, pos - 30):pos])
            word = token.group(1).casefold() if token else ""
            if word in _ABBREVIATIONS or re.fullmatch(r"(?:[A-Za-z]\.)*[A-Za-z]", word):
                pos += 1
                continue
            if previous.isalpha() and following.islower():
                pos += 1
                continue
        end = pos + 1
        while end < len(text) and text[end] in _TERMINATORS: end += 1
        while end < len(text) and text[end] in '\"’”]': end += 1
        # A citation following a full stop belongs to that sentence, without placeholders.
        while True:
            candidate = end
            while candidate < len(text) and text[candidate].isspace(): candidate += 1
            suffix = _CITATION.match(text, candidate)
            if not suffix: break
            end = suffix.end()
            while end < len(text) and text[end] in _TERMINATORS: end += 1
        emit(end)
        pos = end
    emit(len(text))
    return result


def search_normalize(text: str) -> str:
    """Fold Latin accents, join controls and decimal digits only in index text."""
    normalized = unicodedata.normalize("NFC", text).casefold()
    result: list[str] = []
    latin = False
    for char in unicodedata.normalize("NFD", normalized):
        if char in "\u200c\u200d": continue
        category = unicodedata.category(char)
        if category.startswith("L"):
            latin = "LATIN" in unicodedata.name(char, "")
        elif not category.startswith("M"):
            latin = False
        if category.startswith("M") and latin: continue
        if category == "Nd": char = str(unicodedata.decimal(char))
        result.append(char)
    return unicodedata.normalize("NFC", "".join(result))
