"""Conservative extraction diagnostics, not an OCR or font-conversion system."""
import re
import unicodedata
from typing import Any, Iterable

VERSION = "extraction-quality-v1"
_LEGACY = ("krutidev", "kruti dev", "devlys", "chanakya", "shusha")
_SGR = re.compile(r"\x1b\[[0-9;]*m")


def normalize_text_input(text: str):
    edits = []
    removed = 0
    def strip(match):
        nonlocal removed
        edits.append({"normalized_start": match.start()-removed,
                      "source_start": match.start(), "source_end": match.end()})
        removed += match.end()-match.start()
        return ""
    cleaned = _SGR.sub(strip, text).replace("\f", "\n")
    return cleaned, edits


def source_offset(index: int, edits: Iterable[dict], end: bool = False) -> int:
    return index + sum(e["source_end"]-e["source_start"] for e in edits
                       if e["normalized_start"] < index or (not end and e["normalized_start"] == index))


def extraction_quality(text: str, expected_language: str | None = None,
                       font_names: Iterable[str] = ()) -> dict[str, Any]:
    fonts = sorted(set(font_names))
    letters = [c for c in text if unicodedata.category(c)[0] in "LM"]
    count = max(len(letters), 1)
    devanagari = sum("DEVANAGARI" in unicodedata.name(c, "") for c in letters)/count
    latin = sum("LATIN" in unicodedata.name(c, "") for c in letters)/count
    bad = sum(c == "\ufffd" or unicodedata.category(c) == "Co" for c in text)/max(len(text),1)
    controls = sum(unicodedata.category(c) == "Cc" and c not in "\n\r\t\f" for c in text)/max(len(text),1)
    flags: list[str] = []
    if not text.strip(): status = "image_only_or_empty"
    else:
        legacy_font = any(any(name in font.casefold() for name in _LEGACY) for font in fonts)
        malformed = bool(re.search(r"(?:^|\s)[\u093a-\u094f\u0962\u0963]",text))
        if legacy_font: flags.append("legacy_font_name")
        if expected_language == "hi" and len(letters) >= 20 and devanagari < .2:
            flags.append("expected_hindi_script_mismatch")
        if controls > .002: flags.append("unexpected_control_characters")
        if bad > .01: flags.append("replacement_or_private_use_characters")
        if malformed: flags.append("suspicious_combining_sequence")
        if legacy_font or bad > .01 or controls > .002: status = "suspected_legacy_font"
        elif flags: status = "uncertain_or_mixed"
        else: status = "usable_unicode"
    return {"version": VERSION, "status": status, "flags": flags,
            "devanagari_ratio": devanagari, "latin_ratio": latin,
            "replacement_private_use_ratio": bad, "unexpected_control_ratio":controls, "font_names": fonts,
            "expected_language": expected_language,
            "quarantined": status != "usable_unicode"}
