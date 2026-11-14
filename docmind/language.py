"""Offline language hints. Detection scores are not calibrated correctness probabilities."""
from functools import lru_cache
import re
import unicodedata
from typing import Any

VERSION = "language-v1"
_ROMAN_HINDI = frozenset("kya hai hain nahi nahin kaise kitna kitni chahiye hoga mujhe iska uska karna batao".split())


@lru_cache(maxsize=1)
def _detector():
    from lingua import LanguageDetectorBuilder
    return LanguageDetectorBuilder.from_all_languages().with_low_accuracy_mode().build()


def language_info(text: str) -> dict[str, Any]:
    value = unicodedata.normalize("NFC", text)
    words = set(re.findall(r"[a-z]+", value.casefold()))
    devanagari = sum("DEVANAGARI" in unicodedata.name(c, "") for c in value)
    letters = sum(c.isalpha() for c in value)
    result: dict[str, Any] = {"language": "und", "confidence": 0.0, "version": VERSION,
                              "validation": "unmeasured", "flags": []}
    if not devanagari and len(words & _ROMAN_HINDI) >= 2:
        result["flags"] = ["possible_romanized_hindi"]
        return result
    if letters < 20:
        result["flags"] = ["short_text"]
        return result
    try:
        scores = _detector().compute_language_confidence_values(value[:4000])
    except ImportError:
        result["flags"] = ["detector_unavailable"]
        return result
    if not scores: return result
    confidence = float(scores[0].value)
    margin = confidence - (float(scores[1].value) if len(scores) > 1 else 0.0)
    result["confidence"] = confidence
    if confidence >= .8 and margin >= .2:
        result["language"] = scores[0].language.iso_code_639_1.name.lower()
    else:
        result["flags"] = ["ambiguous_language"]
    return result
