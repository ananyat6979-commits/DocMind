"""Offline language hints. Detection scores are not calibrated correctness probabilities."""
from functools import lru_cache
import re
import unicodedata
from typing import Any

VERSION = "language-v2"
_ROMAN_HINDI = frozenset("kya hai hain nahi nahin kaise kitna kitni chahiye hoga mujhe iska uska karna batao".split())


@lru_cache(maxsize=1)
def _detector():
    from lingua import LanguageDetectorBuilder
    return LanguageDetectorBuilder.from_all_languages().with_low_accuracy_mode().build()


def detection_sample(value: str, limit: int = 4000) -> str:
    """Sample across the document and suppress duplicated sentences, without changing source text."""
    parts = re.split(r"(?<=[.!?।॥])\s+|[\r\n]+", value)
    unique = list(dict.fromkeys(part.strip() for part in parts if part.strip()))
    prose = " ".join(unique)
    if len(prose) <= limit:return prose
    width = limit // 3
    middle = max(0, len(prose)//2-width//2)
    return " ".join((prose[:width], prose[middle:middle+width], prose[-width:]))


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
        scores = _detector().compute_language_confidence_values(detection_sample(value))
    except ImportError:
        result["flags"] = ["detector_unavailable"]
        return result
    if not scores: return result
    confidence = float(scores[0].value)
    margin = confidence - (float(scores[1].value) if len(scores) > 1 else 0.0)
    result["confidence"] = confidence
    if confidence >= .8 and margin >= .2:
        code = scores[0].language.iso_code_639_1.name.lower()
        if devanagari and code in ("mr", "ne", "sa"):
            result["flags"] = ["devanagari_language_ambiguous"]
        else:result["language"] = code
    else:
        result["flags"] = ["ambiguous_language"]
    return result


@lru_cache(maxsize=512)
def short_language_hint(text: str) -> str:
    """A conservative extra veto for short prose, never proof of English."""
    if not any(c.isalpha() for c in text):
        return "und"
    try:
        scores = _detector().compute_language_confidence_values(text)
    except ImportError:
        return "und"
    if not scores:
        return "und"
    best = scores[0]
    margin = float(best.value) - (float(scores[1].value) if len(scores) > 1 else 0)
    if float(best.value) >= .6 and margin >= .2:
        return best.language.iso_code_639_1.name.lower()
    return "und"


def document_language(units) -> dict[str, Any]:
    """Weight readable prose, never a caller-supplied language permission."""
    weights: dict[str, int] = {}
    for unit in units:
        if unit.metadata.get("extraction_quality", {}).get("quarantined"):
            continue
        prose = re.sub(r"```.*?```", " ", unit.content, flags=re.S)
        count = sum(c.isalpha() for c in prose)
        hint = language_info(prose)
        code = hint.get("language", "und")
        weights[code] = weights.get(code, 0) + count
    total = sum(weights.values())
    best = max(weights, key=lambda key: weights[key]) if weights else "und"
    fraction = weights.get(best, 0) / total if total else 0.0
    return {"language": best if best != "und" and fraction >= .8 and total >= 40 else "und",
            "fraction": fraction, "prose_letters": total, "version": VERSION}
