"""Offline language hints. Detection scores are not calibrated correctness probabilities."""

import re
import unicodedata
from functools import lru_cache
from typing import Any

VERSION = "language-v4"
_ROMAN_HINDI = frozenset(
    "kya hai hain nahi nahin kaise kitna kitni chahiye hoga mujhe iska uska karna batao".split()
)


@lru_cache(maxsize=1)
def _detector():
    from lingua import LanguageDetectorBuilder

    return LanguageDetectorBuilder.from_all_languages().with_low_accuracy_mode().build()


@lru_cache(maxsize=1)
def _english_context_detector():
    """A corroborating model, never a closed-set permission by itself."""
    from lingua import Language, LanguageDetectorBuilder

    names = (
        "ENGLISH",
        "FRENCH",
        "GERMAN",
        "FINNISH",
        "HINDI",
        "MARATHI",
        "SPANISH",
        "PORTUGUESE",
        "ITALIAN",
        "DUTCH",
        "POLISH",
        "ROMANIAN",
        "SWEDISH",
        "DANISH",
        "BOKMAL",
        "ESTONIAN",
        "LATIN",
        "TURKISH",
    )
    return LanguageDetectorBuilder.from_languages(
        *(getattr(Language, name) for name in names)
    ).build()


def _english_corroborated(value):
    words = set(re.findall(r"[a-z]+", value.casefold()))
    markers = words & {
        "the",
        "is",
        "are",
        "must",
        "which",
        "what",
        "when",
        "where",
        "who",
        "does",
        "should",
        "within",
        "during",
        "with",
        "from",
    }
    if len(markers) < 3:
        return False
    scores = _english_context_detector().compute_language_confidence_values(
        detection_sample(value)
    )
    return bool(
        scores
        and scores[0].language.iso_code_639_1.name.lower() == "en"
        and float(scores[0].value) >= 0.9
        and float(scores[0].value) - float(scores[1].value) >= 0.4
    )


def detection_sample(value: str, limit: int = 4000) -> str:
    """Sample across the document and suppress duplicated sentences, without changing source text."""
    parts = re.split(
        r"(?<=[.!?\u0964\u0965])\s+|[\r\n]+",
        value,
    )
    unique = list(dict.fromkeys(part.strip() for part in parts if part.strip()))
    prose = " ".join(unique)
    if len(prose) <= limit:
        return prose
    width = limit // 3
    middle = max(0, len(prose) // 2 - width // 2)
    return " ".join((prose[:width], prose[middle : middle + width], prose[-width:]))


def language_info(text: str) -> dict[str, Any]:
    value = unicodedata.normalize("NFC", text)
    words = set(re.findall(r"[a-z]+", value.casefold()))
    devanagari = sum("DEVANAGARI" in unicodedata.name(c, "") for c in value)
    letters = sum(c.isalpha() for c in value)
    result: dict[str, Any] = {
        "language": "und",
        "confidence": 0.0,
        "version": VERSION,
        "validation": "unmeasured",
        "flags": [],
    }
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
    if not scores:
        return result
    confidence = float(scores[0].value)
    margin = confidence - (float(scores[1].value) if len(scores) > 1 else 0.0)
    result["confidence"] = confidence
    if confidence >= 0.8 and margin >= 0.2:
        code = scores[0].language.iso_code_639_1.name.lower()
        if devanagari and code in ("mr", "ne", "sa"):
            result["flags"] = ["devanagari_language_ambiguous"]
        else:
            result["language"] = code
    else:
        result["flags"] = ["ambiguous_language"]
    # Only corroborate ambiguous prose or the known technical-English/Latin
    # confusion. Confidently detected contemporary foreign languages retain veto.
    if (
        result["language"] in ("und", "la")
        and not devanagari
        and _english_corroborated(value)
    ):
        result.update(language="en", flags=["english_context_corroborated"])
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
    if float(best.value) >= 0.6 and margin >= 0.2:
        return best.language.iso_code_639_1.name.lower()
    return "und"


@lru_cache(maxsize=512)
def short_foreign_prose(text: str) -> bool:
    """Veto identified foreign phrases, not isolated quoted names."""
    words = re.findall(r"[^\W_]+", text, re.UNICODE)
    if len(words) < 2:
        return False
    scores = _english_context_detector().compute_language_confidence_values(text)
    if not scores:
        return False
    best = scores[0]
    code = best.language.iso_code_639_1.name.lower()
    return (
        code not in ("en", "la")
        and float(best.value) >= 0.6
        and float(best.value) - float(scores[1].value) >= 0.2
    )


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
    language = best if best != "und" and fraction >= 0.8 and total >= 40 else "und"
    # A strict English majority needs independent pooled-prose corroboration.
    # Caller hints and unidentified mass alone never establish English support.
    english_fraction = weights.get("en", 0) / total if total else 0
    if language == "und" and english_fraction > 0.5:
        prose = " ".join(
            u.content
            for u in units
            if not u.metadata.get("extraction_quality", {}).get("quarantined")
        )
        if _english_corroborated(prose):
            language = "en"
    return {
        "language": language,
        "fraction": fraction,
        "prose_letters": total,
        "version": VERSION,
    }
