"""Deterministic quote location checks. Matching text does not prove entailment."""
from dataclasses import asdict, dataclass
import json
import unicodedata
from typing import Any, Sequence

import regex

from docmind.models import SearchResult

VERSION = "quoted-evidence-v1"
_ESCAPED = {0x85, 0x2028, 0x2029, 0x61C, 0x200E, 0x200F, 0x200B, 0x2060}
_ESCAPED.update(range(0x202A, 0x202F))
_ESCAPED.update(range(0x2066, 0x206A))
_ESCAPED.update(range(0xE0000, 0xE0080))


def untrusted_json(value: Any) -> str:
    """Readable Unicode JSON with explicit directional/separator escaping."""
    serialized = json.dumps(value, ensure_ascii=False)
    return "".join(chr(92) + "u" + format(ord(c), "04x") if ord(c) in _ESCAPED and ord(c) <= 0xFFFF else (chr(92) + "u" + format(0xD800 + ((ord(c)-0x10000) >> 10), "04x") +
                         chr(92) + "u" + format(0xDC00 + ((ord(c)-0x10000) & 0x3FF), "04x") if ord(c) in _ESCAPED else c)
                   for c in serialized)


def canonical_text(text: str) -> tuple[str, list[tuple[int, int]]]:
    """Map NFC/whitespace-normalized code points back to the input string."""
    characters: list[str] = []
    positions: list[tuple[int, int]] = []
    for match in regex.finditer(r"\X", text):
        cluster = unicodedata.normalize("NFC", match.group())
        for character in cluster:
            if character == "\u00ad":
                continue  # Explicit discretionary marker only; literal hyphens remain.
            if character.isspace():
                if characters and characters[-1] == " ":
                    positions[-1] = (positions[-1][0], match.end())
                    continue
                character = " "
            characters.append(character)
            positions.append((match.start(), match.end()))
    while characters and characters[-1] == " ":
        characters.pop(); positions.pop()
    if characters and characters[0] == " ":
        characters.pop(0); positions.pop(0)
    return "".join(characters), positions


@dataclass(frozen=True)
class QuoteMatch:
    status: str
    start: int | None = None
    end: int | None = None
    text: str | None = None
    offset_unit: str = "unicode_codepoint"
    offset_scope: str = "chunk_content"


def _locations(text: str, quote: str) -> list[int]:
    result: list[int] = []
    position = text.find(quote)
    while position >= 0 and len(result) < 2:
        result.append(position)
        position = text.find(quote, position + 1)
    return result


def match_quote(text: str, quote: str) -> QuoteMatch:
    if not quote or not quote.strip():
        return QuoteMatch("missing")
    exact = _locations(text, quote)
    if len(exact) > 1:
        return QuoteMatch("ambiguous")
    if exact:
        start = exact[0]
        return QuoteMatch("literal", start, start + len(quote), text[start:start + len(quote)])
    canonical, positions = canonical_text(text)
    needle, _ = canonical_text(quote)
    if not needle:
        return QuoteMatch("missing")
    found = _locations(canonical, needle)
    if len(found) > 1:
        return QuoteMatch("ambiguous")
    if not found:
        return QuoteMatch("missing")
    start, end = positions[found[0]][0], positions[found[0] + len(needle) - 1][1]
    return QuoteMatch("normalized", start, end, text[start:end])


def parse_payload(text: str) -> dict[str, Any]:
    value = text.strip()
    if value.startswith("```") and value.endswith("```"):
        lines = value.splitlines()
        if lines[0].strip().lower() not in ("```", "```json"):
            raise ValueError("Unsupported response wrapper")
        value = "\n".join(lines[1:-1])
    if len(value) > 32000:
        raise ValueError("Response exceeds the structured-output bound")
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise ValueError("Response must be a JSON object")
    return payload


def validate_claims(payload: dict[str, Any], sources: Sequence[SearchResult]) -> dict[str, Any]:
    """Validate every displayed claim; never accept an unlinked answer summary."""
    claims = payload.get("claims", [])
    if not isinstance(claims, list) or len(claims) > 8:
        raise ValueError("Invalid claims array")
    abstained = payload.get("abstained", False)
    if not isinstance(abstained, bool) or (abstained and claims):
        raise ValueError("Abstention and claims conflict")
    if not abstained and not claims:
        raise ValueError("Non-abstaining responses require claims")
    lookup = {result.chunk.chunk_id: result for result in sources}
    if len(lookup) != len(sources):
        raise ValueError("Duplicate evidence identities")
    evidence = []
    errors = []
    texts = []
    for index, claim in enumerate(claims):
        if not isinstance(claim, dict) or not isinstance(claim.get("text"), str):
            raise ValueError("Invalid claim")
        text = claim["text"].strip()
        if not text or len(text) > 4000 or "(source:" in text.casefold():
            raise ValueError("Claim text is empty, oversized or contains inline citations")
        references = claim.get("evidence")
        if not isinstance(references, list) or not 1 <= len(references) <= 4:
            raise ValueError("Every claim requires bounded evidence")
        texts.append(text)
        for reference in references:
            if not isinstance(reference, dict):
                raise ValueError("Invalid evidence reference")
            chunk_id, quote = reference.get("chunk_id"), reference.get("quote")
            if not isinstance(chunk_id, str) or not isinstance(quote, str) or len(quote) > 4000:
                raise ValueError("Invalid quote fields")
            source = lookup.get(chunk_id)
            if source is None:
                errors.append("unknown_citation")
                continue
            matched = match_quote(source.chunk.content, quote)
            if matched.status not in ("literal", "normalized"):
                errors.append("quote_" + matched.status)
            evidence.append({"claim_index": index, "claim": text, "chunk_id": chunk_id,
                "file_id": source.chunk.file_id, "filename": source.chunk.filename(),
                "page": source.chunk.metadata.get("page"), "section": source.chunk.metadata.get("section"),
                **asdict(matched)})
    answer = " ".join(texts)
    if "answer" in payload:
        if not isinstance(payload["answer"], str) or canonical_text(payload["answer"])[0] != canonical_text(answer)[0]:
            raise ValueError("Answer contains text outside the linked claims")
    short_answer = payload.get("short_answer", answer)
    if not isinstance(short_answer, str) or len(short_answer) > 256:
        raise ValueError("Short answer must be a bounded string")
    if abstained and short_answer.strip():raise ValueError("An abstention cannot contain a short answer")
    short_normalized = canonical_text(short_answer)[0]
    if not abstained and (not short_normalized or short_normalized not in canonical_text(answer)[0]):
        raise ValueError("Short answer must be a substring of the displayed claims")
    return {"version": VERSION, "answer": answer, "short_answer": short_answer,
            "short_answer_explicit": "short_answer" in payload, "abstained": abstained,
            "claims": evidence, "errors": sorted(set(errors)),
            "quotes_matched": bool(evidence) and not errors,
            "claim_support": "not_independently_verified"}
