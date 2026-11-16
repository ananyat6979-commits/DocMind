"""Current release policy. NLI is diagnostic and cannot grant or deny release."""

import re
import unicodedata

from docmind.config import CONFIG
from docmind.models import AgentResponse
from docmind.review.policy import _audit_hit, _language_neutral_number

VERSION = "cited-draft-v2"
_COMPLETE = re.compile(
    r"\b(?:list of all|complete list|in total|never mentions|no mention|entire document|all occurrences|exhaustive|comprehensive summary)\b",
    re.I,
)
_ENGLISH_NUMBER = re.compile(
    r"(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|[0-9]+)(?:[ .,-]+(?:seconds?|minutes?|hours?|days?|weeks?|months?|years?|degrees?|percent|centimeters?|millimeters?|meters?|bytes?|tokens?))?[.!]?",
    re.I,
)


def _unvalidated_language(response: AgentResponse, cited) -> list[str]:
    from docmind.language import language_info

    reasons = []
    file_languages = [
        s.chunk.metadata.get("document_language_info", {}).get("language", "und")
        for s in cited
    ]
    context_languages = response.document_context.get("languages", [])
    english_files = bool(file_languages) and all(x == "en" for x in file_languages)
    if response.abstained:
        english_files = bool(context_languages) and all(
            x == "en" for x in context_languages
        )

    def devanagari_prose(value):
        return any(
            "DEVANAGARI" in unicodedata.name(c, "")
            and unicodedata.category(c)[0] in "LM"
            for c in value
        )

    def permitted(value, inherit=False, query=False):
        if devanagari_prose(value):
            return False
        info = language_info(value)
        if "possible_romanized_hindi" in info.get("flags", []):
            return False
        language = info.get("language", "und")
        # Short questions are not answers: interrogatives need not occur in
        # evidence, and proper names do not establish a foreign prose language.
        if (
            inherit
            and query
            and len(value) <= 180
            and re.match(
                r"^(?:what|which|who|whose|when|where|why|how|is|are|was|were|does|do|did|can|could|should)\b",
                value.strip(),
                re.I,
            )
            and all(
                not c.isalpha() or "LATIN" in unicodedata.name(c, "") for c in value
            )
        ):
            return True
        if inherit and not query and len(value) <= 100:
            proper_words = re.findall(r"[^\W_]+", value, re.UNICODE)
            proper_fragment = (
                bool(proper_words)
                and all(w[0].isupper() for w in proper_words)
                and all(
                    not c.isalpha() or "LATIN" in unicodedata.name(c, "") for c in value
                )
            )
            if proper_fragment and value.strip().rstrip(".!?") in " ".join(
                s.chunk.content for s in cited
            ):
                # Explicit contemporary foreign phrase veto still applies.
                from docmind.language import short_language_hint

                hint = short_language_hint(value)
                if len(proper_words) == 1 or hint in ("en", "und", "la"):
                    return True
        if language == "en":
            return True
        if language != "und":
            return False
        if (
            _language_neutral_number(value)
            or _ENGLISH_NUMBER.fullmatch(value.strip())
            or value.strip().casefold()
            in ("yes", "yes.", "no", "no.", "true", "true.", "false", "false.")
        ):
            return inherit
        if inherit and "short_text" in info.get("flags", []):
            # File context is necessary, not blanket permission for Latin script.
            from docmind.language import short_language_hint

            hint = short_language_hint(value)
            if hint != "und":
                return hint == "en"
            # An unresolved Latin-script answer cannot acquire a language merely
            # from its file. Permit source-identical short fragments only.
            words = set(re.findall(r"[^\W_]+", value.casefold()))
            evidence_words = set(
                re.findall(
                    r"[^\W_]+", " ".join(s.chunk.content for s in cited).casefold()
                )
            )
            return bool(words) and words <= evidence_words
        return False

    if not permitted(response.query, english_files, query=True):
        reasons.append("query_language_unvalidated")
    if not response.abstained and not permitted(response.answer, english_files):
        reasons.append("answer_language_unvalidated")
    if (cited and not english_files) or (response.abstained and not english_files):
        reasons.append("language_policy_unvalidated")
    for source in cited:
        language = source.chunk.metadata.get("language_info", {}).get("language", "und")
        if language not in ("en", "und"):
            reasons.append("language_policy_unvalidated")
    return reasons


def apply_policy(response: AgentResponse) -> AgentResponse:
    mode = CONFIG.review.release_mode
    if mode not in ("draft", "strict"):
        raise ValueError("DOCMIND_RELEASE_MODE must be draft or strict")
    response.release_policy_version = VERSION
    response.release_mode = mode
    hold = []
    warnings = []
    cited = response.cited_sources
    partial = bool(response.document_context.get("partial")) or any(
        s.chunk.metadata.get("partial_document") for s in cited
    )
    if partial:
        warnings.append("document_incomplete")
    if response.abstained:
        if partial:
            response.answer = (
                "Not found in the readable pages. Absence cannot be confirmed."
            )
            hold.append("incomplete_document_abstention")
        else:
            response.answer = "Not found in the selected documents. The system may have missed relevant information."
            warnings.append("absence_not_verified")
    else:
        report = response.citation_report or {}
        quote_report = report.get("quote_validation") or {}
        if quote_report.get("support_errors"):
            hold.append("numeric_evidence_conflict")
        if report.get("invalid"):
            hold.append("invalid_citation")
        if not cited:
            hold.append("no_citations")
        if not quote_report.get("quotes_matched"):
            hold.append("invalid_or_missing_quote")
        if any(
            s.chunk.metadata.get("extraction_quality", {}).get("quarantined")
            for s in cited
        ):
            hold.append("cited_page_quarantined")
        if any(
            s.chunk.metadata.get("extraction_quality", {}).get("partial") for s in cited
        ):
            hold.append("cited_page_partial")
        if partial and _COMPLETE.search(response.answer):
            hold.append("incomplete_document_completeness_claim")
        if response.finish_reason in ("fallback", "format_failure", "deadline"):
            hold.append("no_clean_final_answer")
    hold.extend(_unvalidated_language(response, cited))
    if mode == "strict":
        hold.append("strict_human_approval")
    if _audit_hit(response.response_id, CONFIG.review.audit_rate):
        warnings.append("audit_requested")
    response.hold_reasons = sorted(set(hold))
    response.review_reasons = response.hold_reasons.copy()
    response.warnings = sorted(set(warnings))
    response.needs_review = bool(hold)
    response.release_state = "held" if hold else "draft"
    return response
