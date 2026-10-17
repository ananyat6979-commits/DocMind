"""
Human-in-the-loop routing policy. Pure function of an AgentResponse.

An answer goes to the review queue when ANY of these hold:
  abstained               the system said NOT_FOUND (a human checks it was right to)
  no_citations            a factual answer with zero verifiable citations
  invalid_citation        at least one citation did not match retrieved evidence
  low_retrieval_confidence best rerank score below DOCMIND_REVIEW_MIN_SCORE
  no_clean_final_answer   the model never produced a proper Final Answer
  random_audit            a deterministic sample (audit_rate) of confident answers,
                          so reviewers also measure the false-accept rate
"""
import hashlib
from typing import List

from docmind.config import CONFIG
from docmind.models import AgentResponse


def _audit_hit(response_id: str, rate: float) -> bool:
    if rate <= 0:
        return False
    bucket = int(hashlib.sha1(response_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return bucket < rate


def review_reasons(resp: AgentResponse) -> List[str]:
    cfg = CONFIG.review
    reasons: List[str] = []
    if resp.abstained:
        reasons.append("abstained")
    else:
        report = resp.citation_report or {}
        if not report.get("cited"):
            reasons.append("no_citations")
        if report.get("invalid"):
            reasons.append("invalid_citation")
        if resp.finish_reason in ("synthesis", "fallback"):
            reasons.append("no_clean_final_answer")
        scores = [s.score for s in resp.sources if s.retriever == "reranker"]
        if cfg.min_top_score is not None and scores and max(scores) < cfg.min_top_score:
            reasons.append("low_retrieval_confidence")
    if not reasons and _audit_hit(resp.response_id, cfg.audit_rate):
        reasons.append("random_audit")
    return reasons
