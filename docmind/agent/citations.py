"""
Citation verification: the mechanism behind a *measurable* "no hallucinated
sources" claim.

Every observation shows the model the exact citation string for each chunk
(Chunk.citation()). After the model answers we parse every "(Source: ...)" in
the answer and check it against the citations of chunks that were actually
retrieved in this run. Anything that does not match is removed from the answer
and counted. The headline metric is `citation_validity = valid / cited`.
"""
import re
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence, Set, Tuple

from docmind.models import SearchResult

# Matches "(Source: ...)" allowing one level of nested parentheses,
# e.g. "(Source: report (1).pdf, p.3)".
CITATION_RE = re.compile(r"\(\s*Source:\s*((?:[^()]|\([^()]*\))*)\)", re.IGNORECASE)
REMOVED_MARK = "[unverified citation removed]"


def _norm(item: str) -> str:
    s = item.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s*,\s*", ",", s)
    s = re.sub(r"\bp\.\s*", "p.", s)
    s = re.sub(r"§\s*", "§", s)
    return s


def _items(body: str) -> List[str]:
    """'a.pdf, p.1; b.pdf, p.2' -> ['a.pdf,p.1', 'b.pdf,p.2']"""
    return [_norm(x) for x in re.split(r"\s*;\s*", body) if x.strip()]


def allowed_citations(sources: Sequence[SearchResult]) -> Dict[str, str]:
    """normalized citation body -> chunk_id of one chunk that supports it."""
    out: Dict[str, str] = {}
    for s in sources:
        m = CITATION_RE.search(s.chunk.citation())
        body = m.group(1) if m else ""
        for item in _items(body):
            out.setdefault(item, s.chunk.chunk_id)
    return out


@dataclass
class CitationReport:
    cited: List[str] = field(default_factory=list)
    valid: List[str] = field(default_factory=list)
    invalid: List[str] = field(default_factory=list)

    @property
    def validity(self) -> Optional[float]:
        return None if not self.cited else len(self.valid) / len(self.cited)

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["validity"] = self.validity
        return d


def verify_citations(answer: str, sources: Sequence[SearchResult]) -> Tuple[str, CitationReport, Set[str]]:
    """Return (answer with bad citations removed, report, chunk_ids actually cited)."""
    allowed = allowed_citations(sources)
    report = CitationReport()
    cited_chunks: Set[str] = set()

    def _sub(match: "re.Match") -> str:
        items = _items(match.group(1))
        if not items:
            report.cited.append("")
            report.invalid.append("")
            return REMOVED_MARK
        ok = [i for i in items if i in allowed]
        for i in items:
            report.cited.append(i)
            (report.valid if i in allowed else report.invalid).append(i)
        for i in ok:
            cited_chunks.add(allowed[i])
        if len(ok) == len(items):
            return match.group(0)
        if not ok:
            return REMOVED_MARK
        raw_items = [x.strip() for x in re.split(r"\s*;\s*", match.group(1)) if x.strip()]
        kept = [raw for raw in raw_items if _norm(raw) in allowed]
        return f"(Source: {'; '.join(kept)})"

    cleaned = CITATION_RE.sub(_sub, answer)
    return cleaned, report, cited_chunks
