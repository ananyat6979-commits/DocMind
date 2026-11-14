"""
Citation label existence checks. These checks do not establish claim support.

Every observation shows the model the exact citation string for each chunk
(Chunk.citation()). After the model answers we parse every "(Source: ...)" in
the answer and check it against the citations of chunks that were actually
retrieved in this run. Anything that does not match is removed from the answer
and counted. The headline metric is `citation_validity = valid / cited`.
"""
import re
import unicodedata
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence, Set, Tuple

from docmind.models import SearchResult

# Matches "(Source: ...)" allowing one level of nested parentheses,
# e.g. "(Source: report (1).pdf, p.3)".
CITATION_RE = re.compile(r"\(\s*Source:\s*((?:[^()]|\([^()]*\))*)\)", re.IGNORECASE)
REMOVED_MARK = "[unverified citation removed]"


def _norm(item: str) -> str:
    s = unicodedata.normalize("NFC", item).strip().casefold()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s*,\s*", ",", s)
    s = re.sub(r"\bp\.\s*", "p.", s)
    s = re.sub(r"§\s*", "§", s)
    return s


def _items(body: str) -> List[str]:
    """'a.pdf, p.1; b.pdf, p.2' -> ['a.pdf,p.1', 'b.pdf,p.2']"""
    return [_norm(x) for x in re.split(r"\s*;\s*", body) if x.strip()]


def source_citations(chunk):
    citations=[chunk.citation()]
    if chunk.file_id:citations.append(chunk.citation(include_id=True))
    return citations


def allowed_citations(sources: Sequence[SearchResult]) -> Dict[str, str]:
    """Accept unique labels and explicit file identities; reject ambiguous legacy labels."""
    out: Dict[str, str] = {}
    identities: Dict[str, str] = {}
    ambiguous: Set[str] = set()
    for source in sources:
        identity=source.chunk.file_id or source.chunk.source
        for citation in source_citations(source.chunk):
            match=CITATION_RE.search(citation)
            for item in _items(match.group(1) if match else ""):
                if item in out and identities[item]!=identity:ambiguous.add(item)
                out.setdefault(item,source.chunk.chunk_id);identities[item]=identity
    return {key:value for key,value in out.items() if key not in ambiguous}


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
            for source in sources:
                for citation in source_citations(source.chunk):
                    match_source=CITATION_RE.search(citation)
                    if match_source and i in _items(match_source.group(1)):
                        cited_chunks.add(source.chunk.chunk_id)
        if len(ok) == len(items):
            return match.group(0)
        if not ok:
            return REMOVED_MARK
        raw_items = [x.strip() for x in re.split(r"\s*;\s*", match.group(1)) if x.strip()]
        kept = [raw for raw in raw_items if _norm(raw) in allowed]
        return f"(Source: {'; '.join(kept)})"

    cleaned = CITATION_RE.sub(_sub, answer)
    return cleaned, report, cited_chunks
