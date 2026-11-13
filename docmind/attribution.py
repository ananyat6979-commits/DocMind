"""Claim-to-cited-span attribution using a local NLI model.

Scores are model judgments until agreement with independent human labels is measured.
Every evidence offset indexes the original retrieved chunk, never normalized text.
"""
import re
import os
from dataclasses import dataclass, asdict
from docmind.agent.citations import CITATION_RE, _items, allowed_citations


@dataclass
class ClaimEvidence:
    claim: str
    supported: bool
    entailment: float
    chunk_id: str | None = None
    start: int | None = None
    end: int | None = None
    quote: str | None = None


class NLIJudge:
    def __init__(self, model="typeform/distilbert-base-uncased-mnli", revision=None, local_files_only=True):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model, revision=revision, local_files_only=local_files_only)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model, revision=revision, local_files_only=local_files_only).eval()
        labels = {str(v).lower(): int(k) for k, v in self.model.config.id2label.items()}
        if "entailment" not in labels:
            raise ValueError("NLI model must expose an entailment label")
        self.entailment_id = labels["entailment"]
        self.identity = {"model": model, "revision": revision or getattr(self.model.config, "_commit_hash", None)}

    def __call__(self, pairs):
        scores = []
        for offset in range(0, len(pairs), 2):
            batch = pairs[offset:offset+2]
            encoded = self.tokenizer([p for p, _ in batch], [h for _, h in batch],
                padding=True, truncation=True, max_length=256, return_tensors="pt")
            with self.torch.inference_mode():
                scores.extend(self.model(**encoded).logits.softmax(-1)[:, self.entailment_id].tolist())
        return scores


def evidence_spans(text):
    matches = list(re.finditer(r"[^\n.!?]+(?:[.!?]+|$)", text))
    for i, match in enumerate(matches):
        yield match.start(), match.end(), text[match.start():match.end()]
        if i + 1 < len(matches):
            end = matches[i+1].end()
            yield match.start(), end, text[match.start():end]


def attribute(answer, sources, judge, threshold=0.8):
    allowed = allowed_citations(sources)
    claims = []
    # Citations bind to the immediately preceding sentence. Uncited sentences remain unsupported.
    citations_by_token = {}
    def protect(match):
        token = "CITEREF" + str(len(citations_by_token)) + "TOKEN"
        citations_by_token[token] = match.group(0)
        return token
    masked = CITATION_RE.sub(protect, answer)
    parts = re.split(r"(?<=[.!?])\s+|\n+", masked)
    parts = [re.sub(r"CITEREF\d+TOKEN", lambda m: citations_by_token[m.group(0)], part) for part in parts]
    for part in parts:
        citations = [item for m in CITATION_RE.finditer(part) for item in _items(m.group(1))]
        claim = CITATION_RE.sub("", part).strip()
        if not claim:
            continue
        candidates = []
        for source in sources:
            match = CITATION_RE.search(source.chunk.citation())
            labels = _items(match.group(1)) if match else []
            if not any(c in allowed and c in labels for c in citations):
                continue
            for start, end, quote in evidence_spans(source.chunk.content):
                candidates.append((source.chunk.chunk_id, start, end, quote))
        scores = judge([(c[3], claim) for c in candidates]) if candidates else []
        if scores:
            index = max(range(len(scores)), key=scores.__getitem__)
            cid, start, end, quote = candidates[index]
            claims.append(ClaimEvidence(claim, scores[index] >= threshold, scores[index], cid, start, end, quote))
        else:
            claims.append(ClaimEvidence(claim, False, 0.0))
    return {"claims": [asdict(c) for c in claims], "n_claims": len(claims),
            "supported_fraction": sum(c.supported for c in claims)/len(claims) if claims else None,
            "threshold": threshold, "judge": getattr(judge, "identity", "injected"),
            "human_validated": False}


def attribution_metrics(predictions, labels):
    if len(predictions) != len(labels):
        raise ValueError("Predictions and human labels must align")
    tp = sum(bool(p) and bool(g) for p, g in zip(predictions, labels))
    fp = sum(bool(p) and not g for p, g in zip(predictions, labels))
    fn = sum(not p and bool(g) for p, g in zip(predictions, labels))
    return {"precision": tp/(tp+fp) if tp+fp else None,
            "recall": tp/(tp+fn) if tp+fn else None, "tp": tp, "fp": fp, "fn": fn,
            "n": len(labels)}


_JUDGE = None

def default_judge():
    global _JUDGE
    if _JUDGE is None:
        _JUDGE = NLIJudge(model=os.getenv("DOCMIND_NLI_MODEL", "typeform/distilbert-base-uncased-mnli"),
                          revision=os.getenv("DOCMIND_NLI_REVISION") or None)
    return _JUDGE
