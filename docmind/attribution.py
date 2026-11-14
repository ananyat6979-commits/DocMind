"""Claim-to-cited-span attribution using a local NLI model.

Scores are model judgments until agreement with independent human labels is measured.
Every evidence offset indexes the original retrieved chunk, never normalized text.
"""
import re
import os
import threading
from dataclasses import dataclass, asdict
from docmind.agent.citations import CITATION_RE, _items, allowed_citations, source_citations


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
    def __init__(self, model="typeform/distilbert-base-uncased-mnli", revision=None, local_files_only=True, device="cpu", batch_size=2):
        if revision is None and model=="typeform/distilbert-base-uncased-mnli":
            revision="cfa538a0fddbbd978fefe8966c1aeff7ad409c90"
        if not revision: raise ValueError("Pin DOCMIND_NLI_REVISION before selecting a different attribution model")
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.torch = torch
        self.device, self.batch_size = device, batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model, revision=revision, local_files_only=local_files_only)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model, revision=revision, local_files_only=local_files_only).to(device).eval()
        labels = {str(v).lower(): int(k) for k, v in self.model.config.id2label.items()}
        if "entailment" not in labels:
            raise ValueError("NLI model must expose an entailment label")
        self.entailment_id = labels["entailment"]
        self.identity = {"model": model, "revision": revision or getattr(self.model.config, "_commit_hash", None)}

    def __call__(self, pairs):
        scores = []
        for offset in range(0, len(pairs), self.batch_size):
            batch = pairs[offset:offset+self.batch_size]
            if any(len(self.tokenizer.encode(h,add_special_tokens=False))>128 for _,h in batch):
                raise ValueError("Claim exceeds the NLI hypothesis budget")
            encoded = self.tokenizer([p for p, _ in batch], [h for _, h in batch],
                padding=True, truncation="only_first", max_length=256, return_tensors="pt").to(self.device)
            with self.torch.inference_mode():
                scores.extend(self.model(**encoded).logits.softmax(-1)[:, self.entailment_id].tolist())
        return scores


def evidence_spans(text):
    from docmind.text import sentence_spans
    matches = sentence_spans(text)
    for index, span in enumerate(matches):
        yield span.start, span.end, span.text
        if index + 1 < len(matches):
            end = matches[index+1].end
            yield span.start, end, text[span.start:end]


def attribute(answer, sources, judge, threshold=0.8, claim_units=None):
    allowed = allowed_citations(sources)
    claims = []
    from docmind.text import sentence_spans
    parts = [span.text for span in sentence_spans(answer)]
    if claim_units is not None:
        parts=list(claim_units)
    for part in parts:
        citations = [item for m in CITATION_RE.finditer(part) for item in _items(m.group(1))]
        claim = CITATION_RE.sub("", part).strip()
        if not claim:
            continue
        candidates = []
        for source in sources:
            labels=[]
            for citation in source_citations(source.chunk):
                match=CITATION_RE.search(citation)
                if match:labels.extend(_items(match.group(1)))
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
_JUDGE_KEY: tuple[str, str | None, str, int] | None = None
_JUDGE_LOCK = threading.Lock()


def default_judge():
    global _JUDGE, _JUDGE_KEY
    key=(os.getenv("DOCMIND_NLI_MODEL","typeform/distilbert-base-uncased-mnli"),
         os.getenv("DOCMIND_NLI_REVISION") or None,
         os.getenv("DOCMIND_NLI_DEVICE","cpu"),int(os.getenv("DOCMIND_NLI_BATCH_SIZE","2")))
    if _JUDGE is None or _JUDGE_KEY!=key:
        with _JUDGE_LOCK:
            if _JUDGE is None or _JUDGE_KEY!=key:
                _JUDGE=NLIJudge(model=key[0],revision=key[1],device=key[2],batch_size=key[3])
                _JUDGE_KEY=key
    return _JUDGE
