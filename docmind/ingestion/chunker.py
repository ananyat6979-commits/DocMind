"""
Semantic Chunker: the most important differentiator from naive RAG.

The core idea: sentences within the same topic have high cosine similarity
in embedding space. When we see a sharp drop in similarity between adjacent
sentences, that's a semantic boundary, a good place to start a new chunk.

Algorithm:
  1. Find offset-preserving sentence boundaries.
  2. Embed every sentence.
  3. Compute cosine similarity between sentence[i] and sentence[i+1].
  4. Convert to distances (1 - similarity). High distance = topic shift.
  5. Threshold = Nth percentile of all distances. Positions above threshold
     become chunk boundaries.
  6. Group sentences into chunks. Merge tiny chunks, split giant ones.

The overlap_sentences parameter carries the last N sentences from one chunk
into the start of the next. This preserves context at boundaries: crucial
for questions that span a chunk boundary.
"""
import logging
from array import array
import re
import numpy as np
from typing import List
from docmind.models import Document, Chunk
from docmind.config import CONFIG
from docmind.embeddings import get_embedding_model

logger = logging.getLogger(__name__)

def _regex_sentences(text: str) -> List[str]:
    from docmind.text import sentence_spans
    return [span.text for span in sentence_spans(text)]


def _tokenize(text: str) -> List[str]:
    return _regex_sentences(text)


def _find_breakpoints(sentences: List[str], config=None, embedding_model=None) -> List[int]:
    if len(sentences) < 2:
        return []

    cfg = config or CONFIG.chunker
    model = embedding_model or get_embedding_model()
    embeddings = model.encode(
        sentences,
        batch_size=CONFIG.embedding.batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    similarities = np.array([
        float(np.dot(embeddings[i], embeddings[i + 1]))
        for i in range(len(embeddings) - 1)
    ])
    distances = 1.0 - similarities
    threshold = float(np.percentile(distances, cfg.breakpoint_percentile))
    return [i for i, d in enumerate(distances) if d > threshold + 1e-8]


def _build_chunks(sentences: List[str], breakpoints: List[int], doc: Document, config=None) -> List[Chunk]:
    """
    Group sentences into chunks at breakpoints, then:
    - merge chunks smaller than min_chunk_size
    - split chunks larger than max_chunk_size
    Apply sentence-level overlap at boundaries.
    """
    cfg = config or CONFIG.chunker
    bp_set = set(breakpoints)

    # Step 1: split into raw sentence groups at breakpoints
    groups: List[List[str]] = []
    current: List[str] = []

    for i, sent in enumerate(sentences):
        current.append(sent)
        if i in bp_set and i < len(sentences) - 1:
            groups.append(current)
            # Overlap: seed next group with tail of current
            current = []

    if current:
        groups.append(current)

    # Step 2: merge tiny groups, split giant ones, emit Chunks
    chunks: List[Chunk] = []
    pending_sents: List[str] = []
    previous_tail: List[str] = []

    for group in groups:
        # Fold any pending tiny fragment INTO this group. The old code emitted
        # the fragment as its own sub-minimum chunk and then the group separately.
        merged = pending_sents + group
        if len(" ".join(merged)) < cfg.min_chunk_size:
            pending_sents = merged
            continue
        pending_sents = []

        if previous_tail and len(" ".join(previous_tail + merged)) <= cfg.max_chunk_size:
            merged = previous_tail + merged
        previous_tail = merged[-cfg.overlap_sentences:] if cfg.overlap_sentences else []
        if len(" ".join(merged)) > cfg.max_chunk_size:
            _emit_large(chunks, merged, doc, cfg.max_chunk_size)
        else:
            _emit(chunks, " ".join(merged), doc)

    if pending_sents:
        tail = " ".join(pending_sents).strip()
        if chunks and len(chunks[-1].content) + 1 + len(tail) <= cfg.max_chunk_size:
            # Trailing fragment (often a page-number footer): attach to the previous chunk.
            prev = chunks[-1]
            prev.content = f"{prev.content} {tail}"
            prev.metadata["char_count"] = len(prev.content)
            spans,ambiguous=_source_spans(prev.content,doc)
            prev.metadata.update(source_spans=spans,source_span_ambiguous=ambiguous)
        elif not _is_noise(tail):
            _emit(chunks, tail, doc)
        # else: drop pure noise such as a lone "14"

    return chunks


def _is_noise(text: str) -> bool:
    """Short fragments with no real word in them: page numbers, rules, stray bullets."""
    return not text.strip()


def _source_spans(content,doc):
    cached=getattr(doc,"_source_span_index",None)
    if cached is None:
        positions=array("I",(i for i,c in enumerate(doc.content) if not c.isspace()))
        compact="".join(doc.content[i] for i in positions)
        cached=(compact,positions);doc._source_span_index=cached
    compact,positions=cached
    needle="".join(c for c in content if not c.isspace())
    start=compact.find(needle) if needle else -1
    if start<0:return [],False
    if compact.find(needle,start+1)>=0:return [],True
    origin=int(doc.metadata.get("source_offset",0))
    from docmind.ingestion.quality import source_offset
    edits = doc.metadata.get("normalization_edits", [])
    return [{"start":source_offset(origin+positions[start], edits),
             "end":source_offset(origin+positions[start+len(needle)-1]+1, edits, end=True)}],False


def _emit(chunks: List[Chunk], content: str, doc: Document) -> None:
    content = content.strip()
    if content:
        spans,ambiguous=_source_spans(content,doc)
        chunks.append(Chunk(
            content=content,
            doc_id=doc.doc_id,
            source=doc.source,
            chunk_index=len(chunks),
            metadata={**doc.metadata, "char_count": len(content),"source_spans":spans,"source_span_ambiguous":ambiguous},
        ))


def _split_oversized(sentence: str, max_size: int) -> List[str]:
    """A single 'sentence' longer than max_size (tables, code, OCR junk) would
    be silently truncated by the 512-token embedder. Split it on whitespace."""
    if len(sentence) <= max_size:
        return [sentence]
    parts, buf = [], ""
    words = [part for word in sentence.split() for part in (
        [word[i:i + max_size] for i in range(0, len(word), max_size)]
    )]
    for word in words:
        if buf and len(buf) + 1 + len(word) > max_size:
            parts.append(buf)
            buf = word
        else:
            buf = f"{buf} {word}".strip()
    if buf:
        parts.append(buf)
    return parts


def _emit_large(chunks: List[Chunk], sentences: List[str], doc: Document, max_size: int) -> None:
    """Accumulate sentences until we would exceed max_size, then emit."""
    current: List[str] = []
    current_len = 0
    expanded: List[str] = []
    for sent in sentences:
        expanded.extend(_split_oversized(sent, max_size))
    for sent in expanded:
        if current_len + len(sent) > max_size and current:
            _emit(chunks, " ".join(current), doc)
            current = [sent]
            current_len = len(sent)
        else:
            current.append(sent)
            current_len += len(sent) + 1
    if current:
        _emit(chunks, " ".join(current), doc)


def chunk_document(doc: Document, config=None, embedding_model=None) -> List[Chunk]:
    """Public API: take a Document, return a list of semantic Chunks."""
    cfg = config or CONFIG.chunker
    if cfg.strategy == "recursive":
        return _recursive_chunk(doc,cfg)
    if cfg.strategy == "fixed":
        return _fixed_chunk(doc,cfg)

    sentences = _tokenize(doc.content)
    if not sentences:
        return []

    breakpoints = _find_breakpoints(sentences,cfg,embedding_model)
    chunks = _build_chunks(sentences, breakpoints, doc,cfg)

    logger.debug(
        f"Chunked '{doc.metadata.get('filename', '?')}' "
        f"p{doc.metadata.get('page', '-')}: {len(sentences)} sentences → {len(chunks)} chunks"
    )
    return chunks


def _fixed_chunk(doc: Document, config=None) -> List[Chunk]:
    """Simple fixed-size chunking. Used as baseline for evaluation comparisons."""
    cfg = config or CONFIG.chunker
    content = doc.content
    chunks: List[Chunk] = []
    overlap_chars = 100
    start = 0

    while start < len(content):
        end = min(start + cfg.max_chunk_size, len(content))
        # Prefer to break at a sentence boundary
        if end < len(content):
            break_pos = content.rfind(". ", start + cfg.min_chunk_size, end)
            if break_pos > start:
                end = break_pos + 2

        chunk_text = content[start:end].strip()
        if chunk_text:
            _emit(chunks, chunk_text, doc)

        start = max(start + 1, end - min(overlap_chars, cfg.max_chunk_size // 4)) if end < len(content) else end

    return chunks


def _recursive_chunk(doc, config=None):
    """Recursive paragraph, line, sentence, word and character splitting."""
    cfg=config or CONFIG.chunker
    if cfg.max_chunk_size<1:raise ValueError("max_chunk_size must be positive")
    separators=("\n\n","\n",". "," ","")
    def split(text,level=0):
        if len(text)<=cfg.max_chunk_size:return [text]
        separator=separators[level]
        if not separator:
            return [text[i:i+cfg.max_chunk_size] for i in range(0,len(text),cfg.max_chunk_size)]
        if separator not in text:return split(text,level+1)
        pieces=text.split(separator)
        result: list[str] = []
        for i,piece in enumerate(pieces):
            piece+=separator if i<len(pieces)-1 else ""
            result.extend(split(piece,level+1) if len(piece)>cfg.max_chunk_size else [piece])
        return result
    chunks: List[Chunk] = []
    current="";overlap=min(100,cfg.max_chunk_size//4)
    for part in split(doc.content):
        if current and len(current)+len(part)>cfg.max_chunk_size:
            _emit(chunks,current,doc)
            tail=current[-overlap:] if overlap else ""
            current=tail if len(tail)+len(part)<=cfg.max_chunk_size else ""
        current+=part
    _emit(chunks,current,doc)
    return chunks
