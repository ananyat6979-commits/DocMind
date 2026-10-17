"""
Semantic Chunker: the most important differentiator from naive RAG.

The core idea: sentences within the same topic have high cosine similarity
in embedding space. When we see a sharp drop in similarity between adjacent
sentences, that's a semantic boundary, a good place to start a new chunk.

Algorithm:
  1. Tokenize text into sentences (NLTK).
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
import re
import numpy as np
from typing import List
from docmind.models import Document, Chunk
from docmind.config import CONFIG
from docmind.embeddings import get_embedding_model

logger = logging.getLogger(__name__)

_REGEX_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])|\n{2,}")


def _regex_sentences(text: str) -> List[str]:
    return [s.strip() for s in _REGEX_SPLIT.split(text) if s and s.strip()]


def _tokenize(text: str) -> List[str]:
    """NLTK sentence tokenizer. Downloads punkt once; if that is impossible
    (offline machine), falls back to a regex splitter instead of crashing."""
    try:
        import nltk
        from nltk.tokenize import sent_tokenize
        try:
            sentences = sent_tokenize(text)
        except LookupError:
            nltk.download("punkt", quiet=True)
            nltk.download("punkt_tab", quiet=True)
            sentences = sent_tokenize(text)
    except Exception as exc:
        logger.warning(f"NLTK unavailable ({exc}); using regex sentence splitter")
        sentences = _regex_sentences(text)
    return [s.strip() for s in sentences if s.strip()]


def _find_breakpoints(sentences: List[str]) -> List[int]:
    if len(sentences) < 2:
        return []

    model = get_embedding_model()
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
    threshold = float(np.percentile(distances, CONFIG.chunker.breakpoint_percentile))
    return [i for i, d in enumerate(distances) if d >= threshold]


def _build_chunks(sentences: List[str], breakpoints: List[int], doc: Document) -> List[Chunk]:
    """
    Group sentences into chunks at breakpoints, then:
    - merge chunks smaller than min_chunk_size
    - split chunks larger than max_chunk_size
    Apply sentence-level overlap at boundaries.
    """
    cfg = CONFIG.chunker
    bp_set = set(breakpoints)

    # Step 1: split into raw sentence groups at breakpoints
    groups: List[List[str]] = []
    current: List[str] = []

    for i, sent in enumerate(sentences):
        current.append(sent)
        if i in bp_set and i < len(sentences) - 1:
            groups.append(current)
            # Overlap: seed next group with tail of current
            current = current[-cfg.overlap_sentences:] if cfg.overlap_sentences else []

    if current:
        groups.append(current)

    # Step 2: merge tiny groups, split giant ones, emit Chunks
    chunks: List[Chunk] = []
    pending_sents: List[str] = []

    for group in groups:
        # Fold any pending tiny fragment INTO this group. The old code emitted
        # the fragment as its own sub-minimum chunk and then the group separately.
        merged = pending_sents + group
        if len(" ".join(merged)) < cfg.min_chunk_size:
            pending_sents = merged
            continue
        pending_sents = []

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
        elif not _is_noise(tail):
            _emit(chunks, tail, doc)
        # else: drop pure noise such as a lone "14"

    return chunks


def _is_noise(text: str) -> bool:
    """Short fragments with no real word in them: page numbers, rules, stray bullets."""
    return len(text) < 20 and not re.search(r"[A-Za-z]{3,}", text)


def _emit(chunks: List[Chunk], content: str, doc: Document) -> None:
    content = content.strip()
    if content:
        chunks.append(Chunk(
            content=content,
            doc_id=doc.doc_id,
            source=doc.source,
            chunk_index=len(chunks),
            metadata={**doc.metadata, "char_count": len(content)},
        ))


def _split_oversized(sentence: str, max_size: int) -> List[str]:
    """A single 'sentence' longer than max_size (tables, code, OCR junk) would
    be silently truncated by the 512-token embedder. Split it on whitespace."""
    if len(sentence) <= max_size:
        return [sentence]
    parts, buf = [], ""
    for word in sentence.split():
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


def chunk_document(doc: Document) -> List[Chunk]:
    """Public API: take a Document, return a list of semantic Chunks."""
    if CONFIG.chunker.strategy == "fixed":
        return _fixed_chunk(doc)

    sentences = _tokenize(doc.content)
    if not sentences:
        return []

    breakpoints = _find_breakpoints(sentences)
    chunks = _build_chunks(sentences, breakpoints, doc)

    logger.debug(
        f"Chunked '{doc.metadata.get('filename', '?')}' "
        f"p{doc.metadata.get('page', '-')}: {len(sentences)} sentences → {len(chunks)} chunks"
    )
    return chunks


def _fixed_chunk(doc: Document) -> List[Chunk]:
    """Simple fixed-size chunking. Used as baseline for evaluation comparisons."""
    cfg = CONFIG.chunker
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

        start = end - overlap_chars if end < len(content) else end

    return chunks
