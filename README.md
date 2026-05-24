# DocMind 🧠

**Intelligent document intelligence**: hybrid retrieval, cross-encoder reranking, and a from-scratch ReAct agent for grounded, cited answers over your documents.

![Python](https://img.shields.io/badge/python-3.10+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## What it does

Upload a PDF, DOCX, or text file. Ask questions in natural language. Get answers cited to exact pages, with no hallucinated sources.

DocMind doesn't just stuff your document into a context window. It implements a production-grade two-stage retrieval pipeline, then routes each query through a ReAct reasoning loop that calls real search tools before synthesizing an answer.

---

## Architecture

Document (PDF/DOCX/TXT)
│
▼
┌─────────────────┐
│  Semantic        │  NLTK sentence tokenization
│  Chunker         │  bge-small-en-v1.5 embeddings
│                  │  Cosine similarity breakpoints
└────────┬────────┘
│ Chunks
▼
┌─────────────────┐    ┌─────────────────┐
│  FAISS Dense     │    │  BM25 Sparse     │
│  Retriever       │    │  Retriever       │
│  (bi-encoder)    │    │  (keyword)       │
└────────┬────────┘    └────────┬────────┘
│                      │
└──────────┬───────────┘
	      ▼
┌─────────────────┐
│  RRF Fusion      │  Reciprocal Rank Fusion
│  (k=60)          │  Cormack et al. 2009
└────────┬────────┘
│
▼
┌─────────────────┐
│  Cross-Encoder   │  ms-marco-MiniLM-L-6-v2
│  Reranker        │  Query × document scoring
└────────┬────────┘
│ Top-5 chunks
▼
┌─────────────────┐
│  ReAct Agent     │  Thought → Action → Observation
│  (from scratch)  │  No LangChain dependency
└────────┬────────┘
│
▼
Cited answer with source pages

## Why each component matters

| Component | Naive alternative | Why this is better |
|-----------|-------------------|-------------------|
| Semantic chunker | Fixed-size chunks | Boundaries follow topic shifts, not character counts |
| FAISS + BM25 hybrid | Dense-only | BM25 catches exact keyword matches dense retrieval misses |
| RRF fusion | Score normalization | Rank-based, no need to normalize incompatible score scales |
| Cross-encoder reranker | Bi-encoder only | Sees query+document together; 40-60% precision improvement |
| ReAct loop | Single-shot QA | Multi-step reasoning; searches until it has enough evidence |

---

## Setup

```bash
# 1. Clone and create virtualenv
git clone https://github.com/ananyat6979-commits/DocMind.git
cd DocMind
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # Mac/Linux

# 2. Install
pip install -e .
pip install -r requirements.txt

# 3. NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

# 4. Set your Groq API key (free at https://console.groq.com)
cp .env.example .env
# Edit .env: GROQ_API_KEY=gsk_...
```

---

## Usage

### Streamlit UI (recommended)
```bash
streamlit run app.py
# Open http://localhost:8501
# Upload any PDF/DOCX, ask questions
```

### CLI demo
```bash
python scripts/demo.py --doc data/your_document.pdf
# Add --rebuild to force re-ingestion
```

### FastAPI
```bash
uvicorn docmind.api.main:app --reload
# POST /ingest  {"path": "data/your_doc.pdf"}
# POST /query   {"question": "What are the main findings?"}
# GET  /health
```

### Evaluation (run on Kaggle for large testsets)
```bash
python scripts/evaluate.py --testset data/testset.json
# testset format: [{"question": "...", "ground_truth": "..."}, ...]
```

---

## Tech stack

| Layer | Technology | Why |
|-------|-----------|-----|
| Embeddings | BAAI/bge-small-en-v1.5 | Outperforms all-MiniLM on MTEB at same parameter count |
| Vector index | FAISS IndexFlatIP | Exact cosine search; no approximation error at corpus size |
| Sparse index | BM25Okapi (rank-bm25) | Captures exact keyword matches; complementary to dense |
| Reranker | ms-marco-MiniLM-L-6-v2 | Cross-encoder; 90MB, fast on CPU |
| LLM | Groq llama-3.1-8b-instant | 100k tok/min free tier; ~1s response time |
| Metadata store | SQLite | Zero-dependency persistence |
| Experiment tracking | MLflow | Query latency, source count logged per run |
| API | FastAPI + Pydantic v2 | Typed endpoints; auto-generated docs at /docs |
| UI | Streamlit | Rapid iteration; file upload + conversation history |

---

## Project structure

DocMind/
├── docmind/
│   ├── embeddings.py       # Shared model singleton
│   ├── models.py           # Core data types (Document, Chunk, SearchResult)
│   ├── config.py           # Central configuration
│   ├── ingestion/
│   │   ├── loader.py       # PDF, DOCX, TXT/MD loaders
│   │   ├── chunker.py      # Semantic chunker
│   │   └── pipeline.py     # End-to-end ingestion orchestration
│   ├── retrieval/
│   │   ├── dense.py        # FAISS retriever
│   │   ├── sparse.py       # BM25 retriever
│   │   ├── reranker.py     # Cross-encoder reranker
│   │   └── hybrid.py       # RRF fusion + pipeline orchestration
│   ├── llm/
│   │   ├── base.py         # Abstract LLM interface
│   │   └── groq_client.py  # Groq implementation
│   ├── agent/
│   │   ├── prompts.py      # ReAct system prompt + formatters
│   │   └── react.py        # ReAct loop (no LangChain)
│   ├── api/
│   │   └── main.py         # FastAPI endpoints
│   ├── evaluation/
│   │   └── harness.py      # RAGAS evaluation
│   └── tracking/
│       └── tracker.py      # MLflow logging
├── scripts/
│   ├── demo.py             # CLI demo
│   └── evaluate.py         # RAGAS eval runner
├── tests/
│   ├── test_ingestion.py
│   └── test_retrieval.py
├── app.py                  # Streamlit UI
└── requirements.txt
---

## Evaluation

Run RAGAS evaluation on Kaggle (GPU, no RAM constraints):

```python
# data/testset.json format
[
  {"question": "What is the main thesis?", "ground_truth": "..."},
  {"question": "How many experiments are described?", "ground_truth": "5"}
]
```

```bash
python scripts/evaluate.py --testset data/testset.json
```

Target metrics (to be updated after full eval run):

| Metric | Target | Description |
|--------|--------|-------------|
| Faithfulness | > 0.80 | Claims in answer supported by retrieved context |
| Answer relevancy | > 0.75 | Answer addresses the question asked |
| Context precision | > 0.70 | Retrieved chunks are relevant to the question |

---

## Design decisions

**Why not LangChain?** The ReAct agent is built from scratch. This was intentional: it demonstrates understanding of what the framework does, not just which import to use. 

**Why Groq instead of OpenAI?** Free tier at 100k tokens/minute with no credit card. The `BaseLLM` interface makes swapping providers a one-line config change.

**Why FAISS over a vector database?** For a corpus under 100k chunks, exact search with `IndexFlatIP` is fast enough on CPU and adds zero infrastructure. The retrieval layer is designed so `DenseRetriever` can be swapped for Qdrant or Pinecone by implementing the same interface.

---

## License

MIT