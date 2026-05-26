"""
DocMind Streamlit UI.
Run with: streamlit run app.py
"""
import streamlit as st
import tempfile
import shutil
from pathlib import Path
import time

st.set_page_config(
    page_title="DocMind",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.main-header { font-size: 2.2rem; font-weight: 700; color: #1a1a2e; margin-bottom: 0; }
.sub-header  { font-size: 1rem; color: #666; margin-top: 0; margin-bottom: 2rem; }
.source-card {
    background: var(--secondary-background-color, #f8f9ff);
    border-left: 4px solid #4361ee;
    padding: 0.8rem 1rem;
    margin: 0.4rem 0;
    border-radius: 0 8px 8px 0;
    font-size: 0.88rem;
    color: var(--text-color, #262730);
}
.answer-box {
    background: var(--background-color, #ffffff);
    border: 1px solid var(--secondary-background-color, #e0e0e0);
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    margin: 1rem 0;
    line-height: 1.7;
    color: var(--text-color, #262730);
}
.metric-chip {
    display: inline-block;
    background: #e8f4fd;
    color: #1565c0;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.78rem;
    margin-right: 6px;
}
.stAlert { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ── Session state ──────────────────────────────────────────────────────────
if "agent" not in st.session_state:
    st.session_state.agent = None
if "ingested_file" not in st.session_state:
    st.session_state.ingested_file = None
if "history" not in st.session_state:
    st.session_state.history = []


# ── Header ─────────────────────────────────────────────────────────────────
st.markdown('<p class="main-header">🧠 DocMind</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Intelligent document intelligence '
    'hybrid retrieval · ReAct agent · cross-encoder reranking</p>',
    unsafe_allow_html=True
)

col_main, col_sidebar = st.columns([2, 1])

# ── Sidebar: document ingestion ────────────────────────────────────────────
with st.sidebar:
    st.header("📄 Document")

    uploaded = st.file_uploader(
        "Upload a document",
        type=["pdf", "docx", "txt", "md"],
        help="Supported formats: PDF, DOCX, TXT, Markdown"
    )

    rebuild = st.checkbox("Rebuild index", value=False,
                          help="Force re-ingestion even if index exists")

    if uploaded:
        if (st.session_state.ingested_file != uploaded.name) or rebuild:
            with st.spinner(f"Ingesting {uploaded.name}..."):
                # Save upload to data/ dir
                data_dir = Path("data")
                data_dir.mkdir(exist_ok=True)
                save_path = data_dir / uploaded.name

                with open(save_path, "wb") as f:
                    f.write(uploaded.getbuffer())

                try:
                    from docmind.ingestion.pipeline import ingest
                    from docmind.agent.react import ReActAgent
                    from docmind.retrieval.hybrid import HybridRetriever

                    chunks = ingest(save_path, rebuild_index=rebuild)
                    retriever = HybridRetriever()
                    st.session_state.agent = ReActAgent(retriever=retriever)
                    st.session_state.ingested_file = uploaded.name
                    st.session_state.history = []

                    st.success(f"✓ Ingested **{len(chunks) or '(cached)'}** chunks")
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")
        else:
            st.success(f"✓ **{uploaded.name}** loaded")

    st.divider()

    # Pipeline info
    st.caption("**Pipeline**")
    st.caption("• Semantic chunker (bge-small-en-v1.5)")
    st.caption("• FAISS dense retrieval")
    st.caption("• BM25 sparse retrieval")
    st.caption("• RRF fusion (k=60)")
    st.caption("• Cross-encoder reranking")
    st.caption("• ReAct agent (Groq llama-3.1-8b)")

    if st.button("🗑️ Clear history"):
        st.session_state.history = []
        st.rerun()


# ── Main: Q&A interface ────────────────────────────────────────────────────
with col_main:
    if st.session_state.agent is None:
        st.info("👈 Upload a document to get started.")
    else:
        # Question input
        with st.form("question_form", clear_on_submit=True):
            question = st.text_input(
                "Ask a question",
                placeholder="What are the main findings? How many experiments? ...",
                label_visibility="collapsed",
            )
            submitted = st.form_submit_button("Ask →", use_container_width=False)

        if submitted and question.strip():
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.agent.run(question)
                    st.session_state.history.append({
                        "question": question,
                        "response": response,
                    })
                except Exception as e:
                    st.error(f"Agent error: {e}")

        # Display history (newest first)
        for item in reversed(st.session_state.history):
            q = item["question"]
            r = item["response"]

            st.markdown(f"**Q: {q}**")

            # Answer
            st.markdown(
                f'<div class="answer-box">{r.answer}</div>',
                unsafe_allow_html=True
            )

            # Metrics
            latency = f"{r.latency_ms/1000:.1f}s"
            steps = f"{len(r.reasoning_trace)} reasoning steps"
            sources_count = f"{len(r.sources)} sources"
            st.markdown(
                f'<span class="metric-chip">⏱ {latency}</span>'
                f'<span class="metric-chip">🔍 {sources_count}</span>'
                f'<span class="metric-chip">💭 {steps}</span>',
                unsafe_allow_html=True
            )

            # Sources
            if r.sources:
                with st.expander(f"📚 Sources ({len(r.sources)})", expanded=True):
                    seen = set()
                    for s in r.sources:
                        meta = s.chunk.metadata
                        fname = meta.get("filename", s.chunk.source)
                        loc = f"p.{meta['page']}" if "page" in meta else meta.get("section", "")
                        key = f"{fname}_{loc}"
                        if key in seen:
                            continue
                        seen.add(key)
                        preview = s.chunk.content[:250].replace("\n", " ")
                        st.markdown(
                            f'<div class="source-card">'
                            f'<strong>{fname}</strong> {loc} '
                            f'<span style="color:#999">score: {s.score:.3f}</span><br>'
                            f'<span style="color:#444">{preview}...</span>'
                            f'</div>',
                            unsafe_allow_html=True
                        )

            # Reasoning trace
            with st.expander("🧠 Reasoning trace", expanded=False):
                for i, step in enumerate(r.reasoning_trace, 1):
                    st.markdown(f"**Step {i}**")
                    st.code(step, language=None)

            st.divider()


# ── Right column: stats ────────────────────────────────────────────────────
with col_sidebar:
    if st.session_state.history:
        st.subheader("Session stats")
        latencies = [h["response"].latency_ms for h in st.session_state.history]
        avg_lat = sum(latencies) / len(latencies)
        st.metric("Questions asked", len(st.session_state.history))
        st.metric("Avg latency", f"{avg_lat/1000:.1f}s")
        total_sources = sum(len(h["response"].sources) for h in st.session_state.history)
        st.metric("Total sources retrieved", total_sources)