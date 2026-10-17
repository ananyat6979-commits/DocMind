"""
DocMind Streamlit UI.   Run: streamlit run app.py

Fixes vs the original:
  * Multiple documents live in one library; uploading a second file no longer
    reuses the first file's index. You choose which documents to search.
  * "Rebuild index" is a button, not a sticky checkbox (the checkbox re-ingested
    and wiped history on every rerun).
  * Model output is rendered as markdown, never as raw HTML.
  * Flagged answers enter a human review queue (Review tab).
  * Every query, feedback and time estimate is logged for the pilot.
"""
import hmac
import hashlib
import os
import uuid
from pathlib import Path

import streamlit as st

from docmind.config import CONFIG
from docmind.ingestion.loader import LOADERS
from docmind.ingestion.pipeline import ingest
from docmind.review.store import ReviewStore
from docmind.store import CorpusStore
from docmind.telemetry.events import EventLog

st.set_page_config(page_title="DocMind", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")

# Password gate. Set DOCMIND_APP_PASSWORD before exposing the app through any tunnel or host:
# otherwise anyone with the URL can spend your Groq quota and read every uploaded document.
_APP_PASSWORD = os.getenv("DOCMIND_APP_PASSWORD", "")
if _APP_PASSWORD and not st.session_state.get("authed"):
    st.title("DocMind")
    entered = st.text_input("Password", type="password", key="app_pw")
    if entered:
        if hmac.compare_digest(entered.encode(), _APP_PASSWORD.encode()):
            st.session_state.authed = True
            st.rerun()
        else:
            st.error("Wrong password.")
    st.stop()

STATUS_BADGE = {
    "auto_approved": "Auto-approved",
    "pending": "Awaiting human review",
    "approved": "Approved by reviewer",
    "edited": "Corrected by reviewer",
    "rejected": "Rejected by reviewer",
}


@st.cache_resource
def get_services():
    from docmind.agent.react import ReActAgent
    from docmind.retrieval.hybrid import HybridRetriever
    store = CorpusStore()
    return {"store": store, "agent": ReActAgent(retriever=HybridRetriever(store)),
            "reviews": ReviewStore(), "events": EventLog()}


svc = get_services()
store, agent, reviews, events = svc["store"], svc["agent"], svc["reviews"], svc["events"]

if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex[:12]
    st.session_state.history = []
    st.session_state.processed = set()
    st.session_state.logged_start = False


def user() -> str:
    return (st.session_state.get("user_name") or "anonymous").strip() or "anonymous"


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("DocMind")
    st.text_input("Your name (for the pilot log)", key="user_name")
    if not st.session_state.logged_start and user() != "anonymous":
        events.log("session_start", user(), st.session_state.session_id)
        st.session_state.logged_start = True

    if getattr(agent.llm, "is_demo", False):
        st.warning("Demo mode: no GROQ_API_KEY, answers are extractive snippets, not model output.")

    uploads = st.file_uploader("Add documents", type=[e.lstrip(".") for e in LOADERS], accept_multiple_files=True)
    for up in uploads or []:
        key = (up.name, up.size)
        if key in st.session_state.processed:
            continue
        if up.size > CONFIG.max_upload_mb * 1024 * 1024:
            st.error(f"{up.name}: larger than {CONFIG.max_upload_mb} MB")
            st.session_state.processed.add(key)
            continue
        content = bytes(up.getbuffer())
        digest = hashlib.sha256(content).hexdigest()
        dest = CONFIG.data_dir / "uploads" / digest / Path(up.name).name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
        with st.spinner(f"Ingesting {up.name}..."):
            try:
                rep = ingest(dest, store=store)
                f = rep.files[0]
                st.success(f"{up.name}: {f.status} ({f.n_chunks} chunks)")
                events.log("upload", user(), st.session_state.session_id, filename=up.name, status=f.status, n_chunks=f.n_chunks)
            except Exception as exc:
                st.error(f"{up.name}: {exc}")
        st.session_state.processed.add(key)

    st.subheader("Library")
    docs = store.list_documents()
    selected = []
    if not docs:
        st.caption("No documents yet.")
    for d in docs:
        c1, c2 = st.columns([5, 1])
        if c1.checkbox(f"{d['filename']} ({d['n_chunks']} chunks)", value=True, key=f"sel_{d['file_id']}"):
            selected.append(d["file_id"])
        if c2.button("x", key=f"del_{d['file_id']}", help="Remove from library"):
            store.delete_document(d["file_id"])
            st.rerun()
    if docs and st.button("Re-index selected documents"):
        for d in docs:
            if d["file_id"] in selected and Path(d["source"]).exists():
                ingest(d["source"], rebuild_index=True, store=store)
        st.rerun()

    st.divider()
    st.caption("Pipeline: semantic chunks, FAISS + BM25, RRF, cross-encoder rerank, ReAct agent with citation verification.")
    if st.button("Clear history"):
        st.session_state.history = []
        st.rerun()

tab_ask, tab_review = st.tabs(["Ask", "Review queue"])


# ------------------------------------------------------------ feedback hooks
def _feedback(response_id: str, helpful: bool):
    events.log("feedback", user(), st.session_state.session_id, response_id=response_id, helpful=helpful)


def _estimate(response_id: str):
    minutes = st.session_state.get(f"est_{response_id}")
    if minutes:
        events.log("manual_estimate", user(), st.session_state.session_id, response_id=response_id, manual_minutes=minutes)


# -------------------------------------------------------------------- Ask
with tab_ask:
    if not docs:
        st.info("Add a document in the sidebar to get started.")
    elif not selected:
        st.warning("Select at least one document in the library.")
    else:
        with st.form("q", clear_on_submit=True):
            question = st.text_input("Ask a question", placeholder="What are the termination conditions?",
                                     label_visibility="collapsed")
            go = st.form_submit_button("Ask")
        if go and question.strip():
            with st.spinner("Searching and reasoning..."):
                try:
                    resp = agent.run(question.strip(), file_ids=selected)
                    status = reviews.record(resp, user=user())
                    events.log("query", user(), st.session_state.session_id, response_id=resp.response_id,
                               latency_ms=round(resp.latency_ms), llm_calls=resp.llm_calls, abstained=resp.abstained,
                               needs_review=resp.needs_review, n_sources=len(resp.sources), finish=resp.finish_reason,
                               citation_validity=resp.citation_report.get("validity"), n_docs=len(selected))
                    st.session_state.history.append(resp)
                except Exception as exc:
                    st.error(f"Agent error: {exc}")

        for resp in reversed(st.session_state.history):
            status = reviews.status_of(resp.response_id) or "auto_approved"
            st.markdown(f"**Q: {resp.query}**")
            hold = CONFIG.review.hold_until_approved and status == "pending"
            if hold:
                st.info("This answer is waiting for human review and is hidden until a reviewer decides.")
            else:
                item = reviews.get(resp.response_id) or {}
                text = item.get("edited_answer") or resp.answer
                if status == "pending":
                    st.warning(f"{STATUS_BADGE[status]}: treat as unverified. Reasons: {', '.join(resp.review_reasons)}")
                elif status == "rejected":
                    st.error(STATUS_BADGE[status])
                st.markdown(text)               # markdown only, never unsafe HTML
            v = resp.citation_report.get("validity")
            st.caption(f"{resp.latency_ms / 1000:.1f}s | {resp.llm_calls} LLM calls | {len(resp.sources)} passages | "
                       f"{STATUS_BADGE.get(status, status)}"
                       + (f" | citation validity {v:.0%}" if v is not None else ""))

            if resp.sources:
                with st.expander(f"Evidence ({len(resp.cited_sources)} cited of {len(resp.sources)} retrieved)"):
                    cited_ids = {s.chunk.chunk_id for s in resp.cited_sources}
                    for s in resp.sources:
                        mark = "CITED" if s.chunk.chunk_id in cited_ids else "retrieved"
                        st.markdown(f"**{s.chunk.filename()}** {s.chunk.citation()[1:-1]} | {mark} | score {s.score:.2f}")
                        st.text(s.chunk.content[:600])
            with st.expander("Reasoning trace"):
                for i, step in enumerate(resp.reasoning_trace, 1):
                    st.markdown(f"Step {i}")
                    st.code(step, language=None)

            c1, c2, c3 = st.columns([1, 1, 3])
            c1.button("Helpful", key=f"up_{resp.response_id}", on_click=_feedback, args=(resp.response_id, True))
            c2.button("Not helpful", key=f"dn_{resp.response_id}", on_click=_feedback, args=(resp.response_id, False))
            c3.number_input("Minutes this would have taken by hand (optional)", min_value=0.0, step=1.0,
                            key=f"est_{resp.response_id}", on_change=_estimate, args=(resp.response_id,))
            st.divider()


# ----------------------------------------------------------------- Review
with tab_review:
    pin_required = CONFIG.review.reviewer_pin
    ok = True
    if pin_required:
        ok = st.text_input("Reviewer PIN", type="password", key="pin") == pin_required
    if not ok:
        st.info("Enter the reviewer PIN to open the queue.")
    else:
        st.caption(f"Queue stats: {reviews.stats()}")
        reviewer = st.text_input("Reviewer name", key="reviewer_name") or "reviewer"
        pending = reviews.pending(limit=20)
        if not pending:
            st.success("Nothing waiting for review.")
        for item in pending:
            with st.container(border=True):
                st.markdown(f"**Q:** {item['question']}")
                st.caption(f"Asked by {item['user']} | flagged: {', '.join(item['reasons'])}")
                st.markdown(item["answer"])
                with st.expander("Evidence the model saw"):
                    for s in item["sources"]:
                        st.markdown(f"**{s['filename']}** p.{s.get('page') or '-'} | {'CITED' if s['cited'] else 'retrieved'}")
                        st.text(s["content"][:600])
                rid = item["response_id"]
                edited = st.text_area("Corrected answer (only for 'Correct and approve')", key=f"ed_{rid}")
                note = st.text_input("Note", key=f"nt_{rid}")
                b1, b2, b3 = st.columns(3)
                decision = None
                if b1.button("Approve", key=f"ap_{rid}"):
                    decision = ("approved", None)
                if b2.button("Correct and approve", key=f"ce_{rid}") and edited.strip():
                    decision = ("edited", edited)
                if b3.button("Reject", key=f"rj_{rid}"):
                    decision = ("rejected", None)
                if decision:
                    reviews.decide(rid, decision[0], reviewer, decision[1], note)
                    events.log("review_decision", reviewer, st.session_state.session_id, response_id=rid, decision=decision[0])
                    st.rerun()
        if st.button("Export reviewed answers as testset"):
            n = reviews.export_testset(str(CONFIG.data_dir / "reviewed_testset.json"))
            st.success(f"Wrote {n} items to data/reviewed_testset.json")
