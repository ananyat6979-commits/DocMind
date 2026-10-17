"""Items 5, 6, 8: the ReAct loop. The central regression is that a successful
answer must survive to the AgentResponse (the old code overwrote it)."""
from docmind.agent.react import ReActAgent, parse_action, NO_ANSWER_MESSAGE
from docmind.ingestion.pipeline import ingest
from docmind.retrieval.hybrid import HybridRetriever
from docmind.store import CorpusStore
from tests.conftest import ALPHA, BETA


def _agent(make_file, scripted, replies, **kw):
    store = CorpusStore()
    ingest(make_file("alpha.txt", ALPHA), store=store)
    ingest(make_file("beta.txt", BETA), store=store)
    llm = scripted(replies)
    return ReActAgent(retriever=HybridRetriever(store), llm=llm, **kw), llm


SEARCH = 'Thought: search.\nAction: search_documents("coolant temperature startup")'
CITE = "(Source: alpha.txt)"


def test_successful_answer_is_not_overwritten(make_file, scripted):
    agent, _ = _agent(make_file, scripted, [SEARCH, f"Thought: done.\nFinal Answer: Coolant must stay below 300 degrees {CITE}."])
    r = agent.run("What is the coolant limit?")
    assert "300 degrees" in r.answer
    assert "unable to find" not in r.answer.lower()
    assert NO_ANSWER_MESSAGE not in r.answer
    assert not r.abstained and r.finish_reason == "final_answer"
    assert r.citation_report["validity"] == 1.0
    assert r.cited_sources and r.cited_sources[0].chunk.filename() == "alpha.txt"


def test_synthesis_answer_after_exhausted_iterations_is_kept(make_file, scripted):
    # Every step searches, never answers -> synthesis call. Old code overwrote this answer too.
    replies = [SEARCH] * 2 + [f"Final Answer: Below 300 degrees {CITE}."]
    agent, llm = _agent(make_file, scripted, replies, max_iterations=2)
    r = agent.run("coolant?")
    assert "300 degrees" in r.answer
    assert r.finish_reason == "synthesis"
    assert llm.calls[-1]["stop"] is None                  # synthesis call has no stop sequence
    assert "no_clean_final_answer" in r.review_reasons


def test_fallback_text_is_kept_when_model_ignores_format(make_file, scripted):
    replies = [SEARCH, SEARCH, "Coolant stays under 300 degrees."]
    agent, _ = _agent(make_file, scripted, replies, max_iterations=2)
    r = agent.run("coolant?")
    assert "300 degrees" in r.answer
    assert r.finish_reason == "fallback"
    assert r.needs_review


def test_answer_before_any_search_is_rejected_then_grounded(make_file, scripted):
    replies = ["Thought: I know this.\nFinal Answer: 42.", SEARCH,
               f"Final Answer: Below 300 degrees {CITE}."]
    agent, llm = _agent(make_file, scripted, replies)
    r = agent.run("coolant?")
    assert "42" not in r.answer and "300 degrees" in r.answer
    nudges = [m["content"] for m in llm.calls[1]["messages"] if "must call search_documents" in m["content"]]
    assert nudges


def test_ungrounded_final_answer_never_accepted_on_last_step(make_file, scripted):
    agent, _ = _agent(make_file, scripted, ["Final Answer: made up."] * 1, max_iterations=1)
    r = agent.run("anything")
    assert r.abstained and "made up" not in r.answer


def test_not_found_token_abstains_and_routes_to_review(make_file, scripted):
    replies = ['Action: search_documents("quarterly revenue")', "Final Answer: NOT_FOUND: nothing about revenue."]
    agent, _ = _agent(make_file, scripted, replies)
    r = agent.run("quarterly revenue?")
    assert r.abstained and r.answer.startswith(NO_ANSWER_MESSAGE)
    assert "abstained" in r.review_reasons and r.needs_review


def test_fabricated_citation_is_stripped_and_flagged(make_file, scripted):
    replies = [SEARCH, "Final Answer: Below 300 degrees (Source: made_up.pdf, p.9) and (Source: alpha.txt)."]
    agent, _ = _agent(make_file, scripted, replies)
    r = agent.run("coolant?")
    assert "made_up.pdf" not in r.answer
    assert "[unverified citation removed]" in r.answer
    assert r.citation_report["invalid"] and r.citation_report["valid"]
    assert "invalid_citation" in r.review_reasons


def test_answer_without_citations_is_flagged(make_file, scripted):
    agent, _ = _agent(make_file, scripted, [SEARCH, "Final Answer: Below 300 degrees."])
    r = agent.run("coolant?")
    assert "no_citations" in r.review_reasons


def test_get_chunk_context_tool_executes_and_adds_neighbours(make_file, scripted):
    store = CorpusStore()
    ingest(make_file("alpha.txt", ALPHA * 6), store=store)
    retriever = HybridRetriever(store)
    first = retriever.search("coolant temperature startup")[0].chunk
    cid = first.chunk_id[:8]
    llm = scripted([SEARCH, f'Action: get_chunk_context("{cid}")', f"Final Answer: ok {CITE}."])
    r = ReActAgent(retriever=retriever, llm=llm).run("coolant?")
    obs = [m["content"] for m in llm.calls[2]["messages"] if m["content"].startswith("Observation:")][-1]
    assert "Surrounding context" in obs or "already shown" in obs
    assert r.llm_calls == 3


def test_unknown_chunk_id_is_reported_not_crashed(make_file, scripted):
    replies = [SEARCH, 'Action: get_chunk_context("deadbeef")', f"Final Answer: ok {CITE}."]
    agent, llm = _agent(make_file, scripted, replies)
    agent.run("coolant?")
    obs = [m["content"] for m in llm.calls[2]["messages"] if m["content"].startswith("Observation:")][-1]
    assert "Unknown chunk_id" in obs


def test_scope_is_respected_by_the_agent(make_file, scripted):
    store = CorpusStore()
    ra = ingest(make_file("alpha.txt", ALPHA), store=store)
    ingest(make_file("beta.txt", BETA), store=store)
    llm = scripted(['Action: search_documents("soup menu")', "Final Answer: NOT_FOUND: no soup."])
    r = ReActAgent(retriever=HybridRetriever(store), llm=llm).run("soup?", file_ids=ra.file_ids)
    assert all(s.chunk.filename() == "alpha.txt" for s in r.sources)


def test_repeated_chunks_are_not_resent_in_full(make_file, scripted):
    agent, llm = _agent(make_file, scripted, [SEARCH, SEARCH, f"Final Answer: ok {CITE}."])
    agent.run("coolant?")
    second_obs = [m["content"] for m in llm.calls[2]["messages"] if m["content"].startswith("Observation:")][-1]
    assert "already shown above" in second_obs


def test_parse_action_variants():
    assert parse_action('Action: search_documents("a (b) c")') == ("search_documents", "a (b) c")
    assert parse_action("Action: search_documents('x y')") == ("search_documents", "x y")
    assert parse_action('Action: search_documents(query="x y")') == ("search_documents", "x y")
    assert parse_action('Action: get_chunk_context("ab12cd34")') == ("get_chunk_context", "ab12cd34")
    assert parse_action("Action: unknown_tool(\"x\")") is None
    assert parse_action("no action here") is None
