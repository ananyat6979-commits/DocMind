import json
from docmind.agent.react import ReActAgent
from docmind.ingestion.pipeline import ingest
from docmind.retrieval.hybrid import HybridRetriever
from docmind.store import CorpusStore
from tests.conftest import ALPHA


def test_native_protocol_uses_structured_tool_messages(make_file, monkeypatch):
    monkeypatch.setenv("DOCMIND_TOOL_PROTOCOL","native")
    store=CorpusStore();ingest(make_file("alpha.txt",ALPHA),store=store)
    class Native:
        def __init__(self):self.calls=[]
        def complete_tools(self,messages,tools):
            self.calls.append(list(messages))
            if len(self.calls)==1:
                return {"role":"assistant","tool_calls":[{"id":"call1","type":"function",
                    "function":{"name":"search_documents","arguments":json.dumps({"argument":"coolant temperature"})}}]}
            assert messages[-1]["role"]=="tool" and messages[-1]["tool_call_id"]=="call1"
            return {"role":"assistant","content":"The limit is 300 degrees (Source: alpha.txt)."}
    llm=Native();response=ReActAgent(HybridRetriever(store),llm).run("What is the limit?")
    assert not response.abstained and response.cited_sources
    assert response.llm_calls==2 and len(llm.calls)==2
