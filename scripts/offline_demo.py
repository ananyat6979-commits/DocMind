"""One-command offline component demonstration with real cached retrieval, not QA evaluation."""
import json
import os
from pathlib import Path
import sys
import tempfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    with tempfile.TemporaryDirectory(prefix='docmind-offline-') as directory:
        home=Path(directory)
        os.environ.pop('GROQ_API_KEY',None)
        os.environ.update(DOCMIND_HOME=str(home),DOCMIND_SECRETS_FILE=str(home/'absent.env'),DOCMIND_TEST_MODE='0',DOCMIND_ATTRIBUTION='0',DOCMIND_DEVICE='cpu',DOCMIND_USE_RERANKER='0',DOCMIND_RETRIEVAL_MODE='dense',DOCMIND_CHUNKER='recursive',HF_HUB_OFFLINE='1',HF_HUB_DISABLE_PROGRESS_BARS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
        from docmind.config import CONFIG
        from docmind.ingestion.pipeline import ingest
        from docmind.store import CorpusStore
        from docmind.retrieval.hybrid import HybridRetriever
        from docmind.agent.reader import CitedReader
        from docmind.llm.demo import DemoLLM
        assert CONFIG.embedding.backend=='sentence-transformers' and not CONFIG.llm.groq_api_key
        CONFIG.ensure_dirs();store=CorpusStore()
        document=home/'public-example.txt'
        document.write_text('The example procedure governs thermal regulation. Coolant must stay below 300 degrees during startup. Operators inspect the coolant pump every shift. This is an original synthetic demonstration source.',encoding='utf-8')
        ingest(document,store=store)
        result=CitedReader(HybridRetriever(store=store),DemoLLM(),deadline_seconds=60).run('What is the coolant temperature limit during startup?')
        if result.release_state!='draft' or '300' not in result.answer or not result.evidence:raise RuntimeError('Offline demonstration failed its cited-draft contract')
        print(json.dumps({'scope':'Offline extractive component demonstration; real cached E5 retrieval, not generative answer quality or a running service','question':result.query,'answer':result.answer,'release_state':result.release_state,'evidence':result.evidence,'embedding_model':CONFIG.embedding.model_name,'embedding_revision':CONFIG.embedding.model_revision,'provider_calls':0,'human_decisions':0,'paid_compute':False},ensure_ascii=True,indent=2))


if __name__=='__main__':main()
