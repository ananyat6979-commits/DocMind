"""Synthetic same-cluster restore check. Never uploads a database dump."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def execute(manifest,seed=False):
    import numpy as np
    from docmind.postgres import DatabaseScope
    from docmind.store import CorpusStore
    from docmind.models import Chunk,AgentResponse
    from docmind.embeddings import embed_texts
    from docmind.review.store import ReviewStore
    from docmind.telemetry.events import EventLog
    from docmind.jobs import IngestionQueue
    url=os.environ['DOCMIND_TEST_POSTGRES_URL'];scope=DatabaseScope(url,'alpha');store=CorpusStore(scope)
    if seed:
        text='The restoration probe preserves committed service state.';vector=embed_texts([text]);chunk=Chunk(text,'probe','probe.txt',0,chunk_id='restore-probe-chunk',file_id='restore-probe')
        store.add_document({'file_id':'restore-probe','filename':'probe.txt','source':'probe.txt','sha256':hashlib.sha256(text.encode()).hexdigest(),'n_units':1},[chunk],vector)
        response=AgentResponse('Synthetic response',[],[],'Restoration probe?',1,needs_review=True)
        ReviewStore(scope).record(response,'restore-user');EventLog(scope).log('backup_probe','restore-user')
        job=IngestionQueue(scope).enqueue('probe.txt','restore-user')
        manifest.parent.mkdir(parents=True,exist_ok=True);manifest.write_text(json.dumps({'chunk_id':chunk.chunk_id,'response_id':response.response_id,'job_id':job,'vector_sha256':hashlib.sha256(vector.tobytes()).hexdigest()}))
        print('Synthetic restore probe committed')
    else:
        expected=json.loads(manifest.read_text());chunk=store.get_chunk(expected['chunk_id']);assert chunk is not None
        chunks,matrix=store.load_corpus(['restore-probe'])
        assert len(chunks)==1 and hashlib.sha256(np.ascontiguousarray(matrix).tobytes()).hexdigest()==expected['vector_sha256']
        assert ReviewStore(scope).get(expected['response_id'])['status']=='pending'
        assert IngestionQueue(scope).get(expected['job_id'])['status']=='queued'
        assert EventLog(scope).rows('backup_probe')
        other=DatabaseScope(url,'beta');assert CorpusStore(other).get_chunk(expected['chunk_id']) is None
        assert ReviewStore(other).get(expected['response_id']) is None
        result={'scope':'Synthetic same-cluster fresh-database dump/restore; not an operator backup or cross-region disaster-recovery test','corpus_vector_equivalent':True,'pending_review_restored':True,'queued_job_restored':True,'event_restored':True,'tenant_isolation_preserved':True,'source_commit':os.getenv('GITHUB_SHA')}
        manifest.with_name('restore-summary.json').write_bytes(json.dumps(result,indent=2).encode());print(json.dumps(result))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--seed',action='store_true');args=parser.parse_args();execute(args.manifest,args.seed)
