"""Bounded real-embedding persistent ANN profile, without generation or provider calls."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))


def main(out,count=50000):
    import numpy as np
    import pyarrow.parquet as pq
    import torch
    from calibration_closure import acquire,SOURCE_SHA,SOURCE_URL
    from docmind.postgres import DatabaseScope
    from docmind.db import connect
    from docmind.models import Chunk
    from docmind.store import CorpusStore
    from docmind.embeddings import embed_texts,embed_query,embedding_identity
    from docmind.retrieval.postgres import PostgresDenseRetriever
    started=time.perf_counter();out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    source=out/'private/source.parquet';acquire(source)
    rows=pq.read_table(source).to_pylist();unique={}
    for row in rows:
        for title,sentences in zip(row['context']['title'],row['context']['sentences']):
            text=title+'\n'+' '.join(sentences)
            if len(text.strip())>=80:unique.setdefault(hashlib.sha256(text.encode()).hexdigest(),text)
    chosen=sorted(unique)[:count]
    if len(chosen)!=count:raise ValueError('Insufficient unique public paragraphs')
    texts=[unique[k] for k in chosen]
    t=time.perf_counter();vectors=embed_texts(texts,show_progress=True);embedding_seconds=time.perf_counter()-t
    scope=DatabaseScope(os.environ['DOCMIND_TEST_POSTGRES_URL'],'scale-profile');store=CorpusStore(scope)
    with connect(scope) as conn:conn.execute('DELETE FROM documents');conn.execute('DELETE FROM meta')
    t=time.perf_counter()
    for start in range(0,count,1000):
        file_id='profile-'+str(start//1000)
        chunks=[Chunk(texts[i],chosen[i],SOURCE_URL,0,{'filename':'public-paragraphs','page':1},chunk_id=chosen[i],file_id=file_id,ord=i-start) for i in range(start,min(start+1000,count))]
        store.add_document({'file_id':file_id,'filename':'public-paragraphs','source':SOURCE_URL,'sha256':hashlib.sha256(''.join(c.chunk_id for c in chunks).encode()).hexdigest(),'n_units':len(chunks)},chunks,vectors[start:start+1000])
        print('persisted',start+len(chunks),flush=True)
    ingestion_seconds=time.perf_counter()-t
    queries=sorted(rows,key=lambda r:hashlib.sha256(('scale:'+r['id']).encode()).hexdigest())[:100]
    qvectors=np.vstack([embed_query(r['question'])[0] for r in queries])
    retriever=PostgresDenseRetriever(store);recalls=[];ann_ms=[];exact_ms=[]
    for vector in qvectors:
        t=time.perf_counter();approx=retriever.search_vector(vector,10);ann_ms.append((time.perf_counter()-t)*1000)
        t=time.perf_counter();exact=retriever.search_vector(vector,10,exact=True);exact_ms.append((time.perf_counter()-t)*1000)
        recalls.append(len({r.chunk.chunk_id for r in approx}&{r.chunk.chunk_id for r in exact})/10)
    # Exact scoped application path and fresh-connection persistence are measured separately.
    scoped=[]
    for vector in qvectors[:20]:
        t=time.perf_counter();result=retriever.search_vector(vector,10,file_ids=['profile-0']);scoped.append((time.perf_counter()-t)*1000)
        assert all(r.chunk.file_id=='profile-0' for r in result)
    loads=[]
    for concurrency in (1,4,8,16):
        def one(index):
            t=time.perf_counter()
            try:
                result=PostgresDenseRetriever(CorpusStore(scope)).search_vector(qvectors[index%len(qvectors)],10)
                return {'elapsed_ms':(time.perf_counter()-t)*1000,'ok':len(result)==10}
            except Exception as exc:return {'elapsed_ms':(time.perf_counter()-t)*1000,'ok':False,'error':type(exc).__name__}
        t=time.perf_counter()
        with ThreadPoolExecutor(concurrency) as pool:records=list(pool.map(one,range(200)))
        elapsed=time.perf_counter()-t;values=[r['elapsed_ms'] for r in records]
        loads.append({'concurrency':concurrency,'requested':200,'completed':sum(r['ok'] for r in records),'failed':sum(not r['ok'] for r in records),'elapsed_seconds':elapsed,'requests_per_second':200/elapsed,'p50_ms':float(np.percentile(values,50)),'p95_ms':float(np.percentile(values,95)),'records':records})
    rng=np.random.default_rng(7);means=[float(np.mean(rng.choice(recalls,len(recalls),replace=True))) for _ in range(2000)]
    with connect(scope) as conn:
        plan=conn.execute('EXPLAIN (FORMAT JSON) SELECT chunk_id FROM chunks WHERE tenant_id=? ORDER BY embedding_vector <#> ?::vector LIMIT 10',(scope.tenant,json.dumps(qvectors[0].tolist()))).fetchone()[0]
        server=conn.execute("SELECT version() AS server,(SELECT extversion FROM pg_extension WHERE extname='vector') AS vector").fetchone()
        size=conn.execute("SELECT pg_total_relation_size('chunks') AS bytes").fetchone()['bytes']
    if 'Index Scan' not in json.dumps(plan) or 'chunks_hnsw_' not in json.dumps(plan):raise AssertionError('ANN planner did not use the persistent tenant HNSW index')
    import platform
    result={'hardware':{'platform':platform.platform(),'cpu_count':os.cpu_count(),'torch_threads':torch.get_num_threads(),'database':dict(server)},'scope':'Single-host persistent PostgreSQL ANN and database search profile; query embeddings precomputed for search timing; no hosted reader or HTTP SLA',
        'source_sha256':SOURCE_SHA,'source_url':SOURCE_URL,'source_license':'CC BY-SA 4.0, HotpotQA and Wikipedia attribution apply','embedding_identity':json.loads(embedding_identity()),
        'corpus_size':count,'unique_paragraphs':len(chosen),'source_selection_sha256':hashlib.sha256(''.join(chosen).encode()).hexdigest(),'embedding_seconds':embedding_seconds,'ingestion_seconds':ingestion_seconds,
        'queries':100,'recall_at10':statistics.mean(recalls),'recall_bootstrap95_nominal':np.percentile(means,[2.5,97.5]).tolist(),
        'recalls':recalls,'ann_ms':ann_ms,'exact_ms':exact_ms,'scoped_exact_ms':scoped,'load':loads,'planner':plan,'database_chunks_bytes':size,
        'elapsed_seconds':time.perf_counter()-started,'restart_status':'Fresh object/connection persistence verified; database restart and backup restoration are separate tests'}
    (out/'summary.json').write_bytes(json.dumps(result,indent=2).encode())
    print(json.dumps({k:result[k] for k in ('corpus_size','recall_at10','embedding_seconds','ingestion_seconds','elapsed_seconds')}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--verify-restart',action='store_true');parser.add_argument('--out',type=Path,required=True);parser.add_argument('--count',type=int,default=50000);args=parser.parse_args()
    if not 1000<=args.count<=50000:raise ValueError('Profile size outside bounds')
    if args.verify_restart:
        import numpy as np
        from docmind.postgres import DatabaseScope
        from docmind.db import connect
        from docmind.store import CorpusStore
        from docmind.retrieval.postgres import PostgresDenseRetriever
        scope=DatabaseScope(os.environ['DOCMIND_TEST_POSTGRES_URL'],'scale-profile')
        store=CorpusStore(scope)
        with connect(scope) as conn:
            count=conn.execute('SELECT COUNT(*) FROM chunks').fetchone()[0]
            row=conn.execute('SELECT chunk_id,embedding FROM chunks ORDER BY chunk_id LIMIT 1').fetchone()
        assert count==args.count
        result=PostgresDenseRetriever(store).search_vector(np.frombuffer(row['embedding'],dtype='float32'),10)
        assert any(r.chunk.chunk_id==row['chunk_id'] for r in result)
        args.out.mkdir(parents=True,exist_ok=True)
        (args.out/'restart.json').write_text(json.dumps({'chunk_count':count,'fresh_process_vector_search':True,'database_restart':'Performed by CI before this fresh process'}))
    else:main(args.out,args.count)
