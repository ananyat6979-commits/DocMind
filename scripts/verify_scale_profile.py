"""Recompute persistent ANN and load summaries without neural inference."""
import argparse
import json
import math
from pathlib import Path


def verify(directory):
    import numpy as np
    d=json.loads((directory/'summary.json').read_text());restart=json.loads((directory/'restart.json').read_text())
    assert d['corpus_size']==d['unique_paragraphs']==50000 and d['queries']==len(d['recalls'])==100
    recalls=np.asarray(d['recalls']);assert np.isfinite(recalls).all() and ((recalls>=0)&(recalls<=1)).all()
    assert math.isclose(float(recalls.mean()),d['recall_at10'],abs_tol=1e-12)
    rng=np.random.default_rng(7);draws=[float(np.mean(rng.choice(recalls,len(recalls),replace=True))) for _ in range(2000)]
    assert np.allclose(np.percentile(draws,[2.5,97.5]),d['recall_bootstrap95_nominal'],rtol=1e-12)
    assert len(d['ann_ms'])==len(d['exact_ms'])==100 and len(d['scoped_exact_ms'])==20
    assert 'chunks_hnsw_' in json.dumps(d['planner']) and 'Index Scan' in json.dumps(d['planner'])
    assert [r['concurrency'] for r in d['load']]==[1,4,8,16]
    for result in d['load']:
        rows=result['records'];assert result['requested']==len(rows)==200
        assert result['completed']==sum(r['ok'] for r in rows)==200 and result['failed']==0
        times=[r['elapsed_ms'] for r in rows];assert all(math.isfinite(t) and t>0 for t in times)
        assert np.allclose(np.percentile(times,[50,95]),[result['p50_ms'],result['p95_ms']],rtol=1e-12)
        assert math.isclose(result['requests_per_second'],200/result['elapsed_seconds'],rel_tol=1e-12)
    assert restart['chunk_count']==50000 and restart['fresh_process_vector_search'] is True
    assert d['embedding_identity']['model']=='intfloat/multilingual-e5-small'
    return {'unique_paragraphs':50000,'queries':100,'recall_at10':d['recall_at10'],'load_requests_verified':800,'restart_verified':True}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path);args=parser.parse_args();print(json.dumps(verify(args.directory)))
