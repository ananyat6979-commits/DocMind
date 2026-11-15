"""Recompute service-load counts/percentiles and verify bounded recovery records."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path


def verify(root):
    import numpy as np
    summaries=[]
    for directory in sorted(root.glob('http_two_process_*')):
        report=json.loads((directory/'summary.json').read_text());rows=[json.loads(s) for s in (directory/'rows.jsonl').read_text().splitlines()]
        assert report['requested']==len(rows)==24 and {r['index'] for r in rows}==set(range(24))
        assert Counter(r['process_index'] for r in rows)=={0:12,1:12}
        warm=report['warm_load'];durations=[r['latency_ms'] for r in rows]
        assert all(r['status']==200 and r['request_id_present'] for r in rows)
        assert warm['requested']==warm['recorded']==warm['completed']==24 and warm['failed']==warm['unserved']==0
        assert warm['status_counts']=={'200':24}
        assert math.isclose(warm['completed_per_second'],24/warm['elapsed_seconds'],rel_tol=1e-12)
        assert np.allclose([warm['completed_only_p50_ms'],warm['completed_only_p95_ms']],np.percentile(durations,[50,95]),rtol=1e-12)
        assert report['stored_attribution']['responses']==report['stored_attribution']['completed_nli']==26
        assert report['neural_path_verified'] and report['provider_calls']==0 and report['api_processes']==2
        summaries.append({'profile':directory.name,'completed':24,'nli_completed':26})
    restore=json.loads((root/'restore_summary.json').read_text())
    for name in ('corpus_vector_equivalent','pending_review_restored','queued_job_restored','event_restored','tenant_isolation_preserved'):assert restore[name] is True
    return {'http_profiles':summaries,'synthetic_restore_verified':True}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path);args=parser.parse_args();print(json.dumps(verify(args.directory)))
