"""Two-process HTTP profile with shared PostgreSQL, real E5/NLI and demo generation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from load_http_service import summarize


def run(out):
    out.mkdir(parents=True,exist_ok=False);started=time.monotonic();deadline=started+600
    requested=24;rows=[];servers=[];scope='http-profile'
    report={'scope':'Two independent API processes, shared PostgreSQL and private upload directory, real E5/NLI, synthetic fixture and demo generation; no generative-reader or hosted SLA',
        'requested':requested,'concurrency':4,'api_processes':2,'provider_calls':0,'model_downloads':False,'quota_per_minute':1000,
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'source_commit':os.getenv('GITHUB_SHA')}
    dsn=os.environ['DOCMIND_TEST_POSTGRES_URL']
    with tempfile.TemporaryDirectory(prefix='docmind_pg_http_') as tmp:
        root=Path(tmp);key=secrets.token_hex(24);credentials=root/'credentials.json'
        credentials.write_text(json.dumps([{'tenant_id':scope,'actor_id':'load-user','key_sha256':hashlib.sha256(key.encode()).hexdigest(),'roles':['reader','ingestor']}]))
        env={k:v for k,v in os.environ.items() if not k.startswith('DOCMIND_') and k!='GROQ_API_KEY'}
        env.update(DOCMIND_HOME=tmp,DOCMIND_TEST_MODE='0',DOCMIND_DATABASE_URL=dsn,DOCMIND_CREDENTIALS_FILE=str(credentials),
            DOCMIND_SECRETS_FILE=str(root/'missing.env'),DOCMIND_REQUESTS_PER_MINUTE='1000',DOCMIND_REVIEW_AUDIT_RATE='0',
            DOCMIND_DEVICE='cpu',HF_HUB_OFFLINE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',
            DOCMIND_PARSER_MODE='local',DOCMIND_ALLOW_UNSANDBOXED_PARSING='1')
        data=root/'data/tenants'/hashlib.sha256(scope.encode()).hexdigest();data.mkdir(parents=True)
        text='Coolant must stay below 300 degrees during startup. Operators verify the pump every shift. '*4
        (data/'coolant.txt').write_text(text);report['fixture_sha256']=hashlib.sha256(text.encode()).hexdigest()
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}));urls=[]
        def request(index,path,body=None):
            if time.monotonic()>=deadline:raise TimeoutError('Total HTTP profile budget reached')
            req=urllib.request.Request(urls[index]+path,data=json.dumps(body).encode() if body is not None else None,headers={'X-API-Key':key,'Content-Type':'application/json'})
            with opener.open(req,timeout=min(120,max(1,deadline-time.monotonic()))) as response:return response.status,json.load(response),bool(response.headers.get('X-Request-ID'))
        try:
            for i in range(2):
                with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
                urls.append('http://127.0.0.1:'+str(port))
                command="from docmind.config import CONFIG; CONFIG.llm.provider='demo'; import uvicorn; uvicorn.run('docmind.api.main:app',host='127.0.0.1',port="+str(port)+",log_level='warning',proxy_headers=False)"
                with (out/('server-'+str(i)+'.log')).open('wb') as log:servers.append(subprocess.Popen([sys.executable,'-c',command],cwd=ROOT,env=env,stdout=log,stderr=log))
            for i in range(2):
                while True:
                    if servers[i].poll() is not None:raise RuntimeError('API process exited during startup')
                    try:
                        if request(i,'/ready')[0]==200:break
                    except Exception:
                        if time.monotonic()>=deadline:raise
                        time.sleep(.1)
            report['both_ready_seconds']=time.monotonic()-started
            t=time.monotonic();_,job,_=request(0,'/ingest/jobs',{'path':'coolant.txt','expected_language':'en'})
            while True:
                _,state,_=request(1,'/ingest/jobs/'+job['job_id'])
                if state['status']=='completed':break
                if state['status']=='failed':raise RuntimeError('Shared ingestion failed')
                time.sleep(.1)
            report['cross_process_ingestion_seconds']=time.monotonic()-t
            question={'question':'What is the coolant limit during startup?'};cold=[]
            for i in range(2):
                t=time.monotonic();request(i,'/query',question);cold.append(time.monotonic()-t)
            report['cold_query_seconds_by_process']=cold
            def one(index):
                t=time.monotonic()
                if t>=deadline:return {'index':index,'unserved':True}
                try:
                    code,payload,trace=request(index%2,'/query',question)
                    return {'index':index,'process_index':index%2,'status':code,'request_id_present':trace,'latency_ms':(time.monotonic()-t)*1000,'review_status':payload['review_status']}
                except Exception as exc:return {'index':index,'status':getattr(exc,'code',None),'latency_ms':(time.monotonic()-t)*1000,'error':type(exc).__name__}
            t=time.monotonic()
            with ThreadPoolExecutor(4) as pool:rows=list(pool.map(one,range(requested)))
            report['warm_load']=summarize(rows,requested,time.monotonic()-t)
        except Exception as exc:report['setup_error']=type(exc).__name__;report['warm_load']=summarize(rows,requested,time.monotonic()-started)
        finally:
            for server in servers:
                if server.poll() is None:server.terminate()
            for server in servers:
                try:server.wait(timeout=10)
                except subprocess.TimeoutExpired:server.kill();server.wait(timeout=10)
        from docmind.postgres import DatabaseScope
        from docmind.db import connect
        with connect(DatabaseScope(dsn,scope)) as conn:stored=conn.execute('SELECT response_info FROM reviews').fetchall()
        completed=0;judges=set()
        for row in stored:
            attribution=json.loads(row['response_info']).get('citation_report',{}).get('attribution') or {}
            fraction=attribution.get('supported_fraction')
            if not attribution.get('error') and attribution.get('n_claims',0)>0 and isinstance(fraction,(int,float)) and math.isfinite(fraction) and isinstance(attribution.get('judge'),dict):
                completed+=1;judges.add(json.dumps(attribution['judge'],sort_keys=True))
        report['stored_attribution']={'responses':len(stored),'completed_nli':completed,'judges':[json.loads(s) for s in sorted(judges)]}
        report['neural_path_verified']=len(stored)==completed==requested+2
    report['total_wall_seconds']=time.monotonic()-started
    (out/'rows.jsonl').write_bytes(''.join(json.dumps(r)+'\n' for r in rows).encode());(out/'summary.json').write_bytes(json.dumps(report,indent=2).encode())
    print(json.dumps(report,indent=2),flush=True)
    return int(bool(report.get('setup_error')) or report['warm_load']['completed']!=requested or not report['neural_path_verified'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args();raise SystemExit(run(args.out))
