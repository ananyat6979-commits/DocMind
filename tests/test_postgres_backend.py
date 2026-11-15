"""Live database integration tests. Stand-in vectors test isolation, never ANN quality."""
import os
import json
import hashlib
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pytest
from docmind.postgres import DatabaseScope,LeaseLost,translate
from docmind.db import connect
from docmind.models import Chunk,AgentResponse
from docmind.store import CorpusStore
from docmind.jobs import IngestionQueue
from docmind.review.store import ReviewStore
from docmind.telemetry.events import EventLog
from docmind.retrieval.postgres import PostgresDenseRetriever


def test_sql_translation_preserves_quoted_values_and_upserts_tenant_key():
    assert translate("SELECT user FROM reviews WHERE question='user ?' AND response_id=?")=="SELECT \"user\" FROM reviews WHERE question='user ?' AND response_id=%s"
    assert 'ON CONFLICT (tenant_id,key)' in translate('INSERT OR REPLACE INTO meta(key,value) VALUES (?,?)')
    assert 'password' not in repr(DatabaseScope('postgresql://name:password@host/db','alpha'))


@pytest.fixture
def scopes():
    url=os.getenv('DOCMIND_TEST_POSTGRES_URL')
    if not url:pytest.skip('Live PostgreSQL integration profile required')
    values=[DatabaseScope(url,t) for t in ('alpha','beta')]
    for scope in values:
        with connect(scope) as conn:
            for table in ('documents','reviews','events','ingestion_jobs','meta'):conn.execute('DELETE FROM '+table)
    return values


def add(store,file_id='same-file',text='Coolant stays below 300 degrees.'):
    vector=np.zeros((1,384),dtype='float32');vector[0,0]=1
    chunk=Chunk(text,'page1','manual.pdf',0,{'filename':'manual.pdf','page':1},chunk_id='aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa',file_id=file_id)
    store.add_document({'file_id':file_id,'filename':'manual.pdf','source':'manual.pdf','sha256':'hash','n_units':1},[chunk],vector)
    return chunk,vector[0]


def test_tenant_isolation_covers_all_stores_and_vector_paths(scopes):
    left,right=scopes;a,b=CorpusStore(left),CorpusStore(right)
    chunk,query=add(a)
    assert a.get_chunk(chunk.chunk_id) and not b.get_chunk(chunk.chunk_id)
    assert not b.get_chunk(chunk.chunk_id[:8]) and not b.neighbors(chunk)
    assert not b.list_documents() and not b.delete_document(chunk.file_id)
    assert len(PostgresDenseRetriever(a).search_vector(query))==1
    assert not PostgresDenseRetriever(b).search_vector(query)
    assert not PostgresDenseRetriever(a).search_vector(query,file_ids=['other-file'])
    assert len(PostgresDenseRetriever(a).search_vector(query,file_ids=[chunk.file_id]))==1
    assert len(PostgresDenseRetriever(a).search_vector(query,file_ids=iter([chunk.file_id])))==1
    response=AgentResponse('300',[],[],'temperature?',1)
    ReviewStore(left).record(response,'alice')
    assert ReviewStore(right).get(response.response_id) is None
    EventLog(left).log('query','alice')
    assert not EventLog(right).rows()
    job=IngestionQueue(left).enqueue('private.txt','alice')
    assert IngestionQueue(right).get(job) is None
    with pytest.raises(Exception):
        with connect(left) as conn:conn.execute("INSERT INTO meta(tenant_id,key,value) VALUES (?,?,?)",('beta','escape','no'))
    add(b,text='A separate tenant can reuse identifiers.')
    assert a.get_chunk(chunk.chunk_id).content!=b.get_chunk(chunk.chunk_id).content


def test_persistent_changes_and_atomic_versions(scopes):
    store=CorpusStore(scopes[0]);chunk,vector=add(store)
    version=store.version()
    # Fresh objects and connections must observe committed vectors without a rebuild.
    assert PostgresDenseRetriever(CorpusStore(scopes[0])).search_vector(vector)[0].chunk.chunk_id==chunk.chunk_id
    assert store.delete_document(chunk.file_id) and store.version()==version+1
    assert not PostgresDenseRetriever(CorpusStore(scopes[0])).search_vector(vector)
    def bump(_):
        with connect(scopes[0]) as conn:store._bump(conn)
    with ThreadPoolExecutor(8) as pool:list(pool.map(bump,range(20)))
    assert store.version()==version+21


def test_claims_are_unique_and_expired_worker_cannot_commit(scopes):
    scope=scopes[0];queues=[IngestionQueue(scope),IngestionQueue(scope)]
    job=queues[0].enqueue('fixture.txt')
    with ThreadPoolExecutor(2) as pool:claims=list(pool.map(lambda q:q._claim(),queues))
    claimed=[r for r in claims if r is not None];assert len(claimed)==1
    old_token=claimed[0][1]
    with connect(scope) as conn:conn.execute("UPDATE ingestion_jobs SET lease_until=0 WHERE job_id=?",(job,))
    replacement=queues[1]._claim();assert replacement[0]['job_id']==job and replacement[1]!=old_token
    with pytest.raises(LeaseLost):
        with connect(scope.fenced(job,old_token)) as conn:conn.execute("INSERT INTO meta(key,value) VALUES ('stale','no')")
    with connect(scope) as conn:assert conn.execute("SELECT value FROM meta WHERE key='stale'").fetchone() is None
    # Starting another replica must preserve a live lease.
    queues[0].run_one=lambda:False
    queues[0].start();queues[0].close()
    assert queues[1].get(job)['status']=='running'


def test_lease_expiry_during_transaction_rolls_back(scopes):
    scope=scopes[0];queue=IngestionQueue(scope);job=queue.enqueue('fixture.txt');row,token=queue._claim(lease_seconds=.05)
    with pytest.raises(LeaseLost):
        with connect(scope.fenced(job,token)) as conn:
            conn.execute("INSERT INTO meta(key,value) VALUES ('late','no')")
            conn.execute('SELECT pg_sleep(.1)')
    with connect(scope) as conn:assert conn.execute("SELECT value FROM meta WHERE key='late'").fetchone() is None


def test_quota_shared_between_connections_and_principals(scopes):
    scope=scopes[0];key=hashlib.sha256(os.urandom(32)).hexdigest();peer=hashlib.sha256(os.urandom(32)).hexdigest()
    def take(_):
        with connect(scope) as conn:return conn.execute('SELECT take_quota(?,?,?,?) AS allowed',(key,peer,5,20)).fetchone()['allowed']
    with ThreadPoolExecutor(8) as pool:results=list(pool.map(take,range(16)))
    assert sum(results)==5
    with connect(scopes[1]) as conn:
        assert not conn.execute('SELECT take_quota(?,?,?,?) AS allowed',(key,peer,5,20)).fetchone()['allowed']
        with pytest.raises(Exception):conn.execute('SELECT * FROM api_quota')


def test_api_tenant_headers_cannot_override_credentials(scopes,tmp_path,monkeypatch):
    from fastapi.testclient import TestClient
    from docmind.api.main import app
    tokens=['alpha-secret','beta-secret'];registry=tmp_path/'credentials.json'
    registry.write_text(json.dumps([{'tenant_id':tenant,'actor_id':'reader','key_sha256':hashlib.sha256(token.encode()).hexdigest(),'roles':['reader','ingestor','admin']} for tenant,token in zip(('alpha','beta'),tokens)]))
    monkeypatch.setenv('DOCMIND_DATABASE_URL',scopes[0].dsn);monkeypatch.setenv('DOCMIND_CREDENTIALS_FILE',str(registry))
    add(CorpusStore(scopes[0]))
    with TestClient(app) as client:
        left=client.get('/documents',headers={'X-API-Key':tokens[0]})
        right=client.get('/documents',headers={'X-API-Key':tokens[1],'X-Tenant-ID':'alpha'})
        assert left.status_code==right.status_code==200
        assert 'same-file' in left.text and 'same-file' not in right.text
        assert client.get('/documents',headers={'X-API-Key':'invalid'}).status_code==401
        # Direct-insert fixtures have no chunker lineage. Select the new-ingest
        # recipe explicitly, without guessing or changing those existing chunks.
        monkeypatch.setenv('DOCMIND_CHUNKER', 'recursive')
        # Synchronous upload and deletion must use the authenticated backend too.
        response=client.post('/upload',headers={'X-API-Key':tokens[0]},files={'file':('tenant.txt',b'The reactor coolant remains below 300 degrees during startup. Operators inspect the pump each shift.','text/plain')})
        assert response.status_code==200,response.text
        file_id=response.json()['file_ids'][0]
        assert CorpusStore(scopes[0]).has_file(file_id) and not CorpusStore(scopes[1]).has_file(file_id)
        assert client.delete('/documents/'+file_id,headers={'X-API-Key':tokens[1]}).status_code==404
        assert client.delete('/documents/'+file_id,headers={'X-API-Key':tokens[0]}).status_code==200
        assert not CorpusStore(scopes[0]).has_file(file_id)
        stats=client.get('/metrics',headers={'X-API-Key':tokens[1]}).json()
        assert stats['requests_by_status'].get('401',0)==0


def test_two_api_processes_share_quota_and_tenant_isolation(scopes,tmp_path,monkeypatch):
    import socket
    import subprocess
    import sys
    import time
    import httpx
    tokens=['process-alpha-'+os.urandom(8).hex(),'process-beta-'+os.urandom(8).hex()]
    registry=tmp_path/'process-credentials.json'
    registry.write_text(json.dumps([{'tenant_id':tenant,'actor_id':'reader','key_sha256':hashlib.sha256(token.encode()).hexdigest(),'roles':['reader']} for tenant,token in zip(('alpha','beta'),tokens)]))
    add(CorpusStore(scopes[0]))
    env={**os.environ,'DOCMIND_DATABASE_URL':scopes[0].dsn,'DOCMIND_CREDENTIALS_FILE':str(registry),'DOCMIND_REQUESTS_PER_MINUTE':'3','DOCMIND_REQUESTS_PER_IP_PER_MINUTE':'5','DOCMIND_HOME':str(tmp_path/'runtime')}
    processes=[];urls=[]
    try:
        for _ in range(2):
            with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
            log=(tmp_path/('server-'+str(port)+'.log')).open('w')
            process=subprocess.Popen([sys.executable,'-m','docmind.api.server','--host','127.0.0.1','--port',str(port)],env=env,stdout=log,stderr=subprocess.STDOUT)
            log.close();processes.append(process);urls.append('http://127.0.0.1:'+str(port))
        with httpx.Client(timeout=5) as client:
            for url,process in zip(urls,processes):
                deadline=time.monotonic()+30
                while True:
                    try:
                        if client.get(url+'/health').status_code==200:break
                    except httpx.HTTPError:pass
                    if process.poll() is not None or time.monotonic()>deadline:raise AssertionError('API process did not start; inspect test server logs')
                    time.sleep(.1)
            for i in range(8):assert client.get(urls[i%2]+'/documents',headers={'X-API-Key':'bogus-'+str(i)}).status_code==401
            statuses=[client.get(urls[i%2]+'/documents',headers={'X-API-Key':tokens[0]}).status_code for i in range(4)]
            assert statuses==[200,200,200,429]
            response=client.get(urls[1]+'/documents',headers={'X-API-Key':tokens[1],'X-Tenant-ID':'alpha'})
            assert response.status_code==200 and 'same-file' not in response.text
            assert client.get(urls[0]+'/whoami',headers={'X-API-Key':tokens[1],'X-Forwarded-For':'198.51.100.1'}).status_code==200
            assert client.get(urls[1]+'/whoami',headers={'X-API-Key':tokens[1],'X-Forwarded-For':'198.51.100.2'}).status_code==429
    finally:
        for process in processes:process.terminate()
        for process in processes:
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:process.kill();process.wait(timeout=5)


def test_terminated_claimant_is_recovered_by_another_process(scopes,tmp_path):
    import subprocess
    import sys
    import time
    job=IngestionQueue(scopes[0]).enqueue('fixture.txt')
    env={**os.environ,'DOCMIND_TEST_POSTGRES_URL':scopes[0].dsn}
    code="from docmind.jobs import IngestionQueue; from docmind.postgres import DatabaseScope; import os; q=IngestionQueue(DatabaseScope(os.environ['DOCMIND_TEST_POSTGRES_URL'],'alpha')); assert q._claim(lease_seconds=.3); os._exit(137)"
    child=subprocess.run([sys.executable,'-c',code],env=env,timeout=20)
    assert child.returncode!=0
    time.sleep(.4)
    recovered=IngestionQueue(scopes[0])._claim()
    assert recovered and recovered[0]['job_id']==job


def test_deleted_sources_cannot_be_recorded_as_new_answers(scopes):
    from docmind.models import SearchResult
    from docmind.services import erase_document
    store=CorpusStore(scopes[0]);chunk,_=add(store)
    response=AgentResponse('300',[SearchResult(chunk,1,'dense')],[],'temperature?',1)
    assert erase_document(chunk.file_id,store=store)
    with pytest.raises(ValueError,match='Corpus changed'):ReviewStore(scopes[0]).record(response,'reader')


def test_checkpoint_identity_remains_tenant_bound_for_identical_corpora(scopes):
    left,right=[CorpusStore(scope) for scope in scopes]
    add(left);add(right)
    assert left.content_identity()!=right.content_identity()


def test_runtime_cannot_migrate_or_disable_row_policies(scopes):
    statements=['ALTER TABLE chunks DISABLE ROW LEVEL SECURITY','CREATE TABLE docmind.runtime_escape(value TEXT)','SET ROLE postgres']
    for statement in statements:
        with pytest.raises(Exception):
            with connect(scopes[0]) as conn:conn.execute(statement)
