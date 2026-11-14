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
    registry.write_text(json.dumps([{'tenant_id':tenant,'actor_id':'reader','key_sha256':hashlib.sha256(token.encode()).hexdigest(),'roles':['reader']} for tenant,token in zip(('alpha','beta'),tokens)]))
    monkeypatch.setenv('DOCMIND_DATABASE_URL',scopes[0].dsn);monkeypatch.setenv('DOCMIND_CREDENTIALS_FILE',str(registry))
    add(CorpusStore(scopes[0]))
    with TestClient(app) as client:
        left=client.get('/documents',headers={'X-API-Key':tokens[0]})
        right=client.get('/documents',headers={'X-API-Key':tokens[1],'X-Tenant-ID':'alpha'})
        assert left.status_code==right.status_code==200
        assert 'same-file' in left.text and 'same-file' not in right.text
        assert client.get('/documents',headers={'X-API-Key':'invalid'}).status_code==401
