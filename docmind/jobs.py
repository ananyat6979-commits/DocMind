"""Bounded ingestion queue: local SQLite or shared PostgreSQL leases."""
import json
import threading
import time
import uuid
from docmind.db import connect
from docmind.ingestion.pipeline import ingest

SCHEMA = """CREATE TABLE IF NOT EXISTS ingestion_jobs (
    job_id TEXT PRIMARY KEY, path TEXT NOT NULL, status TEXT NOT NULL,
    created_at REAL NOT NULL, updated_at REAL NOT NULL, result TEXT, error TEXT);
"""


class IngestionQueue:
    def __init__(self, db_path=None):
        self.db_path = db_path
        self.stop_event = threading.Event()
        self.thread = None
        with connect(db_path) as conn:
            conn.executescript(SCHEMA)
            if "owner" not in {row["name"] for row in conn.execute("PRAGMA table_info(ingestion_jobs)")}:
                conn.execute("ALTER TABLE ingestion_jobs ADD COLUMN owner TEXT NOT NULL DEFAULT 'shared-api'")

            if "options" not in {row["name"] for row in conn.execute("PRAGMA table_info(ingestion_jobs)")}:
                conn.execute("ALTER TABLE ingestion_jobs ADD COLUMN options TEXT NOT NULL DEFAULT '{}'")

    def enqueue(self, path, owner="shared-api", expected_language=None, encoding=None):
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            count = conn.execute("SELECT COUNT(*) FROM ingestion_jobs WHERE status IN ('queued','running')").fetchone()[0]
            if count >= 20:
                raise ValueError("Ingestion queue is full")
            job = uuid.uuid4().hex
            conn.execute("INSERT INTO ingestion_jobs(job_id,path,status,created_at,updated_at,result,error,owner,options) VALUES (?,?, 'queued', ?,?,NULL,NULL,?,?)",
                         (job, str(path), time.time(), time.time(), owner,json.dumps({"expected_language":expected_language,"encoding":encoding})))
        return job

    def get(self, job):
        with connect(self.db_path) as conn:
            row = conn.execute("SELECT job_id,status,created_at,updated_at,result,error,owner FROM ingestion_jobs WHERE job_id=?", (job,)).fetchone()
        return dict(row) if row else None

    def cancel(self, job):
        with connect(self.db_path) as conn:
            return bool(conn.execute("UPDATE ingestion_jobs SET status='cancelled',updated_at=? WHERE job_id=? AND status='queued'", (time.time(),job)).rowcount)

    def run_one(self):
        from docmind.postgres import DatabaseScope
        if isinstance(self.db_path,DatabaseScope):return self._run_leased()
        from docmind.store import CorpusStore
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT job_id,path,options FROM ingestion_jobs WHERE status='queued' ORDER BY created_at LIMIT 1").fetchone()
            if not row:
                return False
            conn.execute("UPDATE ingestion_jobs SET status='running',updated_at=? WHERE job_id=?", (time.time(), row["job_id"]))
        try:
            report = ingest(row["path"], store=CorpusStore(self.db_path),**json.loads(row["options"]))
            result = json.dumps({"file_ids":report.file_ids,"chunks_added":report.chunks_added})
            status, error = "completed", None
        except Exception as exc:
            status, result, error = "failed", None, type(exc).__name__
        with connect(self.db_path) as conn:
            conn.execute("UPDATE ingestion_jobs SET status=?,result=?,error=?,updated_at=? WHERE job_id=?",
                         (status,result,error,time.time(),row["job_id"]))
        from docmind.uploads import cleanup
        cleanup(row["path"], self.db_path)
        return True

    def _claim(self,lease_seconds=60):
        token=uuid.uuid4().hex
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("UPDATE ingestion_jobs SET status=CASE WHEN attempts>=3 THEN 'failed' ELSE 'queued' END,error='Worker lease expired',lease_token=NULL WHERE status='running' AND lease_until<=EXTRACT(EPOCH FROM clock_timestamp())")
            row=conn.execute("SELECT job_id,path,options FROM ingestion_jobs WHERE status='queued' ORDER BY created_at LIMIT 1 FOR UPDATE SKIP LOCKED").fetchone()
            if row is None:return None
            conn.execute("UPDATE ingestion_jobs SET status='running',lease_token=?,lease_until=EXTRACT(EPOCH FROM clock_timestamp())+?,attempts=attempts+1,updated_at=EXTRACT(EPOCH FROM clock_timestamp()) WHERE job_id=?",(token,lease_seconds,row['job_id']))
        return dict(row),token

    def _run_leased(self):
        from docmind.store import CorpusStore
        claimed=self._claim()
        if claimed is None:return False
        row,token=claimed;done=threading.Event()
        def heartbeat():
            while not done.wait(10):
                try:
                    with connect(self.db_path) as conn:
                        changed=conn.execute("UPDATE ingestion_jobs SET lease_until=EXTRACT(EPOCH FROM clock_timestamp())+60 WHERE job_id=? AND lease_token=? AND status='running' AND lease_until>EXTRACT(EPOCH FROM clock_timestamp())",(row['job_id'],token)).rowcount
                    if not changed:return
                except Exception:return
        pulse=threading.Thread(target=heartbeat,daemon=True);pulse.start()
        try:
            report=ingest(row['path'],store=CorpusStore(self.db_path.fenced(row['job_id'],token)),**json.loads(row['options']))
            status,result,error='completed',json.dumps({'file_ids':report.file_ids,'chunks_added':report.chunks_added}),None
        except Exception as exc:
            status,result,error='failed',None,type(exc).__name__
        finally:done.set();pulse.join(timeout=1)
        with connect(self.db_path) as conn:
            changed=conn.execute("UPDATE ingestion_jobs SET status=?,result=?,error=?,updated_at=EXTRACT(EPOCH FROM clock_timestamp()),lease_token=NULL WHERE job_id=? AND lease_token=? AND status='running' AND lease_until>EXTRACT(EPOCH FROM clock_timestamp())",(status,result,error,row['job_id'],token)).rowcount
        if changed:
            from docmind.uploads import cleanup
            cleanup(row['path'],self.db_path)
        return True

    def start(self):
        from docmind.postgres import DatabaseScope
        if not isinstance(self.db_path,DatabaseScope):
            with connect(self.db_path) as conn:
                conn.execute("UPDATE ingestion_jobs SET status='failed',error='Interrupted during service restart' WHERE status='running'")
        def loop():
            next_sweep=0.0
            while not self.stop_event.is_set():
                try:
                    if time.monotonic()>=next_sweep:
                        from docmind.uploads import sweep
                        sweep(self.db_path)
                        next_sweep=time.monotonic()+60
                    if not self.run_one(): self.stop_event.wait(1 if isinstance(self.db_path,DatabaseScope) else .25)
                except Exception:
                    import logging
                    logging.getLogger(__name__).exception("Ingestion worker failed; retrying after backoff")
                    self.stop_event.wait(1)
        self.thread = threading.Thread(target=loop, name="ingestion-worker", daemon=True)
        self.thread.start()

    def close(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=5)
