"""Bounded, durable ingestion queue for one SQLite service process."""
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

    def enqueue(self, path):
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            count = conn.execute("SELECT COUNT(*) FROM ingestion_jobs WHERE status IN ('queued','running')").fetchone()[0]
            if count >= 20:
                raise ValueError("Ingestion queue is full")
            job = uuid.uuid4().hex
            conn.execute("INSERT INTO ingestion_jobs VALUES (?,?, 'queued', ?,?,NULL,NULL)",
                         (job, str(path), time.time(), time.time()))
        return job

    def get(self, job):
        with connect(self.db_path) as conn:
            row = conn.execute("SELECT job_id,status,created_at,updated_at,result,error FROM ingestion_jobs WHERE job_id=?", (job,)).fetchone()
        return dict(row) if row else None

    def cancel(self, job):
        with connect(self.db_path) as conn:
            return bool(conn.execute("UPDATE ingestion_jobs SET status='cancelled',updated_at=? WHERE job_id=? AND status='queued'", (time.time(),job)).rowcount)

    def run_one(self):
        from docmind.store import CorpusStore
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT job_id,path FROM ingestion_jobs WHERE status='queued' ORDER BY created_at LIMIT 1").fetchone()
            if not row:
                return False
            conn.execute("UPDATE ingestion_jobs SET status='running',updated_at=? WHERE job_id=?", (time.time(), row["job_id"]))
        try:
            report = ingest(row["path"], store=CorpusStore(self.db_path))
            result = json.dumps({"file_ids":report.file_ids,"chunks_added":report.chunks_added})
            status, error = "completed", None
        except Exception as exc:
            status, result, error = "failed", None, type(exc).__name__
        with connect(self.db_path) as conn:
            conn.execute("UPDATE ingestion_jobs SET status=?,result=?,error=?,updated_at=? WHERE job_id=?",
                         (status,result,error,time.time(),row["job_id"]))
        return True

    def start(self):
        with connect(self.db_path) as conn:
            conn.execute("UPDATE ingestion_jobs SET status='failed',error='Interrupted during service restart' WHERE status='running'")
        def loop():
            while not self.stop_event.is_set():
                if not self.run_one():
                    self.stop_event.wait(0.25)
        self.thread = threading.Thread(target=loop, name="ingestion-worker", daemon=True)
        self.thread.start()

    def close(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=5)
