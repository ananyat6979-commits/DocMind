"""Library operations shared by the UI and API."""
import json
from pathlib import Path
from docmind.config import CONFIG
from docmind.db import connect
from docmind.store import CorpusStore


def erase_document(file_id, store=None):
    if store is None:
        from docmind.tenancy import database_scope
        store=CorpusStore(database_scope())
    doc = store.get_document(file_id)
    if not doc:
        return False
    with connect(store.db_path) as conn:
        from docmind.postgres import Connection
        if isinstance(conn,Connection):conn.corpus_lock()
        ids = {r[0] for r in conn.execute("SELECT chunk_id FROM chunks WHERE file_id=?", (file_id,))}
        has_reviews = conn.execute("SELECT 1 FROM sqlite_master WHERE name='reviews'").fetchone()
        if has_reviews:
            rows = conn.execute("SELECT response_id, sources_json FROM reviews").fetchall()
            for row in rows:
                sources = json.loads(row["sources_json"])
                if any(s.get("chunk_id") in ids or s.get("file_id") == file_id for s in sources):
                    conn.execute("DELETE FROM reviews WHERE response_id=?", (row["response_id"],))
        conn.execute("DELETE FROM documents WHERE file_id=?", (file_id,))
        store._bump(conn)
    # Only uploaded copies owned by the service are eligible for filesystem erasure.
    from docmind.tenancy import data_root
    root = (data_root(store.db_path) / "uploads").resolve()
    source = Path(doc["source"]).resolve()
    if root in source.parents:
        from docmind.uploads import cleanup
        cleanup(source,store.db_path,owned_document=True)
        owned=(root/doc["sha256"]).resolve()
        if owned.parent==root and owned.is_dir():
            for copy in owned.iterdir():
                if copy.is_file() and copy.resolve().parent==owned:cleanup(copy,store.db_path)
    return True
