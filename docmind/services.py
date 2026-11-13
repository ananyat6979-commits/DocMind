"""Library operations shared by the UI and API."""
import json
from pathlib import Path
from docmind.config import CONFIG
from docmind.db import connect
from docmind.store import CorpusStore


def erase_document(file_id, store=None):
    store = store or CorpusStore()
    doc = store.get_document(file_id)
    if not doc:
        return False
    with connect(store.db_path) as conn:
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
    root = (CONFIG.data_dir / "uploads").resolve()
    source = Path(doc["source"]).resolve()
    if root in source.parents:
        source.unlink(missing_ok=True)
    return True
