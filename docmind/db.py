"""SQLite helper. One database file holds corpus, reviews and usage events."""
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from docmind.config import CONFIG


@contextmanager
def connect(db_path=None):
    path = Path(db_path or CONFIG.db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
