"""
Usage event log for the pilot. One row per event, payload as JSON.

Event types written by the app:
  session_start | upload | query | feedback | manual_estimate | review_decision
No document text is logged by default; only counts, latencies and flags.
"""
import json
import time
from typing import Dict, List, Optional

from docmind.db import connect

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          REAL NOT NULL,
    user        TEXT NOT NULL,
    session_id  TEXT NOT NULL,
    type        TEXT NOT NULL,
    payload     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(type, ts);
CREATE INDEX IF NOT EXISTS idx_events_user ON events(user, ts);
"""


class EventLog:
    def __init__(self, db_path=None):
        self.db_path = db_path
        with connect(self.db_path) as conn:
            conn.executescript(SCHEMA)

    def log(self, type: str, user: str = "anonymous", session_id: str = "", **payload) -> None:
        try:
            with connect(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO events(ts, user, session_id, type, payload) VALUES (?, ?, ?, ?, ?)",
                    (time.time(), user or "anonymous", session_id, type, json.dumps(payload, default=str)),
                )
        except Exception:  # telemetry must never break the product
            pass

    def rows(self, type: Optional[str] = None) -> List[Dict]:
        sql, params = "SELECT * FROM events", ()
        if type:
            sql, params = sql + " WHERE type=?", (type,)
        with connect(self.db_path) as conn:
            rows = conn.execute(sql + " ORDER BY ts", params).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["payload"] = json.loads(d["payload"])
            out.append(d)
        return out
