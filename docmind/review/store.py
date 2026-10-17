"""
Review queue: persistent human-in-the-loop decisions.

Lifecycle of an answer:
  auto_approved            policy found no reason to review it
  pending                  policy flagged it; waiting for a reviewer
  approved | edited | rejected    reviewer decision (edited keeps the corrected text)

Approved and edited items export to a JSON testset, so every review improves the
evaluation set (the feedback loop that makes this more than a UI button).
"""
import json
import time
from typing import Dict, List, Optional

from docmind.db import connect
from docmind.models import AgentResponse

SCHEMA = """
CREATE TABLE IF NOT EXISTS reviews (
    response_id    TEXT PRIMARY KEY,
    created_at     REAL NOT NULL,
    user           TEXT NOT NULL,
    question       TEXT NOT NULL,
    answer         TEXT NOT NULL,
    sources_json   TEXT NOT NULL,
    reasons_json   TEXT NOT NULL,
    status         TEXT NOT NULL,
    reviewer       TEXT,
    edited_answer  TEXT,
    note           TEXT,
    decided_at     REAL
);
CREATE INDEX IF NOT EXISTS idx_reviews_status ON reviews(status, created_at);
"""

DECISIONS = {"approved", "edited", "rejected"}


def _source_payload(resp: AgentResponse) -> List[Dict]:
    cited = {s.chunk.chunk_id for s in resp.cited_sources}
    return [
        {
            "chunk_id": s.chunk.chunk_id,
            "filename": s.chunk.filename(),
            "page": s.chunk.metadata.get("page"),
            "section": s.chunk.metadata.get("section"),
            "content": s.chunk.content,
            "score": round(float(s.score), 4),
            "cited": s.chunk.chunk_id in cited,
        }
        for s in resp.sources
    ]


class ReviewStore:
    def __init__(self, db_path=None):
        self.db_path = db_path
        with connect(self.db_path) as conn:
            conn.executescript(SCHEMA)

    def record(self, resp: AgentResponse, user: str = "anonymous") -> str:
        status = "pending" if resp.needs_review else "auto_approved"
        with connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO reviews(response_id, created_at, user, question, answer, sources_json,"
                " reasons_json, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (resp.response_id, time.time(), user, resp.query, resp.answer,
                 json.dumps(_source_payload(resp)), json.dumps(resp.review_reasons), status),
            )
        return status

    def get(self, response_id: str) -> Optional[Dict]:
        with connect(self.db_path) as conn:
            row = conn.execute("SELECT * FROM reviews WHERE response_id=?", (response_id,)).fetchone()
        return self._decode(row) if row else None

    def status_of(self, response_id: str) -> Optional[str]:
        item = self.get(response_id)
        return item["status"] if item else None

    def pending(self, limit: int = 50) -> List[Dict]:
        with connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM reviews WHERE status='pending' ORDER BY created_at LIMIT ?", (limit,)
            ).fetchall()
        return [self._decode(r) for r in rows]

    def decide(self, response_id: str, decision: str, reviewer: str,
               edited_answer: Optional[str] = None, note: str = "") -> bool:
        if decision not in DECISIONS:
            raise ValueError(f"decision must be one of {sorted(DECISIONS)}")
        if decision == "edited" and not (edited_answer and edited_answer.strip()):
            raise ValueError("edited decision requires edited_answer")
        with connect(self.db_path) as conn:
            cur = conn.execute(
                "UPDATE reviews SET status=?, reviewer=?, edited_answer=?, note=?, decided_at=?"
                " WHERE response_id=? AND status='pending'",
                (decision, reviewer, edited_answer if decision == "edited" else None, note, time.time(), response_id),
            )
            return cur.rowcount > 0

    def stats(self) -> Dict[str, int]:
        with connect(self.db_path) as conn:
            rows = conn.execute("SELECT status, COUNT(*) AS n FROM reviews GROUP BY status").fetchall()
        out = {r["status"]: r["n"] for r in rows}
        out["total"] = sum(out.values())
        return out

    def export_testset(self, path: str) -> int:
        """Write reviewer-approved/edited Q&A (with their evidence) as a testset."""
        with connect(self.db_path) as conn:
            rows = conn.execute("SELECT * FROM reviews WHERE status IN ('approved','edited')").fetchall()
        items = []
        for r in rows:
            d = self._decode(r)
            items.append({
                "id": d["response_id"],
                "question": d["question"],
                "ground_truth": d["edited_answer"] or d["answer"],
                "answerable": not any(x == "abstained" for x in d["reasons"]),
                "contexts": [s["content"] for s in d["sources"] if s.get("cited")] or [s["content"] for s in d["sources"]],
                "origin": "human_reviewed",
            })
        with open(path, "w", encoding="utf-8") as f:
            json.dump(items, f, indent=2, ensure_ascii=False)
        return len(items)

    @staticmethod
    def _decode(row) -> Dict:
        d = dict(row)
        d["sources"] = json.loads(d.pop("sources_json"))
        d["reasons"] = json.loads(d.pop("reasons_json"))
        return d
