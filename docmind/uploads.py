"""Owned upload leases and reference-aware cleanup for a single corpus process."""
from collections import Counter
from itertools import chain
from pathlib import Path
from typing import Iterator
import re
import threading
import time
from docmind.config import CONFIG
from docmind.db import connect

_LOCK = threading.RLock()
_ACTIVE: Counter[str] = Counter()
_SWEEPS: dict[str, Iterator[Path]] = {}


def acquire(path):
    with _LOCK: _ACTIVE[str(Path(path).resolve())] += 1


def cleanup(path, db_path=None, release=False, owned_document=False):
    path = Path(path).resolve()
    from docmind.tenancy import data_root
    root = (data_root(db_path) / "uploads").resolve()
    if root not in path.parents: return False
    key = str(path)
    with _LOCK:
        if release and _ACTIVE[key]:
            _ACTIVE[key] -= 1
            if not _ACTIVE[key]: del _ACTIVE[key]
        if _ACTIVE[key]: return False
        with connect(db_path) as conn:
            tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if "documents" in tables and conn.execute("SELECT 1 FROM documents WHERE source=? OR sha256=? LIMIT 1", (key,path.parent.name)).fetchone(): return False
            if "ingestion_jobs" in tables and conn.execute("SELECT 1 FROM ingestion_jobs WHERE path=? AND status IN ('queued','running') LIMIT 1",(key,)).fetchone(): return False
        if not owned_document and path.parent != root and not re.fullmatch(r"[a-f0-9]{64}",path.parent.name): return False
        if not path.is_file(): return False
        path.unlink()
        if path.parent != root:
            try: path.parent.rmdir()
            except OSError: pass
        return True


def sweep(db_path=None, min_age_seconds=3600, limit=500):
    from docmind.tenancy import data_root
    root=data_root(db_path)/"uploads"
    if not root.exists():return 0
    removed=0
    with _LOCK:
        key=str(root.resolve())
        if key not in _SWEEPS:
            if len(_SWEEPS)>=16:_SWEEPS.pop(next(iter(_SWEEPS)))
            _SWEEPS[key]=iter(chain(root.glob("*"),root.glob("*/*")))
        iterator=_SWEEPS[key]
        for _ in range(limit):
            try:path=next(iterator)
            except StopIteration:
                _SWEEPS.pop(key,None)
                break
            try:
                if path.is_file() and time.time()-path.stat().st_mtime>=min_age_seconds:
                    removed+=int(cleanup(path,db_path))
            except FileNotFoundError:continue
    return removed
