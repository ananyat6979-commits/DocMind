"""Server-derived principals. Shared credentials are never individual identities."""
from dataclasses import dataclass
import hashlib
import hmac
import os
import json
import re
from pathlib import Path


@dataclass(frozen=True)
class Principal:
    actor_id: str
    key_id: str
    roles: frozenset[str] = frozenset({"reader", "ingestor"})
    tenant_id: str = "default"


def authenticate(token: str | None) -> tuple[Principal | None, int]:
    credential_path = os.getenv("DOCMIND_CREDENTIALS_FILE")
    if credential_path:
        try:
            path=Path(credential_path)
            if path.stat().st_size>65536: return None,503
            records=json.loads(path.read_text(encoding="utf-8-sig"))
            if not isinstance(records,list) or not 1<=len(records)<=200: return None,503
            seen_actors=set();seen_keys=set()
            candidate=None
            token_hash=hashlib.sha256((token or "").encode()).hexdigest()
            for record in records:
                actor=record["actor_id"];key=record["key_sha256"];roles=record["roles"]
                tenant=record.get("tenant_id","default")
                if not isinstance(tenant,str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}",tenant): return None,503
                if os.getenv("DOCMIND_DATABASE_URL") and "tenant_id" not in record:return None,503
                if not isinstance(actor,str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}",actor): return None,503
                if not isinstance(key,str) or not re.fullmatch(r"[a-f0-9]{64}",key): return None,503
                if (tenant,actor) in seen_actors or key in seen_keys: return None,503
                if not isinstance(roles,list) or not set(roles)<={"reader","ingestor","reviewer","admin"}: return None,503
                seen_actors.add((tenant,actor));seen_keys.add(key)
                if token and hmac.compare_digest(token_hash,key): candidate=Principal(actor,key,frozenset(roles),tenant)
            return (candidate,200) if candidate else (None,401)
        except (OSError,ValueError,KeyError,TypeError): return None,503
    if os.getenv("DOCMIND_DATABASE_URL"):return None,503
    expected = os.getenv("DOCMIND_API_KEY", "")
    if not expected: return None, 503
    if not token or not hmac.compare_digest(token.encode(), expected.encode()): return None, 401
    return Principal("shared-api", hashlib.sha256(token.encode()).hexdigest()), 200
