"""Server-derived principals. Shared credentials are never individual identities."""
from dataclasses import dataclass
import hashlib
import hmac
import os


@dataclass(frozen=True)
class Principal:
    actor_id: str
    key_id: str
    roles: frozenset[str] = frozenset({"reader", "ingestor"})


def authenticate(token: str | None) -> tuple[Principal | None, int]:
    expected = os.getenv("DOCMIND_API_KEY", "")
    if not expected: return None, 503
    if not token or not hmac.compare_digest(token.encode(), expected.encode()): return None, 401
    return Principal("shared-api", hashlib.sha256(token.encode()).hexdigest()), 200
