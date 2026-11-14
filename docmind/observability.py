"""Authenticate before quota allocation; trace every response, including rejections."""
import json
import logging
import threading
import time
import uuid
from collections import defaultdict, deque
from starlette.responses import JSONResponse
from docmind.auth import authenticate


class BodyTooLarge(Exception):
    pass


class RequestMonitor:
    def __init__(self, limit_per_minute=60, limit_per_ip=300):
        self.limit = limit_per_minute
        self.ip_limit = limit_per_ip
        self.buckets = defaultdict(deque)
        self.ip_buckets = defaultdict(deque)
        self.lock = threading.Lock()
        self.counts = defaultdict(int)

    def _quota(self, buckets, identity, limit, now):
        for old in list(buckets):
            if not buckets[old] or buckets[old][-1] <= now-60: buckets.pop(old, None)
        if identity not in buckets and len(buckets) >= 1000: return False
        bucket = buckets[identity]
        while bucket and bucket[0] <= now-60: bucket.popleft()
        if len(bucket) >= limit: return False
        bucket.append(now)
        return True

    async def __call__(self, request, call_next):
        request_id = uuid.uuid4().hex
        request.state.request_id = request_id
        started = time.perf_counter()
        response = None
        if request.url.path != "/health":
            principal, status = authenticate(request.headers.get("x-api-key"))
            if principal is None:
                response = JSONResponse({"detail": "API authentication is not configured" if status == 503 else "Invalid or missing X-API-Key"}, status_code=status)
            else:
                request.state.principal = principal
                if request.url.path != "/metrics":
                    # Only the actual peer address is used. No untrusted forwarding headers.
                    peer = request.client.host if request.client else "local"
                    with self.lock:
                        now = time.monotonic()
                        allowed = self._quota(self.buckets, principal.key_id, self.limit, now)
                        allowed = self._quota(self.ip_buckets, peer, self.ip_limit, now) and allowed
                    if not allowed: response = JSONResponse({"detail":"Authenticated request quota exceeded"},status_code=429)
        if response is None:
            try:
                length = int(request.headers.get("content-length", "0"))
                if length < 0: raise ValueError("Negative content length")
                if length > 52*1024*1024: response = JSONResponse({"detail":"Request body too large"},status_code=413)
            except ValueError:
                response = JSONResponse({"detail":"Invalid content length"},status_code=400)
        if response is None:
            original_receive = request._receive
            received = 0
            async def bounded_receive():
                nonlocal received
                message = await original_receive()
                received += len(message.get("body", b""))
                if received > 52*1024*1024: raise BodyTooLarge("Request body exceeds limit")
                return message
            request._receive = bounded_receive
            try:
                response = await call_next(request)
            except BodyTooLarge:
                response = JSONResponse({"detail":"Invalid or oversized request body"},status_code=413)
            except Exception:
                logging.getLogger(__name__).exception("Request failed; request_id=%s",request_id)
                response = JSONResponse({"detail":"Request failed; consult server logs"},status_code=500)
        response.headers["X-Request-ID"] = request_id
        with self.lock: self.counts[str(response.status_code)] += 1
        logging.getLogger("docmind.requests").info(json.dumps({"request_id":request_id,"method":request.method,
            "route":getattr(request.scope.get("route"),"path",request.url.path),"status":response.status_code,
            "latency_ms":round((time.perf_counter()-started)*1000,2)}))
        return response
