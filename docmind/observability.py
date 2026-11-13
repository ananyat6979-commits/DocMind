"""Request tracing, bounded quotas and JSON request logs without question text."""
import hashlib
import json
import logging
import threading
import time
import uuid
from collections import defaultdict, deque
from starlette.responses import JSONResponse


class BodyTooLarge(Exception):
    pass


class RequestMonitor:
    def __init__(self, limit_per_minute=60):
        self.limit = limit_per_minute
        self.buckets = defaultdict(deque)
        self.lock = threading.Lock()
        self.counts = defaultdict(int)

    async def __call__(self, request, call_next):
        request_id = uuid.uuid4().hex
        started = time.perf_counter()
        token = request.headers.get("x-api-key", "anonymous")
        identity = hashlib.sha256(token.encode()).hexdigest()
        if request.url.path not in ("/health", "/metrics"):
            with self.lock:
                now = time.monotonic()
                for old_key in list(self.buckets):
                    if not self.buckets[old_key] or self.buckets[old_key][-1] <= now-60:
                        self.buckets.pop(old_key, None)
                if identity not in self.buckets and len(self.buckets) >= 1000:
                    return JSONResponse({"detail":"Request capacity exceeded"},status_code=429)
                bucket = self.buckets[identity]
                while bucket and bucket[0] <= now-60:
                    bucket.popleft()
                if len(bucket) >= self.limit:
                    return JSONResponse({"detail":"Per-key request quota exceeded"},status_code=429)
                bucket.append(now)
        try:
            if int(request.headers.get("content-length","0")) > 52*1024*1024:
                return JSONResponse({"detail":"Request body too large"},status_code=413)
        except ValueError:
            return JSONResponse({"detail":"Invalid content length"},status_code=400)
        # Starlette parses uploads incrementally; enforce actual stream length as well.
        original_receive = request._receive
        received = 0
        async def bounded_receive():
            nonlocal received
            message = await original_receive()
            received += len(message.get("body", b""))
            if received > 52*1024*1024:
                raise BodyTooLarge("Request body exceeds limit")
            return message
        request._receive = bounded_receive
        try:
            response = await call_next(request)
        except BodyTooLarge:
            response = JSONResponse({"detail":"Invalid or oversized request body"},status_code=413)
        response.headers["X-Request-ID"] = request_id
        path = getattr(request.scope.get("route"),"path",request.url.path)
        with self.lock:
            self.counts[str(response.status_code)] += 1
        logging.getLogger("docmind.requests").info(json.dumps({
            "request_id":request_id,"method":request.method,"route":path,
            "status":response.status_code,"latency_ms":round((time.perf_counter()-started)*1000,2)}))
        return response
