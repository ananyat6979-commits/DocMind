"""Request-owned accounting and bounded async inference adapters."""
import asyncio
from dataclasses import dataclass, field
import threading
import time
import weakref
from typing import Any


class DeadlineExceeded(TimeoutError):
    pass


class CapacityExceeded(RuntimeError):
    pass


_LOCK = threading.Lock()
_LIMITS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def limit(name, capacity):
    loop = asyncio.get_running_loop()
    with _LOCK:
        values = _LIMITS.setdefault(loop, {})
        if name not in values:values[name] = asyncio.Semaphore(capacity)
        return values[name]


async def cpu_work(function, *args, **kwargs):
    semaphore = limit("cpu", 2)
    await semaphore.acquire()
    task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    def finished(completed):
        semaphore.release()
        if not completed.cancelled():completed.exception()
    task.add_done_callback(finished)
    # A deadline cancels the waiter, never pretends the native worker stopped.
    return await asyncio.shield(task)


@dataclass
class CompletionSession:
    backend: Any
    deadline: float
    records: list[dict] = field(default_factory=list)

    async def complete(self, messages, **kwargs):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:raise DeadlineExceeded("Query deadline exceeded")
        started = time.monotonic()
        record: dict[str, Any] = {"status": "error", "input_tokens": None, "output_tokens": None}
        before = len(getattr(self.backend, "call_records", []))
        try:
            async with asyncio.timeout(remaining):
                asynchronous = getattr(self.backend, "acomplete", None)
                if asynchronous is not None:
                    result = await asynchronous(messages, deadline=self.deadline, **kwargs)
                    usage = getattr(self.backend, "last_usage", {})
                    record.update(usage)
                else:
                    def invoke():
                        # Local model state is shared, while network I/O is never locked here.
                        guard = getattr(self.backend, "inference_lock", None)
                        if guard is not None:
                            with guard:
                                if time.monotonic() >= self.deadline:raise DeadlineExceeded("Query deadline exceeded")
                                return self.backend.complete(messages, deadline=self.deadline, **kwargs)
                        return self.backend.complete(messages, **kwargs)
                    result = await asyncio.to_thread(invoke)
            record["status"] = "completed"
            return result
        except TimeoutError as exc:
            raise DeadlineExceeded("Query deadline exceeded") from exc
        finally:
            record["elapsed_seconds"] = time.monotonic() - started
            attempts = getattr(self.backend, "call_records", [])[before:]
            if attempts:
                self.records.extend(dict(attempt) for attempt in attempts)
            else:
                self.records.append(record)

    @property
    def calls(self):return len(self.records)

    def usage(self):
        return {"prompt_tokens": sum(r["input_tokens"] for r in self.records if r["input_tokens"] is not None),
                "completion_tokens": sum(r["output_tokens"] for r in self.records if r["output_tokens"] is not None),
                "usage_unknown_calls": sum(r["input_tokens"] is None or r["output_tokens"] is None for r in self.records)}
