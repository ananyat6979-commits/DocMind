import socket
import threading
import time
import pytest
from docmind.ingestion.parser_service import handle


def test_drip_sender_cannot_reset_total_receive_deadline():
    reader, writer = socket.socketpair()
    def drip():
        try:
            for _ in range(30):
                writer.sendall(b"\x00");time.sleep(.015)
        except OSError:pass
    thread = threading.Thread(target=drip, daemon=True)
    start=time.monotonic();thread.start()
    try:
        with pytest.raises((TimeoutError, ValueError)):
            handle(reader, receive_seconds=.035)
        assert time.monotonic()-start < .3
    finally:reader.close();writer.close();thread.join(timeout=1)
