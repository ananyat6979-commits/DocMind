"""Bounded Unix-socket transport. A missing isolated parser fails closed."""
import json
import os
import socket
import struct
from pathlib import Path
from docmind.models import Document

MAX_INPUT = 50 * 1024 * 1024
MAX_OUTPUT = 48 * 1024 * 1024

class ParserUnavailable(RuntimeError):
    pass

def receive_exact(stream, count):
    pieces = []
    remaining = count
    while remaining:
        part = stream.recv(min(remaining, 65536))
        if not part:
            raise ValueError("Truncated parser response")
        pieces.append(part)
        remaining -= len(part)
    return b"".join(pieces)

def parse_remote(path: Path, expected_language: str | None = None) -> list[Document]:
    endpoint = os.getenv("DOCMIND_PARSER_SOCKET", "/run/docmind-parser/parser.sock")
    payload = path.read_bytes()
    if len(payload) > MAX_INPUT:
        raise ValueError("Parser input exceeds 50 MiB")
    header = json.dumps({"suffix": path.suffix.lower(), "size": len(payload),
                         "expected_language": expected_language}).encode()
    try:
        with socket.socket(getattr(socket, "AF_UNIX"), socket.SOCK_STREAM) as stream:
            stream.settimeout(65)
            stream.connect(endpoint)
            stream.sendall(struct.pack("!I", len(header)) + header + payload)
            size = struct.unpack("!I", receive_exact(stream, 4))[0]
            if size > MAX_OUTPUT:
                raise ValueError("Parser output exceeds safety limit")
            result = json.loads(receive_exact(stream, size))
    except (OSError, AttributeError) as exc:
        raise ParserUnavailable("Isolated parser is unavailable; native parsing requires an explicit trusted-input override") from exc
    if not isinstance(result, dict) or not result.get("ok"):
        raise ValueError("Document rejected by isolated parser")
    documents = [Document(**item) for item in result["documents"]]
    for document in documents:
        document.source = str(path.resolve())
        document.metadata["filename"] = path.name
    return documents
