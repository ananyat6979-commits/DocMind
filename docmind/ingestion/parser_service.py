"""Single-client bounded parser transport. Deploy only inside the confined container."""
import json
import os
from pathlib import Path
import selectors
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import time
from docmind.ingestion.parser_client import MAX_INPUT, MAX_OUTPUT, receive_exact
from docmind.ingestion.loader import parser_environment


def run_bounded(arguments, timeout=60):
    process=subprocess.Popen(arguments,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,
                             env=parser_environment(),start_new_session=True)
    output=bytearray()
    try:
        assert process.stdout is not None
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout,selectors.EVENT_READ)
            deadline=time.monotonic()+timeout
            while True:
                remaining=deadline-time.monotonic()
                if remaining<=0: raise TimeoutError("Parser deadline exceeded")
                if not selector.select(min(remaining,1)): continue
                chunk=os.read(process.stdout.fileno(),65536)
                if not chunk: break
                if len(output)+len(chunk)>MAX_OUTPUT: raise ValueError("Parser output too large")
                output.extend(chunk)
        process.wait(timeout=max(.01,deadline-time.monotonic()))
        if process.returncode: raise ValueError("Parser rejected document or exceeded resource limits")
        return json.loads(output)
    finally:
        if process.poll() is None:
            getattr(os,"killpg")(process.pid,getattr(signal,"SIGKILL"))
            process.wait()
        if process.stdout is not None: process.stdout.close()


class ReceiveDeadline:
    def __init__(self, stream, deadline):
        self.stream, self.deadline = stream, deadline
    def recv(self, count):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:raise TimeoutError("Total parser receive deadline exceeded")
        if hasattr(self.stream, "settimeout"):self.stream.settimeout(remaining)
        return self.stream.recv(count)


def handle(stream, receive_seconds=10, total_seconds=75):
    total_deadline = time.monotonic() + total_seconds
    stream = ReceiveDeadline(stream, min(total_deadline, time.monotonic() + receive_seconds))
    size=struct.unpack("!I",receive_exact(stream,4))[0]
    if not 1<=size<=4096: raise ValueError("Invalid header length")
    header=json.loads(receive_exact(stream,size))
    suffix=header.get("suffix")
    size=header.get("size")
    if suffix not in (".pdf",".docx") or type(size) is not int or not 0<size<=MAX_INPUT:
        raise ValueError("Invalid document envelope")
    language=header.get("expected_language")
    if language not in (None,"en","hi","fr","de","fi"): raise ValueError("Invalid language hint")
    with tempfile.TemporaryDirectory(prefix="parse-") as directory:
        path=Path(directory)/("input"+suffix)
        remaining=size
        with path.open("wb") as destination:
            while remaining:
                chunk=stream.recv(min(remaining,65536))
                if not chunk: raise ValueError("Truncated input")
                destination.write(chunk);remaining-=len(chunk)
        remaining_time = total_deadline - time.monotonic()
        if remaining_time <= 0:raise TimeoutError("Total parser connection deadline exceeded")
        documents=run_bounded([sys.executable,"-m","docmind.ingestion.parse_worker",str(path),language or ""], timeout=min(60, remaining_time))
        return {"ok":True,"documents":documents}


def main():
    if os.name!="posix": raise RuntimeError("The parser service requires a Linux isolation boundary")
    endpoint=Path(os.getenv("DOCMIND_PARSER_SOCKET","/run/docmind-parser/parser.sock"))
    endpoint.parent.mkdir(parents=True,exist_ok=True)
    endpoint.unlink(missing_ok=True)
    os.umask(0o007)
    with socket.socket(getattr(socket,"AF_UNIX"),socket.SOCK_STREAM) as server:
        server.bind(str(endpoint));os.chmod(endpoint,0o660);server.listen(4)
        while True:
            stream,_=server.accept()
            with stream:
                connection_deadline=time.monotonic()+75
                stream.settimeout(10)
                try: result=handle(stream)
                except Exception: result={"ok":False,"error":"Document rejected or resource limit exceeded"}
                encoded=json.dumps(result,ensure_ascii=True).encode()
                if len(encoded)>MAX_OUTPUT: encoded=b'{"ok":false,"error":"Output limit exceeded"}'
                stream.settimeout(max(.01, connection_deadline-time.monotonic()))
                try: stream.sendall(struct.pack("!I",len(encoded))+encoded)
                except OSError: pass

if __name__=="__main__": main()
