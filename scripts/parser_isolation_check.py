"""Run inside the confined parser image, without application mounts."""
import os
from pathlib import Path
import socket
import sys
from docmind.ingestion.parser_service import run_bounded


def main():
    try: socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    except PermissionError: pass
    else: raise AssertionError("Internet-family sockets must be denied by seccomp")
    for secret in ("/data/docmind.db","/run/secrets/application","/app/.env"):
        assert not Path(secret).exists(),"Application secret or state is mounted in parser"
    os.environ["DOCMIND_ADMIN_PIN"]="sentinel"
    os.environ["DOCMIND_SECRETS_FILE"]="/run/secrets/application"
    result=run_bounded([sys.executable,"-c",'import json,os;print(json.dumps({k:v for k,v in os.environ.items() if "DOCMIND_ADMIN" in k or "SECRETS" in k}))'])
    assert result=={},"Worker inherited application credentials"
    try:
        run_bounded([sys.executable,"-c","from docmind.ingestion.parse_worker import bound_resources;bound_resources();bytearray(1024**3)"])
    except ValueError: pass
    else: raise AssertionError("Memory exhaustion was not bounded")
    assert run_bounded([sys.executable,"-c",'print("{}")'])=={},"Service did not recover after worker failure"
    print("Network, secret isolation, memory limit and recovery checks passed")

if __name__=="__main__":main()
