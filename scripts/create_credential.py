"""Create a personal service key in private files. The token is never printed."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import secrets


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--actor",required=True)
    parser.add_argument("--roles",nargs="+",default=["reader"],choices=["reader","ingestor","reviewer","admin"])
    parser.add_argument("--out",type=Path,default=Path.home()/".config"/"docmind"/"credentials.json")
    args=parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}",args.actor):parser.error("Invalid actor ID")
    records=json.loads(args.out.read_text(encoding="utf-8")) if args.out.exists() else []
    if any(r["actor_id"]==args.actor for r in records):parser.error("Actor already exists; revoke explicitly before replacement")
    token=secrets.token_urlsafe(32)
    records.append({"actor_id":args.actor,"key_sha256":hashlib.sha256(token.encode()).hexdigest(),"roles":sorted(set(args.roles)|{"reader"})})
    args.out.parent.mkdir(parents=True,exist_ok=True)
    private_key=args.out.parent/(args.actor+".key")
    descriptor=os.open(private_key,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(descriptor,"w",encoding="utf-8") as stream:stream.write(token+"\n")
    temporary=args.out.with_suffix(".tmp")
    descriptor=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(descriptor,"w",encoding="utf-8") as stream:
        if os.name == "posix": os.fchmod(stream.fileno(),0o640)
        json.dump(records,stream,indent=2)
    temporary.replace(args.out)
    if os.name == "posix": print("Registry group ID for DOCMIND_CREDENTIALS_GID:",args.out.stat().st_gid)
    print("Credentials updated. Read the personal key privately from",private_key)

if __name__=="__main__":main()
