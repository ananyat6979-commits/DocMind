"""Explicit tenant-scoped PostgreSQL adapter. Runtime credentials cannot migrate."""
from contextlib import contextmanager
from dataclasses import dataclass, replace, field
import re


@dataclass(frozen=True)
class DatabaseScope:
    dsn: str = field(repr=False)
    tenant: str
    lease_job: str | None = None
    lease_token: str | None = None

    def __post_init__(self):
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}",self.tenant):raise ValueError("Invalid tenant scope")
        if not self.dsn.startswith(("postgresql://","postgres://")):raise ValueError("PostgreSQL URL required")

    def fenced(self,job,token):return replace(self,lease_job=job,lease_token=token)


class LeaseLost(RuntimeError):
    pass


class Record(dict):
    def __getitem__(self,key):
        return list(self.values())[key] if isinstance(key,int) else super().__getitem__(key)
    def __iter__(self):return iter(self.values())
    def keys(self):return super().keys()


def row_factory(cursor):
    names=[c.name for c in cursor.description] if cursor.description else []
    return lambda values: Record(zip(names,values))


# Existing storage statements contain no escaped or dollar-quoted SQL literals.
# Tokenize quoted strings before converting SQLite bind markers and the user column.
def translate(sql):
    parts=re.split(r"('(?:''|[^'])*'|\"(?:\"\"|[^\"])*\")",sql)
    for i in range(0,len(parts),2):
        parts[i]=re.sub(r"\buser\b",'"user"',parts[i],flags=re.I).replace("?","%s")
    result="".join(parts)
    match=re.match(r"\s*INSERT OR REPLACE INTO (meta|documents|reviews)\(([^)]+)\)(.*)",result,re.I|re.S)
    if match:
        table,columns,tail=match.groups();names=[c.strip() for c in columns.split(',')]
        primary={"meta":"key","documents":"file_id","reviews":"response_id"}[table.lower()]
        updates=','.join(f'{c}=EXCLUDED.{c}' for c in names if c!=primary)
        result=f"INSERT INTO {table}({columns}){tail} ON CONFLICT (tenant_id,{primary}) DO UPDATE SET {updates}"
    return result


class Connection:
    def __init__(self,raw,scope):self.raw=raw;self.scope=scope
    def execute(self,sql,params=()):
        if sql=="BEGIN IMMEDIATE":
            return self.raw.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s,0))",("queue:"+self.scope.tenant,))
        match=re.fullmatch(r"PRAGMA table_info\((\w+)\)",sql)
        if match:
            return self.raw.execute("SELECT column_name AS name FROM information_schema.columns WHERE table_schema='docmind' AND table_name=%s",(match[1],))
        if "FROM sqlite_master" in sql:
            return self.raw.execute("SELECT table_name AS name FROM information_schema.tables WHERE table_schema='docmind'")
        return self.raw.execute(translate(sql),params)
    def executemany(self,sql,params):
        if sql.startswith("INSERT INTO chunks("):
            import numpy as np
            import json
            sql=sql.replace("metadata, embedding)","metadata, embedding, embedding_vector)").replace("?, ?, ?, ?, ?, ?, ?, ?, ?)","?, ?, ?, ?, ?, ?, ?, ?, ?, ?::vector)")
            params=[(*row,json.dumps(np.frombuffer(row[-1],dtype="float32").tolist())) for row in params]
        with self.raw.cursor() as cursor:cursor.executemany(translate(sql),params)
    def executescript(self,script):
        names=re.findall(r"CREATE TABLE IF NOT EXISTS (\w+)",script)
        for name in names:
            if not self.raw.execute("SELECT to_regclass(%s) AS name",('docmind.'+name,)).fetchone()["name"]:
                raise RuntimeError("Database migration required before runtime startup")
    def corpus_lock(self,shared=False):
        function="pg_advisory_xact_lock_shared" if shared else "pg_advisory_xact_lock"
        self.raw.execute(f"SELECT {function}(hashtextextended(%s,0))",("corpus:"+self.scope.tenant,))
    def check_lease(self):
        if not self.scope.lease_job:return
        row=self.raw.execute("SELECT 1 FROM ingestion_jobs WHERE job_id=%s AND status='running' AND lease_token=%s AND lease_until>EXTRACT(EPOCH FROM clock_timestamp()) FOR SHARE",
            (self.scope.lease_job,self.scope.lease_token)).fetchone()
        if not row:raise LeaseLost("Ingestion lease expired or replaced")


@contextmanager
def connect(scope):
    import psycopg
    with psycopg.connect(scope.dsn,row_factory=row_factory,connect_timeout=10,prepare_threshold=None) as raw:
        role=raw.execute("SELECT rolsuper,rolbypassrls FROM pg_roles WHERE rolname=current_user").fetchone()
        if role["rolsuper"] or role["rolbypassrls"]:raise RuntimeError("Runtime database role must not bypass row policies")
        raw.execute("SELECT set_config('search_path','docmind,public,pg_catalog',true)")
        raw.execute("SELECT set_config('docmind.tenant',%s,true)",(scope.tenant,))
        raw.execute("SELECT set_config('statement_timeout','15000',true)")
        raw.execute("SELECT set_config('lock_timeout','5000',true)")
        conn=Connection(raw,scope);conn.check_lease()
        yield conn
        conn.check_lease()
