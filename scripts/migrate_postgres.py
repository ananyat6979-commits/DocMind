"""Provision the single-host multi-replica database using separate administrator credentials."""
import argparse
import os
from pathlib import Path
import re
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def migrate(admin_dsn,runtime_role,runtime_password,tenants):
    import psycopg
    from psycopg import sql
    from docmind.store import SCHEMA as corpus
    from docmind.review.store import SCHEMA as reviews
    from docmind.telemetry.events import SCHEMA as events
    from docmind.jobs import SCHEMA as jobs
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,47}",runtime_role):raise ValueError("Invalid runtime role")
    if len(runtime_password)<24:raise ValueError("Runtime password must have at least 24 characters")
    if not tenants or any(not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}",t) for t in tenants):raise ValueError("Explicit valid tenant list required")
    with psycopg.connect(admin_dsn) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute("CREATE SCHEMA IF NOT EXISTS docmind")
        conn.execute("REVOKE ALL ON SCHEMA public FROM PUBLIC")
        conn.execute("SET search_path=docmind,public,pg_catalog")
        if not conn.execute("SELECT 1 FROM pg_roles WHERE rolname=%s",(runtime_role,)).fetchone():
            conn.execute(sql.SQL("CREATE ROLE {} LOGIN NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE").format(sql.Identifier(runtime_role)))
        conn.execute(sql.SQL("ALTER ROLE {} PASSWORD {}").format(sql.Identifier(runtime_role),sql.Literal(runtime_password)))
        for schema in (corpus,reviews,events,jobs):
            for statement in schema.split(';'):
                if not statement.strip():continue
                table=re.search(r"CREATE TABLE IF NOT EXISTS (\w+)",statement)
                if table:
                    name=table[1]
                    statement=statement.replace('INTEGER PRIMARY KEY AUTOINCREMENT','BIGINT GENERATED ALWAYS AS IDENTITY')
                    statement=re.sub(r'\bBLOB\b','BYTEA',statement)
                    statement=re.sub(r'\bREAL\b','DOUBLE PRECISION',statement)
                    statement=re.sub(r'\buser\b','"user"',statement)
                    statement=statement.replace(' TEXT PRIMARY KEY',' TEXT NOT NULL')
                    statement=statement.replace('REFERENCES documents(file_id) ON DELETE CASCADE','')
                    primary={'documents':'file_id','chunks':'chunk_id','meta':'key','reviews':'response_id','events':'id','ingestion_jobs':'job_id'}[name]
                    position=statement.rfind(')')
                    extra=", tenant_id TEXT NOT NULL DEFAULT current_setting('docmind.tenant',true), PRIMARY KEY(tenant_id,"+primary+")"
                    if name=='chunks':
                        statement=statement.replace('UNIQUE(file_id, ord)','UNIQUE(tenant_id,file_id,ord)')
                        extra+=", embedding_vector vector(384) NOT NULL, FOREIGN KEY(tenant_id,file_id) REFERENCES documents(tenant_id,file_id) ON DELETE CASCADE"
                    position=statement.rfind(')')
                    statement=statement[:position]+extra+statement[position:]
                else:statement=re.sub(r'\buser\b','"user"',statement)
                conn.execute(statement)
        conn.execute("ALTER TABLE reviews ADD COLUMN IF NOT EXISTS response_info TEXT NOT NULL DEFAULT '{}'")
        conn.execute("ALTER TABLE ingestion_jobs ADD COLUMN IF NOT EXISTS owner TEXT NOT NULL DEFAULT 'shared-api'")
        conn.execute("ALTER TABLE ingestion_jobs ADD COLUMN IF NOT EXISTS options TEXT NOT NULL DEFAULT '{}'")
        conn.execute("ALTER TABLE ingestion_jobs ADD COLUMN IF NOT EXISTS lease_token TEXT")
        conn.execute("ALTER TABLE ingestion_jobs ADD COLUMN IF NOT EXISTS lease_until DOUBLE PRECISION NOT NULL DEFAULT 0")
        conn.execute("ALTER TABLE ingestion_jobs ADD COLUMN IF NOT EXISTS attempts INTEGER NOT NULL DEFAULT 0")
        for table in ('documents','chunks','meta','reviews','events','ingestion_jobs'):
            name=sql.Identifier(table)
            conn.execute(sql.SQL('ALTER TABLE {} ENABLE ROW LEVEL SECURITY').format(name))
            conn.execute(sql.SQL('ALTER TABLE {} FORCE ROW LEVEL SECURITY').format(name))
            conn.execute(sql.SQL('DROP POLICY IF EXISTS tenant_scope ON {}').format(name))
            conn.execute(sql.SQL("CREATE POLICY tenant_scope ON {} USING (tenant_id=current_setting('docmind.tenant',true)) WITH CHECK (tenant_id=current_setting('docmind.tenant',true))").format(name))
            conn.execute(sql.SQL('REVOKE ALL ON {} FROM PUBLIC').format(name))
            conn.execute(sql.SQL('GRANT SELECT,INSERT,UPDATE,DELETE ON {} TO {}').format(name,sql.Identifier(runtime_role)))
        for tenant in set(tenants):
            import hashlib
            index='chunks_hnsw_'+hashlib.sha256(tenant.encode()).hexdigest()[:20]
            conn.execute(sql.SQL('CREATE INDEX IF NOT EXISTS {} ON chunks USING hnsw (embedding_vector vector_ip_ops) WHERE tenant_id={}').format(sql.Identifier(index),sql.Literal(tenant)))
        conn.execute(sql.SQL('GRANT USAGE ON SCHEMA docmind,public TO {}').format(sql.Identifier(runtime_role)))
        conn.execute(sql.SQL('GRANT USAGE,SELECT ON ALL SEQUENCES IN SCHEMA docmind TO {}').format(sql.Identifier(runtime_role)))
        conn.execute("CREATE TABLE IF NOT EXISTS api_quota(identity TEXT PRIMARY KEY, times DOUBLE PRECISION[] NOT NULL)")
        conn.execute("REVOKE ALL ON api_quota FROM PUBLIC")
        conn.execute("""CREATE OR REPLACE FUNCTION take_quota(key_hash TEXT,ip_hash TEXT,key_limit INTEGER,ip_limit INTEGER)
        RETURNS BOOLEAN LANGUAGE plpgsql SECURITY DEFINER SET search_path=docmind,pg_catalog AS $$
        DECLARE stamp DOUBLE PRECISION := EXTRACT(EPOCH FROM clock_timestamp()); item TEXT; maximum INTEGER; values_now DOUBLE PRECISION[];
        BEGIN
          IF key_limit<1 OR ip_limit<1 OR length(key_hash)<>64 OR length(ip_hash)<>64 THEN RETURN false; END IF;
          PERFORM pg_advisory_xact_lock(7628143952);
          DELETE FROM api_quota WHERE times[array_length(times,1)]<=stamp-60;
          IF (SELECT count(*) FROM api_quota)>2000 THEN RETURN false; END IF;
          FOR item,maximum IN SELECT 'key:'||key_hash,key_limit UNION ALL SELECT 'ip:'||ip_hash,ip_limit LOOP
            SELECT ARRAY(SELECT t FROM unnest(times) t WHERE t>stamp-60) INTO values_now FROM api_quota WHERE identity=item;
            IF coalesce(array_length(values_now,1),0)>=maximum THEN RETURN false; END IF;
          END LOOP;
          FOR item IN SELECT 'key:'||key_hash UNION ALL SELECT 'ip:'||ip_hash LOOP
            INSERT INTO api_quota VALUES(item,ARRAY[stamp]) ON CONFLICT(identity) DO UPDATE
              SET times=ARRAY(SELECT t FROM unnest(api_quota.times) t WHERE t>stamp-60)||stamp;
          END LOOP;
          RETURN true;
        END $$""")
        conn.execute("REVOKE ALL ON FUNCTION take_quota(TEXT,TEXT,INTEGER,INTEGER) FROM PUBLIC")
        conn.execute(sql.SQL('GRANT EXECUTE ON FUNCTION take_quota(TEXT,TEXT,INTEGER,INTEGER) TO {}').format(sql.Identifier(runtime_role)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--tenant',action='append',required=True);args=parser.parse_args()
    migrate(os.environ['DOCMIND_MIGRATION_DSN'],os.environ.get('DOCMIND_DB_ROLE','docmind_runtime'),os.environ['DOCMIND_DB_PASSWORD'],args.tenant)
    print('Schema, forced tenant policies and runtime grants verified')
