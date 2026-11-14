"""Create private single-host deployment credentials without printing secrets."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import secrets


def initialize(directory,tenants):
    if not tenants or len(set(tenants))>16 or any(not re.fullmatch(r'[A-Za-z0-9_.-]{1,80}',t) for t in tenants):raise ValueError('One to sixteen valid tenant identifiers required')
    directory=Path(directory).resolve();directory.mkdir(mode=0o700,parents=True,exist_ok=True)
    if any(directory.iterdir()):raise FileExistsError('Credential directory must be empty; existing credentials are never overwritten')
    def write(name,text,mode=0o600):
        descriptor=os.open(directory/name,os.O_WRONLY|os.O_CREAT|os.O_EXCL,mode)
        with os.fdopen(descriptor,'w',encoding='utf-8',newline='\n') as f:f.write(text)
    admin=secrets.token_hex(32);runtime=secrets.token_hex(32);registry=[];keys=[]
    for tenant in sorted(set(tenants)):
        key=secrets.token_urlsafe(40)
        registry.append({'tenant_id':tenant,'actor_id':'initial-admin','roles':['reader','ingestor','reviewer','admin'],'key_sha256':hashlib.sha256(key.encode()).hexdigest()})
        keys.append(tenant+' '+key)
    write('credentials.json',json.dumps(registry,indent=2),0o640)
    write('api-keys.txt','\n'.join(keys)+'\n')
    values={'DOCMIND_DATABASE_URL':'postgresql://docmind_runtime:'+runtime+'@postgres:5432/docmind',
        'DOCMIND_MIGRATION_DSN':'postgresql://postgres:'+admin+'@postgres:5432/docmind','DOCMIND_DB_PASSWORD':runtime,
        'DOCMIND_DB_ADMIN_PASSWORD':admin,'DOCMIND_DB_TENANTS':','.join(sorted(set(tenants))),
        'DOCMIND_CREDENTIALS_PATH':str(directory/'credentials.json').replace('\\','/'),
        'DOCMIND_CREDENTIALS_GID':str(os.getgid() if hasattr(os,'getgid') else 1000)}
    write('compose.env','\n'.join(k+'='+v for k,v in values.items())+'\n')
    return directory


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--directory',default='.secrets/production');parser.add_argument('--tenant',action='append',required=True);args=parser.parse_args()
    directory=initialize(args.directory,args.tenant)
    print('Private credential files created in '+str(directory))
