"""Tenant identity comes from the server credential registry, never request fields."""
from contextvars import ContextVar
import hashlib
import os
from docmind.config import CONFIG
from docmind.auth import Principal
from docmind.postgres import DatabaseScope

PRINCIPAL: ContextVar[Principal | None]=ContextVar('docmind_principal',default=None)


def database_scope():
    dsn=os.getenv('DOCMIND_DATABASE_URL')
    if not dsn:return str(CONFIG.db_path)
    principal=PRINCIPAL.get()
    if principal is None:raise RuntimeError('Authenticated tenant scope required')
    return DatabaseScope(dsn,principal.tenant_id)


def data_root(scope=None):
    if scope is None:
        scope=database_scope() if os.getenv('DOCMIND_DATABASE_URL') else None
    if isinstance(scope,DatabaseScope):
        return CONFIG.data_dir/'tenants'/hashlib.sha256(scope.tenant.encode()).hexdigest()
    return CONFIG.data_dir
