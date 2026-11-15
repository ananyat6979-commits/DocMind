import importlib.util
import json
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('production_init',Path(__file__).resolve().parents[1]/'scripts/init_production.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_private_credentials_are_explicit_and_never_overwritten(tmp_path):
    path=module.initialize(tmp_path/'private',['alpha','beta'])
    records=json.loads((path/'credentials.json').read_text())
    assert {r['tenant_id'] for r in records}=={'alpha','beta'}
    assert len({r['key_sha256'] for r in records})==2
    assert 'key_sha256' not in (path/'api-keys.txt').read_text()
    original=(path/'compose.env').read_bytes()
    with pytest.raises(FileExistsError):module.initialize(path,['alpha'])
    assert (path/'compose.env').read_bytes()==original
    with pytest.raises(ValueError):module.initialize(tmp_path/'invalid',['../../other'])
