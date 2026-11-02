import importlib.util
import json
import pytest
from pathlib import Path

def api():
    assert importlib.util.find_spec('reproduce.proxy_cloud') is not None,'diagnostic payload builder not implemented'
    return __import__('reproduce.proxy_cloud',fromlist=['*'])

def test_payload_is_private_pinned_and_routes_to_diagnostics(tmp_path):
    a=api();root=Path(__file__).parents[1]
    job=a.prepare_diagnostic(root,tmp_path/'payload','owner123','paper-p05-test-20261028','ot',
        'owner123/state-private/1','a'*64,'b'*64)
    meta,verified=a.verify_diagnostic(tmp_path/'payload',root)
    assert meta['is_private'] and meta['dataset_sources'][-1]=='owner123/state-private/1'
    assert verified['diagnostic']['stage']=='diagnose' and verified['diagnostic']['max_seconds']==7200
    source=''.join(json.loads((tmp_path/'payload/notebook.ipynb').read_text())['cells'][0]['source'])
    assert "'reproduce.proxy_recovery'" in source
    assert "'-m','reproduce.run'" not in source

def test_payload_rejects_unowned_state_and_excess_diagnostic_budget(tmp_path):
    a=api();root=Path(__file__).parents[1]
    with pytest.raises(ValueError):a.prepare_diagnostic(root,tmp_path/'bad','owner123','paper-p05-test-20261028','ot',
        'another/state/1','a'*64,'b'*64)
    with pytest.raises(ValueError):a.prepare_diagnostic(root,tmp_path/'bad2','owner123','paper-p05-test-20261028','ot',
        'owner123/state/1','a'*64,'b'*64,max_seconds=18000)
