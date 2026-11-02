import hashlib,importlib.util,json
from pathlib import Path
import pytest

def api():
    assert importlib.util.find_spec('reproduce.tri_task_cloud') is not None,'three-task cloud not implemented'
    return __import__('reproduce.tri_task_cloud',fromlist=['*'])

def test_all_task_payloads_bind_public_proxy_datasets_and_new_runner(tmp_path):
    a=api();from reproduce.tri_task_data import config,PUBLIC_PROXY
    root=Path(__file__).parents[1]
    for task in ('is','od','ot'):
        p=tmp_path/task;job=a.prepare(root,p,'owner123','tri-'+task+'-foreground-20261101',config(task,'foreground'))
        meta,verified=a.verify(p,root)
        assert meta['is_private'] and meta['enable_gpu'] and meta['enable_internet']
        assert PUBLIC_PROXY['ref'] in meta['dataset_sources'] and verified['config']['task']==task
        assert meta['dataset_sources'][0].endswith('/1' if task=='ot' else '/2')
        code=''.join(json.loads((p/'notebook.ipynb').read_text())['cells'][0]['source'])
        assert "'reproduce.tri_task_study'" in code and "'-m','reproduce.run'" not in code
        assert "if TASK in ('od','ic','is','ot'):" in code

def test_cloud_rejects_configuration_and_literal_input_tampering(tmp_path):
    a=api();from reproduce.tri_task_data import config
    root=Path(__file__).parents[1];cfg=config('is','control');cfg['identity_fallback']=True
    with pytest.raises(ValueError):a.prepare(root,tmp_path/'bad','owner123','tri-is-bad-20261101',cfg)
    p=tmp_path/'ok';a.prepare(root,p,'owner123','tri-is-ok-20261101',config('is','control'))
    n=json.loads((p/'notebook.ipynb').read_text());text=''.join(n['cells'][0]['source']).replace("TASK='is'","TASK='ot'")
    n['cells'][0]['source']=text.splitlines(True);(p/'notebook.ipynb').write_text(json.dumps(n))
    j=json.loads((p/'job.json').read_text());j['payload_hashes']['notebook.ipynb']=hashlib.sha256((p/'notebook.ipynb').read_bytes()).hexdigest();(p/'job.json').write_text(json.dumps(j))
    with pytest.raises(ValueError):a.verify(p)
