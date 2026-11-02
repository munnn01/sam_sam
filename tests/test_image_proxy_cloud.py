import base64,hashlib,importlib.util,json,zlib
from pathlib import Path
import pytest

def api():
    assert importlib.util.find_spec('reproduce.image_proxy_cloud') is not None,'image payload not implemented'
    return __import__('reproduce.image_proxy_cloud',fromlist=['*'])

def legacy():
    contents={'reproduce/run.py':'# immutable legacy fixture\n'}
    hashes={n:hashlib.sha256(v.encode()).hexdigest() for n,v in contents.items()}
    sha=hashlib.sha256(json.dumps(hashes,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    bundle=base64.b64encode(zlib.compress(json.dumps(contents).encode())).decode()
    return bundle,hashes,sha

def test_image_payload_pins_owned_kernel_and_preserves_separate_source_roots(tmp_path):
    a=api();bundle,hashes,sha=legacy();root=Path(__file__).parents[1]
    job=a.prepare_image_diagnostic(root,tmp_path/'payload','owner123','image-audit-20261028','od',
        'owner123/old-image-kernel/1','a'*64,sha,bundle,hashes)
    meta,verified=a.verify_image_diagnostic(tmp_path/'payload',root)
    assert meta['is_private'] and meta['kernel_sources']==['owner123/old-image-kernel/1']
    assert job['image_diagnostic']['max_seconds']==18000 and job['image_diagnostic']['audit_seconds']==5400
    code=''.join(json.loads((tmp_path/'payload/notebook.ipynb').read_text())['cells'][0]['source'])
    assert "'reproduce.image_proxy_audit'" in code and 'legacy_image_reproduction' in code

def test_image_payload_rejects_wrong_owner_or_unverified_frozen_source(tmp_path):
    a=api();bundle,hashes,sha=legacy();root=Path(__file__).parents[1]
    with pytest.raises(ValueError):a.prepare_image_diagnostic(root,tmp_path/'bad','owner123','image-audit-20261028','ic',
        'another/old-image-kernel/1','a'*64,sha,bundle,hashes)
    with pytest.raises(ValueError):a.prepare_image_diagnostic(root,tmp_path/'bad2','owner123','image-audit-20261028','ic',
        'owner123/old-image-kernel/1','a'*64,'b'*64,bundle,hashes)
