import hashlib,importlib.util,json
from pathlib import Path
import pytest
from reproduce.image_study import study_config

def api():
    assert importlib.util.find_spec('reproduce.image_study_cloud') is not None,'study cloud not implemented'
    return __import__('reproduce.image_study_cloud',fromlist=['*'])

def test_sam_payload_binds_offline_teacher_and_matched_configuration(tmp_path):
    from reproduce.sam2_masks import sam_config
    a=api();root=Path(__file__).parents[1];cfg=sam_config('od','foreground_bg')
    p=tmp_path/'sam';spec=dict(kind='dataset',ref='owner123/image-state/1',member='od_proxy.pt',sha256='a'*64)
    a.prepare_study(root,p,'owner123','sam2-od-foreground-bg-20261031',cfg,spec)
    _,job=a.verify_study(p,root)
    assert job['config']['sam2']['source_commit']=='2b90b9f5ceec907a1c18123530e92e794ad901a4'
    assert job['config']['fixed_steps']==256 and job['config']['adaptive_steps']==256
    cfg['sam2']['checkpoint_sha256']='0'*64
    with pytest.raises(ValueError):a.prepare_study(root,tmp_path/'bad-sam','owner123','sam2-od-bad-20261031',cfg,spec)

def test_explicit_public_dataset_can_be_shared_across_accounts_but_private_cannot(tmp_path):
    from reproduce.sam2_masks import sam_config
    a=api();root=Path(__file__).parents[1];cfg=sam_config('ic','foreground')
    spec=dict(kind='dataset',ref='sharedowner/image-state/1',member='ic_proxy.pt',sha256='b'*64,visibility='public')
    p=tmp_path/'public';a.prepare_study(root,p,'owner123','sam2-ic-public-20261031',cfg,spec)
    meta,_=a.verify_study(p)
    assert spec['ref'] in meta['dataset_sources'] and meta['is_private']
    with pytest.raises(ValueError):a.input_valid('owner123',cfg,dict(spec,kind='kernel'))
    with pytest.raises(ValueError):a.input_valid('owner123',cfg,{k:v for k,v in spec.items() if k!='visibility'})

def test_study_cloud_binds_private_versioned_input_and_exact_config(tmp_path):
    a=api();root=Path(__file__).parents[1];cfg=study_config('od','screen',.5)
    spec=dict(kind='dataset',ref='owner123/image-proxy-state/1',member='od_proxy.pt',sha256='a'*64)
    job=a.prepare_study(root,tmp_path/'payload','owner123','image-od-l050-20261031',cfg,spec)
    meta,verified=a.verify_study(tmp_path/'payload',root)
    assert meta['is_private'] and meta['dataset_sources'][-1]==spec['ref']
    assert verified['config']==cfg and job['input_spec']==spec
    text=''.join(json.loads((tmp_path/'payload/notebook.ipynb').read_text())['cells'][0]['source'])
    assert "'reproduce.image_study'" in text and "'-m','reproduce.run'" not in text

def test_flickr_payload_has_only_flickr_and_no_downstream_checkpoint(tmp_path):
    a=api();root=Path(__file__).parents[1]
    job=a.prepare_study(root,tmp_path/'proxy','owner123','image-flickr-proxy-20261031',study_config('proxy','proxy',1),None)
    meta,_=a.verify_study(tmp_path/'proxy')
    assert meta['dataset_sources']==['eeshawn/flickr30k/1'] and not meta['kernel_sources']

def test_cloud_rejects_foreign_owner_unsafe_member_and_literal_tampering(tmp_path):
    a=api();root=Path(__file__).parents[1];cfg=study_config('ic','screen',1.)
    for ref,member in [('someone/image-state/1','ic_proxy.pt'),('owner123/image-state/1','../ic_proxy.pt')]:
        with pytest.raises(ValueError):a.prepare_study(root,tmp_path/'bad','owner123','image-ic-l100-20261031',cfg,
            dict(kind='dataset',ref=ref,member=member,sha256='a'*64))
    p=tmp_path/'good';a.prepare_study(root,p,'owner123','image-ic-l100-20261031',cfg,
        dict(kind='kernel',ref='owner123/image-state/1',member='paper_results/legacy_results/last.pt',sha256='b'*64))
    notebook=json.loads((p/'notebook.ipynb').read_text());text=''.join(notebook['cells'][0]['source'])
    text=text.replace("TASK='ic'","TASK='od'");notebook['cells'][0]['source']=text.splitlines(True)
    (p/'notebook.ipynb').write_text(json.dumps(notebook));job=json.loads((p/'job.json').read_text())
    job['payload_hashes']['notebook.ipynb']=hashlib.sha256((p/'notebook.ipynb').read_bytes()).hexdigest();(p/'job.json').write_text(json.dumps(job))
    with pytest.raises(ValueError):a.verify_study(p)
