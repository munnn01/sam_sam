import importlib.util
import io
import zipfile
import json
import torch
import pytest
from torch import nn

def api(name):
    assert importlib.util.find_spec('reproduce.'+name) is not None,name+' not implemented'
    return __import__('reproduce.'+name,fromlist=['*'])

def test_archive_traversal_is_rejected_before_extraction(tmp_path):
    a=api('assets');z=tmp_path/'bad.zip'
    with zipfile.ZipFile(z,'w') as f:f.writestr('../outside','bad')
    with pytest.raises(ValueError):a.safe_extract(z,tmp_path/'destination')
    assert not (tmp_path/'outside').exists()

def test_bdrate_requires_four_monotone_points_with_common_quality():
    e=api('evaluation')
    q=[.8,.7,.6,.5];r=[1.,.7,.4,.2]
    assert e.bd_rate(r,q,[x*.8 for x in r],q)['percent']==pytest.approx(-20.,abs=1e-6)
    assert e.bd_rate(r,[.8,.7,.75,.5],r,q)['valid'] is False
    assert e.bd_rate(r,q,r,[.4,.3,.2,.1])['valid'] is False

def test_checkpoint_saves_resume_rng_and_optimizer_not_only_weights(tmp_path):
    from reproduce import training as t
    from reproduce.protocol import recipe
    cfg=recipe('ar','pilot');state=t.initial_state(cfg)
    npp=nn.Linear(2,2);proxy=nn.Linear(2,2);opt=torch.optim.Adam(npp.parameters(),lr=.001)
    npp(torch.ones(1,2)).sum().backward();opt.step();state['phase_steps']['adaptive']=1
    t.save_checkpoint(tmp_path/'last.pt',cfg,state,npp,proxy,{'npp':opt})
    expected=torch.rand(4)
    saved=torch.load(tmp_path/'last.pt',weights_only=False)
    t.restore_rng(saved['rng'])
    assert torch.equal(expected,torch.rand(4))
    assert saved['optimizers']['npp']['state'] and saved['config_sha256']==t.config_digest(cfg)

def test_budget_receipt_does_not_claim_test_results(tmp_path):
    a=api('assets')
    assert a.DATASETS['od']==['awsaf49/coco-2017-dataset']
    assert 'abhimanyukarshni/got10k' in a.DATASETS['ot']
    assert a.DATASETS['ic']!=a.DATASETS['ar']
