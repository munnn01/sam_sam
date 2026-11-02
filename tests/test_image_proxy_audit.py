import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from reproduce.protocol import recipe

def api():
    assert importlib.util.find_spec('reproduce.image_proxy_audit') is not None,'BPG audit not implemented'
    return __import__('reproduce.image_proxy_audit',fromlist=['*'])

def test_image_checkpoint_requires_task_hash_source_and_proxy_only_state(tmp_path):
    a=api();cfg=recipe('od');state=dict(phase_steps={'proxy':295236,'fixed':0,'adaptive':0},completed=False)
    saved=dict(config=cfg,state=state,source_manifest_sha256='a'*64,optimizers={'proxy/30':{}},rng={'torch':torch.get_rng_state()})
    p=tmp_path/'last.pt';torch.save(saved,p);sha=hashlib.sha256(p.read_bytes()).hexdigest()
    assert a.load_image_checkpoint(p,sha,'a'*64,'od')['state']==state
    with pytest.raises(ValueError):a.load_image_checkpoint(p,sha,'a'*64,'ic')
    saved['state']['phase_steps']['fixed']=1;torch.save(saved,p);sha=hashlib.sha256(p.read_bytes()).hexdigest()
    with pytest.raises(ValueError):a.load_image_checkpoint(p,sha,'a'*64,'od')

def test_proxy_restore_uses_network_load_override_for_entropy_buffers():
    a=api()
    class Network(nn.Module):
        def __init__(self):super().__init__();self.register_buffer('cdf',torch.empty(0));self.calls=0
        def load_state_dict(self,state,strict=True):
            self.calls+=1;self.cdf=torch.empty_like(state['cdf']);return super().load_state_dict(state,strict=strict)
    bank=nn.Module();bank.networks=nn.ModuleDict({str(q):Network() for q in (30,35,40,45)})
    state={f'networks.{q}.cdf':torch.arange(5.) for q in (30,35,40,45)}
    a.restore_proxy_state(bank,state)
    assert all(m.calls==1 and m.cdf.shape==(5,) for m in bank.networks.values())

def test_bpg_gate_handles_all_qps_and_static_scale_without_video_formula():
    a=api();rows=[]
    for q in (30,35,40,45):
        for i in range(8):rows.append(dict(source=str(i),qp=q,variant='blur',raw_bpp=.2,bpp=.1,proxy_raw=2.,proxy_bpp=1.9,pixels=65536))
    result=a.image_rate_gate(rows)
    assert result['passed'] and len(result['groups'])==4
    for r in rows:r['proxy_bpp']=2.1
    assert not a.image_rate_gate(rows)['passed']

def test_proxy_only_guard_preserves_optimizer_rng_and_blocks_npp(tmp_path):
    a=api();schedule=lambda cfg:[('proxy',400000),('fixed',120000)]
    module=SimpleNamespace(phase_schedule=schedule)
    payload=dict(state=dict(phase_steps={'proxy':400000,'fixed':0,'adaptive':0},completed=False),
        optimizers={'adam':{'step':7}},rng={'seed':123})
    def train(cfg,rows,npp,proxy,analyzer,codec,out,resume=None,max_seconds=None):
        assert module.phase_schedule(cfg)==[('proxy',400000)]
        torch.save(payload,Path(out)/'last.pt')
        return dict(payload['state'],completed=True),npp
    state,_=a.proxy_only_train(module,train,recipe('od'),[],object(),None,None,None,tmp_path)
    assert not state['completed'] and state['phase_steps']['fixed']==0
    saved=torch.load(tmp_path/'last.pt',weights_only=False)
    assert saved['optimizers']==payload['optimizers'] and saved['rng']==payload['rng']
    assert module.phase_schedule is schedule

def test_controlled_variants_are_bounded_repeatable_and_not_untrained_npp():
    a=api();x=torch.full((1,1,3,8,8),.5)
    variants=a.controlled_inputs(x,seed=5)
    assert set(variants)=={'raw','gaussian_0.5','gaussian_1.0','perturb_1byte'}
    assert torch.equal(variants['perturb_1byte'],a.controlled_inputs(x,5)['perturb_1byte'])
    assert all(v.shape==x.shape and v.min()>=0 and v.max()<=1 for v in variants.values())

def test_audit_reconstruction_quality_has_finite_perfect_case():
    a=api();x=torch.ones(1,1,3,8,8)
    perfect=a.reconstruction_quality(x,x)
    assert perfect['mse']==0 and perfect['psnr_db'] is None and perfect['perfect']

def test_continuation_watchdog_bounds_setup_and_keeps_console(tmp_path):
    a=api();start=time.monotonic();log=tmp_path/'console.log'
    result=a.bounded_process([sys.executable,'-u','-c',"import time;print('setup_started',flush=True);time.sleep(20)"],
        tmp_path,None,log,seconds=.5,grace=.2)
    assert result['timed_out'] and time.monotonic()-start<5
    assert 'setup_started' in log.read_text()

def test_completed_proxy_checkpoint_still_exports_state_without_native_save(tmp_path):
    a=api();source=tmp_path/'input.pt';out=tmp_path/'out';out.mkdir()
    payload=dict(state=dict(phase_steps={'proxy':400000,'fixed':0,'adaptive':0},completed=False),
        optimizers={'proxy/30':{'step':400000}},rng={'seed':7})
    torch.save(payload,source);module=SimpleNamespace(phase_schedule=lambda cfg:[])
    def native(*args,**kwargs):return dict(payload['state'],completed=True),args[2]
    state,_=a.proxy_only_train(module,native,recipe('od'),[],object(),None,None,None,out,resume=source)
    saved=torch.load(out/'last.pt',weights_only=False)
    assert not state['completed'] and saved['optimizers']==payload['optimizers'] and saved['rng']==payload['rng']

def test_continuation_watchdog_stops_descendants(tmp_path):
    a=api();marker=tmp_path/'orphan.txt'
    child="import time;from pathlib import Path;time.sleep(1.5);Path("+repr(str(marker))+").write_text('orphan')"
    parent="import subprocess,sys,time;subprocess.Popen([sys.executable,'-c',"+repr(child)+"]);print('child_started',flush=True);time.sleep(20)"
    result=a.bounded_process([sys.executable,'-u','-c',parent],tmp_path,None,tmp_path/'tree.log',seconds=.5,grace=.2)
    assert result['timed_out']
    time.sleep(1.6)
    assert not marker.exists(),'worker descendant escaped the budget'
