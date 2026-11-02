import hashlib,importlib.util,json,random
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch

def api():
    assert importlib.util.find_spec('reproduce.image_study') is not None,'image study not implemented'
    return __import__('reproduce.image_study',fromlist=['*'])

def test_study_grid_has_matched_screening_and_distinct_full_targets():
    a=api();configs=[a.study_config('od','screen',s) for s in (.5,1.,2.)]
    assert all(c['fixed_steps']==1000 and c['adaptive_steps']==1000 and c['proxy_steps']==0 for c in configs)
    assert all(c['backbone']=='fcos_resnet50_fpn' and c['qps']==[30,35,40,45] and not c['identity_fallback'] for c in configs)
    assert {c['lambda_scale'] for c in configs}=={.5,1.,2.}
    full=a.study_config('ic','full',1.)
    assert (full['fixed_steps'],full['adaptive_steps'],full['max_session_seconds'])==(120000,100000,21600)
    with pytest.raises(ValueError):a.study_config('ar','screen',1)
    with pytest.raises(ValueError):a.study_config('od','screen',0)

def test_image_real_forward_loss_keeps_proxy_gradient_with_changed_task_scale():
    a=api();from reproduce.protocol import real_forward,task_lambda
    cfg=a.study_config('ic','screen',.5);y=torch.tensor(.2,requires_grad=True);x=torch.tensor(.4)
    rate=real_forward(y*3,torch.tensor(.1));decoded=real_forward(y*2,torch.tensor(.7))
    loss=a.study_objective(cfg,rate,decoded.square(),x,y,35)
    lam=.5*task_lambda(35)
    assert float(loss)==pytest.approx(.1+lam*.49+.5*.04)
    loss.backward();assert float(y.grad)==pytest.approx(3+2.8*lam-.2)

def test_native_adaptations_restore_even_on_error_and_random_qps_stay_configured():
    a=api();import reproduce.training as training
    before=training.image_objective,training.qp_for_step,training.load_sample
    cfg=a.study_config('od','screen',2)
    with pytest.raises(RuntimeError):
        with a.study_context(cfg):
            assert training.qp_for_step(20,cfg,'fixed')==35
            draws=[training.qp_for_step(i,cfg,'adaptive') for i in range(100)]
            assert set(draws)=={30,35,40,45}
            raise RuntimeError('interrupt')
    assert before==(training.image_objective,training.qp_for_step,training.load_sample)

def test_proxy_import_binds_task_and_does_not_migrate_untrained_npp_or_adam(tmp_path):
    a=api();p=tmp_path/'proxy.pt';torch.save(dict(schema=1,task='od',qps=[30,35,40,45],
        proxy_quality={'30':5,'35':4,'40':3,'45':2},proxy={'weights':torch.ones(2)},
        original_checkpoint_sha256='b'*64,original_training_source='c'*64,
        original_phase_steps={'proxy':315254,'fixed':0,'adaptive':0},optimizers={'bad':1},npp={'bad':2}),p)
    sha=hashlib.sha256(p.read_bytes()).hexdigest();result=a.load_proxy_input(p,sha,'od')
    assert torch.equal(result['proxy']['weights'],torch.ones(2))
    assert 'optimizers' not in result and 'npp' not in result
    assert result['original_phase_steps']['proxy']==315254
    with pytest.raises(ValueError):a.load_proxy_input(p,sha,'ic')
    with pytest.raises(ValueError):a.load_proxy_input(p,'0'*64,'od')

def test_batch_codec_uses_total_paid_bytes_and_original_pixel_count():
    a=api();x=torch.stack([torch.full((1,3,2,2),v) for v in (.1,.2,.3,.4)])
    def codec(x,*args):
        count=round(float(x.mean())*100)
        return SimpleNamespace(reconstruction=x*.8,bytes=count,sha256=str(count))
    actual=a.BatchCodec(codec)(x,'bpg',35)
    assert actual.bytes==100 and actual.bpp==50
    assert torch.allclose(actual.reconstruction,x*.8)

def test_flickr_holdout_is_source_disjoint_and_deterministic():
    a=api();rows=[{'id':'flickr/'+str(i),'path':str(i)} for i in range(40)]
    train,dev=a.flickr_split(rows,8)
    assert len(train)==32 and len(dev)==8
    assert not {x['id'] for x in train}&{x['id'] for x in dev}
    assert a.flickr_split(list(reversed(rows)),8)==(train,dev)

def test_untrained_npp_cannot_be_exported_for_evaluation(tmp_path):
    a=api();model=torch.nn.Linear(1,1);cfg=a.study_config('ic','screen',1.)
    with pytest.raises(ValueError):a.export_evaluated_model(tmp_path,cfg,{'phase_steps':{'fixed':0,'adaptive':0}},model)
    path=a.export_evaluated_model(tmp_path,cfg,{'phase_steps':{'fixed':3,'adaptive':0}},model)
    saved=torch.load(path,weights_only=False)
    assert saved['trained_npp_steps']==3 and not saved['full_curriculum_complete']

def test_frozen_analyzer_uses_the_same_digest_contract_as_native_preflight(monkeypatch):
    a=api();import reproduce.tasks as tasks
    model=torch.nn.Linear(1,1);proxy=torch.nn.Linear(1,1)
    monkeypatch.setattr(tasks,'fingerprint',lambda m:'native/'+str(float(m.weight.sum())))
    analyzer=SimpleNamespace(model=model,weight_digest=tasks.fingerprint(model))
    assert a.frozen_models_unchanged(proxy,a.model_digest(proxy),analyzer)
    with torch.no_grad():model.weight.add_(1)
    assert not a.frozen_models_unchanged(proxy,a.model_digest(proxy),analyzer)

def test_proxy_mask_canonicalization_preserves_function_and_stabilizes_frozen_digest():
    a=api()
    class MaskedConv2d(torch.nn.Conv2d):
        def __init__(self):
            super().__init__(1,1,3,padding=1,bias=False)
            self.register_buffer('mask',torch.ones_like(self.weight));self.mask[...,2,:]=0
        def forward(self,x):
            self.weight.data*=self.mask
            return super().forward(x)
    proxy=MaskedConv2d().requires_grad_(False);x=torch.ones(1,1,4,4,requires_grad=True)
    raw=proxy.weight.detach().clone();expected=torch.nn.functional.conv2d(x,raw*proxy.mask,padding=1)
    report=a.canonicalize_proxy_masks(proxy);before=a.model_digest(proxy)
    actual=proxy(x)
    assert report['changed_entries']>0 and torch.allclose(expected,actual)
    assert a.model_digest(proxy)==before
    actual.sum().backward();assert x.grad is not None and proxy.weight.grad is None

def test_guarded_adam_stops_nonfinite_gradients_before_weight_update_and_tracks_real_steps():
    a=api();cfg=a.study_config('ic','screen',1);progress={'npp_updates':0};p=torch.nn.Parameter(torch.tensor(1.))
    cls=a.guarded_adam_class(cfg,progress);optimizer=cls([p],lr=1e-4)
    p.square().backward();optimizer.step();assert progress['npp_updates']==1
    before=p.detach().clone();p.grad=torch.tensor(float('nan'))
    with pytest.raises(RuntimeError):optimizer.step()
    assert torch.equal(p.detach(),before) and progress['npp_updates']==1

def test_proxy_nonfinite_forward_and_auxiliary_values_are_rejected_in_context():
    a=api();cfg=a.study_config('proxy','proxy');model=torch.nn.Linear(1,1)
    model.forward=lambda x:SimpleNamespace(bpp=torch.tensor(float('inf')),reconstruction=x)
    model.aux_loss=lambda q:torch.tensor(float('nan'))
    with a.study_context(cfg,proxy=model):
        with pytest.raises(RuntimeError):model(torch.ones(1))
        with pytest.raises(RuntimeError):model.aux_loss(35)

def test_error_report_distinguishes_actual_updates_and_recoverable_checkpoint(tmp_path):
    a=api();torch.save({'state':{'phase_steps':{'fixed':2,'adaptive':0,'proxy':0}}},tmp_path/'last.pt')
    report={'npp_updates':3};a.record_error_progress(tmp_path,report)
    assert report['npp_updates']==3 and report['recoverable_npp_updates']==2

def test_proxy_validation_budget_is_partial_completion_not_error(tmp_path,monkeypatch):
    a=api();from reproduce.proxy_recovery import BudgetExpired
    def run(*args):
        (tmp_path/'proxy_validation.json').write_text('[{"source":"one"}]')
        raise BudgetExpired('proxy_validation')
    monkeypatch.setattr(a,'proxy_validation',run)
    result=a.bounded_proxy_validation(None,[],{},None,tmp_path,None)
    assert result['complete'] is False and result['points']==[{'source':'one'}] and result['stop_reason']=='time_budget'
