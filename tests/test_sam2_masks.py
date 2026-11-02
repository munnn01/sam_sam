import importlib.util,json
from pathlib import Path
import pytest
import torch

def api():
    assert importlib.util.find_spec('reproduce.sam2_masks') is not None,'SAM 2 mask extension missing'
    return __import__('reproduce.sam2_masks',fromlist=['*'])

def test_matched_sam_arms_keep_tasks_qps_initialization_and_phase_budgets():
    a=api();configs=[a.sam_config('od',arm) for arm in ('control','foreground','foreground_bg')]
    assert all(c['qps']==[30,35,40,45] and c['fixed_steps']==256 and c['adaptive_steps']==256 for c in configs)
    assert len({c['seed'] for c in configs})==1
    assert all(c['lr']==1e-4 and c['train_sources']==256 and not c['identity_fallback'] for c in configs)
    assert [c['sam2']['foreground_extra'] for c in configs]==[0,3,3]
    assert [c['sam2']['gamma'] for c in configs]==[0,0,1]
    with pytest.raises(ValueError):a.sam_config('ar','control')

def test_control_is_exact_paper_objective_and_mask_gradient_is_direct():
    a=api();from reproduce.image_study import study_objective
    x=torch.zeros(1,1,3,4,4);y=torch.full_like(x,.2,requires_grad=True);m=torch.zeros(1,1,1,4,4,requires_grad=True)
    m.data[:,:,:,:2]=1
    cfg=a.sam_config('ic','control');loss,parts=a.mask_objective(cfg,torch.tensor(.1),torch.tensor(.3),x,y,35,m)
    assert torch.allclose(loss,study_objective(cfg,torch.tensor(.1),torch.tensor(.3),x,y,35))
    cfg=a.sam_config('ic','foreground');loss,parts=a.mask_objective(cfg,torch.tensor(.1),torch.tensor(.3),x,y,35,m)
    loss.backward()
    assert m.grad is None and y.grad is not None
    assert float(y.grad[:,:,:,:2].mean()/y.grad[:,:,:,2:].mean())==pytest.approx(4)
    for value in (torch.zeros_like(m),torch.ones_like(m)):
        l,p=a.mask_objective(a.sam_config('ic','foreground_bg'),torch.tensor(.1),torch.tensor(.3),x,y,35,value)
        assert torch.isfinite(l) and all(torch.isfinite(v) for v in p.values())

def test_background_penalty_has_no_foreground_gradient_and_penalizes_high_frequency():
    a=api();x=torch.zeros(1,1,3,8,8);y=x.clone();y[...,::2,::2]=.2;y.requires_grad_()
    cfg=a.sam_config('od','foreground_bg');mask=torch.zeros(1,1,1,8,8)
    _,parts=a.mask_objective(cfg,torch.tensor(0.),torch.tensor(0.),x,y,30,mask)
    assert parts['background_highpass']>0
    _,fg=a.mask_objective(cfg,torch.tensor(0.),torch.tensor(0.),x,y,30,torch.ones_like(mask))
    assert float(fg['background_highpass'])==0

def test_automatic_binary_masks_become_finite_soft_weights_with_area_quality_filters():
    a=api();settings=a.sam_config('ic','foreground')['sam2']
    whole=torch.ones(10,10,dtype=torch.bool);small=torch.zeros_like(whole);small[2:6,2:6]=1
    rows=[dict(segmentation=whole,predicted_iou=.99,stability_score=.99),
          dict(segmentation=small,predicted_iou=.9,stability_score=.99),
          dict(segmentation=small,predicted_iou=.1,stability_score=.99)]
    mask,receipt=a.soft_mask(rows,(10,10),settings)
    assert receipt['accepted']==1 and mask.shape==(1,1,1,10,10)
    assert 0<float(mask.max())<1 and float(mask.min())==0
    empty,receipt=a.soft_mask([], (10,10),settings)
    assert empty.sum()==0 and receipt['empty']

def test_cache_binds_rendered_pixels_source_bytes_teacher_and_mask_bytes(tmp_path):
    a=api();source=tmp_path/'image.jpg';source.write_bytes(b'source')
    row={'id':'train/1','path':str(source)};x=torch.zeros(1,1,3,4,4);settings=a.sam_config('ic','foreground')['sam2']
    cache=a.MaskCache(tmp_path/'cache',settings);mask=torch.full((1,1,1,4,4),.5)
    cache.store(row,x,mask);assert torch.allclose(cache.load(row,x),mask)
    with pytest.raises(ValueError):cache.load(row,x+1)
    source.write_bytes(b'changed')
    with pytest.raises(ValueError):cache.load(row,x)
    source.write_bytes(b'source');different=dict(settings,checkpoint_sha256='0'*64)
    with pytest.raises(ValueError):a.MaskCache(tmp_path/'cache',different).load(row,x)
    path=next((tmp_path/'cache').glob('*.pt'));saved=torch.load(path,weights_only=False);saved['mask'].zero_();torch.save(saved,path)
    with pytest.raises(ValueError):cache.load(row,x)

def test_sam_context_binds_current_sample_restores_native_hooks_and_never_runs_teacher(tmp_path):
    a=api();import reproduce.image_study as base;import reproduce.training as training
    source=tmp_path/'a';source.write_bytes(b'x');row={'id':'one','path':str(source),'sam_cache':str(tmp_path/'cache')}
    x=torch.zeros(1,1,3,4,4);cfg=a.sam_config('ic','foreground');cache=a.MaskCache(row['sam_cache'],cfg['sam2']);cache.store(row,x,torch.ones(1,1,1,4,4))
    def sample(*args,**kwargs):return x,torch.tensor([0]),{'id':'one'}
    before=training.load_sample,training.image_objective
    with a.mask_context(cfg,{},tmp_path,sample):
        actual,_,_=training.load_sample(row,cfg,True)
        loss=training.image_objective(torch.tensor(.1),torch.tensor(.1),actual,actual+.1,35)
        assert torch.isfinite(loss)
        with pytest.raises(ValueError):training.image_objective(torch.tensor(.1),torch.tensor(.1),actual+1,actual,35)
    assert before==(training.load_sample,training.image_objective)

def test_runtime_context_installs_sam_loss_and_uses_cached_crop(tmp_path):
    a=api();import reproduce.image_study as base;import reproduce.training as training
    source=tmp_path/'a';source.write_bytes(b'x');cfg=a.sam_config('ic','foreground')
    row={'id':'one','path':str(source),'sam_cache':str(tmp_path/'cache')}
    x=torch.zeros(1,1,3,4,4);a.MaskCache(row['sam_cache'],cfg['sam2']).store(row,x,torch.ones(1,1,1,4,4))
    old=base.study_sample;base.study_sample=lambda *args,**kwargs:(x,torch.tensor([0]),{'id':'one'})
    try:
        with base.study_context(cfg,{},tmp_path):
            training.load_sample(row,cfg,True)
            y=(x+.2).requires_grad_();loss=training.image_objective(torch.tensor(.1),torch.tensor(.1),x,y,35)
            assert torch.isfinite(loss)
            assert (tmp_path/'progress.jsonl').is_file()
    finally:base.study_sample=old

def test_control_prepares_complete_matched_cache_without_loading_sam(tmp_path,monkeypatch):
    a=api();from types import SimpleNamespace
    cfg=a.sam_config('ic','control');cfg['train_sources']=3
    sources=[]
    for i in range(4):
        path=tmp_path/str(i);path.write_bytes(str(i).encode());sources.append({'id':str(i),'path':str(path)})
    monkeypatch.setattr(a,'build_teacher',lambda *args:pytest.fail('control loaded SAM'))
    def sample(row,*args):
        import random
        return torch.full((1,1,3,4,4),random.random()),torch.tensor([0]),{'id':row['id']}
    selected=a.prepare_training(cfg,sources,tmp_path,SimpleNamespace(require=lambda stage:None),sample)
    assert len(selected)==3 and 'sam2_runtime' in cfg
    for row in selected:
        x,_,_=a.rendered_sample(row,cfg,sample)
        assert a.MaskCache(row['sam_cache'],cfg['sam2']).load(row,x).sum()==0

def test_corrupt_sam_training_cache_stops_instead_of_skipping_source(tmp_path,monkeypatch):
    a=api();import reproduce.image_study as base;import reproduce.training as training
    path=tmp_path/'image';path.write_bytes(b'original');cfg=a.sam_config('ic','foreground')
    row={'id':'one','path':str(path),'sam_cache':str(tmp_path/'cache')};x=torch.zeros(1,1,3,4,4)
    a.MaskCache(row['sam_cache'],cfg['sam2']).store(row,x,torch.ones(1,1,1,4,4))
    path.write_bytes(b'changed')
    monkeypatch.setattr(base,'study_sample',lambda *args:(x,torch.tensor([0]),{'id':'one'}))
    before=training.sample_training
    with base.study_context(cfg,{},tmp_path):
        with pytest.raises(ValueError,match='mask cache identity/content differs'):
            training.sample_training([row],cfg,tmp_path)
    assert training.sample_training is before
    assert not (tmp_path/'progress.jsonl').exists()
