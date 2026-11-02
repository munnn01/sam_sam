import importlib.util,json,math
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from PIL import Image

def api(module):
    name='reproduce.tri_task_'+module
    assert importlib.util.find_spec(name) is not None,name+' missing'
    return __import__(name,fromlist=['*'])

def test_three_tasks_keep_forward_pipeline_and_declared_gates():
    a=api('data')
    for task,codec in [('is','bpg'),('od','bpg'),('ot','h264')]:
        control=a.config(task,'control');masked=a.config(task,'foreground')
        assert control['qps']==[30,35,40,45] and control['codecs']==[codec]
        assert not control['identity_fallback'] and control['warmup_steps']==12000 and control['warmup_min_steps']==2000
        assert control['task_steps']==256 and control['gate_psnr']==25 and control['collapse_psnr']==20
        assert control['seed']==masked['seed'] and masked['sam2']['foreground_extra']==3
        assert control['sam2']['gamma']==0 and masked['sam2']['gamma']==.1
    with pytest.raises(ValueError):a.config('ic','foreground')

def test_coco_masks_resize_with_boxes_and_keep_original_category_ids(tmp_path):
    a=api('data');path=tmp_path/'image.png';Image.new('RGB',(12,8),(100,80,50)).save(path)
    row={'id':'coco/1','task':'is','path':str(path),'width':12,'height':8,'image_id':1,'split':'train',
        'instances':[{'bbox':[2,1,6,4],'category_id':17,'iscrowd':0,'segmentation':[[2,1,8,1,8,5,2,5]]},
                     {'bbox':[0,0,1,1],'category_id':1,'iscrowd':1,'segmentation':[[0,0,1,0,1,1]]}]}
    cfg=a.config('is','control');cfg['npp_size']=8;cfg['max_size']=12
    x,target,meta=a.load(row,cfg)
    assert x.shape==(1,1,3,8,12) and target['masks'].shape==(1,8,12)
    assert target['labels'].tolist()==[17] and target['boxes'].tolist()==[[2,1,8,5]]
    assert target['masks'].dtype==torch.uint8 and target['masks'].sum()>0
    assert meta['original_width']==12 and meta['original_height']==8

def test_tracking_fixed_clip_cache_binds_all_frames_and_gt(tmp_path):
    a=api('data');seq=tmp_path/'seq';seq.mkdir()
    for i in range(6):Image.new('RGB',(12,8),(i*20,80,50)).save(seq/f'{i:08}.jpg')
    (seq/'groundtruth.txt').write_text('\n'.join(['2,1,4,3']*6));(seq/'cover.label').write_text('\n'.join(['1']*6))
    row={'id':'got10k/one','task':'ot','path':str(seq),'split':'train'};cfg=a.config('ot','foreground');cfg['npp_size']=16
    x,target,meta=a.load(row,cfg);xx,tt,mm=a.load(row,cfg)
    assert torch.equal(x,xx) and meta['frame_indices']==mm['frame_indices'] and len(meta['frame_indices'])==4
    cache=a.VideoMaskCache(tmp_path/'cache',cfg['sam2']);mask=torch.ones(1,4,1,16,16);cache.store(row,x,mask)
    assert torch.equal(cache.load(row,x),mask)
    changed=seq/f"{meta['frame_indices'][1]:08}.jpg";Image.new('RGB',(12,8),(255,0,0)).save(changed)
    with pytest.raises(ValueError):cache.load(row,x)

def test_ste_clip_preserves_codec_forward_and_has_no_identity_input_path():
    a=api('models');raw=torch.tensor([-2.,.4,2.],requires_grad=True)
    y=a.clip_output(raw,True);assert torch.equal(y.detach(),torch.tensor([0.,.4,1.]))
    y.sum().backward();assert torch.equal(raw.grad,torch.ones(3))
    raw.grad.zero_();a.clip_output(raw,False).sum().backward();assert torch.equal(raw.grad,torch.tensor([0.,1.,0.]))

def test_frame_proxy_preserves_temporal_geometry_and_positive_rate_gradients():
    a=api('models')
    class Bank(torch.nn.Module):
        def forward(self,x,qp):return SimpleNamespace(reconstruction=x*.8,bpp=x.square().mean()+.1)
    proxy=a.FrameProxy(Bank());x=torch.full((1,4,3,8,8),.5,requires_grad=True)
    out=proxy(x,35)
    assert out.reconstruction.shape==x.shape and torch.allclose(out.reconstruction,x*.8) and out.bpp>0
    out.bpp.backward();assert x.grad.abs().sum()>0

def test_frozen_detection_loss_keeps_bn_fixed_and_predict_never_gets_targets():
    a=api('models')
    class Detector(torch.nn.Module):
        def __init__(self):super().__init__();self.bn=torch.nn.BatchNorm2d(3);self.calls=[]
        def forward(self,images,targets=None):
            self.calls.append(targets is not None)
            if self.training:return {'loss_mask':self.bn(images[0][None]).square().mean()}
            assert targets is None;return [{'scores':torch.ones(1)}]
    m=Detector();head=a.CocoTask('is',m);before=m.bn.running_mean.clone();x=torch.rand(1,1,3,8,8,requires_grad=True)
    head.loss(x,{'boxes':torch.zeros(0,4),'labels':torch.zeros(0,dtype=torch.long),'masks':torch.zeros(0,8,8,dtype=torch.uint8)}).backward()
    assert x.grad is not None and not any(p.requires_grad for p in m.parameters()) and torch.equal(before,m.bn.running_mean)
    head.predict(x.detach());assert m.calls==[True,False] and not m.training

def test_fidelity_gate_and_stage_accounting_do_not_call_warmup_task_training():
    a=api('study');assert a.fidelity_result([.001,.001],25)['passed']
    assert not a.fidelity_result([.1],25)['passed']
    s={'warmup_updates':2000,'task_updates':0,'stop_reason':'fidelity_gate_failed'}
    assert not a.task_trained(s)
    s['task_updates']=1;assert a.task_trained(s)

def test_fixed_to_adaptive_preserves_output_before_task_and_bn_stats_can_be_frozen():
    a=api('models');b=api('study');torch.manual_seed(1)
    npp=a.PaperNPP().eval();x=torch.rand(1,1,3,32,32)
    adaptive=a.PaperNPP.from_fixed(npp).eval()
    with torch.no_grad():assert torch.allclose(npp(x,35),adaptive(x,35),atol=1e-6)
    adaptive.train();b.freeze_npp_bn(adaptive)
    assert all(not m.training for m in adaptive.modules() if isinstance(m,torch.nn.BatchNorm2d))

def test_real_forward_task_chain_and_mask_loss_have_finite_npp_gradient():
    a=api('models');from reproduce.protocol import real_forward
    from reproduce.sam2_masks import mask_objective
    d=api('data');cfg=d.config('od','foreground');x=torch.rand(1,1,3,8,8);weight=torch.nn.Parameter(torch.tensor(.8))
    y=x*weight;proxy=a.FrameProxy(torch.nn.Identity())
    reconstruction=real_forward(y*.9,torch.full_like(y,.6));rate=real_forward(y.mean(),torch.tensor(.05))
    loss,parts=mask_objective(cfg,rate,reconstruction.square().mean(),x,y,35,torch.ones(1,1,1,8,8))
    loss.backward();assert torch.isfinite(weight.grad) and weight.grad.abs()>0
    assert torch.allclose(reconstruction,torch.full_like(y,.6)) and float(rate)==pytest.approx(.05)

def test_task_gradient_check_rejects_detached_or_zero_task_paths():
    a=api('study');model=torch.nn.Linear(1,1,bias=False);x=torch.ones(1,1)
    loss=model(x).square().mean();assert a.task_gradient_norm(loss,model)>0
    with pytest.raises(RuntimeError):a.task_gradient_norm(torch.tensor(1.),model)
    with pytest.raises(RuntimeError):a.task_gradient_norm(model(x).sum()*0,model)

def test_resume_rejects_changed_config_source_or_proxy_before_state_use(tmp_path):
    a=api('study');cfg={'task':'od'};saved={'config_sha256':'bad','source_sha256':'source'}
    with pytest.raises(ValueError):a.validate_resume(saved,cfg,'source')

def test_resumed_task_uses_task_fidelity_threshold_and_clears_previous_stop():
    a=api('study');cfg=api('data').config('od','control')
    old={'warmup_updates':2000,'task_updates':100,'stop_reason':'task_budget','warmup_calibrated':True,
         'warmup_fidelity':{'psnr_db':27.,'passed':True}}
    state=a.continuation_state(old)
    assert old['stop_reason']=='task_budget' and state['stop_reason'] is None
    assert state['previous_session_stop_reason']=='task_budget' and state['warmup_calibrated']
    assert a.stage_fidelity_threshold(cfg,'task',state)==20
    assert a.fidelity_result([10**(-22/10)],a.stage_fidelity_threshold(cfg,'task',state))['passed']
    assert a.stage_fidelity_threshold(cfg,'warmup',{'task_updates':0})==25

def test_task_constructor_isolated_from_saved_training_rng():
    a=api('study');torch.manual_seed(31);saved=torch.get_rng_state().clone()
    model=a.initialize_task(lambda:torch.nn.Linear(3,2))
    assert isinstance(model,torch.nn.Linear) and torch.equal(saved,torch.get_rng_state())

def test_warmup_can_converge_before_cap_and_never_restarts_task_resume():
    a=api('study');cfg=api('data').config('od','control')
    state={'warmup_updates':2000,'task_updates':0,'warmup_converged':False}
    assert a.continue_warmup(cfg,state,'warmup')
    assert a.warmup_ready(cfg,state,{'passed':True})
    assert not a.warmup_ready(cfg,state,{'passed':False})
    assert not a.continue_warmup(cfg,state,'task')
    state['warmup_converged']=True;assert not a.continue_warmup(cfg,state,'warmup')
    state['warmup_updates']=1000;assert not a.warmup_ready(cfg,state,{'passed':True})

def test_coco_uncompressed_and_compressed_rle_are_decoded_in_original_geometry():
    a=api('data');from pycocotools import mask as cm
    rle={'size':[2,3],'counts':[0,2,4]};expected=torch.tensor([[1,0,0],[1,0,0]],dtype=torch.uint8)
    assert torch.equal(a.decode_mask(rle,2,3),expected)
    encoded=cm.frPyObjects(rle,2,3);encoded['counts']=encoded['counts'].decode('ascii')
    assert torch.equal(a.decode_mask(encoded,2,3),expected)

def test_tracking_prediction_interface_accepts_only_initial_box():
    a=api('models');calls=[]
    backend=SimpleNamespace(track=lambda x,box:calls.append(box) or [[1,2,3,4]]*x.shape[1])
    head=a.TrackingTask.__new__(a.TrackingTask);head.backend=backend
    result=head.predict(torch.zeros(1,4,3,8,8),[1,2,3,4])
    assert calls==[[1,2,3,4]] and len(result)==4

def test_predicted_instance_mask_is_encoded_at_original_coco_resolution():
    a=api('study');d=api('data');mask=torch.zeros(1,1,4,6);mask[:,:,1:3,2:4]=1
    pred={'boxes':torch.tensor([[2.,1.,4.,3.]]),'scores':torch.tensor([.9]),'labels':torch.tensor([17]),'masks':mask}
    rows=a.coco_predictions(pred,{'image_id':9,'original_width':12,'original_height':8},(4,6),'is')
    assert rows[0]['category_id']==17 and rows[0]['image_id']==9 and 'bbox' not in rows[0]
    binary=d.decode_mask(rows[0]['segmentation'],8,12)
    assert binary.shape==(8,12) and binary.sum()>0

def test_real_torchvision_maskrcnn_has_frozen_weights_and_input_gradient():
    a=api('models');from torchvision.models.detection import maskrcnn_resnet50_fpn
    old_threads=torch.get_num_threads();torch.set_num_threads(2)
    try:
        model=maskrcnn_resnet50_fpn(weights=None,weights_backbone=None,min_size=64,max_size=64,
            rpn_pre_nms_top_n_train=32,rpn_post_nms_top_n_train=16,rpn_batch_size_per_image=16,box_batch_size_per_image=16)
        task=a.CocoTask('is',model);x=torch.rand(1,1,3,64,64,requires_grad=True);mask=torch.zeros(1,64,64,dtype=torch.uint8);mask[:,16:48,16:48]=1
        target={'boxes':torch.tensor([[16.,16.,48.,48.]]),'labels':torch.tensor([1]),'masks':mask}
        loss=task.loss(x,target);loss.backward()
        assert torch.isfinite(loss) and x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum()>0
        assert all(p.grad is None for p in task.model.parameters())
    finally:torch.set_num_threads(old_threads)
