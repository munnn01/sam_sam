import copy
import pytest
import torch
from reproduce.coco_codec_study import config,instance_edges,weighted_task_loss,validate_warm_start

def test_profiles_are_codec_specific_and_keep_pipeline():
 for task in ('is','od'):
  for codec in ('h264','h265'):
   c=config(task,'foreground',codec)
   assert c['codecs']==[codec] and c['qps']==[30,35,40,45]
   assert not c['identity_fallback'] and c['proxy_steps']==256 and c['task_steps']==768
   assert c['sam2']['gamma'] < .1 and c['sam2']['foreground_extra'] < 3
 with pytest.raises(ValueError):config('is','foreground','bpg')

def test_instance_edges_preserve_touching_instance_boundary():
 m=torch.zeros(2,9,9);m[0,2:7,1:4]=1;m[1,2:7,4:7]=1
 e=instance_edges(m)
 assert e.shape==(1,1,1,9,9) and e[0,0,0,4,3]>0 and e[0,0,0,4,4]>0
 assert e[0,0,0,0,0]==0

def test_mask_loss_weight_and_gradient():
 x=torch.tensor(2.,requires_grad=True)
 loss=weighted_task_loss({'loss_classifier':x,'loss_mask':x*2},3.)
 loss.backward();assert x.grad.item()==7

def test_warmstart_rejects_codec_source_and_cohort_changes():
 cfg=config('is','control','h264');cfg['runtime_identity']={'train':'a'}
 saved=dict(config=cfg,source_sha256='s',optimizer_phase='warmup',state={'task_updates':0,'warmup_converged':True},proxy={'a':1},npp={'a':1},rng={'a':1},optimizer={'a':1})
 target=config('is','foreground','h264');target['runtime_identity']={'train':'a'}
 validate_warm_start(saved,target,'s')
 for key,value in [('codecs',['h265']),('runtime_identity',{'train':'b'})]:
  wrong=copy.deepcopy(target);wrong[key]=value
  with pytest.raises(ValueError):validate_warm_start(saved,wrong,'s')
 with pytest.raises(ValueError):validate_warm_start(saved,target,'bad')

def test_pair_cannot_be_complete_with_unequal_or_truncated_updates():
 from reproduce.coco_codec_study import paired_complete
 full=dict(task_trained=True,metrics={'complete':True},state={'task_updates':768,'stop_reason':'task_schedule_finished'})
 partial=dict(task_trained=True,metrics={'complete':True},state={'task_updates':50,'stop_reason':'task_budget'})
 assert paired_complete([full,full],768)
 assert not paired_complete([full,partial],768)
 assert not paired_complete([partial,partial],768)
