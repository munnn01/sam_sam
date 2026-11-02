"""Offline frozen SAM 2 mask regularization; no teacher in the NPP gradient path."""
from contextlib import contextmanager
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import torch
from torch.nn import functional as F

from .assets import digest,download
from .data import ordered_subset
from .protocol import task_lambda
from .training import config_digest,log_event,rng_state,restore_rng
from .tensor_bytes import raw_tensor_bytes

SAM_COMMIT='2b90b9f5ceec907a1c18123530e92e794ad901a4'
SAM_CHECKPOINT='7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69'


def sam_config(task,arm):
    from .image_study import study_config
    if task not in ('od','ic') or arm not in ('control','foreground','foreground_bg'):raise ValueError('image task and declared SAM arm required')
    cfg=study_config(task,'screen',1.)
    cfg.update(study_profile='sam2',sam_arm=arm,train_sources=256,fixed_steps=256,adaptive_steps=256,
        validation_sources=64 if task=='od' else 512,max_session_seconds=7200,evaluation_reserve=1800,
        mask_prepare_seconds=1800,scope='sam2_offline_train_subset_screen_not_paper_schedule')
    cfg['sam2']=dict(source_commit=SAM_COMMIT,checkpoint_sha256=SAM_CHECKPOINT,
        model='sam2.1_hiera_tiny',model_cfg='configs/sam2.1/sam2.1_hiera_t.yaml',
        points_per_side=16,points_per_batch=32,pred_iou_thresh=.8,stability_score_thresh=.95,
        min_area_fraction=.005,max_area_fraction=.6,max_masks=8,smoothing=5,
        foreground_extra=0 if arm=='control' else 3,gamma=1 if arm=='foreground_bg' else 0,
        mask_kind='confidence_weighted_smoothed_binary_AMG_not_probability')
    cfg['fidelity_notes']+=['SAM 2 extension, not original paper; TRAIN-only automatic prompts.',
        'Matched 256-source fixed rendered crops, 256 fixed +256 adaptive updates; validation subset.']
    return cfg


def mask_settings(settings):
    # Loss-arm changes do not change teacher/cache identity.
    return {k:v for k,v in settings.items() if k not in ('foreground_extra','gamma')}


def tensor_digest(value):
    return hashlib.sha256(str((value.dtype,tuple(value.shape))).encode()+raw_tensor_bytes(value)).hexdigest()


def soft_mask(rows,size,settings):
    h,w=size;accepted=[]
    for row in rows:
        mask=torch.as_tensor(row['segmentation'],dtype=torch.float32)
        if mask.shape!=(h,w) or not torch.isfinite(mask).all():raise ValueError('SAM mask geometry/nonfinite mismatch')
        area=float(mask.mean());iou=float(row['predicted_iou']);stability=float(row['stability_score'])
        if (settings['min_area_fraction']<=area<=settings['max_area_fraction']
            and settings['pred_iou_thresh']<=iou<=1 and settings['stability_score_thresh']<=stability<=1):
            accepted.append((iou*stability,mask))
    accepted=sorted(accepted,key=lambda v:v[0],reverse=True)[:settings['max_masks']]
    m=torch.zeros(h,w)
    for confidence,value in accepted:m=torch.maximum(m,confidence*value)
    k=settings['smoothing'];m=F.avg_pool2d(m[None,None],k,stride=1,padding=k//2)[None].clamp(0,1)
    return m,dict(accepted=len(accepted),candidates=len(rows),coverage=float(m.mean()),empty=not accepted)


class MaskCache:
    def __init__(self,root,settings):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True);self.settings=mask_settings(settings)
    def path(self,row):return self.root/(hashlib.sha256(row['id'].encode()).hexdigest()+'.pt')
    def identity(self,row,x):
        return dict(source=row['id'],source_sha256=digest(row['path']),pixels_sha256=tensor_digest(x),
            shape=list(x.shape),teacher=self.settings)
    def store(self,row,x,mask):
        if mask.shape!=(*x.shape[:2],1,*x.shape[-2:]) or not torch.isfinite(mask).all() or mask.min()<0 or mask.max()>1:
            raise ValueError('finite aligned mask in [0,1] required')
        mask=mask.detach().cpu().to(torch.float16).contiguous()
        identity=self.identity(row,x)
        torch.save(dict(identity=identity,identity_sha256=config_digest(identity),mask=mask,mask_sha256=tensor_digest(mask)),self.path(row))
    def load(self,row,x):
        value=torch.load(self.path(row),map_location='cpu',weights_only=False);identity=self.identity(row,x);m=value['mask']
        if (value['identity']!=identity or value['identity_sha256']!=config_digest(identity)
            or value['mask_sha256']!=tensor_digest(m) or m.shape!=(*x.shape[:2],1,*x.shape[-2:])
            or not torch.isfinite(m).all() or m.min()<0 or m.max()>1):raise ValueError('mask cache identity/content differs')
        return m.float()


def mask_objective(cfg,rate,task,x,y,qp,mask):
    m=mask.detach().to(device=y.device,dtype=y.dtype)
    if m.shape!=(*y.shape[:2],1,*y.shape[-2:]) or not torch.isfinite(m).all() or m.min()<0 or m.max()>1:raise ValueError('aligned finite mask required')
    residual=y-x;w=1+cfg['sam2']['foreground_extra']*m
    distortion=(w*residual.square()).mean()/w.mean()
    flat=residual.flatten(0,1);highpass=(flat-F.avg_pool2d(F.pad(flat,(1,1,1,1),mode='replicate'),3,stride=1)).reshape_as(residual)
    background=((1-m)*highpass.square()).mean()
    loss=rate+cfg['lambda_scale']*task_lambda(qp)*task+cfg['beta']*distortion+cfg['sam2']['gamma']*background
    return loss,dict(weighted_distortion=distortion,background_highpass=background,mask_coverage=m.mean(),residual_mse=residual.square().mean())


def rendered_sample(row,cfg,sample):
    # Fixed per-source crop makes offline masks exact for IC as well as OD.
    state=random.getstate()
    random.seed(int(hashlib.sha256((str(cfg['seed'])+'/'+row['id']).encode()).hexdigest()[:16],16))
    try:return sample(row,cfg,True)
    finally:random.setstate(state)


@contextmanager
def mask_context(cfg,progress,out,sample):
    from . import training
    before=(training.load_sample,training.image_objective,training.sample_training);current={};calls=0
    def load(row,settings,training=False):
        if 'sam_cache' not in row:raise ValueError('training source has no offline mask cache')
        x,target,meta=rendered_sample(row,settings,sample)
        current.update(mask=MaskCache(row['sam_cache'],cfg['sam2']).load(row,x),pixels=tensor_digest(x),source=row['id'])
        return x,target,meta
    def objective(rate,task,x,y,qp):
        nonlocal calls
        if current.get('pixels')!=tensor_digest(x):raise ValueError('current mask and NPP pixels differ')
        loss,parts=mask_objective(cfg,rate,task,x,y,qp,current['mask'])
        if not torch.isfinite(loss):raise RuntimeError('nonfinite SAM objective')
        calls+=1
        if calls==1 or calls%101==0:
            log_event(out,'sam_mask_loss',arm=cfg['sam_arm'],qp=qp,source=current['source'],
                **{k:float(v.detach()) for k,v in parts.items()})
        return loss
    training.load_sample=load;training.image_objective=objective
    # Native generic-data retries would silently drop a corrupt cached source.
    training.sample_training=lambda rows,settings,destination:load(rows[random.randrange(len(rows))],settings,True)
    try:yield
    finally:training.load_sample,training.image_objective,training.sample_training=before


def build_teacher(settings,out):
    # Install pinned official source without replacing the Kaggle Torch environment.
    environment=dict(os.environ,SAM2_BUILD_CUDA='0')
    commands=[['hydra-core==1.3.2','iopath==0.1.10'],
        ['--no-deps',f'https://github.com/facebookresearch/sam2/archive/{SAM_COMMIT}.zip']]
    with (out/'sam_install.log').open('w') as f:
        for packages in commands:
            p=subprocess.run([sys.executable,'-m','pip','install','--no-build-isolation',*packages],env=environment,stdout=f,stderr=subprocess.STDOUT,timeout=900)
            if p.returncode:raise RuntimeError('SAM dependency install failed; see sam_install.log')
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    checkpoint=out/'sam2.1_hiera_tiny.pt'
    receipt=download('https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt',checkpoint)
    if receipt['sha256']!=settings['checkpoint_sha256']:raise ValueError('official SAM checkpoint SHA differs')
    model=build_sam2(settings['model_cfg'],str(checkpoint),device='cuda',apply_postprocessing=False).eval().requires_grad_(False)
    generator=SAM2AutomaticMaskGenerator(model,points_per_side=settings['points_per_side'],points_per_batch=settings['points_per_batch'],
        pred_iou_thresh=settings['pred_iou_thresh'],stability_score_thresh=settings['stability_score_thresh'],
        crop_n_layers=0,min_mask_region_area=0,output_mode='binary_mask')
    (out/'sam_teacher.json').write_text(json.dumps(dict(source_commit=SAM_COMMIT,checkpoint=receipt,settings=mask_settings(settings),
        frozen=not any(p.requires_grad for p in model.parameters()),precision='float32_no_amp_T4_compatible'),indent=2))
    return generator


def prepare_training(cfg,rows,out,budget,sample):
    selected=ordered_subset(rows,cfg['train_sources'],cfg['seed']);cache=MaskCache(out/'sam_masks',cfg['sam2'])
    saved=rng_state();started=time.monotonic();teacher=None;receipts=[];result=[]
    try:
        if cfg['sam_arm']!='control':teacher=build_teacher(cfg['sam2'],out)
        for row in selected:
            budget.require('sam_mask_prepare')
            if time.monotonic()-started>=cfg['mask_prepare_seconds']:raise RuntimeError('SAM mask budget exhausted before complete matched cohort')
            x,_,_=rendered_sample(row,cfg,sample)
            if teacher is None:mask=torch.zeros(*x.shape[:2],1,*x.shape[-2:]);receipt=dict(accepted=0,coverage=0,empty=True,control_no_teacher=True)
            else:
                rgb=(x[0,0].permute(1,2,0)*255).round().clamp(0,255).byte().numpy()
                with torch.inference_mode():generated=teacher.generate(rgb)
                mask,receipt=soft_mask(generated,x.shape[-2:],cfg['sam2'])
            cache.store(row,x,mask);receipts.append(dict(source=row['id'],pixels_sha256=tensor_digest(x),**receipt))
            result.append(dict(row,sam_cache=str(cache.root)))
            if len(result)==1 or len(result)%32==0:log_event(out,'sam_cache_progress',arm=cfg['sam_arm'],sources=len(result),target=len(selected))
    finally:
        del teacher;gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
        restore_rng(saved)
    info=dict(sources=receipts,rendered_cohort_sha256=config_digest([(r['source'],r['pixels_sha256']) for r in receipts]),
        teacher_settings=mask_settings(cfg['sam2']),empty_sources=sum(r['empty'] for r in receipts),
        mean_coverage=sum(r['coverage'] for r in receipts)/len(receipts),wall_seconds=time.monotonic()-started)
    (out/'sam_cache.json').write_text(json.dumps(info,indent=2))
    cfg['sam2_runtime']=dict(rendered_cohort_sha256=info['rendered_cohort_sha256'],teacher_settings=info['teacher_settings'])
    log_event(out,'sam_cache_ready',arm=cfg['sam_arm'],sources=len(result),empty_sources=info['empty_sources'],mean_coverage=info['mean_coverage'])
    return result
