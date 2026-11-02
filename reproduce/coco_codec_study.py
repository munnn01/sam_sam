"""Matched COCO intra-image H264/H265 experiments, with shared warm-up/proxy."""
import argparse
import gc
import json
from pathlib import Path
import sys
import torch
from torch.nn import functional as F
from . import tri_task_study as study
from .tri_task_data import config as base_config
from .tri_task_models import CocoTask
from .sam2_masks import mask_objective as base_objective

def config(task,arm,codec='h264'):
    if task not in ('is','od') or codec not in ('h264','h265'):raise ValueError('COCO task and real video codec required')
    cfg=base_config(task,arm)
    cfg.update(study_profile='coco_codecs',codecs=[codec],lr=3e-5,task_steps=768,proxy_steps=256,
        validation_sources=64,max_session_seconds=6000,evaluation_reserve=1800,
        mask_task_weight=3. if task=='is' else 1.,boundary_weight=2. if task=='is' and arm=='foreground' else 0.,
        scope='COCO_independent_single_frame_intra_not_temporal_video')
    cfg['sam2'].update(foreground_extra=(1. if task=='is' else 1.5) if arm=='foreground' else 0.,
        gamma=.02 if task=='od' and arm=='foreground' else 0.)
    cfg['fidelity_notes']+=['Codec-specific proxy refit; exact shared warm-up/proxy/RNG per paired worker.',
        'Each COCO image is one independent H264/H265 stream; include all stream headers; no inter-frame claim.',
        'Mask task loss weight3 for both IS arms; improved IS adds TRAIN-GT instance boundary fidelity; bundled intervention.']
    return cfg

def instance_edges(masks):
    if masks.ndim!=3:raise ValueError('N,H,W instance masks required')
    if not len(masks):return masks.new_zeros((1,1,1,*masks.shape[-2:]),dtype=torch.float32)
    v=masks.float()[:,None]
    edges=(F.max_pool2d(v,3,1,1)+F.max_pool2d(-v,3,1,1)).clamp(0,1)
    return edges.amax(0,keepdim=True)[None]

def weighted_task_loss(parts,weight):return sum(v*(weight if k=='loss_mask' else 1.) for k,v in parts.items())

def paired_complete(reports,target):
    return len(reports)==2 and all(v.get('task_trained') and v.get('metrics',{}).get('complete')
        and v.get('state',{}).get('task_updates')==target
        and v.get('state',{}).get('stop_reason')=='task_schedule_finished' for v in reports)

def direction_probe(proxy,samples,cfg,codec,out,budget):
    """Finite codec comparison in the proxy rate-gradient direction, never a gain claim."""
    rows=[];proxy.eval();proxy.bank.train()  # noise relaxation carries rate gradients; eval rounding does not
    for sample,_,meta in samples[:2]:
        for q in cfg['qps']:
            budget.require('codec_gradient_probe')
            x=sample.to('cuda').detach().requires_grad_(True)
            with torch.random.fork_rng(devices=[x.device]):
                torch.manual_seed(cfg['seed']+q)
                rate=proxy(x,q).bpp;gradient=torch.autograd.grad(rate,x)[0]
            if not torch.isfinite(gradient).all() or gradient.abs().sum()==0:raise RuntimeError('invalid proxy rate gradient')
            step=gradient.sign()*(2/255)
            with torch.no_grad():
                minus=codec((x-step).clamp(0,1),cfg['codecs'][0],q).bpp
                plus=codec((x+step).clamp(0,1),cfg['codecs'][0],q).bpp
            rows.append(dict(source=meta['id'],qp=q,minus_bpp=minus,plus_bpp=plus,aligned=minus<plus,tied=minus==plus))
    result=dict(rows=rows,aligned=sum(r['aligned'] for r in rows),tied=sum(r['tied'] for r in rows),total=len(rows),
        scope='original TRAIN-dev inputs, finite step2/255; local diagnostic, not proof on NPP distribution')
    (out/'rate_direction.json').write_text(json.dumps(result,indent=2))
    proxy.eval()
    return result

def validate_warm_start(saved,cfg,source):
    a=dict(saved['config']);b=dict(cfg)
    for v in (a,b):
        v.pop('sam_arm');v.pop('boundary_weight');v['sam2']=dict(v['sam2'])
        for key in ('foreground_extra','gamma'):v['sam2'].pop(key)
    if a!=b or saved.get('source_sha256')!=source:raise ValueError('warm-start source/config/cohort differs')
    if saved.get('optimizer_phase')!='warmup' or saved['state'].get('task_updates')!=0 or not saved['state'].get('warmup_converged'):
        raise ValueError('converged task-free warm-up required')
    if not all(saved.get(k) for k in ('npp','proxy','optimizer','rng')):raise ValueError('incomplete warm-up snapshot')

class WeightedTask(CocoTask):
    def loss(self,x,target):
        self.model.train()
        for m in self.model.modules():
            if isinstance(m,torch.nn.modules.batchnorm._BatchNorm):m.eval()
        try:
            parts=self.model([x[0,0]],[{k:v.to(x.device) for k,v in target.items()}])
            self.last_components={k:float(v.detach()) for k,v in parts.items()}
            study._active_task_components=self.last_components
            study._codec_edge=instance_edges(target['masks']).to(x.device) if 'masks' in target else None
            return weighted_task_loss(parts,3. if self.task=='is' else 1.)
        finally:self.model.eval()

def objective(cfg,rate,task,x,y,q,mask):
    loss,parts=base_objective(cfg,rate,task,x,y,q,mask)
    edge=getattr(study,'_codec_edge',None)
    boundary=rate.new_zeros(()) if edge is None else (((y-x).square()*edge).sum()/(edge.sum()*3).clamp_min(1))
    parts['boundary_distortion']=boundary
    for k,v in getattr(study,'_active_task_components',{}).items():parts['native_'+k]=rate.new_tensor(v)
    return loss+cfg['boundary_weight']*boundary,parts

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--proxy',required=True)
    parser.add_argument('--proxy-sha256',required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();initial=json.loads(args.config.read_text());task=initial['task'];codec=initial['codecs'][0]
    if initial!=config(task,'control',codec):raise ValueError('paired recipe differs')
    study.config=lambda t,a:config(t,a,codec);study.CocoTask=WeightedTask;study.mask_objective=objective
    args.out.mkdir(parents=True,exist_ok=True);warm=args.out/'shared_warmup.pt'
    for arm in ('control','foreground'):
        cfg=config(task,arm,codec);path=args.out/(arm+'_config.json');path.write_text(json.dumps(cfg,indent=2))
        sys.argv=['tri_task_study','--config',str(path),'--proxy',args.proxy,'--proxy-sha256',args.proxy_sha256,
            '--out',str(args.out/arm),'--export-warmup',str(warm)]
        if arm=='foreground':sys.argv+=['--warm-start',str(warm)]
        study.main();gc.collect();torch.cuda.empty_cache()
    a=json.loads((args.out/'control/study_run.json').read_text());b=json.loads((args.out/'foreground/study_run.json').read_text())
    if not paired_complete([a,b],initial['task_steps']):
        raise RuntimeError('paired study incomplete; inspect per-arm reports')
    if a['warmup_npp_sha256']!=b['warmup_npp_sha256'] or a['frozen_proxy_sha256']!=b['frozen_proxy_sha256']:
        raise RuntimeError('paired warm-up/proxy differs')
    (args.out/'paired_results.json').write_text(json.dumps(dict(task=task,codec=codec,complete=True,
        warmup_npp_sha256=a['warmup_npp_sha256'],arms={arm:v['metrics'] for arm,v in (('control',a),('foreground',b))},
        scope=initial['scope']),indent=2))

if __name__=='__main__':main()
