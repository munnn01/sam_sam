"""Bounded image-paper recipes and declared parameter screens; original core unchanged."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from .protocol import recipe,task_lambda,QPS
from .assets import digest,ensure_bpg,records_for_task,dataset_mount
from .codecs import RealCodec,CodecOutput
from .data import load_sample as native_sample,ordered_subset
from .training import config_digest,records_digest,log_event,trained_npp_steps
from .proxy_recovery import Budget,BudgetExpired,model_digest,filtered


def study_config(task,profile,lambda_scale=1.):
    if task not in ('od','ic','proxy') or profile not in ('screen','full','proxy'):
        raise ValueError('image tasks/profiles required')
    if (task=='proxy')!=(profile=='proxy') or not math.isfinite(lambda_scale) or lambda_scale<=0:
        raise ValueError('invalid proxy profile/lambda scale')
    cfg=recipe('ic' if task=='proxy' else task)
    cfg.update(task=task,study_task=task,study_profile=profile,lambda_scale=float(lambda_scale),
        backbone='fcos_resnet50_fpn' if task=='od' else 'resnet50',log_every=101,
        proxy_steps=0,fixed_steps=120000 if profile=='full' else 1000,
        adaptive_steps=100000 if profile=='full' else 1000,
        npp_size=800 if task=='od' else 224,checkpoint_every=250,
        max_session_seconds=21600 if profile=='full' else 7200,
        evaluation_reserve=3600 if profile=='full' else 1800,
        validation_sources=(512 if profile=='full' else 128) if task=='od' else (2048 if profile=='full' else 1024),
        scope='image_paper_warmstart_'+profile,random_adaptive_qp=True,
        fidelity_notes=cfg['fidelity_notes']+[
            'Separate study: frozen proxy-only warmstart; original NPP/Adam/counters not migrated.',
            'FCOS family replaces FasterRCNN for OD; Torchvision checkpoint differs from original OpenMMLab.',
            'QP35 substitutes paper fixedQP34; lambda multiplier is an explicit experiment.',
            'OD800/1333 geometry; adaptive QPs randomly sampled; short screens are not full schedules.'])
    if profile=='proxy':
        cfg.update(backbone='none',proxy_steps=400000,fixed_steps=0,adaptive_steps=0,
            npp_size=256,batch_size=4,max_session_seconds=14400,evaluation_reserve=1800,
            validation_sources=16,scope='Flickr30k_proxy_only_global400k',
            fidelity_notes=['Flickr30k/version1 source-disjoint crops; no downstream NPP update.',
                '400k counts global bank updates; per-QP counts explicit; on-the-fly crops rather than a released800k patch manifest.'])
    return cfg


def study_objective(cfg,rate,task,x,y,qp):
    return rate+cfg['lambda_scale']*task_lambda(qp)*task+cfg['beta']*(x-y).square().mean()


def expected_config(cfg):
    if cfg.get('study_profile')=='sam2':
        from .sam2_masks import sam_config
        return sam_config(cfg['study_task'],cfg['sam_arm'])
    return study_config(cfg['study_task'],cfg['study_profile'],cfg['lambda_scale'])


def load_proxy_input(path,sha,task):
    if digest(path)!=sha:raise ValueError('proxy input SHA differs')
    saved=torch.load(path,map_location='cpu',weights_only=False)
    if 'config' in saved:
        cfg=saved['config']
        if saved.get('config_sha256')!=config_digest(cfg):raise ValueError('checkpoint config digest differs')
        saved=dict(task=cfg['task'],qps=cfg['qps'],proxy_quality=cfg['proxy_quality'],proxy=saved['proxy'],
            original_checkpoint_sha256=sha,original_training_source=saved['source_manifest_sha256'],
            original_phase_steps=saved['state']['phase_steps'])
    if saved['task']!=task or saved['qps']!=list(QPS):raise ValueError('proxy task/QP differs')
    return {k:saved[k] for k in ('task','qps','proxy_quality','proxy','original_checkpoint_sha256',
        'original_training_source','original_phase_steps')}


def study_sample(record,cfg,training=False):
    if record['task']=='flickr':
        from PIL import Image
        from .data import _tensor
        with Image.open(record['path']) as source:image=source.convert('RGB')
        w,h=image.size
        if min(w,h)<256:
            scale=256/min(w,h);image=image.resize((math.ceil(w*scale),math.ceil(h*scale)),Image.Resampling.BILINEAR)
        w,h=image.size;xx=random.randrange(w-255) if training else (w-256)//2;yy=random.randrange(h-255) if training else (h-256)//2
        return _tensor([image.crop((xx,yy,xx+256,yy+256))]),None,dict(id=record['id'])
    x,target,meta=native_sample(record,cfg,training)
    if record['task']=='od' and max(x.shape[-2:])>1333:
        h,w=x.shape[-2:];scale=1332/max(h,w);nh=max(2,2*round(h*scale/2));nw=max(2,2*round(w*scale/2))
        x=F.interpolate(x[:,0],size=(nh,nw),mode='bilinear',align_corners=False)[:,None]
        target=dict(target,boxes=target['boxes']*target['boxes'].new_tensor([nw/w,nh/h,nw/w,nh/h]))
    return x,target,meta


def guarded_adam_class(cfg,progress,out=None):
    native=torch.optim.Adam
    class GuardedAdam(native):
        def __init__(self,params,*args,**kwargs):
            params=list(params);super().__init__(params,*args,**kwargs)
            self.initial=[p.detach().clone() for p in params] if cfg['study_profile']!='proxy' else None
        def step(self,*args,**kwargs):
            params=[p for group in self.param_groups for p in group['params']]
            grads=[p.grad for p in params if p.grad is not None]
            if not grads or not all(torch.isfinite(g).all() for g in grads):raise RuntimeError('optimizer rejected nonfinite/missing gradients')
            result=super().step(*args,**kwargs)
            key='proxy_optimizer_updates' if cfg['study_profile']=='proxy' else 'npp_updates'
            progress[key]=progress.get(key,0)+1
            if out is not None and (progress[key]==1 or progress[key]%101==0):
                row=dict(optimizer_updates=progress[key],phase=progress.get('current_phase'),
                    lr=[group['lr'] for group in self.param_groups])
                if self.initial is not None:
                    row['theta_delta_norm']=float(torch.sqrt(sum((p.detach()-v).square().sum() for p,v in zip(params,self.initial))))
                log_event(out,'study_optimizer',**row)
                (Path(out)/'optimizer_progress.json').write_text(json.dumps(dict(npp_updates=progress.get('npp_updates',0),**row)))
            return result
    return GuardedAdam


def require_finite(value,label):
    if not torch.isfinite(value).all():raise RuntimeError('nonfinite '+label)
    return value


@contextmanager
def study_context(cfg,progress=None,out=None,proxy=None):
    from . import training,evaluation
    originals=(training.image_objective,training.qp_for_step,training.load_sample,training.sample_training,evaluation.load_sample,
        torch.optim.Adam,training.log_event)
    if progress is None:progress={}
    torch.optim.Adam=guarded_adam_class(cfg,progress,out)
    def emit(path,stage,**fields):
        if stage=='phase_start':progress['current_phase']=fields['phase']
        return originals[-1](path,stage,**fields)
    training.log_event=emit
    training.image_objective=lambda rate,task,x,y,qp:require_finite(study_objective(cfg,rate,task,x,y,qp),'image objective')
    training.qp_for_step=lambda step,settings,phase:settings['fixed_qp'] if phase=='fixed' else random.choice(settings['qps'])
    training.load_sample=study_sample;evaluation.load_sample=study_sample
    if cfg['study_profile']=='proxy':
        def sample(rows,settings,out):
            selected=[rows[random.randrange(len(rows))] for _ in range(cfg['batch_size'])]
            samples=[study_sample(r,settings,True) for r in selected]
            return torch.cat([s[0] for s in samples]),None,dict(id=selected[0]['id'])
        training.sample_training=sample
        # Proxy accounting stays round-robin: exactly balanced update counts.
        training.qp_for_step=lambda step,settings,phase:settings['qps'][step%len(settings['qps'])]
    saved_proxy=None
    if proxy is not None and cfg['study_profile']=='proxy':
        saved_proxy=(proxy.forward,proxy.aux_loss)
        def forward(*args,**kwargs):
            value=saved_proxy[0](*args,**kwargs)
            require_finite(value.bpp,'proxy rate');require_finite(value.reconstruction,'proxy reconstruction');return value
        proxy.forward=forward;proxy.aux_loss=lambda qp:require_finite(saved_proxy[1](qp),'proxy auxiliary loss')
    try:
        if cfg.get('study_profile')=='sam2':
            from .sam2_masks import mask_context
            with mask_context(cfg,progress,out,study_sample):yield
        else:yield
    finally:
        training.image_objective,training.qp_for_step,training.load_sample,training.sample_training,evaluation.load_sample,torch.optim.Adam,training.log_event=originals
        if saved_proxy is not None:proxy.forward,proxy.aux_loss=saved_proxy


class BatchCodec:
    def __init__(self,codec):self.codec=codec
    def __call__(self,x,codec,qp):
        cpu=x.detach().cpu()
        with ThreadPoolExecutor(max_workers=2) as pool:
            values=list(pool.map(lambda i:self.codec(cpu[i:i+1],codec,qp),range(len(cpu))))
        count=sum(v.bytes for v in values);pixels=len(x)*x.shape[1]*x.shape[-2]*x.shape[-1]
        return CodecOutput(torch.cat([v.reconstruction for v in values]).to(x.device),count*8/pixels,count,'',
            hashlib.sha256(''.join(v.sha256 for v in values).encode()).hexdigest())


def flickr_split(rows,count=64):
    if len({r['id'] for r in rows})!=len(rows) or len(rows)<=count:raise ValueError('unique Flickr train/dev sources required')
    ordered=sorted(rows,key=lambda r:hashlib.sha256(('20261031/'+r['id']).encode()).hexdigest())
    return ordered[count:],ordered[:count]


def export_evaluated_model(out,cfg,state,npp):
    steps=trained_npp_steps(state)
    if steps<=0:raise ValueError('untrained NPP cannot be evaluated/exported')
    path=Path(out)/'evaluated_npp.pt'
    torch.save(dict(npp=npp.state_dict(),adaptive=getattr(npp,'adaptive',False),config=cfg,state=state,
        trained_npp_steps=steps,full_curriculum_complete=cfg['study_profile']=='full' and bool(state.get('completed')),
        source_sha256=os.environ.get('PAPER_SOURCE_SHA256'),npp_sha256=model_digest(npp)),path)
    return path


def canonicalize_proxy_masks(proxy):
    """MaskedConv2d applies this projection on every forward, even while frozen."""
    count=0
    with torch.no_grad():
        for module in proxy.modules():
            if module.__class__.__name__!='MaskedConv2d':continue
            count+=int(((module.mask==0)&(module.weight!=0)).sum())
            module.weight.mul_(module.mask)
    return dict(changed_entries=count,scope='functionally null masked weights; original input SHA retained')


class StudyAnalyzer:
    def __new__(cls,task,device='cuda'):
        from .tasks import Analyzer,freeze,fingerprint
        if task=='ic':return Analyzer(task,device)
        from torchvision.models.detection import fcos_resnet50_fpn,FCOS_ResNet50_FPN_Weights
        result=Analyzer.__new__(Analyzer);result.task=task;result.device=torch.device(device);result.tracking_checkpoint=None
        result.model=freeze(fcos_resnet50_fpn(weights=FCOS_ResNet50_FPN_Weights.COCO_V1)).to(device)
        result.description='Torchvision FCOS ResNet50 FPN COCO V1; paper default family, different OpenMMLab checkpoint'
        result.mean=torch.tensor([.485,.456,.406],device=device).view(1,3,1,1)
        result.std=torch.tensor([.229,.224,.225],device=device).view(1,3,1,1)
        result.weight_digest=fingerprint(result.model);return result


def frozen_models_unchanged(proxy,expected,analyzer):
    from .tasks import fingerprint
    return model_digest(proxy)==expected and fingerprint(analyzer.model)==analyzer.weight_digest


class GaussianNPP(nn.Module):
    def __init__(self,device):super().__init__();self.anchor=nn.Parameter(torch.zeros(1,device=device),requires_grad=False)
    def forward(self,x,qp):return filtered(x,'gaussian_0.5')


def proxy_validation(proxy,rows,cfg,codec,out,budget):
    points=[];proxy.eval()
    for i,record in enumerate(rows[:16]):
        x,_,_=study_sample(record,cfg,False);x=x.to('cuda')
        for qp in QPS:
            budget.require('proxy_validation')
            with torch.no_grad():prediction=proxy(x,qp)
            actual=codec(x,'bpg',qp);mse=float((prediction.reconstruction-actual.reconstruction).square().mean())
            points.append(dict(source=record['id'],qp=qp,real_bpp=actual.bpp,proxy_bpp=float(prediction.bpp),
                proxy_bpg_mse=mse,proxy_bpg_psnr=-10*math.log10(mse) if mse else None,bytes=actual.bytes))
        (out/'proxy_validation.json').write_text(json.dumps(points,indent=2));log_event(out,'proxy_dev_source',source=i+1,total=16)
    return points


def bounded_proxy_validation(proxy,rows,cfg,codec,out,budget):
    try:return dict(complete=True,points=proxy_validation(proxy,rows,cfg,codec,out,budget))
    except BudgetExpired as error:
        path=Path(out)/'proxy_validation.json'
        return dict(complete=False,points=json.loads(path.read_text()) if path.is_file() else [],
            stop_reason='time_budget',stopped_stage=str(error))


def record_error_progress(out,report):
    path=Path(out)/'last.pt';report['recoverable_npp_updates']=0
    if path.is_file():
        try:
            saved=torch.load(path,map_location='cpu',weights_only=False)
            report['recoverable_phase_steps']=saved['state']['phase_steps']
            report['recoverable_npp_updates']=trained_npp_steps(saved['state'])
        except Exception as error:report['checkpoint_inspection_error']=str(error)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--proxy',type=Path)
    parser.add_argument('--proxy-sha256');parser.add_argument('--resume',type=Path)
    parser.add_argument('--inputs',type=Path,default=Path('/kaggle/input'))
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    cfg=json.loads(args.config.read_text());expected=expected_config(cfg)
    if cfg!=expected:raise ValueError('study config differs from frozen recipe')
    out=args.out;out.mkdir(parents=True,exist_ok=True);budget=Budget(cfg['max_session_seconds']);started=time.time()
    report=dict(task=cfg['study_task'],profile=cfg['study_profile'],scientific_complete=False,execution_complete=False,npp_updates=0,started=started)
    try:
        if not torch.cuda.is_available():raise RuntimeError('CUDA required; no silent CPU training')
        import numpy as np
        from . import training,evaluation
        from .models import ImageNPP
        from .image_proxy import ImageProxyBank
        from .image_proxy_audit import restore_proxy_state
        from .run import preflight
        random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed']);torch.set_num_threads(2)
        torch.backends.cudnn.benchmark=False
        paths=ensure_bpg(Path('/kaggle/working/paper_assets/bpg'));codec=RealCodec(bpgenc=paths['bpgenc'],bpgdec=paths['bpgdec'])
        if cfg['study_profile']=='proxy':
            root=dataset_mount(args.inputs,'eeshawn/flickr30k')/'flickr30k_images'
            rows=[dict(id='flickr/'+p.stem,path=str(p),task='flickr',split='train') for p in sorted(root.glob('*.jpg'))]
            if not 30000<=len(rows)<=40000:raise ValueError('complete Flickr30k JPEG corpus required')
            train,validation=flickr_split(rows)
            cfg['runtime_identity']=dict(dataset_versions={'eeshawn/flickr30k':1},train_records_sha256=records_digest(train),
                validation_records_sha256=records_digest(validation))
            proxy=ImageProxyBank(cfg['proxy_quality'],pretrained=True).to('cuda')
            npp=nn.Linear(1,1).to('cuda');analyzer=SimpleNamespace(model=nn.Identity(),receipt=lambda:dict(task='proxy_only_no_analyzer'))
            runner_codec=BatchCodec(codec)
        else:
            saved=load_proxy_input(args.proxy,args.proxy_sha256,cfg['task'])
            proxy=ImageProxyBank(saved['proxy_quality'],pretrained=False);restore_proxy_state(proxy,saved['proxy'])
            report['proxy_origin']={k:v for k,v in saved.items() if k!='proxy'};del saved
            report['proxy_mask_canonicalization']=canonicalize_proxy_masks(proxy)
            train,validation,inventory=records_for_task(cfg['task'],args.inputs)
            if cfg['study_profile']=='sam2':
                from .sam2_masks import prepare_training
                train=prepare_training(cfg,train,out,budget,study_sample)
            proxy=proxy.to('cuda')
            analyzer=StudyAnalyzer(cfg['task']);cfg['runtime_identity']=training.runtime_identity(train,validation,inventory,analyzer.weight_digest)
            (out/'analyzer.json').write_text(json.dumps(analyzer.receipt(),indent=2))
            torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed']);npp=ImageNPP().to('cuda');runner_codec=codec
            report['initial_npp_sha256']=model_digest(npp);report['frozen_proxy_sha256']=model_digest(proxy)
        (out/'config.json').write_text(json.dumps(cfg,indent=2));selected=ordered_subset(validation,cfg['validation_sources'],cfg['seed'])
        (out/'cohort.json').write_text(json.dumps(dict(train_sources=len(train),validation_sources=len(validation),
            selected=[r['id'] for r in selected],runtime_identity=cfg['runtime_identity']),indent=2))
        log_event(out,'study_ready',task=cfg['task'],profile=cfg['study_profile'],lambda_scale=cfg['lambda_scale'],
            train_sources=len(train),validation_selected=len(selected),source=os.environ.get('PAPER_SOURCE_SHA256'))
        if args.resume:
            previous_state=torch.load(args.resume,map_location='cpu',weights_only=False)['state']
            report['npp_updates']=trained_npp_steps(previous_state)
        with study_context(cfg,report,out,proxy):
            if cfg['study_profile']!='proxy':
                # Preflight imports native load_sample; adapt its bound reference too.
                import reproduce.run as native_runner
                previous=native_runner.load_sample;native_runner.load_sample=training.load_sample
                try:preflight(cfg,train,npp,proxy,analyzer,codec,out)
                finally:native_runner.load_sample=previous
            remaining=max(0,budget.remaining()-cfg['evaluation_reserve'])
            state,npp=training.train(cfg,train,npp,proxy,analyzer,runner_codec,out,args.resume,max_seconds=remaining)
            report['training']=state;report['npp_updates']=trained_npp_steps(state)
            if cfg['study_profile']=='proxy':
                report['updates_by_qp']={str(q):state['phase_steps']['proxy']//4+int(i<state['phase_steps']['proxy']%4) for i,q in enumerate(QPS)}
                report['proxy_validation']=bounded_proxy_validation(proxy,selected,cfg,codec,out,budget)
            elif report['npp_updates']>0:
                export_evaluated_model(out,cfg,state,npp)
                if not frozen_models_unchanged(proxy,report['frozen_proxy_sha256'],analyzer):
                    raise RuntimeError('frozen proxy/analyzer changed')
                annotation=None
                if cfg['task']=='od':
                    root=dataset_mount(args.inputs,'awsaf49/coco-2017-dataset')
                    annotations=list(root.rglob('instances_val2017.json'))
                    if len(annotations)!=1:raise ValueError('one official COCO validation annotation required')
                    annotation=annotations[0]
                report['metrics']=evaluation.evaluate(cfg,selected,npp,analyzer,codec,out/'validation',annotation,max_seconds=budget.remaining())
                if report['metrics']['complete'] and budget.remaining()>60:
                    result=evaluation.evaluate(cfg,selected,GaussianNPP('cuda'),analyzer,codec,out/'gaussian_validation',annotation,max_seconds=budget.remaining())
                    report['gaussian_metrics']=result
            report['execution_complete']=True
    except BaseException as error:
        report['error']=f'{type(error).__name__}: {error}';record_error_progress(out,report);raise
    finally:
        report['wall_seconds']=time.time()-started;(out/'study_run.json').write_text(json.dumps(report,indent=2))
        log_event(out,'study_end',task=cfg['task'],execution_complete=report['execution_complete'],npp_updates=report['npp_updates'],error=report.get('error'))


if __name__=='__main__':main()
