"""Run original-task image/video paper recipes on real data and real standard codecs."""
import argparse
import json
import os
from pathlib import Path
import random
import shutil
import time
import traceback
import numpy as np
import torch
from .protocol import recipe,real_forward
from .assets import ensure_bpg,ensure_tracking,records_for_task,digest
from .models import ImageNPP,VideoNPP
from .virtual_codec import VirtualCodec
from .image_proxy import ImageProxyBank
from .tasks import Analyzer,freeze,fingerprint
from .codecs import RealCodec
from .data import load_sample,ordered_subset
from .training import (train,log_event,rng_state,restore_rng,image_objective,video_objective,
                       trained_npp_steps,scientific_complete,runtime_identity)

def gradient_receipt(task_loss,total,npp):
    params=list(npp.parameters())
    def norm(loss,name,retain):
        if not loss.requires_grad:raise RuntimeError(name+' gradient path is detached')
        grads=torch.autograd.grad(loss,params,allow_unused=True,retain_graph=retain)
        grads=[g.detach() for g in grads if g is not None]
        if not grads or not all(torch.isfinite(g).all() for g in grads):raise RuntimeError(name+' gradient is missing/nonfinite')
        value=float(torch.sqrt(sum(g.square().sum() for g in grads)))
        if value<=0 or not np.isfinite(value):raise RuntimeError(name+' gradient is zero/nonfinite')
        return value
    return dict(task_npp_grad_norm=norm(task_loss,'task',True),npp_grad_norm=norm(total,'objective',False))

def restore_validation_cache(resume,out):
    previous=Path(resume).parent/'validation';destination=Path(out)/'validation'
    if previous.is_dir() and previous.resolve()!=destination.resolve():
        shutil.copytree(previous,destination,dirs_exist_ok=True)

def preflight(cfg,rows,npp,proxy,analyzer,codec,out):
    """Check real analyzers/gradients without mutating full-run initialization."""
    device=next(npp.parameters()).device;state=rng_state();before=fingerprint(npp)
    npp.eval();freeze(proxy)
    if cfg['paper']=='image':proxy.train()  # differentiable quantization noise; no weight update
    record=ordered_subset(rows,1,cfg['seed'])[0]
    x,target,meta=load_sample(record,cfg,False);x=x.to(device)
    y=npp(x,35);pred=proxy(y,35)
    streams=[]
    for cod in cfg['codecs']:
        for qp in cfg['qps']:
            real=codec(y,cod,qp,out/'preflight_streams'/cod/str(qp))
            streams.append(dict(codec=cod,qp=qp,bytes=real.bytes,bpp=real.bpp,sha256=real.sha256))
    if cfg['paper']=='image':
        actual=codec(y,'bpg',35)
        decoded=real_forward(pred.reconstruction,actual.reconstruction)
        rate=real_forward(pred.bpp,pred.bpp.new_tensor(actual.bpp))
        task_loss=analyzer.loss(decoded,target);loss=image_objective(rate,task_loss,x,y,35)
    else:
        task_loss=analyzer.loss(pred.reconstruction,target)
        loss=video_objective(pred.bpp,task_loss,x,pred.reconstruction)
    gradients=gradient_receipt(task_loss,loss,npp)
    if fingerprint(npp)!=before or fingerprint(analyzer.model)!=analyzer.weight_digest:
        raise RuntimeError('preflight changed model/BN state')
    receipt=dict(passed=True,source=meta['id'],**gradients,task_loss=float(task_loss.detach()),
                 analyzer=analyzer.receipt(),streams=streams,npp_state_unchanged=True,scope='training-pipeline preflight, not RD results')
    (out/'preflight.json').write_text(json.dumps(receipt,indent=2))
    restore_rng(state);log_event(out,'preflight_passed',task=cfg['task'],**receipt)
    if torch.cuda.is_available():torch.cuda.empty_cache()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',required=True,choices=['od','ic','ar','ot'])
    parser.add_argument('--mode',choices=['full','pilot'],default='full')
    parser.add_argument('--inputs',type=Path,default=Path('/kaggle/input'))
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--assets',type=Path,default=Path('/kaggle/working/paper_assets'))
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--preflight-only',action='store_true')
    args=parser.parse_args();cfg=recipe(args.task,args.mode)
    out=args.out;out.mkdir(parents=True,exist_ok=True)
    random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed'])
    torch.set_num_threads(2)
    if not torch.cuda.is_available():raise RuntimeError('CUDA GPU is required; no silent CPU experiment')
    torch.cuda.manual_seed_all(cfg['seed']);torch.backends.cudnn.benchmark=False
    device='cuda';receipt=dict(task=args.task,mode=args.mode,started=time.time(),execution_complete=False,scientific_complete=False)
    (out/'config.json').write_text(json.dumps(cfg,indent=2))
    try:
        log_event(out,'bootstrap',task=args.task,gpu=torch.cuda.get_device_name(),torch=torch.__version__,
                  qps=cfg['qps'],source_manifest_sha256=os.environ.get('PAPER_SOURCE_SHA256'))
        assets={};codec_args=dict(preset=cfg['preset'],gop=cfg['gop'],fps=cfg['fps'])
        if cfg['paper']=='image':
            assets['bpg']=ensure_bpg(args.assets/'bpg')
            codec_args.update(bpgenc=assets['bpg']['bpgenc'],bpgdec=assets['bpg']['bpgdec'])
        elif args.task=='ot':assets['tracking']=ensure_tracking(args.assets/'tracking')
        (out/'assets.json').write_text(json.dumps(assets,indent=2))
        log_event(out,'assets_ready',task=args.task)
        rows,validation,inventory=records_for_task(args.task,args.inputs,args.assets)
        selected=ordered_subset(validation,cfg['validation_sources'],cfg['seed'])
        inventory['validation_selected']=[r['id'] for r in selected]
        inventory['train_id_sha256']=__import__('hashlib').sha256('\n'.join(sorted(r['id'] for r in rows)).encode()).hexdigest()
        inventory['validation_id_sha256']=__import__('hashlib').sha256('\n'.join(sorted(r['id'] for r in validation)).encode()).hexdigest()
        (out/'data_inventory.json').write_text(json.dumps(inventory,indent=2))
        log_event(out,'data_ready',task=args.task,train_sources=len(rows),validation_sources=len(validation),
                  validation_selected=len(selected),full_original_training_corpus=inventory['full_original_training_corpus'])
        analyzer=Analyzer(args.task,device,assets.get('tracking',{}).get('checkpoint'))
        cfg['runtime_identity']=runtime_identity(rows,validation,inventory,analyzer.weight_digest)
        (out/'config.json').write_text(json.dumps(cfg,indent=2))
        (out/'analyzer.json').write_text(json.dumps(analyzer.receipt(),indent=2))
        log_event(out,'analyzer_ready',**analyzer.receipt())
        if cfg['paper']=='image':
            npp=ImageNPP().to(device);proxy=ImageProxyBank(cfg['proxy_quality']).to(device)
        else:
            npp=VideoNPP(cfg['video_width']).to(device)
            proxy=VirtualCodec(cfg['block_size'],cfg['motion_radius']).to(device)
        codec=RealCodec(**codec_args)
        preflight(cfg,rows,npp,proxy,analyzer,codec,out)
        if args.preflight_only:
            receipt.update(execution_complete=True,preflight_only=True,scientific_complete=False)
            return
        remaining=max(0,cfg['max_session_seconds']-(time.time()-receipt['started']))
        state,npp=train(cfg,rows,npp,proxy,analyzer,codec,out,args.resume,max_seconds=remaining)
        receipt.update(training=state,trained_npp_steps=trained_npp_steps(state))
        if state['completed']:
            from .evaluation import evaluate
            annotation=None
            if args.task=='od':
                from .assets import dataset_mount,DATASETS
                files=list(dataset_mount(args.inputs,DATASETS['od'][0]).rglob('instances_val2017.json'))
                if len(files)!=1:raise ValueError('one official COCO val GT required')
                annotation=files[0]
            if args.resume:
                restore_validation_cache(args.resume,out)
            remaining=max(0,cfg['max_session_seconds']-(time.time()-receipt['started']))
            metrics=evaluate(cfg,selected,npp,analyzer,codec,out/'validation',annotation,max_seconds=remaining)
            state['validation_complete']=metrics['complete'] and len(selected)==cfg['full_validation_sources']
            receipt.update(metrics=metrics,execution_complete=True,
                scientific_complete=scientific_complete(cfg,state) and inventory['full_original_training_corpus'])
            if not metrics['complete']:receipt['needs_resume']=True
        else:receipt.update(execution_complete=True,scientific_complete=False,needs_resume=True)
        if fingerprint(analyzer.model)!=analyzer.weight_digest:raise RuntimeError('frozen analyzer changed during NPP training/evaluation')
        receipt['frozen_analyzer_verified']=True
    except BaseException as error:
        receipt['error']=f'{type(error).__name__}: {error}';receipt['traceback']=traceback.format_exc()
        raise
    finally:
        receipt['wall_seconds']=time.time()-receipt['started']
        (out/'run.json').write_text(json.dumps(receipt,indent=2))
        log_event(out,'run_end',task=args.task,execution_complete=receipt['execution_complete'],
                  scientific_complete=receipt['scientific_complete'],wall_seconds=receipt['wall_seconds'],error=receipt.get('error'))

if __name__=='__main__':main()
