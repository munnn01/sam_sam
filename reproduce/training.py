"""Paper objectives, phase accounting and resumable optimization without fallback."""
import hashlib
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from .protocol import task_lambda,proxy_lambda,real_forward
from .data import load_sample
from .models import ImageNPP
from .tasks import freeze

def image_objective(rate,task,x,y,qp):
    return rate+task_lambda(qp)*task+.5*(x-y).square().mean()

def video_objective(rate,task,x,reconstruction):
    return 10*((x-reconstruction).square().mean()+.001*rate)+task

def phase_schedule(cfg):
    if cfg['paper']=='image':
        return [('proxy',cfg['proxy_steps']),('fixed',cfg['fixed_steps']),('adaptive',cfg['adaptive_steps'])]
    return [('prior',cfg['prior_steps']),('adaptive',cfg['adaptive_steps'])]

def initial_state(cfg):
    return dict(task=cfg['task'],mode=cfg['mode'],phase_steps={p:0 for p,_ in phase_schedule(cfg)},
                selected_step=None,completed=False,validation_complete=False)

def trained_npp_steps(state):
    return sum(state['phase_steps'].get(p,0) for p in ('fixed','adaptive'))

def scientific_complete(cfg,state):
    return (cfg['mode']=='full' and state.get('validation_complete',False)
            and all(state['phase_steps'].get(p,0)>=count for p,count in phase_schedule(cfg)))

def qp_for_step(step,cfg,phase):
    return cfg['fixed_qp'] if phase=='fixed' else cfg['qps'][step%len(cfg['qps'])]

def log_event(out,stage,**fields):
    row=dict(stage=stage,time=time.time(),**fields)
    value=json.dumps(row,allow_nan=False)
    print(value,flush=True)
    with (Path(out)/'progress.jsonl').open('a',encoding='utf-8') as f:f.write(value+'\n')

def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])

def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'].cpu())
    if torch.cuda.is_available() and state['cuda']:torch.cuda.set_rng_state_all(state['cuda'])

def config_digest(cfg):
    return hashlib.sha256(json.dumps(cfg,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def records_digest(rows):
    # Dataset versions pin contents; this also binds actual discovered identities,
    # labels and loaded COCO geometry. Absolute mount paths may vary by account.
    h=hashlib.sha256()
    for row in rows:
        value={k:v for k,v in row.items() if k!='path'}
        h.update(json.dumps(value,sort_keys=True,separators=(',',':')).encode());h.update(b'\n')
    return h.hexdigest()

def runtime_identity(rows,validation,inventory,analyzer_digest):
    return dict(analyzer_weights_sha256=analyzer_digest,train_records_sha256=records_digest(rows),
                validation_records_sha256=records_digest(validation),dataset_versions=inventory['dataset_versions'],
                validation_annotations=inventory.get('validation_annotations',{}).get('sha256'))

def save_checkpoint(path,cfg,state,npp,proxy,optimizers):
    data=dict(schema=1,config=cfg,config_sha256=config_digest(cfg),state=state,
              npp=npp.state_dict(),npp_adaptive=getattr(npp,'adaptive',False),proxy=proxy.state_dict(),
              optimizers={k:o.state_dict() for k,o in optimizers.items()},rng=rng_state(),
              source_manifest_sha256=os.environ.get('PAPER_SOURCE_SHA256'))
    path=Path(path);tmp=path.with_suffix('.writing')
    torch.save(data,tmp);os.replace(tmp,path)

def sample_training(rows,cfg,out):
    for _ in range(12):
        row=rows[random.randrange(len(rows))]
        try:return load_sample(row,cfg,True)
        except (ValueError,OSError,RuntimeError) as error:
            log_event(out,'data_skip',source=row['id'],error=str(error)[:250])
    raise RuntimeError('12 consecutive unusable training sources; data pipeline must be fixed')

def train(cfg,rows,npp,proxy,analyzer,codec,out,resume=None,max_seconds=None):
    """Return state + current NPP; budget stops retain all optimizer/RNG state."""
    out=Path(out);device=next(npp.parameters()).device
    state=initial_state(cfg);optimizers={};saved=None
    if resume:
        saved=torch.load(resume,map_location='cpu',weights_only=False)
        if (saved['config_sha256']!=config_digest(cfg) or saved['config']!=cfg):
            raise ValueError('resume config drift; phase counters must not migrate silently')
        expected=os.environ.get('PAPER_SOURCE_SHA256')
        if expected and saved.get('source_manifest_sha256')!=expected:
            raise ValueError('resume source manifest differs')
        if cfg['paper']=='image' and saved['npp_adaptive']:
            npp=ImageNPP(adaptive=True).to(device)
        npp.load_state_dict(saved['npp'],strict=True);proxy.load_state_dict(saved['proxy'],strict=True)
        state=saved['state'];restore_rng(saved['rng'])
    start=time.monotonic();previous_phase=None
    budget=cfg['max_session_seconds'] if max_seconds is None else max(0,max_seconds)
    for phase,target in phase_schedule(cfg):
        if state['phase_steps'][phase]>=target:continue
        # Completed-phase Adam moments must not occupy GPU memory during NPP training.
        optimizers={}
        if phase=='adaptive' and cfg['paper']=='image' and not npp.adaptive:
            npp=ImageNPP.from_fixed(npp).to(device)
        if phase=='proxy':
            proxy.train()
            for p in proxy.parameters():p.requires_grad_(True)
            for qp in cfg['qps']:
                optimizers[f'proxy/{qp}']=torch.optim.Adam(proxy.main_parameters(qp),lr=cfg['lr'])
                optimizers[f'aux/{qp}']=torch.optim.Adam(proxy.aux_parameters(qp),lr=1e-3)
        elif phase=='prior':
            for p in proxy.parameters():p.requires_grad_(True)
            optimizers['prior']=torch.optim.Adam(proxy.prior.parameters(),lr=1e-3)
        else:
            freeze(proxy)
            if cfg['paper']=='image':proxy.train()  # additive-uniform quantization surrogate; weights remain frozen
            npp.train();optimizers['npp']=torch.optim.Adam(npp.parameters(),lr=cfg['lr'])
        saved_phase=None
        if saved:
            keys=saved['optimizers']
            saved_phase=('adaptive' if saved['npp_adaptive'] or cfg['paper']=='video' else 'fixed') if 'npp' in keys else ('prior' if 'prior' in keys else 'proxy')
        if saved and previous_phase is None and phase==saved_phase:
            for key,opt in optimizers.items():
                if key in saved['optimizers']:opt.load_state_dict(saved['optimizers'][key])
        previous_phase=phase
        log_event(out,'phase_start',task=cfg['task'],phase=phase,completed=state['phase_steps'][phase],target=target,
                  npp_parameters=sum(p.numel() for p in npp.parameters()),analyzer=analyzer.receipt())
        if phase in ('proxy','prior'):analyzer.model.cpu()
        else:analyzer.model.to(device)
        while state['phase_steps'][phase]<target:
            if time.monotonic()-start>=budget:
                state['stop_reason']='session_budget';state['completed']=False
                save_checkpoint(out/'last.pt',cfg,state,npp,proxy,optimizers)
                log_event(out,'budget_stop',task=cfg['task'],phase=phase,phase_steps=state['phase_steps'],
                          trained_npp_steps=trained_npp_steps(state),scientific_complete=False)
                return state,npp
            step=state['phase_steps'][phase];qp=qp_for_step(step,cfg,phase)
            sample_cfg=dict(cfg)
            if phase=='proxy':sample_cfg['npp_size']=256
            x,target_data,meta=sample_training(rows,sample_cfg,out);x=x.to(device)
            if phase=='proxy':
                # A conventional256px codec-training crop, measured through real BPG.
                h,w=x.shape[-2:];ch=min(256,h);cw=min(256,w)
                yy=random.randrange(h-ch+1);xx=random.randrange(w-cw+1);x=x[...,yy:yy+ch,xx:xx+cw]
                optimizer=optimizers[f'proxy/{qp}'];aux=optimizers[f'aux/{qp}']
                for group in optimizer.param_groups:group['lr']=cfg['lr']*(.1 if step>=int(target*.8) else 1)
                optimizer.zero_grad(set_to_none=True);aux.zero_grad(set_to_none=True)
                actual=codec(x,'bpg',qp);prediction=proxy(x,qp)
                loss=prediction.bpp+proxy_lambda(qp)*(prediction.reconstruction-actual.reconstruction).square().mean()
                loss.backward();optimizer.step()
                aux_loss=proxy.aux_loss(qp);aux_loss.backward();aux.step()
                rate=float(prediction.bpp.detach());task_loss=None
            elif phase=='prior':
                optimizer=optimizers['prior'];optimizer.zero_grad(set_to_none=True)
                prediction=proxy(x,qp);loss=prediction.bpp
                loss.backward();optimizer.step();rate=float(prediction.bpp.detach());task_loss=None
            else:
                optimizer=optimizers['npp']
                decay={'fixed':80000,'adaptive':60000}.get(phase,int(target*.6))
                if cfg['mode']=='pilot':decay=int(target*.6)
                for group in optimizer.param_groups:group['lr']=cfg['lr']*(.1 if step>=decay else 1)
                optimizer.zero_grad(set_to_none=True)
                y=npp(x,qp);prediction=proxy(y,qp)
                if cfg['paper']=='image':
                    actual=codec(y,'bpg',qp)
                    reconstructed=real_forward(prediction.reconstruction,actual.reconstruction)
                    rate_tensor=real_forward(prediction.bpp,prediction.bpp.new_tensor(actual.bpp))
                    task_loss=analyzer.loss(reconstructed,target_data)
                    loss=image_objective(rate_tensor,task_loss,x,y,qp)
                else:
                    reconstructed=prediction.reconstruction;rate_tensor=prediction.bpp
                    task_loss=analyzer.loss(reconstructed,target_data)
                    loss=video_objective(rate_tensor,task_loss,x,reconstructed)
                if not torch.isfinite(loss):raise RuntimeError('nonfinite paper objective; no rollback or fallback applied')
                loss.backward()
                grads=[p.grad for p in npp.parameters() if p.grad is not None]
                if not grads or not all(torch.isfinite(g).all() for g in grads):
                    raise RuntimeError('NPP input-gradient chain is missing/nonfinite')
                grad_norm=float(torch.sqrt(sum(g.detach().square().sum() for g in grads)))
                optimizer.step();state['selected_step']=trained_npp_steps(state)+1
                rate=float(rate_tensor.detach())
            state['phase_steps'][phase]+=1
            if state['phase_steps'][phase]%cfg['log_every']==0 or step==0:
                log_event(out,'train',task=cfg['task'],phase=phase,step=state['phase_steps'][phase],qp=qp,
                          source=meta['id'],loss=float(loss.detach()),rate=rate,
                          task_loss=float(task_loss.detach()) if task_loss is not None else None,
                          npp_grad_norm=grad_norm if phase not in ('proxy','prior') else None,
                          elapsed_seconds=time.monotonic()-start)
            if state['phase_steps'][phase]%cfg['checkpoint_every']==0:
                save_checkpoint(out/'last.pt',cfg,state,npp,proxy,optimizers)
        save_checkpoint(out/'last.pt',cfg,state,npp,proxy,optimizers)
        log_event(out,'phase_finished',task=cfg['task'],phase=phase,phase_steps=state['phase_steps'])
    state['completed']=True;state['stop_reason']='schedule_finished'
    analyzer.model.to(device)
    if trained_npp_steps(state)>0:
        # Export only a trained model. Resume snapshots may legitimately contain a not-yet-trained NPP during proxy fitting.
        torch.save({'npp':npp.state_dict(),'adaptive':getattr(npp,'adaptive',False),'step':trained_npp_steps(state),
                    'config':cfg},out/'trained_npp.pt')
    return state,npp
