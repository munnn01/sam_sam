"""Task-specific BPG diagnostics followed by bounded, frozen-source proxy continuation."""
import argparse
import gc
import hashlib
import inspect
import json
import math
import os
import signal
from pathlib import Path
from statistics import mean,median
import subprocess
import sys
import threading
import time

import torch

from .assets import digest,ensure_bpg,records_for_task
from .codecs import RealCodec
from .data import load_sample
from .protocol import QPS
from .proxy_diagnostics import bind_cache,calibration_split,gradient_vectors,noise_gradient_summary
from .proxy_recovery import Budget,BudgetExpired,filtered
from .training import config_digest,records_digest,log_event


def load_image_checkpoint(path,sha,source,task):
    if digest(path)!=sha:raise ValueError('image checkpoint SHA differs')
    saved=torch.load(path,map_location='cpu',weights_only=False);cfg=saved['config']
    if task not in ('od','ic') or cfg['task']!=task or cfg['paper']!='image' or saved['source_manifest_sha256']!=source:
        raise ValueError('image task/training source differs')
    if cfg['qps']!=list(QPS) or cfg['identity_fallback']:raise ValueError('image QP/identity protocol differs')
    if saved.get('config_sha256',config_digest(cfg))!=config_digest(cfg):raise ValueError('image config digest differs')
    phases=saved['state']['phase_steps']
    if phases.get('fixed',0)!=0 or phases.get('adaptive',0)!=0 or not 0<=phases['proxy']<=cfg['proxy_steps']:
        raise ValueError('proxy-only image checkpoint required')
    if not saved.get('optimizers') or not saved.get('rng'):raise ValueError('original optimizer/RNG missing')
    return saved


def restore_proxy_state(bank,state):
    consumed=set()
    for qp,network in bank.networks.items():
        prefix='networks.'+qp+'.'
        subset={name[len(prefix):]:value for name,value in state.items() if name.startswith(prefix)}
        if not subset:raise ValueError('QP-specific proxy state missing')
        # CompressionModel.load_state_dict resizes entropy/CDF buffers. A parent
        # Module.load_state_dict would bypass that public override.
        network.load_state_dict(subset,strict=True)
        consumed.update(prefix+name for name in subset)
    if consumed!=set(state):raise ValueError('unexpected proxy-bank state keys')


def controlled_inputs(x,seed=20261028):
    generator=torch.Generator(device=x.device).manual_seed(seed)
    perturb=(torch.rand(x.shape,device=x.device,dtype=x.dtype,generator=generator)*2-1)/255
    return dict(raw=x,**{'gaussian_0.5':filtered(x,'gaussian_0.5'),'gaussian_1.0':filtered(x,'gaussian_1.0'),
        'perturb_1byte':(x+perturb).clamp(0,1)})


def reconstruction_quality(prediction,target):
    mse=float((prediction.detach()-target.detach()).square().mean())
    return dict(mse=mse,psnr_db=-10*math.log10(mse) if mse else None,perfect=mse==0)


def image_rate_gate(rows,min_pairs=8,agreement=.7):
    groups=[];seen=set()
    for row in rows:
        key=(row['source'],row['qp'],row['variant'])
        if key in seen:raise ValueError('duplicate BPG pair')
        seen.add(key)
        if row['pixels']<=0 or not all(math.isfinite(row[k]) and row[k]>0 for k in ('raw_bpp','bpp','proxy_raw','proxy_bpp')):
            raise ValueError('positive finite BPG rates required')
    for qp in QPS:
        selected=[r for r in rows if r['qp']==qp and r['variant']!='raw'];resolved=[]
        for row in selected:
            actual=row['bpp']-row['raw_bpp'];predicted=row['proxy_bpp']-row['proxy_raw']
            if abs(actual)>=max(8/row['pixels'],.005*row['raw_bpp']):resolved.append((row['source'],actual,predicted))
        sources=len({source for source,_,_ in resolved})
        signs=mean(float(actual*predicted>0) for _,actual,predicted in resolved) if resolved else None
        coverage=len(resolved)/len(selected) if selected else 0.
        groups.append(dict(qp=qp,pairs=len(selected),resolved=len(resolved),resolved_sources=sources,
            coverage=coverage,sign_agreement=signs,passed=sources>=min_pairs and coverage>=.5 and signs>=agreement))
    return dict(passed=all(r['passed'] for r in groups),groups=groups,
        scope='BPG controlled-input directions; no downstream accuracy/NPP claim')


def proxy_only_train(training_module,native,cfg,rows,npp,proxy,analyzer,codec,out,resume=None,max_seconds=None):
    """Operational phase gate; retain original targets/config/optimizer/RNG."""
    original=training_module.phase_schedule
    training_module.phase_schedule=lambda settings:[('proxy',settings['proxy_steps'])]
    try:
        state,npp=native(cfg,rows,npp,proxy,analyzer,codec,out,resume,max_seconds)
    finally:training_module.phase_schedule=original
    if state['completed']:
        path=Path(out)/'last.pt'
        # A phase already at target performs no native save in a fresh directory.
        snapshot=torch.load(path if path.is_file() else resume,map_location='cpu',weights_only=False)
        state=dict(snapshot['state'],completed=False,stop_reason='proxy_complete_waiting_npp_audit')
        snapshot['state']=state;temporary=path.with_suffix('.phase_gate')
        torch.save(snapshot,temporary);os.replace(temporary,path)
    return state,npp


def measure_bpg(proxy,rows,cfg,codec,out,budget,identity):
    directory=out/'bpg_cache';bind_cache(directory,identity);points=[]
    device=next(proxy.parameters()).device;proxy.eval()
    for index,record in enumerate(rows):
        budget.require('bpg_pairs');x,_,_=load_sample(record,dict(cfg,npp_size=256,frames=1),False);x=x.to(device)
        variants=controlled_inputs(x,20261028+index)
        for qp in QPS:
            with torch.no_grad():raw=proxy(x,qp)
            raw_key=directory/f'{index:04}_{qp}_raw.json'
            raw_actual=codec(x,'bpg',qp)
            for name,y in variants.items():
                budget.require('bpg_pairs')
                with torch.no_grad():prediction=raw if name=='raw' else proxy(y,qp)
                actual=raw_actual if name=='raw' else codec(y,'bpg',qp)
                row=dict(source=record['id'],variant=name,qp=qp,codec='bpg',pixels=int(y.shape[-2]*y.shape[-1]),
                    height=y.shape[-2],width=y.shape[-1],raw_bpp=raw_actual.bpp,bpp=actual.bpp,
                    proxy_raw=float(raw.bpp),proxy_bpp=float(prediction.bpp),bytes=actual.bytes,
                    stream_sha256=actual.sha256,reconstruction_vs_bpg=reconstruction_quality(prediction.reconstruction,actual.reconstruction),
                    reconstruction_vs_source=reconstruction_quality(actual.reconstruction,y))
                points.append(row)
                (directory/f'{index:04}_{qp}_{name}.json').write_text(json.dumps(row))
        log_event(out,'bpg_source',source=index+1,total=len(rows))
        (out/'bpg_pairs_partial.json').write_text(json.dumps(points,indent=2))
    result=dict(points=points,gate=image_rate_gate(points),npp_evaluated=False,geometry='proxy-scale256; OD aspect ratio preserved')
    (out/'bpg_pairs.json').write_text(json.dumps(result,indent=2));return result


def image_gradient_audit(proxy,rows,cfg,codec,out,budget):
    results=[];device=next(proxy.parameters()).device
    for index,record in enumerate(rows[:2]):
        x,_,_=load_sample(record,dict(cfg,npp_size=256,frames=1),False);x=x.to(device).requires_grad_()
        for qp in QPS:
            budget.require('bpg_gradients');proxy.eval()
            actual=codec(x.detach(),'bpg',qp)
            prediction=proxy(x,qp);vectors=gradient_vectors({'rate':prediction.bpp},[x]);eval_gradient=vectors['rate']
            quality=reconstruction_quality(prediction.reconstruction,actual.reconstruction)
            del prediction,vectors
            proxy.train();samples=[];rates=[]
            for repeat in range(8):
                budget.require('bpg_gradient_samples')
                # Save no model gradients; each stochastic estimate is through the input only.
                with torch.random.fork_rng(devices=[x.device.index or 0] if x.is_cuda else []):
                    torch.manual_seed(20261028+repeat)
                    prediction=proxy(x,qp)
                    samples.append(gradient_vectors({'rate':prediction.bpp},[x])['rate'])
                    rates.append(float(prediction.bpp.detach()))
                del prediction
            report=noise_gradient_summary(samples,{})
            expected=torch.stack(samples).mean(0).reshape_as(x);rms=expected.square().mean().sqrt()
            descent=[]
            if float(rms)>0:
                for epsilon in (1/255,4/255):
                    budget.require('bpg_direction')
                    candidate=(x.detach()-epsilon*expected/rms).clamp(0,1)
                    measured=codec(candidate,'bpg',qp)
                    descent.append(dict(epsilon=epsilon,bpp=measured.bpp,delta_bpp=measured.bpp-actual.bpp))
            row=dict(source=record['id'],qp=qp,eval_rate_gradient_norm=float(eval_gradient.norm()),
                noisy_expected=report,noisy_rate_mean=mean(rates),reconstruction_vs_bpg=quality,
                real_bpg_descent=descent,scope='input gradient only; no NPP exists yet')
            results.append(row);(out/'bpg_gradient_audit.json').write_text(json.dumps(results,indent=2))
            log_event(out,'bpg_gradient',source=record['id'],qp=qp,expected_norm=report['norms']['rate'],samples=8)
            del samples,expected
    proxy.eval();return results


def bounded_process(args,cwd,environment,log_path,seconds,grace=30):
    """Bound setup as well as training, preserving streamed console on timeout."""
    with Path(log_path).open('w',encoding='utf-8') as log:
        grouping=({'creationflags':subprocess.CREATE_NEW_PROCESS_GROUP} if os.name=='nt'
                  else {'start_new_session':True})
        process=subprocess.Popen(args,cwd=cwd,env=environment,stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,text=True,bufsize=1,**grouping)
        job=None
        if os.name=='nt':
            import ctypes
            from ctypes import wintypes
            kernel=ctypes.WinDLL('kernel32',use_last_error=True)
            kernel.CreateJobObjectW.argtypes=[ctypes.c_void_p,wintypes.LPCWSTR];kernel.CreateJobObjectW.restype=wintypes.HANDLE
            kernel.AssignProcessToJobObject.argtypes=[wintypes.HANDLE,wintypes.HANDLE]
            kernel.TerminateJobObject.argtypes=[wintypes.HANDLE,wintypes.UINT]
            kernel.CloseHandle.argtypes=[wintypes.HANDLE]
            job=kernel.CreateJobObjectW(None,None)
            if not job or not kernel.AssignProcessToJobObject(job,int(process._handle)):
                process.kill()
                if job:kernel.CloseHandle(job)
                raise OSError(ctypes.get_last_error(),'cannot contain worker process tree')
        def relay():
            for line in process.stdout:
                print(line,end='',flush=True);log.write(line);log.flush()
        reader=threading.Thread(target=relay,daemon=True);reader.start();timed_out=False
        try:code=process.wait(timeout=max(.01,seconds))
        except subprocess.TimeoutExpired:
            timed_out=True
            if os.name=='nt':
                if not kernel.TerminateJobObject(job,1):raise OSError(ctypes.get_last_error(),'cannot stop worker tree')
            else:os.killpg(process.pid,signal.SIGTERM)
            try:code=process.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                process.kill();code=process.wait(timeout=10)
            if os.name!='nt':
                # The group can outlive its leader, including descendants ignoring TERM.
                try:os.killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError:pass
        reader.join(timeout=10)
        if job:kernel.CloseHandle(job)
    return dict(exit_code=code,timed_out=timed_out)


def continue_legacy(checkpoint,legacy_root,training_source,task,out,budget):
    budget.require('legacy_proxy_continuation');legacy_out=out/'legacy_results';legacy_out.mkdir(exist_ok=True)
    worker=out/'legacy_proxy_worker.py'
    body="""import os,sys,time,torch
from pathlib import Path
import reproduce.run as runner
import reproduce.training as training
"""+inspect.getsource(proxy_only_train)+"""
native=training.train
deadline=time.monotonic()+float(os.environ['PROXY_CONTINUATION_SECONDS'])
def guarded(*args,**kwargs):
    kwargs['max_seconds']=max(0,min(kwargs.get('max_seconds',1e20),deadline-time.monotonic()-60))
    return proxy_only_train(training,native,*args,**kwargs)
runner.train=guarded
runner.main()
"""
    worker.write_text(body,encoding='utf-8')
    seconds=min(14400,max(0,budget.remaining()-120))
    env=dict(os.environ,PYTHONPATH=str(legacy_root),PAPER_SOURCE_SHA256=training_source,
        PROXY_CONTINUATION_SECONDS=str(seconds),PYTHONUNBUFFERED='1')
    log_event(out,'legacy_proxy_start',seconds=seconds,legacy_source=training_source,
        worker_sha256=hashlib.sha256(body.encode()).hexdigest())
    args=[sys.executable,'-u',str(worker),'--task',task,'--mode','full','--out',str(legacy_out),'--resume',str(checkpoint)]
    execution=bounded_process(args,legacy_root,env,out/'legacy_console.log',seconds)
    result=json.loads((legacy_out/'run.json').read_text()) if (legacy_out/'run.json').is_file() else {'error':'legacy report missing'}
    if execution['timed_out']:
        return dict(stop_reason='parent_time_budget',partial=result,execution=execution,
            checkpoint_available=(legacy_out/'last.pt').is_file(),scientific_complete=False)
    if execution['exit_code']:raise RuntimeError('legacy proxy continuation failed: '+str(result.get('error')))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',choices=['od','ic'],required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--checkpoint-sha256',required=True)
    parser.add_argument('--training-source-sha256',required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--inputs',type=Path,default=Path('/kaggle/input'))
    parser.add_argument('--assets',type=Path,default=Path('/kaggle/working/paper_assets'))
    parser.add_argument('--max-seconds',type=int,default=18000)
    parser.add_argument('--audit-seconds',type=int,default=5400)
    parser.add_argument('--legacy-root',type=Path)
    parser.add_argument('--continue-proxy',action='store_true')
    args=parser.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=True);budget=Budget(args.max_seconds)
    report=dict(task=args.task,scope='image proxy/BPG audit; no trained NPP metrics',scientific_complete=False,
        npp_updates=0,audit_complete=False,started=time.time(),checkpoint_sha256=args.checkpoint_sha256)
    try:
        torch.set_num_threads(2)
        if not torch.cuda.is_available():raise RuntimeError('CUDA required for cloud image diagnostic')
        saved=load_image_checkpoint(args.checkpoint,args.checkpoint_sha256,args.training_source_sha256,args.task)
        cfg=saved['config'];report['initial_phase_steps']=saved['state']['phase_steps']
        from .image_proxy import ImageProxyBank
        proxy=ImageProxyBank(cfg['proxy_quality'],pretrained=False);restore_proxy_state(proxy,saved['proxy'])
        proxy=proxy.to('cuda').requires_grad_(False);del saved;gc.collect()
        versions={name:value._version for name,value in proxy.state_dict(keep_vars=True).items()}
        rows,validation,inventory=records_for_task(args.task,args.inputs,args.assets)
        if inventory['dataset_versions']!=cfg['runtime_identity']['dataset_versions'] or records_digest(rows)!=cfg['runtime_identity']['train_records_sha256']:
            raise ValueError('image training inventory/version differs')
        calibration,dev=calibration_split(rows,32,16,20261028)
        identity=dict(task=args.task,checkpoint=args.checkpoint_sha256,config=config_digest(cfg),
            datasets=inventory['dataset_versions'],sources=[r['id'] for r in dev],geometry='proxy256',
            diagnostic_source=os.environ.get('PAPER_SOURCE_SHA256'))
        bind_cache(out/'binding',identity)
        (out/'cohort.json').write_text(json.dumps(dict(identity=identity,calibration=[r['id'] for r in calibration],
            dev=[r['id'] for r in dev],train_dev_disjoint=True),indent=2))
        log_event(out,'image_diagnostic_ready',task=args.task,proxy_steps=report['initial_phase_steps']['proxy'],
            dev_sources=len(dev),npp_trained=False)
        audit=Budget(min(args.audit_seconds,budget.remaining()))
        try:
            paths=ensure_bpg(args.assets/'bpg');audit.require('bpg_assets')
            codec=RealCodec(bpgenc=paths['bpgenc'],bpgdec=paths['bpgdec'])
            paired=measure_bpg(proxy,dev,cfg,codec,out,audit,identity);report['rate_gate']=paired['gate']
            report['gradient_audit']=image_gradient_audit(proxy,dev,cfg,codec,out,audit)
            report['audit_complete']=True
        except BudgetExpired as error:report['audit_stop']=str(error)
        if any(value._version!=versions[name] for name,value in proxy.state_dict(keep_vars=True).items()):
            raise RuntimeError('frozen image proxy mutated during audit')
        report['proxy_unchanged']=True
        del proxy;gc.collect();torch.cuda.empty_cache()
        (out/'audit_summary.json').write_text(json.dumps(report,indent=2))
        if args.continue_proxy:
            if args.legacy_root is None:raise ValueError('exact frozen legacy source root required')
            report['legacy_continuation']=continue_legacy(args.checkpoint,args.legacy_root,args.training_source_sha256,args.task,out,budget)
    except BudgetExpired as error:report['stop_reason']='time_budget';report['stopped_stage']=str(error)
    except BaseException as error:
        report['error']=f'{type(error).__name__}: {error}';raise
    finally:
        report['wall_seconds']=time.time()-report['started'];(out/'run.json').write_text(json.dumps(report,indent=2))
        log_event(out,'image_diagnostic_end',task=args.task,audit_complete=report['audit_complete'],error=report.get('error'),npp_updates=0)


if __name__=='__main__':main()
