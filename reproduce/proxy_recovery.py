"""Checkpoint-bound P0.5 video diagnostics and explicitly gated supplementary training."""
import argparse
import copy
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import random
from statistics import mean
import time

import torch
import numpy as np
from torch.nn import functional as F

from .assets import digest,ensure_tracking,records_for_task
from .codecs import RealCodec
from .data import load_sample
from .models import VideoNPP
from .protocol import QPS
from .proxy_variants import DiagnosticCodec,quantizer_probe
from .proxy_diagnostics import (bind_cache,calibration_split,paired_gate,fit_prior,
    component_gradients,finetune,residual_statistics,rate_ao_margin)
from .proxy_diagnostics import (margin_interval,rd_should_stop,gradient_vectors,
    summarize_gradients,noise_gradient_summary,calibrate_rd_tolerance)
from .training import config_digest,records_digest,log_event
from .virtual_codec import VirtualCodec
from .tensor_bytes import raw_tensor_bytes


class BudgetExpired(TimeoutError):pass


class Budget:
    def __init__(self,seconds):self.deadline=time.monotonic()+max(0,seconds)
    def remaining(self):return max(0,self.deadline-time.monotonic())
    def require(self,stage):
        if self.remaining()<=0:raise BudgetExpired(stage)


@contextmanager
def paired_rng(source,seed=20261028):
    python=random.getstate();numpy=np.random.get_state();cpu=torch.get_rng_state()
    cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    paired_seed=int(hashlib.sha256((str(seed)+'/'+source).encode()).hexdigest()[:8],16)
    random.seed(paired_seed);np.random.seed(paired_seed);torch.manual_seed(paired_seed)
    if cuda is not None:torch.cuda.manual_seed_all(paired_seed)
    try:yield
    finally:
        random.setstate(python);np.random.set_state(numpy);torch.set_rng_state(cpu)
        if cuda is not None:torch.cuda.set_rng_state_all(cuda)


def model_digest(model):
    h=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        h.update(name.encode());h.update(str(tuple(value.shape)).encode())
        h.update(raw_tensor_bytes(value))
    return h.hexdigest()


def load_checkpoint(path,sha,source,task):
    if digest(path)!=sha:raise ValueError('checkpoint SHA differs')
    saved=torch.load(path,map_location='cpu',weights_only=False)
    cfg=saved['config']
    if saved.get('source_manifest_sha256')!=source or cfg['task']!=task or cfg['paper']!='video':
        raise ValueError('legacy training source/task differs')
    if cfg['qps']!=list(QPS) or cfg['identity_fallback']:raise ValueError('QP/identity protocol differs')
    if saved.get('config_sha256',config_digest(cfg))!=config_digest(cfg):raise ValueError('checkpoint config digest differs')
    counts=saved['state']['phase_steps']
    if counts.get('adaptive',0)!=cfg['adaptive_steps'] or counts.get('prior',0)!=cfg['prior_steps']:
        raise ValueError('completed original video training required')
    return saved


def sample_symbols(symbols,positions=512,seed=0):
    if positions<1:raise ValueError('positive symbol cache bound required')
    values=torch.cat([s.detach().cpu().permute(1,0,2,3).reshape(s.shape[1],-1) for s in symbols],dim=1)
    generator=torch.Generator().manual_seed(seed)
    indices=torch.randperm(values.shape[1],generator=generator)[:positions]
    return values[:,indices].contiguous().view(1,values.shape[0],1,-1)


def output_summary(out):
    return {key:float(getattr(out,key).detach()) for key in ('bpp','entropy_bpp','mode_bpp',
        'zero_reference_bpp','excess_bpp','zero_fraction','zero_symbol_cost_bpp','nonzero_symbol_cost_bpp')}


def recovery_decision(gate,converged=True):
    if not gate['passed']:return 'supervised_rate_head_required'
    return 'eligible_for_bounded_recovery' if converged else 'prior_convergence_unverified'


def load_probe(record,cfg,frames,seed):
    state=random.getstate()
    try:
        random.seed(int(hashlib.sha256((str(seed)+'/'+record['id']+'/'+str(frames)).encode()).hexdigest()[:16],16))
        settings=dict(cfg,frames=frames)
        return load_sample(record,settings,True)
    finally:random.setstate(state)


def eligible_records(rows,count,frames=64):
    eligible=[]
    for row in sorted(rows,key=lambda r:hashlib.sha256(r['id'].encode()).hexdigest()):
        if row['task']=='ot':
            path=Path(row['path'])
            cover_path=path/'cover.label'
            if not cover_path.is_file():continue
            cover=[float(x)>0 for x in cover_path.read_text().splitlines() if x.strip()]
            if len(cover)<frames:continue
            if not any(cover[i] and any(cover[i+1:i+frames]) for i in range(len(cover)-frames+1)):continue
        elif row['task']=='ar':
            import cv2
            capture=cv2.VideoCapture(row['path'])
            try:
                if not capture.isOpened() or int(capture.get(cv2.CAP_PROP_FRAME_COUNT))<(frames-1)*2+1:continue
            finally:capture.release()
        eligible.append(row)
        if len(eligible)>=count:return eligible
    raise ValueError('insufficient independently eligible TRAIN sources')


def paired_measurements(npp,proxy,rows,cfg,codec,out,budget,identity,label='original',collect=None,positions=512):
    """Actual rates are reused only for identical NPP, clip protocol and cohort."""
    out=Path(out);device=next(npp.parameters()).device
    real_identity=dict(identity,npp=model_digest(npp),sources=[r['id'] for r in rows],frames=16,
        codec_config={k:cfg[k] for k in ('preset','fps','gop','npp_size')})
    actual_cache=out/'actual'/real_identity['npp'][:16]/hashlib.sha256(json.dumps(real_identity,sort_keys=True).encode()).hexdigest()[:16]
    bind_cache(actual_cache,real_identity);points=[];spectra=[]
    for index,record in enumerate(rows):
        budget.require('paired/'+label)
        x,target,meta=load_probe(record,cfg,16,20261028);x=x.to(device)
        for qp in QPS:
            with torch.no_grad():
                y=npp(x,qp);raw=proxy(x,qp);processed=proxy(y,qp)
            if qp==30:spectra.append(dict(source=record['id'],**residual_statistics(x,y)))
            summaries={'raw':output_summary(raw),'npp':output_summary(processed)}
            if collect is not None:
                for kind,value in [('raw',raw),('npp',processed)]:
                    collect.append(dict(source=record['id'],qp=qp,kind=kind,
                        codes=sample_symbols(value.symbols,positions,20261028+index)))
            for cod in cfg['codecs']:
                budget.require('paired/'+label)
                cache=actual_cache/f'{index:04}_{cod}_{qp}.json'
                if cache.is_file():actual=json.loads(cache.read_text())
                else:
                    a=codec(x,cod,qp);b=codec(y,cod,qp)
                    actual=dict(source=record['id'],codec=cod,qp=qp,raw_bpp=a.bpp,npp_bpp=b.bpp,
                        raw_bytes=a.bytes,npp_bytes=b.bytes,pixels=int(x.shape[1]*x.shape[-2]*x.shape[-1]),
                        raw_stream_sha256=a.sha256,npp_stream_sha256=b.sha256)
                    cache.write_text(json.dumps(actual,indent=2))
                if actual['source']!=record['id']:raise ValueError('cached source differs')
                points.append(dict(actual,proxy_raw=summaries['raw']['bpp'],proxy_npp=summaries['npp']['bpp'],
                    proxy_details=summaries))
        log_event(out,'paired_source',variant=label,source=index+1,total=len(rows))
        (out/f'paired_{label}_partial.json').write_text(json.dumps(points,indent=2))
    result=dict(points=points,gate=paired_gate(points),residuals=spectra,identity=real_identity,
                proxy_sha256=model_digest(proxy))
    (out/f'paired_{label}.json').write_text(json.dumps(result,indent=2))
    return result


def build_train_cache(npp,proxy,rows,cfg,budget,positions,log):
    batches=[];device=next(npp.parameters()).device
    for index,record in enumerate(rows):
        budget.require('cache_train')
        x,_,_=load_probe(record,cfg,16,20261028);x=x.to(device)
        with torch.no_grad():
            for qp in QPS:
                y=npp(x,qp)
                for kind,value in [('raw',x),('npp',y)]:
                    result=proxy(value,qp)
                    batches.append(dict(source=record['id'],qp=qp,kind=kind,
                        codes=sample_symbols(result.symbols,positions,20261028+index)))
        if sum(r['codes'].numel()*r['codes'].element_size() for r in batches)>512*1024**2:
            raise RuntimeError('symbol cache exceeded512MiB')
        log(dict(stage='symbol_cache',sources=index+1,total=len(rows)))
    return batches


def fit_candidates(original,train_cache,dev_cache,out,budget,args,phase=None):
    started=time.monotonic();phase=phase or Budget(min(5400,budget.remaining()));reports=[]
    reserve=min(900,phase.remaining()*.2)
    def emit(row):log_event(out,row.pop('stage'),**row)
    trials=[]
    for lr in (1e-3,3e-3,1e-2):
        phase.require('prior_lr_screen');budget.require('prior_lr_screen')
        candidate=DiagnosticCodec.from_legacy(original)
        seconds=min(300,phase.remaining()-reserve,budget.remaining())
        if seconds<=0:raise BudgetExpired('prior_evaluation_reserve')
        report=fit_prior(candidate,train_cache,dev_cache,steps=args.lr_probe_steps,lr=lr,
            eval_every=max(4,args.lr_probe_steps),max_seconds=seconds,log=emit)
        trials.append(report)
    selected=min(trials,key=lambda r:r['best_loss'])['lr']
    (out/'prior_lr_screen.json').write_text(json.dumps(trials,indent=2))
    candidates=[]
    for per_qp in (False,True):
        phase.require('prior_fit');budget.require('prior_fit')
        candidate=DiagnosticCodec.from_legacy(original,per_qp=per_qp)
        seconds=min((phase.remaining()-reserve)/(1 if per_qp else 2),budget.remaining())
        if seconds<=0:raise BudgetExpired('prior_evaluation_reserve')
        report=fit_prior(candidate,train_cache,dev_cache,steps=args.prior_updates,lr=selected,
            eval_every=256,max_seconds=seconds,log=emit)
        name='per_qp' if per_qp else 'shared';reports.append(dict(variant=name,**report))
        torch.save(dict(proxy=candidate.state_dict(),per_qp=per_qp,report=report),out/f'prior_{name}.pt')
        candidates.append((name,candidate,report))
    (out/'prior_fits.json').write_text(json.dumps(dict(candidates=reports,lr_screen=trials,
        seconds=time.monotonic()-started),indent=2))
    return candidates


def gradient_audit(npp,proxy,analyzer,rows,cfg,out,budget):
    results=[];device=next(npp.parameters()).device
    for record in rows[:2]:
        for qp in (30,45):
            budget.require('gradient_audit');npp.requires_grad_(True);proxy.requires_grad_(False)
            x,target,_=load_probe(record,cfg,16,20261028);x=x.to(device);y=npp(x,qp)
            prediction=proxy(y,qp,rate_mode='ste');task=analyzer.loss(prediction.reconstruction,target)
            parts=dict(rate=.01*prediction.bpp,distortion=10*(prediction.reconstruction-x).square().mean(),task=task)
            parameters=gradient_vectors(parts,npp.parameters());output=gradient_vectors(parts,[y])
            parameter_samples=[];output_samples=[]
            for index in range(8):
                budget.require('gradient_noise_samples')
                noisy=proxy(y,qp,rate_mode='uniform',noise_sign=1 if index%2==0 else -1,
                    generator=torch.Generator(device=device).manual_seed(20261028+index//2))
                loss={'rate':.01*noisy.bpp}
                parameter_samples.append(gradient_vectors(loss,npp.parameters())['rate'])
                output_samples.append(gradient_vectors(loss,[y])['rate'])
                del noisy,loss
            cases=[('ste',summarize_gradients(parameters),summarize_gradients(output)),
                ('uniform_expected',noise_gradient_summary(parameter_samples,{k:v for k,v in parameters.items() if k!='rate'}),
                 noise_gradient_summary(output_samples,{k:v for k,v in output.items() if k!='rate'}))]
            for mode,parameter_report,output_report in cases:
                report=dict(source=record['id'],qp=qp,mode=mode,parameters=parameter_report,output=output_report,
                    sampling='four antithetic pairs; norm of mean gradient vector')
                results.append(report);log_event(out,'gradient_probe',**report)
            (out/'gradient_audit.json').write_text(json.dumps(results,indent=2))
            del prediction,task,parts,y,parameters,output,parameter_samples,output_samples,cases
    npp.requires_grad_(False)
    (out/'gradient_audit.json').write_text(json.dumps(results,indent=2))
    return results


def filtered(x,name):
    if name=='raw':return x
    flat=x.flatten(0,1)
    if name.startswith('gaussian'):
        sigma=float(name.split('_')[1]);coordinates=torch.arange(-2,3,device=x.device,dtype=x.dtype)
        one=torch.exp(-coordinates.square()/(2*sigma*sigma));one=one/one.sum();kernel=(one[:,None]*one[None,:]).expand(3,1,5,5)
        return F.conv2d(F.pad(flat,(2,2,2,2),mode='replicate'),kernel,groups=3).reshape_as(x)
    if name=='bilateral':
        patches=F.unfold(F.pad(flat,(1,1,1,1),mode='replicate'),3).reshape(len(flat),3,9,*flat.shape[-2:])
        spatial=x.new_tensor([2,1,2,1,0,1,2,1,2]).view(1,1,9,1,1)
        weights=torch.exp(-spatial/2-(patches-flat[:,:,None]).square()/(2*.05**2))
        return ((patches*weights).sum(2)/weights.sum(2)).reshape_as(x)
    raise ValueError('unknown deterministic filter')


def filter_rate_audit(rows,cfg,codec,out,budget):
    points=[]
    for index,record in enumerate(rows):
        x,_,_=load_probe(record,cfg,16,20261028)
        for name in ('gaussian_0.5','gaussian_1.0','bilateral'):
            y=filtered(x,name)
            for cod in cfg['codecs']:
                for qp in QPS:
                    budget.require('filter_rate')
                    actual=codec(y,cod,qp)
                    points.append(dict(source=record['id'],filter=name,codec=cod,qp=qp,bpp=actual.bpp,bytes=actual.bytes))
        log_event(out,'filter_source',source=index+1,total=len(rows))
    result=dict(scope='rate-only16-frame diagnostic; no task/AO/BD-rate claim',points=points)
    (out/'filter_rates.json').write_text(json.dumps(result,indent=2))
    return result


def evaluate_probe(npp,rows,cfg,analyzer,codec,out,budget,label,
                   arms=('codec_only','npp','gaussian_0.5','gaussian_1.0','bilateral'),tracker_seed=20261028):
    """Matched 64-frame task operating points; never reuse the16-frame rate probe."""
    from .evaluation import aggregate_tracking,tracking_metrics,bd_rate
    device=next(npp.parameters()).device;curves=[]
    identity=dict(npp=model_digest(npp),analyzer=analyzer.weight_digest,
        cfg=config_digest(cfg),frames=64,seed=tracker_seed,torch=torch.__version__,
        diagnostic_source=__import__('os').environ.get('PAPER_SOURCE_SHA256'))
    for arm in arms:
        cache_identity=dict(identity,arm=arm,npp=identity['npp'] if arm=='npp' else 'static_filter_v1')
        key=hashlib.sha256(json.dumps(cache_identity,sort_keys=True).encode()).hexdigest()
        cache=out/'dev_cache'/key;bind_cache(cache,cache_identity)
        for cod in cfg['codecs']:
            for qp in QPS:
                measured=[]
                for index,record in enumerate(rows):
                    budget.require('rd_probe/'+label)
                    source_key=hashlib.sha256(record['id'].encode()).hexdigest()
                    saved=cache/f'{source_key}_{cod}_{qp}.json'
                    record_digest=records_digest([record])
                    if saved.is_file():
                        point=json.loads(saved.read_text())
                        if point['source']!=record['id'] or point.get('record_sha256')!=record_digest:
                            raise ValueError('RD cached source differs')
                        measured.append(point);continue
                    x,target,_=load_probe(record,cfg,64,20261028)
                    with torch.no_grad():
                        if arm=='npp':
                            y=torch.cat([npp(x[:,i:i+16].to(device),qp).cpu() for i in range(0,64,16)],dim=1)
                        else:y=x if arm=='codec_only' else filtered(x,arm)
                        actual=codec(y,cod,qp)
                    if cfg['task']=='ot':
                        with paired_rng(record['id'],tracker_seed),torch.enable_grad():
                            prediction=analyzer.track(actual.reconstruction,target['boxes_xywh'][0].tolist())
                        metric=tracking_metrics(prediction,target['boxes_xywh'].numpy(),target['cover'].numpy(),(x.shape[-1],x.shape[-2]))
                        weight=metric['frames'];quality_sum=metric['ao']*weight if weight else 0.
                    else:
                        with paired_rng(record['id'],tracker_seed),torch.no_grad():logits=analyzer.predict(actual.reconstruction.to(device))
                        weight=1;quality_sum=float(logits.argmax(1).item()==int(target.item()))
                    point=dict(source=record['id'],record_sha256=record_digest,bytes=actual.bytes,pixels=int(x.shape[1]*x.shape[-2]*x.shape[-1]),
                        quality_sum=quality_sum,quality_weight=weight)
                    saved.write_text(json.dumps(point));measured.append(point)
                quality=sum(p['quality_sum'] for p in measured)/sum(p['quality_weight'] for p in measured)
                curves.append(dict(arm=arm,codec=cod,qp=qp,bpp=8*sum(r['bytes'] for r in measured)/sum(r['pixels'] for r in measured),
                    quality=quality,sources=len(rows),frames=64,points=measured))
                log_event(out,'dev_point',label=label,**{k:v for k,v in curves[-1].items() if k!='points'})
                (out/f'dev_{label}_partial.json').write_text(json.dumps(curves,indent=2))
    rates={}
    for cod in cfg['codecs']:
        anchor=[p for p in curves if p['arm']=='codec_only' and p['codec']==cod]
        if len(anchor)!=4:continue
        for arm in ('npp','gaussian_0.5','gaussian_1.0','bilateral'):
            points=[p for p in curves if p['arm']==arm and p['codec']==cod]
            if len(points)!=4:continue
            rates[cod+'/'+arm]=bd_rate([p['bpp'] for p in anchor],[p['quality'] for p in anchor],
                [p['bpp'] for p in points],[p['quality'] for p in points])
    result=dict(scope='TRAIN-heldout64-frame diagnostic only',curves=curves,bd_rate=rates)
    (out/f'dev_{label}.json').write_text(json.dumps(result,indent=2))
    return result


def run_recovery(npp,proxy,analyzer,train,dev,cfg,codec,out,budget,invocation):
    reports=[];phase=Budget(min(10800,budget.remaining()));e0_phase=Budget(min(3600,phase.remaining()))
    progress_path=out/'lambda_runs.json'
    if progress_path.exists():raise ValueError('recovery output already exists; use a new invocation directory')
    base=evaluate_probe(npp,dev[:16],cfg,analyzer,codec,out,e0_phase,'recovery_base')
    def margins(result,baseline):
        values={}
        for point in result['curves']:
            if point['arm']!='npp':continue
            anchor=[p for p in baseline['curves'] if p['arm']=='codec_only' and p['codec']==point['codec']]
            values[point['codec']+'/'+str(point['qp'])]=margin_interval(anchor,point)
        return values
    initial_margins=margins(base,base)
    repeated=[initial_margins]
    try:
        for repeat in (1,2):
            result=evaluate_probe(npp,dev[:16],cfg,analyzer,codec,out,e0_phase,'e0_'+str(repeat),
                arms=('codec_only','npp'),tracker_seed=20261028+repeat)
            repeated.append(margins(result,result))
        e0=calibrate_rd_tolerance(repeated)
    except BudgetExpired:
        e0=dict(available=False,reason='E0 time budget; quality early stop disabled')
    e0.update(seeds_measured=len(repeated),repeated_margins=repeated,invocation_id=invocation,
        base_npp=model_digest(npp),sources=[r['id'] for r in dev[:16]])
    (out/'e0_calibration.json').write_text(json.dumps(e0,indent=2))
    phase=Budget(min(7200,phase.remaining(),budget.remaining()))
    original=model_digest(npp)
    def persist():
        progress_path.write_text(json.dumps(dict(invocation_id=invocation,base_npp=original,branches=reports),indent=2))
    x_probe,_,_=load_probe(dev[0],cfg,16,20261028);device=next(npp.parameters()).device;x_probe=x_probe.to(device)
    with torch.no_grad():y_initial=npp(x_probe,35).detach()
    for rate_lambda in (.001,.01,.1,1.):
        budget.require('lambda');phase.require('lambda')
        branch=copy.deepcopy(npp)
        torch.manual_seed(20261028);torch.cuda.manual_seed_all(20261028)
        histories={key:[] for key in initial_margins}
        progress_state=dict(additional_steps=0,global_steps=100000,optimizer=None)
        def losses(model,qp,index):
            record=train[(index//4)%len(train)]
            x,target,_=load_probe(record,cfg,16,20261028+index//(4*len(train)));x=x.to(device)
            y=model(x,qp);pred=proxy(y,qp,rate_mode='uniform')
            return dict(rate=pred.bpp,distortion=(pred.reconstruction-x).square().mean(),task=analyzer.loss(pred.reconstruction,target))
        def emit(row):
            with torch.no_grad():row['output_change_rms']=float((branch(x_probe,35)-y_initial).square().mean().sqrt())
            log_event(out,row.pop('stage'),**row)
        def probe(model,step):
            tag=('lambda_'+str(rate_lambda)+'_'+str(step)).replace('.','p')
            result=evaluate_probe(model,dev[:16],cfg,analyzer,codec,out,phase,tag,arms=('npp',))
            values=margins(result,base)
            for key,value in values.items():histories[key].append(value)
            report=dict(step=step,rate_lambda=rate_lambda,margins=values)
            (out/(tag+'_margins.json')).write_text(json.dumps(report,indent=2))
            if not e0['available']:return False
            supported=[key for key in histories if key in e0['spreads_pp'] and initial_margins[key].get('valid')]
            return len(supported)>=4 and all(rd_should_stop(histories[key],initial_margins[key]['margin_pp'],e0['delta_pp']) for key in supported)
        def progress(value,optimizer):
            progress_state.update(value,optimizer=optimizer)
            path=out/'recovery_progress.json'
            path.write_text(json.dumps(dict(invocation_id=invocation,base_npp=original,rate_lambda=rate_lambda,**value)))
        def save_branch(report):
            name=('lambda_'+str(rate_lambda)).replace('.','p')
            torch.save(dict(npp=branch.state_dict(),report=report,base_npp=original,invocation_id=invocation,
                optimizer=progress_state['optimizer'].state_dict() if progress_state['optimizer'] else None,
                rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                    cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])),out/(name+'.pt'))
        try:
            report=finetune(branch,losses,steps=1000,rate_lambda=rate_lambda,
                max_seconds=min(phase.remaining(),budget.remaining()),log=emit,probe=probe,probe_every=500,progress=progress)
        except BaseException as error:
            partial=dict(additional_steps=progress_state['additional_steps'],global_steps=progress_state['global_steps'],
                rate_lambda=rate_lambda,margin_history=histories,error=f'{type(error).__name__}: {error}',stop_reason='error')
            save_branch(partial);reports.append(partial);persist()
            raise
        report['margin_history']=histories
        report['rd_quality_stop_available']=e0['available'];save_branch(report)
        reports.append(report);persist()
    # Expand only the two best finite endpoint comparisons; reuse exact existing baselines.
    def score(report):
        points=[v[-1]['margin_pp'] for v in report['margin_history'].values() if v and v[-1].get('valid')]
        return mean(points) if points else -float('inf')
    finalists=sorted(reports,key=score,reverse=True)[:2]
    for count in (32,48):
        if len(dev)<count or phase.remaining()<=0:break
        for final in finalists:
            if not math.isfinite(score(final)):continue
            latest=[v[-1] for v in final['margin_history'].values() if v and v[-1].get('valid')]
            if latest and all(v['low_pp']>0 or v['high_pp']<-.5 for v in latest):continue
            rate_lambda=final['rate_lambda'];name=('lambda_'+str(rate_lambda)).replace('.','p')
            branch=copy.deepcopy(npp)
            branch.load_state_dict(torch.load(out/(name+'.pt'),map_location=device,weights_only=False)['npp'])
            anchor=evaluate_probe(npp,dev[:count],cfg,analyzer,codec,out,phase,'expanded_base_'+str(count),arms=('codec_only',))
            result=evaluate_probe(branch,dev[:count],cfg,analyzer,codec,out,phase,name+'_expanded_'+str(count),arms=('npp',))
            final.setdefault('expanded',[]).append(dict(sources=count,margins=margins(result,anchor)))
            persist()
    return reports


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',choices=['ot','ar'],required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--checkpoint-sha256',required=True)
    p.add_argument('--training-source-sha256',required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--inputs',type=Path,default=Path('/kaggle/input'))
    p.add_argument('--assets',type=Path,default=Path('/kaggle/working/paper_assets'))
    p.add_argument('--stage',choices=['diagnose','recover','evaluate'],default='diagnose')
    p.add_argument('--max-seconds',type=int,default=7200)
    p.add_argument('--calibration-sources',type=int,default=32)
    p.add_argument('--dev-sources',type=int,default=16,choices=[16,32,48])
    p.add_argument('--symbol-positions',type=int,default=512)
    p.add_argument('--prior-updates',type=int,default=8192)
    p.add_argument('--lr-probe-steps',type=int,default=256)
    return p


def main():
    args=parser().parse_args();out=args.out;out.mkdir(parents=True,exist_ok=True)
    budget=Budget(args.max_seconds);report=dict(task=args.task,stage=args.stage,diagnostic_complete=False,
        scientific_complete=False,npp_updates=0,started=time.time(),checkpoint_sha256=args.checkpoint_sha256)
    npp=None;before=None;recovery_invocation=None
    try:
        torch.set_num_threads(2)
        if not torch.cuda.is_available():raise RuntimeError('Kaggle CUDA run required; CPU unit probes are separate')
        random.seed(20261028);torch.manual_seed(20261028);device='cuda'
        saved=load_checkpoint(args.checkpoint,args.checkpoint_sha256,args.training_source_sha256,args.task)
        cfg=saved['config'];npp=VideoNPP(cfg['video_width']).to(device);npp.load_state_dict(saved['npp']);npp.eval().requires_grad_(False)
        original=VirtualCodec(cfg['block_size'],cfg['motion_radius']).to(device);original.load_state_dict(saved['proxy'])
        proxy=DiagnosticCodec.from_legacy(original).requires_grad_(False);before=model_digest(npp)
        rows,validation,inventory=records_for_task(args.task,args.inputs,args.assets)
        if inventory['dataset_versions']!=cfg['runtime_identity']['dataset_versions']:
            raise ValueError('dataset version differs from training')
        if records_digest(rows)!=cfg['runtime_identity']['train_records_sha256']:
            raise ValueError('TRAIN source inventory differs from training')
        eligible=eligible_records(rows,args.calibration_sources+48)
        train,dev_all=calibration_split(eligible,args.calibration_sources,48,20261028);dev=dev_all[:args.dev_sources]
        identity=dict(checkpoint=args.checkpoint_sha256,legacy_source=args.training_source_sha256,
            diagnostic_source=__import__('os').environ.get('PAPER_SOURCE_SHA256'),
            train_records=records_digest(train),dev_records=records_digest(dev_all),seed=20261028,
            datasets=inventory['dataset_versions'],symbol_positions=args.symbol_positions)
        import subprocess
        identity['ffmpeg_version']=subprocess.check_output([str(codec_path) for codec_path in ('ffmpeg','-version')],text=True).splitlines()[0]
        bind_cache(out/'binding',identity)
        (out/'cohort.json').write_text(json.dumps(dict(identity=identity,
            train=[r['id'] for r in train],dev_nested=[r['id'] for r in dev_all],
            selected_dev=[r['id'] for r in dev],crop_policy='TRAIN deterministic visible-reference crops'),indent=2))
        codec=RealCodec(preset=cfg['preset'],fps=cfg['fps'],gop=cfg['gop'])
        log_event(out,'diagnostic_ready',task=args.task,train_sources=len(train),dev_sources=len(dev),npp=before,
            checkpoint_steps=saved['state']['phase_steps'])
        dev_cache=[]
        paired=paired_measurements(npp,proxy,dev,cfg,codec,out,budget,identity,collect=dev_cache,positions=args.symbol_positions)
        report['original_gate']=paired['gate']
        prior_phase=Budget(min(5400,budget.remaining()))
        train_cache=build_train_cache(npp,proxy,train,cfg,prior_phase,args.symbol_positions,
            lambda row:log_event(out,row.pop('stage'),**row))
        torch.save(dict(identity=identity,train=train_cache,dev=dev_cache),out/'symbol_cache.pt')
        candidates=fit_candidates(original,train_cache,dev_cache,out,budget,args,phase=prior_phase);ranked=[]
        for name,candidate,fit in candidates:
            result=paired_measurements(npp,candidate,dev,cfg,codec,out,prior_phase,identity,label=name)
            ranked.append((not result['gate']['passed'],fit['best_loss'],name,candidate,fit,result['gate']))
        _,_,name,proxy,fit,gate=min(ranked,key=lambda item:item[:2])
        report.update(selected_prior=name,selected_gate=gate,prior_converged=fit['converged'],
            next_action=recovery_decision(gate,fit['converged']),quantizer_probe={str(q):quantizer_probe(proxy.selected_prior(q)) for q in QPS})
        (out/'diagnostic_summary.json').write_text(json.dumps(report,indent=2))
        report['filter_rate_audit']=filter_rate_audit(dev,cfg,codec,out,budget)['scope']
        budget.require('analyzer_setup')
        tracking=ensure_tracking(args.assets/'tracking') if args.task=='ot' else {}
        from .tasks import Analyzer,fingerprint
        analyzer=Analyzer(args.task,device,tracking.get('checkpoint'))
        if analyzer.weight_digest!=cfg['runtime_identity']['analyzer_weights_sha256']:
            raise ValueError('frozen analyzer differs from training')
        report['gradient_audit']=gradient_audit(npp,proxy,analyzer,dev,cfg,out,budget)
        if args.stage=='recover':
            if gate['passed'] and fit['converged']:
                recovery_invocation=hashlib.sha256((args.checkpoint_sha256+str(time.time_ns())).encode()).hexdigest()
                report['recovery_invocation']=recovery_invocation
                report['recovery']=run_recovery(npp,proxy,analyzer,train,dev_all,cfg,codec,out,budget,recovery_invocation)
                report['npp_updates']=sum(r['additional_steps'] for r in report['recovery'])
            else:report['recovery_skipped']='paired_gate_failed' if not gate['passed'] else 'prior_convergence_unverified'
        if args.stage=='evaluate':report['dev_rd']=evaluate_probe(npp,dev,cfg,analyzer,codec,out,budget,'checkpoint')
        if fingerprint(analyzer.model)!=analyzer.weight_digest:raise RuntimeError('analyzer weights changed')
        report['diagnostic_complete']=True
    except BudgetExpired as error:
        report.update(stop_reason='time_budget',stopped_stage=str(error))
    except BaseException as error:
        report['error']=f'{type(error).__name__}: {error}'
        raise
    finally:
        if recovery_invocation is not None and (out/'lambda_runs.json').is_file():
            progress=json.loads((out/'lambda_runs.json').read_text())
            if isinstance(progress,dict) and progress.get('invocation_id')==recovery_invocation and progress.get('base_npp')==before:
                report['recovery']=progress['branches']
                report['npp_updates']=sum(r['additional_steps'] for r in report['recovery'])
        if npp is not None and before is not None:
            report['base_npp_unchanged']=model_digest(npp)==before
            if not report['base_npp_unchanged']:report['error']='base NPP changed unexpectedly'
        report['wall_seconds']=time.time()-report['started']
        (out/'run.json').write_text(json.dumps(report,indent=2))
        log_event(out,'diagnostic_end',task=args.task,complete=report['diagnostic_complete'],
            stop_reason=report.get('stop_reason'),error=report.get('error'),npp_updates=report['npp_updates'])


if __name__=='__main__':main()
