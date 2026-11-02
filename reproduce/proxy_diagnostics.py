"""Bounded, source-disjoint diagnostics and optimization for proxy recovery."""
import copy
import hashlib
import json
import math
from pathlib import Path
import random
from statistics import mean,median
import time

import torch

from .protocol import QPS


def calibration_split(rows,train_count,dev_count,seed):
    if any(r.get('split')!='train' for r in rows):raise ValueError('only TRAIN sources may enter diagnostics')
    if len({r['id'] for r in rows})!=len(rows):raise ValueError('duplicate source identities')
    if min(train_count,dev_count)<1 or train_count+dev_count>len(rows):raise ValueError('insufficient independent sources')
    ordered=sorted(rows,key=lambda r:hashlib.sha256((str(seed)+'/'+r['id']).encode()).hexdigest())
    return ordered[:train_count],ordered[train_count:train_count+dev_count]


def bind_cache(directory,identity):
    directory=Path(directory);path=directory/'identity.json'
    if path.is_file():
        if json.loads(path.read_text())!=identity:raise ValueError('cache identity differs')
    elif directory.is_dir() and any(directory.iterdir()):
        raise ValueError('nonempty cache is missing its identity')
    directory.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(identity,sort_keys=True,indent=2),encoding='utf-8')


def paired_gate(rows,min_pairs=8,agreement=.7,min_coverage=.5):
    groups=[];keys=set()
    for row in rows:
        key=(row['source'],row['codec'],row['qp'])
        if key in keys:raise ValueError('duplicate paired source')
        keys.add(key)
        if row['pixels']<=0 or not all(math.isfinite(row[k]) and row[k]>0 for k in ('raw_bpp','npp_bpp','proxy_raw','proxy_npp')):
            raise ValueError('finite positive measured/proxy rates required')
    for codec in ('h264','h265'):
        for qp in QPS:
            selected=[r for r in rows if r['codec']==codec and r['qp']==qp]
            resolved=[]
            for r in selected:
                delta=r['npp_bpp']-r['raw_bpp'];pred=r['proxy_npp']-r['proxy_raw']
                threshold=max(8/r['pixels'],.005*r['raw_bpp'])
                if abs(delta)>=threshold:resolved.append((delta,pred))
            coverage=len(resolved)/len(selected) if selected else 0
            signs=mean(float(a*b>0) for a,b in resolved) if resolved else None
            passed=len(resolved)>=min_pairs and coverage>=min_coverage and signs>=agreement
            groups.append(dict(codec=codec,qp=qp,pairs=len(selected),resolved=len(resolved),coverage=coverage,
                sign_agreement=signs,passed=passed,
                mean_real_delta_bpp=mean(a for a,_ in resolved) if resolved else None,
                mean_proxy_delta_bpp=mean(b for _,b in resolved) if resolved else None,
                median_scale=median(r['proxy_npp']/r['npp_bpp'] for r in selected) if selected else None))
    return dict(passed=all(g['passed'] for g in groups),groups=groups,
        rule='absolute paired differences; no rejection based on relative rate scale alone')


def rate_ao_margin(baseline,rate,ao):
    ordered=sorted(baseline)
    if (len(ordered)<2 or rate<=0 or not math.isfinite(rate) or not math.isfinite(ao)
        or any(r<=0 or not math.isfinite(r) or not math.isfinite(q) for r,q in ordered)):
        return dict(valid=False,reason='invalid operating points')
    if any(b[0]<=a[0] or b[1]<a[1] for a,b in zip(ordered,ordered[1:])):
        return dict(valid=False,reason='baseline is nonmonotone or has tied rates')
    if rate<ordered[0][0] or rate>ordered[-1][0]:return dict(valid=False,reason='outside baseline rate support')
    a,b=next((a,b) for a,b in zip(ordered,ordered[1:]) if a[0]<=rate<=b[0])
    fraction=math.log(rate/a[0])/math.log(b[0]/a[0])
    reference=a[1]+fraction*(b[1]-a[1])
    return dict(valid=True,reference_ao=reference,margin_pp=100*(ao-reference))


def margin_interval(baseline_curves,candidate_curve,samples=256,seed=20261028):
    ids=[p['source'] for p in candidate_curve['points']]
    if len(ids)<2 or len(set(ids))!=len(ids):raise ValueError('independent source points required')
    if any([p['source'] for p in c['points']]!=ids for c in baseline_curves):
        raise ValueError('RD bootstrap sources must be paired')
    def aggregate(points,indices):
        selected=[points[i] for i in indices]
        pixels=sum(p['pixels'] for p in selected);weight=sum(p['quality_weight'] for p in selected)
        if pixels<=0 or weight<=0:raise ValueError('positive RD aggregation weights required')
        return 8*sum(p['bytes'] for p in selected)/pixels,sum(p['quality_sum'] for p in selected)/weight
    def measure(indices):
        base=[aggregate(c['points'],indices) for c in baseline_curves]
        rate,quality=aggregate(candidate_curve['points'],indices)
        return rate_ao_margin(base,rate,quality)
    point=measure(list(range(len(ids))))
    if not point['valid']:return point
    rng=random.Random(seed);draws=[]
    for _ in range(samples):
        row=measure([rng.randrange(len(ids)) for _ in ids])
        if row['valid']:draws.append(row['margin_pp'])
    if len(draws)<samples*.5:return dict(valid=False,reason='insufficient overlapping monotone bootstrap draws')
    ordered=sorted(draws)
    return dict(point,low_pp=ordered[int(.025*(len(ordered)-1))],high_pp=ordered[int(.975*(len(ordered)-1))],
        bootstrap_valid=len(draws),bootstrap_total=samples,sources=len(ids),scope='source bootstrap; one paired tracker seed')


def rd_should_stop(history,initial_margin_pp,tolerance_pp=.5):
    if len(history)<2:return False
    recent=history[-2:]
    return all(r.get('valid') and r['high_pp'] < -tolerance_pp
        and r['margin_pp']<=initial_margin_pp+tolerance_pp for r in recent)


def calibrate_rd_tolerance(repeated):
    if len(repeated)<3:return dict(available=False,reason='three paired tracker seeds required')
    keys=set.intersection(*(set(row) for row in repeated))
    spreads={key:max(row[key]['margin_pp'] for row in repeated)-min(row[key]['margin_pp'] for row in repeated)
        for key in keys if all(row[key].get('valid') for row in repeated)}
    if not spreads:return dict(available=False,reason='no comparable in-support repeated margins')
    return dict(available=True,delta_pp=max(.5,max(spreads.values())),spreads_pp=spreads,
        seeds=len(repeated),scope='observed repeated-seed range with0.5pp minimum; not a formal stochastic CI')


def evaluate_prior(model,batches):
    device=next(model.parameters()).device;values={}
    with torch.no_grad():
        for row in batches:
            codes=row['codes'].to(device)
            value=float(-model.selected_prior(row['qp'])(codes).log2().mean()*codes.shape[1]/model.block**2)
            values.setdefault(str(row['qp'])+'/'+row['kind'],[]).append(value)
    if not values:raise ValueError('nonempty prior dev probe required')
    groups={k:mean(v) for k,v in values.items()}
    return dict(loss=mean(groups.values()),groups=groups)


def fit_prior(model,train_batches,dev_batches,steps=256,lr=1e-3,eval_every=256,max_seconds=5400,log=None):
    train_ids={r['source'] for r in train_batches};dev_ids={r['source'] for r in dev_batches}
    if not train_ids or not dev_ids or train_ids&dev_ids:raise ValueError('prior TRAIN/dev sources must be disjoint')
    if steps<0 or eval_every<1 or lr<=0 or not math.isfinite(lr):raise ValueError('valid prior fit limits required')
    buckets={q:[r for r in train_batches if r['qp']==q] for q in QPS}
    if any(not v for v in buckets.values()):raise ValueError('TRAIN cache must cover all four QPs')
    if set(r['qp'] for r in dev_batches)!=set(QPS):raise ValueError('dev cache must cover all four QPs')
    started=time.monotonic();model.requires_grad_(False);parameters=[];seen=set()
    for qp in QPS:
        prior=model.selected_prior(qp).requires_grad_(True)
        for p in prior.parameters():
            if id(p) not in seen:parameters.append(p);seen.add(id(p))
    optimizer=torch.optim.Adam(parameters,lr=lr);device=next(model.parameters()).device
    initial=evaluate_prior(model,dev_batches);best=initial['loss'];best_step=0
    best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    history=[dict(step=0,**initial)];counts={str(q):0 for q in QPS};stop='step_budget';updates=0;symbols=0
    for step in range(steps):
        if time.monotonic()-started>=max_seconds:stop='time_budget';break
        qp=QPS[step%4];row=buckets[qp][(step//4)%len(buckets[qp])]
        codes=row['codes'].to(device);optimizer.zero_grad(set_to_none=True)
        loss=-model.selected_prior(qp)(codes).log2().mean()*codes.shape[1]/model.block**2
        if not torch.isfinite(loss):raise RuntimeError('nonfinite prior loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(parameters,10.,error_if_nonfinite=True);optimizer.step()
        updates+=1;counts[str(qp)]+=1;symbols+=codes.numel()
        if updates%eval_every==0 or updates==steps:
            point=dict(step=updates,**evaluate_prior(model,dev_batches));history.append(point)
            if point['loss']<best:
                best=point['loss'];best_step=updates
                best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            if log:log(dict(stage='prior_probe',lr=lr,per_qp=model.per_qp,**point))
            if len(history)>=4:
                tail=history[-4:]
                stable=all(abs(b['loss']-a['loss'])/max(abs(a['loss']),1e-9)<.005 for a,b in zip(tail,tail[1:]))
                best_groups=next(p['groups'] for p in history if p['step']==best_step)
                stable=stable and all(v<=best_groups[k]*1.02 for k,v in point['groups'].items())
                if stable:stop='converged_dev';break
    model.load_state_dict(best_state);model.requires_grad_(False)
    return dict(updates=updates,updates_by_qp=counts,symbols_processed=symbols,lr=lr,
        initial_loss=initial['loss'],best_loss=best,best_step=best_step,history=history,
        stop_reason=stop,converged=stop=='converged_dev',seconds=time.monotonic()-started)


def gradient_vectors(losses,parameters):
    parameters=list(parameters);vectors={}
    for key,loss in losses.items():
        grads=torch.autograd.grad(loss,parameters,allow_unused=True,retain_graph=True) if loss.requires_grad else [None]*len(parameters)
        vector=torch.cat([(g.detach() if g is not None else torch.zeros_like(p)).flatten() for p,g in zip(parameters,grads)])
        if not torch.isfinite(vector).all():raise RuntimeError('nonfinite component gradient')
        vectors[key]=vector
    return vectors


def summarize_gradients(vectors):
    norms={key:float(value.norm()) for key,value in vectors.items()}
    cosines={}
    names=list(vectors)
    for i,left in enumerate(names):
        for right in names[i+1:]:
            denominator=norms[left]*norms[right]
            cosines[left+'/'+right]=float(torch.dot(vectors[left],vectors[right])/denominator) if denominator else None
    return dict(norms=norms,cosines=cosines,rate_to_task=norms.get('rate',0)/norms['task'] if norms.get('task') else None)


def component_gradients(losses,parameters):
    return summarize_gradients(gradient_vectors(losses,parameters))


def noise_gradient_summary(samples,fixed):
    if len(samples)<2:raise ValueError('multiple gradient samples required')
    stacked=torch.stack(samples);vectors=dict(fixed,rate=stacked.mean(0))
    report=summarize_gradients(vectors)
    report.update(noise_samples=len(samples),noise_sample_norm_mean=float(stacked.norm(dim=1).mean()),
        noise_variance=float(stacked.var(0,unbiased=False).sum()),
        noise_gradient_rms=float(stacked.square().sum(1).mean().sqrt()))
    return report


def parameter_drift(model,initial):
    parameters=list(model.parameters())
    delta=float(torch.sqrt(sum((p.detach()-a.to(p.device)).square().sum() for p,a in zip(parameters,initial))))
    base=float(torch.sqrt(sum(a.square().sum() for a in initial)))
    return dict(theta_delta_norm=delta,theta_relative_delta=delta/base if base else None)


def finetune(model,loss_function,steps=1000,lr=1e-4,rate_lambda=.001,original_steps=100000,
             max_seconds=7200,log=None,probe=None,probe_every=250,progress=None):
    if steps<1 or lr<=0 or rate_lambda<=0:raise ValueError('positive supplementary steps/lr/lambda required')
    model.requires_grad_(True);model.train();parameters=list(model.parameters())
    initial=[p.detach().cpu().clone() for p in parameters]
    optimizer=torch.optim.Adam(parameters,lr=lr)
    started=time.monotonic();updates=0;stop='step_budget';last=None
    for index in range(steps):
        if time.monotonic()-started>=max_seconds:stop='time_budget';break
        qp=QPS[index%4];optimizer.zero_grad(set_to_none=True)
        parts=loss_function(model,qp,index)
        loss=10*parts['distortion']+10*rate_lambda*parts['rate']+parts['task']
        if not torch.isfinite(loss):raise RuntimeError('nonfinite supplementary objective')
        loss.backward()
        grads=[p.grad for p in parameters if p.grad is not None]
        if not grads or not all(torch.isfinite(g).all() for g in grads):raise RuntimeError('invalid NPP gradient')
        optimizer.step();updates+=1
        if progress:progress(dict(additional_steps=updates,global_steps=original_steps+updates),optimizer)
        if (updates-1)%100<4 or updates==steps:
            last=dict(stage='npp_update',additional_step=updates,global_step=original_steps+updates,qp=qp,
                lr=optimizer.param_groups[0]['lr'],rate_lambda=rate_lambda,loss=float(loss.detach()),
                rate=float(parts['rate'].detach()),task_loss=float(parts['task'].detach()),**parameter_drift(model,initial))
            if log:log(last)
        if probe and (updates%probe_every==0 or updates==steps):
            try:
                if probe(model,updates):stop='rd_probe';break
            except TimeoutError:
                stop='probe_time_budget';break
    return dict(additional_steps=updates,global_steps=original_steps+updates,lr=lr,
        rate_lambda=rate_lambda,stop_reason=stop,seconds=time.monotonic()-started,
        **parameter_drift(model,initial))


def residual_statistics(x,y):
    r=(y-x).detach().float();spectrum=torch.fft.fft2(r,norm='ortho').abs().square()
    h,w=r.shape[-2:]
    fy=torch.fft.fftfreq(h,device=r.device)[:,None];fx=torch.fft.fftfreq(w,device=r.device)[None,:]
    high=(fy.square()+fx.square()).sqrt()>.25
    energy=float(spectrum.sum())
    return dict(rms=float(r.square().mean().sqrt()),l1=float(r.abs().mean()),
        high_frequency_energy=float(spectrum[...,high].sum()),
        high_frequency_fraction=float(spectrum[...,high].sum())/energy if energy else 0.,
        temporal_residual_rms=float((r[:,1:]-r[:,:-1]).square().mean().sqrt()) if r.shape[1]>1 else 0.,
        clamp_fraction=float(((y<=0)|(y>=1)).float().mean()))
