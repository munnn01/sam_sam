"""Full-geometry codec content calibration and matched paired NPP trials."""
import json
import math
from pathlib import Path
import torch
from . import coco_codec_study as paired
from . import tri_task_study as study
from .codec_content_rate import NALCodec,ContentProxy,fit_scales,fit_mode_power_scales,screen_gate
from .protocol import QPS,proxy_lambda
from .tasks import freeze
from .training import log_event,rng_state,restore_rng
from .proxy_recovery import model_digest

_original_config=paired.config

def config(task,arm,codec='h264'):
    cfg=_original_config(task,arm,codec)
    cfg.update(rate_profile='nal_content_v1',rate_fit_sources=16,
        rate_median_error_max=.35,rate_p90_error_max=.8,rate_direction_min=.75)
    cfg['fidelity_notes']+=['Full deployment geometry decoder-only reconstruction refit; frozen analysis/entropy.',
        'Exact Annex-B non-VCL paid bytes separated; positive content scale fitted on TRAIN x/NPP(x).',
        'Disjoint TRAIN-dev paired rate/reconstruction/direction screen before NPP task updates.']
    return cfg

def _device(proxy):return next(proxy.bank.parameters()).device

def _relaxed(proxy,x,q,seed,gradient=False):
    proxy.eval();proxy.bank.train()
    devices=[x.device.index or 0] if x.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        value=proxy(x,q)
    return value

def reconstruction_fit(proxy,samples,cfg,codec,out,budget):
    proxy.requires_grad_(False);proxy.bank.train()
    entropy_before={k:v.detach().cpu().clone() for k,v in proxy.bank.state_dict().items() if '.g_s.' not in k}
    optimizers={}
    for q in QPS:
        decoder=proxy.bank.networks[str(q)].g_s
        decoder.requires_grad_(True);optimizers[q]=torch.optim.Adam(decoder.parameters(),lr=1e-4)
    updates=0
    for i in range(cfg['proxy_steps']):
        if budget.remaining()<cfg['evaluation_reserve']+900:break
        q=QPS[i%4];x=samples[(i//4)%len(samples)][0].to(_device(proxy))
        with torch.no_grad():actual=codec(x,cfg['codecs'][0],q)
        optimizers[q].zero_grad(set_to_none=True);value=proxy(x,q)
        loss=proxy_lambda(q)*(value.reconstruction-actual.reconstruction).square().mean()
        if not torch.isfinite(loss):raise RuntimeError('nonfinite full-geometry reconstruction loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(proxy.bank.networks[str(q)].g_s.parameters(),10.,error_if_nonfinite=True)
        optimizers[q].step();updates+=1
        if i<4 or i%51==0:
            log_event(out,'content_decoder_fit',step=updates,qp=q,source=samples[(i//4)%len(samples)][2]['id'],
                height=x.shape[-2],width=x.shape[-1],loss=float(loss.detach()),
                real_content_bpp=actual.content_bytes*8/(x.shape[-2]*x.shape[-1]),
                real_overhead_bpp=actual.overhead_bytes*8/(x.shape[-2]*x.shape[-1]))
    for k,v in proxy.bank.state_dict().items():
        if k in entropy_before and not torch.equal(v.cpu(),entropy_before[k]):raise RuntimeError('analysis/entropy changed during decoder-only refit')
    del entropy_before
    freeze(proxy);proxy.eval()
    return dict(updates=updates,target=cfg['proxy_steps'],codec=cfg['codecs'][0],
        scope='decoder-only reconstruction; full TRAIN geometry; analysis and entropy unchanged')

def calibrate_content(proxy,npp,samples,cfg,codec,out,budget):
    rows=[];saved=rng_state();training=npp.training;npp.eval()
    try:
        with torch.no_grad():
            for i,(sample,_,meta) in enumerate(samples[:cfg['rate_fit_sources']]):
                x=sample.to(_device(proxy))
                for q in QPS:
                    budget.require('full_geometry_content_calibration')
                    for variant,z in (('x',x),('npp',npp(x,q))):
                        actual=codec(z,cfg['codecs'][0],q);pixels=z.shape[1]*z.shape[-2]*z.shape[-1]
                        proxy.eval();rounded=proxy(z,q)
                        relaxed=_relaxed(proxy,z,q,cfg['seed']+i*101+q)
                        for mode,value in (('rounded',rounded),('relaxed',relaxed)):
                            rows.append(dict(source=meta['id'],variant=variant,mode=mode,qp=q,
                                height=z.shape[-2],width=z.shape[-1],raw_bpp=float(value.raw_bpp),
                                content_bpp=actual.content_bytes*8/pixels,overhead_bytes=actual.overhead_bytes,
                                paid_bytes=actual.bytes))
                if (i+1)%4==0:log_event(out,'content_rate_fit_cache',sources=i+1,target=cfg['rate_fit_sources'])
        if cfg['rate_profile']=='nal_content_v2_power':
            fits=fit_mode_power_scales(rows)
            for mode,values in fits.items():
                getattr(proxy,mode+'_scale').copy_(proxy.content_scale.new_tensor(values['scales']))
                getattr(proxy,mode+'_power').copy_(proxy.content_scale.new_tensor(values['powers']))
            proxy.separate_modes=True
            scales=None;method='separate(mode,QP) positive power laws; TRAIN-only, non-VCL excluded'
        else:
            scales=fit_scales(rows);proxy.content_scale.copy_(proxy.content_scale.new_tensor(scales));fits=None
            method='geometric-mean positive content scale per QP; non-VCL excluded from target'
        receipt=dict(sources=[s[2]['id'] for s in samples[:cfg['rate_fit_sources']]],rows=rows,scales=scales,
            mode_fits=fits,method=method)
        (out/'content_calibration.json').write_text(json.dumps(receipt,indent=2))
        log_event(out,'content_rate_calibrated',scales=scales,mode_fits=fits,rows=len(rows))
        return dict(scales=scales,mode_fits=fits,rows=len(rows),sources=len(receipt['sources']))
    finally:
        npp.train(training);proxy.eval();restore_rng(saved)

def audit_content(proxy,npp,samples,cfg,codec,out,budget,label):
    rows=[];directions=[];saved=rng_state();training=npp.training;npp.eval()
    try:
        for i,(sample,_,meta) in enumerate(samples):
            x=sample.to(_device(proxy))
            for q in QPS:
                budget.require('paired_content_audit')
                with torch.no_grad():variants=(('x',x),('npp',npp(x,q)))
                for variant,z in variants:
                    z=z.detach()
                    with torch.no_grad():
                        actual=codec(z,cfg['codecs'][0],q)
                        proxy.eval();rounded=proxy(z,q)
                        relaxed=_relaxed(proxy,z,q,cfg['seed']+i*101+q)
                        pixels=z.shape[1]*z.shape[-2]*z.shape[-1]
                        rows.append(dict(source=meta['id'],variant=variant,qp=q,real_bpp=actual.bpp,
                            real_content_bpp=actual.content_bytes*8/pixels,real_overhead_bpp=actual.overhead_bytes*8/pixels,
                            content_bytes=actual.content_bytes,overhead_bytes=actual.overhead_bytes,total_bytes=actual.bytes,
                            proxy_bpp=float(rounded.bpp),relaxed_bpp=float(relaxed.bpp),
                            proxy_content_bpp=float(rounded.content_bpp),relaxed_content_bpp=float(relaxed.content_bpp),proxy_overhead_bpp=float(rounded.overhead_bpp),
                            reconstruction_mse=float((rounded.reconstruction-actual.reconstruction).square().mean()),
                            relaxed_reconstruction_mse=float((relaxed.reconstruction-actual.reconstruction).square().mean())))
                    if i<2:
                        v=z.clone().requires_grad_(True)
                        value=_relaxed(proxy,v,q,cfg['seed']+i*101+q)
                        gradient=torch.autograd.grad(value.bpp,v)[0]
                        if not torch.isfinite(gradient).all() or gradient.abs().sum()==0:raise RuntimeError('invalid content rate gradient')
                        with torch.no_grad():
                            delta=gradient.sign()*(2/255)
                            minus=codec((v-delta).clamp(0,1),cfg['codecs'][0],q).bpp
                            plus=codec((v+delta).clamp(0,1),cfg['codecs'][0],q).bpp
                        directions.append(dict(source=meta['id'],variant=variant,qp=q,minus_bpp=minus,plus_bpp=plus,
                            aligned=minus<plus,tied=minus==plus))
        paired_changes=[]
        for i in range(0,len(rows),2):
            a,b=rows[i:i+2]
            assert a['variant']=='x' and b['variant']=='npp'
            real=b['real_bpp']/a['real_bpp']-1;pred=b['relaxed_bpp']/a['relaxed_bpp']-1
            paired_changes.append(dict(source=a['source'],qp=a['qp'],real_relative_change=real,proxy_relative_change=pred,
                same_sign=real*pred>0,near_tie=abs(real)<.02))
        direction=dict(aligned=sum(r['aligned'] for r in directions),untied=sum(not r['tied'] for r in directions),
            total=len(directions),rows=directions)
        gate=screen_gate(rows,direction,cfg)
        result=dict(label=label,rows=rows,paired_changes=paired_changes,direction=direction,gate=gate,
            scope='disjoint TRAIN-dev full geometry x/NPP; no VAL fitting; local finite direction test',
            proxy_sha256=model_digest(proxy))
        (out/('content_audit_'+label+'.json')).write_text(json.dumps(result,indent=2))
        log_event(out,'content_rate_audit',label=label,**gate)
        return result
    finally:
        npp.train(training);proxy.eval();restore_rng(saved)

def main():
    paired.config=config
    codecs=[]
    def make_codec(**kwargs):
        # The pinned config file is the authority for the codec.
        import sys
        cfg=json.loads(Path(sys.argv[sys.argv.index('--config')+1]).read_text())
        codec=NALCodec(cfg['codecs'][0],**kwargs);codecs.append(codec);return codec
    study.RealCodec=make_codec
    study.FrameProxy=lambda bank:ContentProxy(bank,codecs[-1])
    study.fit_video_proxy=reconstruction_fit
    paired.main()

if __name__=='__main__':main()
