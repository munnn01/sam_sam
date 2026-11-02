"""Exact Annex-B accounting and deployment-geometry content rate proxies."""
from pathlib import Path
from types import SimpleNamespace
import math
import re
import tempfile
import os
import torch
from torch import nn
from .codecs import RealCodec
from .protocol import QPS

def nal_accounting(data,codec):
    if codec not in ('h264','h265'):raise ValueError('Annex-B AVC or HEVC required')
    starts=list(re.finditer(b'\x00\x00(?:\x00)?\x01',data))
    if not starts or any(data[:starts[0].start()]):raise ValueError('invalid Annex-B prefix')
    content=0;overhead=starts[0].start();types=[]
    for i,start in enumerate(starts):
        end=starts[i+1].start() if i+1<len(starts) else len(data)
        payload=data[start.end():end]
        if len(payload)<(1 if codec=='h264' else 2):raise ValueError('truncated NAL header')
        typ=payload[0]&31 if codec=='h264' else (payload[0]>>1)&63
        vcl=1<=typ<=5 if codec=='h264' else 0<=typ<=31
        size=end-start.start()
        if vcl:content+=size
        else:overhead+=size
        types.append(dict(type=typ,vcl=vcl,bytes=size))
    if not content or content+overhead!=len(data):raise ValueError('paid stream must contain VCL and exact byte accounting')
    return dict(content_bytes=content,overhead_bytes=overhead,total_bytes=len(data),nals=types)

class NALCodec(RealCodec):
    def __init__(self,codec_name,**kwargs):
        super().__init__(**kwargs);self.codec_name=codec_name;self.overheads={}
    def __call__(self,x,codec,qp,directory=None):
        if codec!=self.codec_name or x.shape[1]!=1:raise ValueError('configured independent one-frame codec required')
        if directory is None:
            with tempfile.TemporaryDirectory(prefix='content-codec-',dir=os.environ.get('PAPER_CODEC_TMP')) as tmp:
                value=self(x,codec,qp,Path(tmp));value.stream='';return value
        value=super().__call__(x,codec,qp,directory)
        parts=nal_accounting(Path(value.stream).read_bytes(),codec)
        key=(qp,*x.shape[1:2],*x.shape[-2:])
        old=self.overheads.get(key)
        if old is not None and old!=parts['overhead_bytes']:raise ValueError('non-VCL byte count changed within codec/QP/geometry')
        self.overheads[key]=parts['overhead_bytes']
        value.content_bytes=parts['content_bytes'];value.overhead_bytes=parts['overhead_bytes']
        return value
    def overhead(self,x,qp):
        key=(qp,*x.shape[1:2],*x.shape[-2:])
        if key not in self.overheads:
            self(torch.full_like(x.detach(),.5),self.codec_name,qp)
        return self.overheads[key]

class ContentProxy(nn.Module):
    def __init__(self,bank,codec):
        super().__init__();self.bank=bank;self.codec=codec
        self.register_buffer('content_scale',torch.ones(4))
        self.register_buffer('rounded_scale',torch.ones(4));self.register_buffer('rounded_power',torch.ones(4))
        self.register_buffer('relaxed_scale',torch.ones(4));self.register_buffer('relaxed_power',torch.ones(4))
        self.register_buffer('mode_power_enabled',torch.tensor(False))
    @property
    def separate_modes(self):return bool(self.mode_power_enabled)
    @separate_modes.setter
    def separate_modes(self,value):self.mode_power_enabled.fill_(bool(value))
    def forward(self,x,qp):
        if qp not in QPS or x.ndim!=5 or x.shape[1]!=1:raise ValueError('configured one-frame proxy input required')
        b,t,c,h,w=x.shape
        value=self.bank(x.reshape(b*t,1,c,h,w),qp)
        i=QPS.index(qp)
        if self.separate_modes:
            scale=self.relaxed_scale[i] if self.bank.training else self.rounded_scale[i]
            power=self.relaxed_power[i] if self.bank.training else self.rounded_power[i]
            raw=value.bpp.clamp_min(1e-9)
            content=scale*raw.pow(power)
            if self.bank.training and x.requires_grad:
                eff_slope=torch.maximum(scale*power*raw.detach().pow(power-1.),.70*scale)
                content=content.detach()+eff_slope*(raw-raw.detach())
        else:content=self.content_scale[i]*value.bpp
        overhead=x.new_tensor(self.codec.overhead(x,qp)*8/(t*h*w))
        recon=value.reconstruction.reshape_as(x)
        if x.requires_grad and not any(p.requires_grad for p in self.bank.parameters()):
            recon=x+(recon-x).detach()
        return SimpleNamespace(reconstruction=recon,bpp=content+overhead,
            content_bpp=content,overhead_bpp=overhead,raw_bpp=value.bpp)

def fit_scales(rows):
    values=[]
    for q in QPS:
        selected=[r for r in rows if r['qp']==q]
        if not selected:raise ValueError('all four QPs required')
        residual=[]
        for row in selected:
            a,b=row['raw_bpp'],row['content_bpp']
            if not (math.isfinite(a) and math.isfinite(b) and a>0 and b>0):raise ValueError('positive finite content rates required')
            residual.append(math.log(b/a))
        value=math.exp(sum(residual)/len(residual))
        if not 1e-4<value<1e4:raise ValueError('content calibration scale outside declared bounds')
        values.append(value)
    return values

def gradient_balance(rate,task,fidelity,parameters):
    params=[p for p in parameters if p.requires_grad]
    vectors=[]
    for loss in (rate,task,fidelity):
        grads=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True) if loss.requires_grad else [None]*len(params)
        if any(g is not None and not torch.isfinite(g).all() for g in grads):raise RuntimeError('nonfinite component gradient')
        vectors.append([g.detach() if g is not None else torch.zeros_like(p) for g,p in zip(grads,params)])
    norms=[float(torch.sqrt(sum(g.float().square().sum() for g in v))) for v in vectors]
    dot=float(sum((a.float()*b.float()).sum() for a,b in zip(vectors[0],vectors[1])))
    return dict(gradient_rate_norm=norms[0],gradient_task_norm=norms[1],gradient_fidelity_norm=norms[2],
        gradient_rate_task_ratio=norms[0]/max(norms[1],1e-12),
        gradient_rate_task_cosine=dot/max(norms[0]*norms[1],1e-12))

def screen_gate(rows,direction,cfg):
    if not rows:raise ValueError('held-out paired probe rows required')
    def summary(actual,predicted):
        errors=[]
        for r in rows:
            a=r.get(actual,r['real_bpp']);b=r.get(predicted,r['relaxed_bpp'])
            if not (math.isfinite(a) and math.isfinite(b) and a>0 and b>0):return float('inf'),float('inf')
            errors.append(abs(b/a-1))
        errors.sort()
        def quantile(q):
            pos=(len(errors)-1)*q;i=int(pos);f=pos-i
            return errors[i]*(1-f)+errors[min(i+1,len(errors)-1)]*f
        return quantile(.5),quantile(.9)
    median,p90=summary('real_bpp','relaxed_bpp')
    content_median,content_p90=summary('real_content_bpp','relaxed_content_bpp')
    alignment=direction['aligned']/direction['untied'] if direction['untied'] else 0.
    return dict(passed=max(median,content_median)<=cfg['rate_median_error_max']
        and max(p90,content_p90)<=cfg['rate_p90_error_max'] and alignment>=cfg['rate_direction_min'],
        median_relative_error=median,p90_relative_error=p90,
        content_median_relative_error=content_median,content_p90_relative_error=content_p90,
        direction_alignment=alignment,direction_untied=direction['untied'])

def fit_mode_power_scales(rows,allow_power=True):
    result={}
    for mode in ('rounded','relaxed'):
        scales=[];powers=[]
        for q in QPS:
            selected=[r for r in rows if r.get('qp')==q and r.get('mode')==mode]
            if not selected:raise ValueError('both modes and all QPs required')
            xs=[];ys=[]
            for r in selected:
                a,b=r['raw_bpp'],r['content_bpp']
                if not (math.isfinite(a) and math.isfinite(b) and a>0 and b>0):
                    raise ValueError('positive finite calibration rates required')
                xs.append(math.log(a));ys.append(math.log(b))
            mx=sum(xs)/len(xs);my=sum(ys)/len(ys);power=1.
            var=sum((x-mx)**2 for x in xs)
            if allow_power and var>1e-12:
                power=min(4.,max(.25,sum((x-mx)*(y-my) for x,y in zip(xs,ys))/var))
            log_scale=my-power*mx
            if not math.log(1e-4)<log_scale<math.log(1e4):raise ValueError('power calibration scale outside declared bounds')
            scales.append(math.exp(log_scale));powers.append(power)
        result[mode]=dict(scales=scales,powers=powers)
    return result
