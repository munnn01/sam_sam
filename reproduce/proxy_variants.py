"""Isolated diagnostic extensions; original virtual-codec source stays unchanged."""
import copy
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from .protocol import QPS
from .virtual_codec import VirtualCodec, blocks, block_dct, block_idct


@dataclass
class DiagnosticOutput:
    reconstruction: torch.Tensor
    bpp: torch.Tensor
    symbols: tuple
    latents: tuple
    entropy_bpp: torch.Tensor
    mode_bpp: torch.Tensor
    zero_reference_bpp: torch.Tensor
    excess_bpp: torch.Tensor
    zero_fraction: torch.Tensor
    zero_symbol_cost_bpp: torch.Tensor
    nonzero_symbol_cost_bpp: torch.Tensor


class DiagnosticCodec(VirtualCodec):
    def __init__(self,block=8,motion_radius=4,per_qp=False):
        super().__init__(block,motion_radius)
        self.per_qp=bool(per_qp)
        self.priors=nn.ModuleDict({str(q):copy.deepcopy(self.prior) for q in QPS}) if per_qp else nn.ModuleDict()

    @classmethod
    def from_legacy(cls,codec,per_qp=False):
        out=cls(codec.block,codec.motion_radius,per_qp).to(next(codec.parameters()).device)
        out.prior.load_state_dict(codec.prior.state_dict())
        for prior in out.priors.values():prior.load_state_dict(codec.prior.state_dict())
        return out

    def selected_prior(self,qp):
        if qp not in QPS:raise ValueError('configured QP required')
        return self.priors[str(qp)] if self.per_qp else self.prior

    def forward(self,x,qp,rate_mode='ste',generator=None,noise_sign=1):
        if qp not in QPS or x.ndim!=5 or x.shape[2]!=3 or rate_mode not in ('ste','uniform') or noise_sign not in (-1,1):
            raise ValueError('configured QP, rate mode and N,T,RGB,H,W required')
        b,t,c,h,w=x.shape;n=self.block
        if min(b,t,h,w)<1:raise ValueError('nonempty video required')
        prior=self.selected_prior(qp)
        p0=prior(x.new_zeros((1,3*n*n,1,1)))
        per_block_zero=-p0.log2().sum()
        previous=None;outputs=[];symbols=[];latents=[]
        bits=x.new_zeros(());entropy=x.new_zeros(());mode_bits=x.new_zeros(())
        zero_cost=x.new_zeros(());nonzero_cost=x.new_zeros(());zeros=x.new_zeros(())
        symbol_count=0;block_count=0
        step=2**((qp-4)/6)/255.
        for i in range(t):
            current=F.pad(x[:,i],(0,(-w)%n,0,(-h)%n),mode='replicate')
            prediction,modes=self.prediction(current,previous)
            latent=block_dct(current-prediction,n)/step
            quantized=latent+(latent.round()-latent).detach()
            bh,bw=current.shape[-2]//n,current.shape[-1]//n
            def pack(values):
                return blocks(values,n).permute(0,1,4,5,2,3).reshape(b,3*n*n,bh,bw)
            codes=pack(quantized);raw_codes=pack(latent)
            rate_codes=codes
            if rate_mode=='uniform':
                noise=torch.rand(raw_codes.shape,device=x.device,dtype=x.dtype,generator=generator)-.5
                rate_codes=raw_codes+noise_sign*noise
            cost=-prior(rate_codes).log2()
            overhead=b*bh*bw*math.log2(modes)
            # Preserve the original arithmetic ordering for exact STE parity.
            bits=bits+cost.sum()+overhead
            entropy=entropy+cost.sum();mode_bits=mode_bits+overhead
            mask=codes.detach()==0
            zero_cost=zero_cost+cost[mask].sum();nonzero_cost=nonzero_cost+cost[~mask].sum()
            zeros=zeros+mask.sum();symbol_count+=codes.numel();block_count+=b*bh*bw
            previous=(prediction+block_idct(quantized*step,n)).clamp(0,1)
            outputs.append(previous[...,:h,:w]);symbols.append(codes);latents.append(raw_codes)
        pixels=b*t*h*w
        entropy=entropy/pixels;reference=per_block_zero*block_count/pixels
        return DiagnosticOutput(torch.stack(outputs,dim=1),bits/pixels,tuple(symbols),tuple(latents),
            entropy,mode_bits/pixels,reference,entropy-reference,zeros/symbol_count,
            zero_cost/pixels,nonzero_cost/pixels)


def quantizer_probe(prior,positions=(0.,.1,.25,.49),samples=2048):
    """Midpoint quadrature of expected NLL derivatives, not E[absolute derivative]."""
    if samples<2:raise ValueError('at least two quadrature points required')
    parameter=next(prior.parameters());rows=[]
    u=(torch.arange(samples,device=parameter.device,dtype=parameter.dtype)+.5)/samples-.5
    for position in positions:
        z=torch.full((1,prior.channels,1,samples),float(position),device=parameter.device,
                     dtype=parameter.dtype,requires_grad=True)
        rounded=z+(z.round()-z).detach()
        ste,=torch.autograd.grad((-prior(rounded).log2()).sum(),z)
        noisy=z+u.view(1,1,1,-1)
        noise,=torch.autograd.grad((-prior(noisy).log2()).sum(),z)
        expected=noise.mean(-1)
        rows.append(dict(latent=float(position),ste_gradient_abs=float(ste.abs().mean()),
            noise_expected_gradient_abs=float(expected.abs().mean()),
            noise_sample_gradient_abs=float(noise.abs().mean()),noise_variance=float(noise.var(-1).mean())))
    return rows
