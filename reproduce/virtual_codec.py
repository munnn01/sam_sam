"""Structured differentiable virtual video codec, with declared missing-paper choices.

Not an x264/x265 implementation. All reported final rates come from real streams.
"""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F
from .protocol import QPS

def dct_matrix(n,reference):
    k=torch.arange(n,device=reference.device,dtype=reference.dtype)[:,None]
    i=torch.arange(n,device=reference.device,dtype=reference.dtype)[None,:]
    matrix=torch.cos(math.pi*(i+.5)*k/n)*math.sqrt(2/n)
    matrix[0]=matrix[0]/math.sqrt(2)
    return matrix

def blocks(x,n):
    b,c,h,w=x.shape
    if h%n or w%n:raise ValueError('block dimensions must divide padded frame')
    return x.reshape(b,c,h//n,n,w//n,n).permute(0,1,2,4,3,5)

def frames(x):
    b,c,bh,bw,n,_=x.shape
    return x.permute(0,1,2,4,3,5).reshape(b,c,bh*n,bw*n)

def block_dct(x,n=8):
    m=dct_matrix(n,x); return frames(torch.matmul(torch.matmul(m,blocks(x,n)),m.T))

def block_idct(x,n=8):
    m=dct_matrix(n,x); return frames(torch.matmul(torch.matmul(m.T,blocks(x,n)),m))

class FactorizedCDF(nn.Module):
    """Independent-channel monotone neural CDF, in the factorized-prior family.

    Fitted on TRAIN transform residuals and then frozen for NPP optimization.
    The authors did not provide entropy-prior weights or an initialization recipe.
    """
    def __init__(self,channels,filters=(1,3,3,1)):
        super().__init__(); self.channels=channels
        self.matrices=nn.ParameterList(); self.biases=nn.ParameterList();self.factors=nn.ParameterList()
        for i,(a,b) in enumerate(zip(filters,filters[1:])):
            self.matrices.append(nn.Parameter(torch.full((channels,b,a),math.log(math.expm1(1/b)))))
            self.biases.append(nn.Parameter(torch.zeros(channels,b,1)))
            if i<len(filters)-2:self.factors.append(nn.Parameter(torch.zeros(channels,b,1)))

    def cumulative_logits(self,x):
        for i,(m,b) in enumerate(zip(self.matrices,self.biases)):
            x=torch.bmm(F.softplus(m),x)+b
            if i<len(self.factors):x=x+torch.tanh(self.factors[i])*torch.tanh(x)
        return x

    def forward(self,symbols):
        if symbols.shape[1]!=self.channels:raise ValueError('entropy-channel mismatch')
        shape=symbols.shape
        values=symbols.permute(1,0,2,3).reshape(self.channels,1,-1)
        lo=self.cumulative_logits(values-.5); hi=self.cumulative_logits(values+.5)
        # Stable tails: sigmoid(-lo)-sigmoid(-hi) when positive.
        sign=torch.where((lo+hi).detach()>0,-torch.ones_like(lo),torch.ones_like(lo))
        likelihood=(torch.sigmoid(sign*hi)-torch.sigmoid(sign*lo)).abs().clamp_min(1e-9)
        return likelihood.reshape(shape[1],shape[0],shape[2],shape[3]).permute(1,0,2,3)

@dataclass
class VirtualOutput:
    reconstruction: torch.Tensor
    bpp: torch.Tensor
    symbols: tuple

class VirtualCodec(nn.Module):
    def __init__(self,block=8,motion_radius=4):
        super().__init__()
        if block<2 or motion_radius<0:raise ValueError('invalid block/search parameters')
        self.block=block;self.motion_radius=motion_radius
        self.prior=FactorizedCDF(3*block*block)

    def prediction(self,current,reference):
        candidates=[torch.full_like(current,.5)]  # declared independent-block intra DC candidate
        if reference is not None:
            r=self.motion_radius
            shifts=[(0,0)] if r==0 else [(dy,dx) for dy in (-r,0,r) for dx in (-r,0,r)]
            h,w=current.shape[-2:]
            padded=F.pad(reference,(r,r,r,r),mode='replicate')
            candidates += [padded[...,r+dy:r+dy+h,r+dx:r+dx+w] for dy,dx in shifts]
        stack=torch.stack(candidates,dim=1)
        b,k,c,h,w=stack.shape;n=self.block
        patches=stack.reshape(b*k,c,h,w)
        patches=blocks(patches,n).reshape(b,k,c,h//n,w//n,n,n).permute(0,3,4,1,2,5,6)
        target=blocks(current,n).permute(0,2,3,1,4,5).unsqueeze(3)
        best=(patches-target).square().mean(dim=(-1,-2,-3)).detach().argmin(dim=3)
        indices=best[...,None,None,None,None].expand(b,h//n,w//n,1,c,n,n)
        selected=patches.gather(3,indices).squeeze(3).permute(0,3,1,2,4,5)
        return frames(selected),k

    def forward(self,x,qp):
        if qp not in QPS or x.ndim!=5 or x.shape[2]!=3:
            raise ValueError('configured QP and N,T,RGB,H,W required')
        b,t,c,h,w=x.shape;n=self.block
        previous=None;outputs=[];symbols=[];bits=x.new_zeros(())
        step=2**((qp-4)/6)/255.  # declared approximation to conventional QP spacing
        for i in range(t):
            current=F.pad(x[:,i],(0,(-w)%n,0,(-h)%n),mode='replicate')
            prediction,modes=self.prediction(current,previous)
            transformed=block_dct(current-prediction,n)
            latent=transformed/step
            quantized=latent+(latent.round()-latent).detach()
            bh,bw=current.shape[-2]//n,current.shape[-1]//n
            codes=blocks(quantized,n).permute(0,1,4,5,2,3).reshape(b,3*n*n,bh,bw)
            likelihood=self.prior(codes)
            bits=bits-likelihood.log2().sum()+b*bh*bw*math.log2(modes)
            previous=(prediction+block_idct(quantized*step,n)).clamp(0,1)
            outputs.append(previous[...,:h,:w]);symbols.append(codes)
        return VirtualOutput(torch.stack(outputs,dim=1),bits/(b*t*h*w),tuple(symbols))
