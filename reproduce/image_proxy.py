"""QP-specific Minnen2018 autoregressive/hyperprior proxies; actual BPG targets."""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from .protocol import QPS

def likelihood_bpp(output,h,w):
    values=list(output['likelihoods'].values())
    if not values or h<1 or w<1:raise ValueError('likelihood tensors and image dimensions required')
    batch=values[0].shape[0]
    return sum((-p.clamp_min(1e-9).log2()).sum() for p in values)/(batch*h*w)

@dataclass
class ImageProxyOutput:
    reconstruction:torch.Tensor
    bpp:torch.Tensor

class ImageProxyBank(nn.Module):
    def __init__(self,qualities=None,pretrained=True):
        super().__init__()
        from compressai.zoo import mbt2018
        qualities=qualities or {'30':5,'35':4,'40':3,'45':2}
        self.networks=nn.ModuleDict({str(q):mbt2018(quality=qualities[str(q)],metric='mse',pretrained=pretrained)
                                    for q in QPS})

    def forward(self,x,qp):
        if qp not in QPS or x.ndim!=5 or x.shape[1]!=1:raise ValueError('image proxy expects one frame and configured QP')
        b,_,c,h,w=x.shape;flat=x[:,0]
        padded=F.pad(flat,(0,(-w)%64,0,(-h)%64),mode='replicate')
        out=self.networks[str(qp)](padded)
        return ImageProxyOutput(out['x_hat'][...,:h,:w].clamp(0,1)[:,None],likelihood_bpp(out,h,w))

    def main_parameters(self,qp):
        return [p for n,p in self.networks[str(qp)].named_parameters() if not n.endswith('.quantiles')]

    def aux_parameters(self,qp):
        return [p for n,p in self.networks[str(qp)].named_parameters() if n.endswith('.quantiles')]

    def aux_loss(self,qp):return self.networks[str(qp)].aux_loss()
