"""Official image NPP wrapper and explicitly reconstructed video-paper NPP."""
import torch
from torch import nn
from torch.nn import functional as F
from .protocol import QPS,task_lambda
from .vendor.fixed.Preprocessor import Preprocessor as FixedNPP
from .vendor.adaptive.Preprocessor import Preprocessor as AdaptiveNPP

class ImageNPP(nn.Module):
    def __init__(self,adaptive=False):
        super().__init__(); self.adaptive=bool(adaptive)
        self.network=AdaptiveNPP() if adaptive else FixedNPP()

    def forward(self,x,qp):
        if qp not in QPS or x.ndim!=5 or x.shape[2]!=3:
            raise ValueError('N,T,RGB,H,W and configured QP required')
        b,t,c,h,w=x.shape
        flat=x.reshape(b*t,c,h,w)
        flat=F.pad(flat,(0,(-w)%16,0,(-h)%16),mode='replicate')
        if self.adaptive:
            # Released source uses loss-lambda/8; documented separately from raw-QP description.
            condition=x.new_tensor([task_lambda(qp)/8])
            out=self.network(flat,condition)
        else:out=self.network(flat)
        return out[...,:h,:w].clamp(0,1).reshape(b,t,c,h,w)

    @classmethod
    def from_fixed(cls,fixed):
        if fixed.adaptive:raise ValueError('fixed-phase NPP required')
        out=cls(adaptive=True).to(next(fixed.parameters()).device)
        missing,unexpected=out.network.load_state_dict(fixed.network.state_dict(),strict=False)
        if unexpected or any('Scaling_Net' not in k for k in missing):
            raise ValueError('non-QA layers differ at phase transition')
        # Insert QA layers as exact unit scales. This preserves learned fixed weights.
        for name,module in out.network.named_modules():
            if name.endswith('layer') and isinstance(module,nn.Sequential):
                final=module[-2]
                if isinstance(final,nn.Linear):
                    nn.init.zeros_(final.weight); nn.init.ones_(final.bias)
        out.train(fixed.training)
        return out

class ResidualBlock(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.layers=nn.Sequential(nn.Conv3d(width,width,(1,3,3),padding=(0,1,1)),nn.ReLU(),
                                  nn.Conv3d(width,width,(1,3,3),padding=(0,1,1)))
    def forward(self,x):return F.relu(x+self.layers(x))

class VideoNPP(nn.Module):
    """Figure2 reconstruction, not an unreleased author's exact implementation.

    Spatial/temporal branches, conditional channel attention and RGB residual are
    published. Width16, kernels, residual-block count and attention equation are
    explicit choices where the paper is incomplete. No ConvGRU or ROI bank.
    """
    def __init__(self,width=16):
        super().__init__()
        self.spatial=nn.Sequential(nn.Conv3d(3,width,(1,3,3),padding=(0,1,1)),nn.ReLU(),ResidualBlock(width))
        self.temporal=nn.Sequential(nn.Conv3d(3,width,3,padding=1),nn.ReLU(),ResidualBlock(width))
        self.attention=nn.Sequential(nn.AdaptiveAvgPool3d(1),nn.Conv3d(width*2,width,1),nn.Sigmoid())
        self.rgb=nn.Conv3d(width,3,(1,3,3),padding=(0,1,1))

    def forward(self,x,qp):
        if qp not in QPS or x.ndim!=5 or x.shape[2]!=3:
            raise ValueError('N,T,RGB,H,W and configured QP required')
        frames=x.transpose(1,2)
        s=self.spatial(frames); t=self.temporal(frames)
        a=self.attention(torch.cat((s,t),dim=1))
        return (frames+self.rgb(a*s+(1-a)*t)).clamp(0,1).transpose(1,2)
