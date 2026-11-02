"""Released NPP forward, framewise proxy and frozen task adapters."""
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn import functional as F
from .models import ImageNPP
from .protocol import QPS
from .proxy_recovery import model_digest
from .tasks import freeze

def clip_output(raw,training):
    clipped=raw.clamp(0,1)
    return raw+(clipped-raw).detach() if training else clipped

class PaperNPP(ImageNPP):
    def forward(self,x,qp):
        if qp not in QPS or x.ndim!=5 or x.shape[2]!=3:raise ValueError('configured QP and N,T,RGB,H,W required')
        from .protocol import task_lambda
        b,t,c,h,w=x.shape;flat=F.pad(x.reshape(b*t,c,h,w),(0,(-w)%16,0,(-h)%16),mode='replicate')
        raw=self.network(flat,x.new_tensor([task_lambda(qp)/8])) if self.adaptive else self.network(flat)
        return clip_output(raw[...,:h,:w],self.training).reshape_as(x)

class FrameProxy(nn.Module):
    def __init__(self,bank):
        super().__init__();self.bank=bank
        self.log_scale=nn.Parameter(torch.zeros(4));self.log_floor=nn.Parameter(torch.full((4,),-12.))
    def forward(self,x,qp):
        b,t,c,h,w=x.shape;value=self.bank(x.reshape(b*t,1,c,h,w),qp);i=QPS.index(qp)
        return SimpleNamespace(reconstruction=value.reconstruction.reshape_as(x),
            bpp=self.log_scale[i].clamp(-9,9).exp()*value.bpp+self.log_floor[i].clamp(-18,0).exp(),raw_bpp=value.bpp)

class CocoTask:
    def __init__(self,task,model=None,device='cpu'):
        if task not in ('is','od'):raise ValueError('COCO task is segmentation or detection')
        if model is None:
            if task=='is':
                from torchvision.models.detection import maskrcnn_resnet50_fpn,MaskRCNN_ResNet50_FPN_Weights
                model=maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.COCO_V1)
            else:
                from torchvision.models.detection import fcos_resnet50_fpn,FCOS_ResNet50_FPN_Weights
                model=fcos_resnet50_fpn(weights=FCOS_ResNet50_FPN_Weights.COCO_V1)
        self.task=task;self.device=torch.device(device);self.model=freeze(model).to(device);self.weight_digest=model_digest(model)
    def loss(self,x,target):
        self.model.train()
        for m in self.model.modules():
            if isinstance(m,nn.modules.batchnorm._BatchNorm):m.eval()
        try:return sum(self.model([x[0,0]],[{k:v.to(x.device) for k,v in target.items()}]).values())
        finally:self.model.eval()
    def predict(self,x):
        self.model.eval();return self.model([x[0,0]])[0]
    def receipt(self):return dict(task=self.task,model=type(self.model).__name__,weight_sha256=self.weight_digest,
        trainable_parameters=sum(p.numel() for p in self.model.parameters() if p.requires_grad))

class TrackingTask:
    def __init__(self,checkpoint,device='cuda'):
        from .tasks import Analyzer
        self.backend=Analyzer('ot',device,checkpoint);self.model=self.backend.model;self.weight_digest=model_digest(self.model)
    def loss(self,x,target):return self.backend.loss(x,target)
    def predict(self,x,initial_box):return self.backend.track(x,initial_box)
    def receipt(self):return dict(self.backend.receipt(),weight_sha256=self.weight_digest)
