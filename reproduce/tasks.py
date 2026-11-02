"""Pretrained, frozen original-task analyzers with differentiable input paths."""
import contextlib
import hashlib
import math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

def freeze(module):
    module.eval()
    for p in module.parameters():p.requires_grad_(False)
    return module

def fingerprint(module):
    h=hashlib.sha256()
    for name,value in sorted(module.state_dict().items()):
        h.update(name.encode());h.update(value.detach().contiguous().cpu().numpy().tobytes())
    return h.hexdigest()

def tracking_score_loss(scores,boxes,image_size):
    if scores.ndim!=4 or scores.shape[1]!=1 or len(boxes)!=scores.shape[0]:
        raise ValueError('tracking scores and GT frames must align')
    h,w=image_size;sh,sw=scores.shape[-2:]
    yy=torch.linspace(0,h,sh,device=scores.device,dtype=scores.dtype)
    xx=torch.linspace(0,w,sw,device=scores.device,dtype=scores.dtype)
    cy=boxes[:,1]+boxes[:,3]/2;cx=boxes[:,0]+boxes[:,2]/2
    sigma=(boxes[:,2]*boxes[:,3]).sqrt().clamp_min(1)*.25
    target=torch.exp(-.5*((yy[None,:,None]-cy[:,None,None])**2+
                           (xx[None,None,:]-cx[:,None,None])**2)/sigma[:,None,None]**2)
    return F.mse_loss(scores[:,0],target)

@contextlib.contextmanager
def trusted_legacy_checkpoint_load():
    # Only used for the official DiMP checkpoint downloaded from the author's model zoo.
    original=torch.load
    def load(*args,**kwargs):
        kwargs.setdefault('weights_only',False)
        return original(*args,**kwargs)
    torch.load=load
    try:yield
    finally:torch.load=original

class Analyzer:
    def __init__(self,task,device='cuda',tracking_checkpoint=None):
        self.task=task;self.device=torch.device(device);self.tracking_checkpoint=tracking_checkpoint
        from torchvision import models
        if task=='ic':
            self.model=models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            self.description='Torchvision ResNet50 ImageNet1K V1; representative paper backbone family'
        elif task=='od':
            from torchvision.models.detection import fasterrcnn_resnet50_fpn,FasterRCNN_ResNet50_FPN_Weights
            self.model=fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
            self.description='Torchvision FasterRCNN ResNet50 FPN COCO V1; differs from original MMDetection implementation'
        elif task=='ar':
            from torchvision.models.video import swin3d_t,Swin3D_T_Weights
            self.model=swin3d_t(weights=Swin3D_T_Weights.KINETICS400_V1)
            self.description='Torchvision Swin3D-T Kinetics400 V1; representative paper Swin family'
        elif task=='ot':
            if not tracking_checkpoint or not Path(tracking_checkpoint).is_file():raise ValueError('official DiMP50 checkpoint required')
            import ltr.admin.loading as loading
            with trusted_legacy_checkpoint_load():
                self.model,_=loading.load_network(str(tracking_checkpoint),backbone_pretrained=False)
            self.description='Official pytracking DiMP50; pretrained weights frozen; declared classification+IoU NPP task loss'
        else:raise ValueError('unknown original task')
        self.model=freeze(self.model).to(self.device)
        self.mean=torch.tensor([.485,.456,.406],device=self.device).view(1,3,1,1)
        self.std=torch.tensor([.229,.224,.225],device=self.device).view(1,3,1,1)
        self.weight_digest=fingerprint(self.model)

    def normalize(self,x):return (x-self.mean)/self.std

    def logits(self,x):
        if self.task=='ic':return self.model(self.normalize(x[:,0]))
        if self.task=='ar':
            b,t,c,h,w=x.shape
            return self.model(self.normalize(x.reshape(b*t,c,h,w)).reshape(b,t,c,h,w).transpose(1,2))
        raise ValueError('logits only for classification tasks')

    def loss(self,x,target):
        if self.task in ('ic','ar'):return F.cross_entropy(self.logits(x),target.to(self.device))
        if self.task=='od':
            # Torchvision exposes supervised detection losses only in training mode.
            # FrozenBatchNorm and all ordinary BN layers stay in eval mode.
            self.model.train()
            for module in self.model.modules():
                if isinstance(module,nn.modules.batchnorm._BatchNorm):module.eval()
            try:
                target={k:v.to(self.device) for k,v in target.items()}
                losses=self.model([x[0,0]],[target])
                return sum(losses.values())
            finally:self.model.eval()
        boxes=target['boxes_xywh'].to(self.device)
        frames=x[0]
        if len(frames)<2:raise ValueError('tracking needs a reference and test frames')
        cover=target.get('cover',torch.ones(len(frames))).to(self.device)
        visible=cover[1:]>0
        if cover[0]<=0 or not visible.any():raise ValueError('tracking task loss needs visible reference/test frames')
        reference=self.normalize(frames[:1])[:,None]
        tests=self.normalize(frames[1:][visible])[:,None]
        train_bb=boxes[:1,None]
        test_boxes=boxes[1:][visible];proposals=test_boxes[:,None,None]
        # The paper leaves its analyzer task loss unspecified; report this reconstruction choice.
        scores,iou=self.model(reference,tests,train_bb,proposals)
        cls=tracking_score_loss(scores[-1],test_boxes,frames.shape[-2:])
        return cls+F.mse_loss(iou,torch.ones_like(iou))

    def predict(self,x):
        if self.task in ('ic','ar'):return self.logits(x)
        if self.task=='od':return self.model([x[0,0]])[0]
        raise ValueError('tracking predictions use sequential track(), not GT-conditioned training forward')

    def track(self,x,initial_box):
        """Actual DiMP inference: only the first GT box is supplied, never later GT."""
        if self.task!='ot':raise ValueError('tracking analyzer required')
        from pytracking.parameter.dimp.dimp50 import parameters
        from pytracking.features.net_wrappers import NetWithBackbone
        from pytracking.tracker.dimp import get_tracker_class
        params=parameters();params.use_gpu=self.device.type=='cuda';params.debug=0;params.visualization=False
        params.net=NetWithBackbone(net_path=str(self.tracking_checkpoint),use_gpu=params.use_gpu)
        tracker=get_tracker_class()(params)
        arrays=(x.detach().cpu()[0].permute(0,2,3,1).clamp(0,1).numpy()*255).round().astype(np.uint8)
        results=[list(map(float,initial_box))]
        with trusted_legacy_checkpoint_load():
            tracker.initialize(arrays[0],{'init_bbox':results[0]})
        for frame in arrays[1:]:results.append(tracker.track(frame)['target_bbox'])
        return np.asarray(results,dtype=np.float64)

    def receipt(self):
        return dict(task=self.task,analyzer=self.description,weight_sha256=self.weight_digest,
                    parameters=sum(p.numel() for p in self.model.parameters()),
                    trainable_parameters=sum(p.numel() for p in self.model.parameters() if p.requires_grad))
