"""Explicit COCO/GOT protocols and exact offline mask/clip identities."""
import gc
import hashlib
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F
from PIL import Image

from .assets import digest
from .data import ordered_subset
from .protocol import recipe
from .sam2_masks import sam_config,MaskCache,soft_mask,tensor_digest,build_teacher
from .training import rng_state,restore_rng,config_digest,log_event

PUBLIC_PROXY=dict(kind='dataset',ref='wagur124705/image-paper-proxy-inputs-20261031/1',member='od_proxy.pt',
    sha256='ab9fde3e3418be44b532eb9c4492382a4a8204e0d64ab5d2b15dad1b2f535dbf',visibility='public')

def config(task,arm):
    if task not in ('is','od','ot') or arm not in ('control','foreground'):raise ValueError('three declared tasks and two arms required')
    cfg=recipe('ot' if task=='ot' else 'od')
    cfg.update(task=task,study_profile='tri_task',sam_arm=arm,seed=20261101,
        backbone={'is':'maskrcnn_resnet50_fpn','od':'fcos_resnet50_fpn','ot':'dimp50'}[task],
        codecs=['h264'] if task=='ot' else ['bpg'],npp_size=288 if task=='ot' else 512,max_size=768,
        frames=4 if task=='ot' else 1,evaluation_frames=32,train_sources=64,dev_sources=8,
        validation_sources=8 if task=='ot' else 32,warmup_steps=12000,warmup_min_steps=2000,warmup_probe_every=1000,task_steps=256,
        proxy_steps=256 if task=='ot' else 0,proxy_crop=128,checkpoint_every=100,log_every=50,
        max_session_seconds=7200,evaluation_reserve=2400,gate_psnr=25.,collapse_psnr=20.,
        beta=5.,lambda_scale=.1,scope='three_task_bounded_study_not_full_reproduction')
    cfg['sam2']=sam_config('od',arm)['sam2']
    cfg['sam2'].update(clip_frames=cfg['frames'],clip_seed=cfg['seed'],gamma=.1 if arm=='foreground' else 0)
    cfg['fidelity_notes']+=['Released image NPP for all tasks; frozen offline per-frame SAM2 masks.',
        'MSE warm-up and fidelity gates precede short task training; no identity fallback.',
        'Training-only STE clipping preserves clipped forward values; task coefficient0.1, beta5 are declared trials.',
        'Tracking uses framewise Minnen proxy recalibrated to H264, not an exact motion-aware codec model.']
    return cfg

def image_tensor(image):
    image=image.convert('RGB');w,h=image.size
    return torch.frombuffer(bytearray(image.tobytes()),dtype=torch.uint8).reshape(h,w,3).permute(2,0,1).float()/255

def clip_indices(row,frames,seed):
    directory=Path(row['path']);paths=sorted(directory.glob('*.jpg'))
    gt=_gt(directory/'groundtruth.txt')
    if len(paths)!=len(gt):raise ValueError('GOT frame/GT alignment differs')
    cover=_cover(directory,len(paths),row['split'])
    if len(paths)<frames:raise ValueError('tracking source shorter than declared clip')
    if row['split']!='train':return paths,list(range(frames)),gt,cover
    valid=[i for i in range(len(paths)-frames+1) if cover[i]>0 and any(v>0 for v in cover[i+1:i+frames])]
    if not valid:raise ValueError('no visible initialized tracking clip')
    selected=int(hashlib.sha256((str(seed)+'/'+row['id']).encode()).hexdigest()[:16],16)%len(valid)
    start=valid[selected];return paths,list(range(start,start+frames)),gt,cover

def _gt(path):
    rows=[[float(v) for v in line.replace('\t',',').split(',')] for line in path.read_text().splitlines() if line.strip()]
    if any(len(v)!=4 or not all(math.isfinite(n) for n in v) for v in rows):raise ValueError('finite GOT xywh rows required')
    return rows

def _cover(directory,count,split):
    path=directory/'cover.label'
    if not path.is_file():
        if split=='val':raise ValueError('official GOT validation visibility required')
        return [1]*count
    values=[float(v) for v in path.read_text().split()]
    if len(values)!=count or not all(math.isfinite(v) for v in values):raise ValueError('GOT visibility alignment differs')
    return values

def decode_mask(segmentation,h,w):
    from pycocotools import mask as coco_mask
    if isinstance(segmentation,list):rle=coco_mask.merge(coco_mask.frPyObjects(segmentation,h,w))
    elif isinstance(segmentation,dict):
        rle=dict(segmentation)
        if isinstance(rle['counts'],list):rle=coco_mask.frPyObjects(rle,h,w)
        elif isinstance(rle['counts'],str):rle['counts']=rle['counts'].encode('ascii')
    else:raise ValueError('COCO polygon or RLE segmentation required')
    decoded=coco_mask.decode(rle)
    if decoded.shape!=(h,w):raise ValueError('COCO decoded mask geometry differs')
    return torch.frombuffer(bytearray(decoded.tobytes()),dtype=torch.uint8).reshape(h,w)

def load(row,cfg):
    if row['task']=='ot':
        paths,indices,gt,cover=clip_indices(row,cfg['frames'],cfg['seed']);images=[];boxes=[];size=cfg['npp_size']
        for i in indices:
            with Image.open(paths[i]) as im:
                im=im.convert('RGB');w,h=im.size;images.append(image_tensor(im.resize((size,size),Image.Resampling.BILINEAR)))
            box=[v*s for v,s in zip(gt[i],[size/w,size/h,size/w,size/h])]
            if cover[i]>0 and (box[2]<=0 or box[3]<=0):raise ValueError('invalid visible GOT box')
            boxes.append(box)
        return torch.stack(images)[None],dict(boxes_xywh=torch.tensor(boxes),cover=torch.tensor([cover[i] for i in indices])),dict(
            id=row['id'],frame_indices=indices,original_width=w,original_height=h)
    with Image.open(row['path']) as im:
        image=im.convert('RGB');w,h=image.size
        if (w,h)!=(row['width'],row['height']):raise ValueError('COCO source/annotation geometry differs')
        scale=min(cfg['npp_size']/min(w,h),cfg['max_size']/max(w,h));nw=max(2,2*round(w*scale/2));nh=max(2,2*round(h*scale/2))
        x=image_tensor(image.resize((nw,nh),Image.Resampling.BILINEAR))[None,None]
    instances=row.get('instances')
    if instances is None:
        if cfg['task']=='is':raise ValueError('COCO instance segmentation annotations missing')
        instances=[dict(bbox=[x1,y1,x2-x1,y2-y1],category_id=c,iscrowd=0) for x1,y1,x2,y2,c in row['objects']]
    boxes=[];labels=[];masks=[]
    for obj in instances:
        xx,yy,ww,hh=obj['bbox']
        if obj.get('iscrowd',0) or min(ww,hh)<=0:continue
        x1=max(0,xx)*nw/w;y1=max(0,yy)*nh/h;x2=min(w,xx+ww)*nw/w;y2=min(h,yy+hh)*nh/h
        if x2<=x1 or y2<=y1:continue
        boxes.append([x1,y1,x2,y2]);labels.append(obj['category_id'])
        if cfg['task']=='is':masks.append(F.interpolate(decode_mask(obj['segmentation'],h,w)[None,None].float(),(nh,nw),mode='nearest')[0,0].byte())
    target=dict(boxes=torch.tensor(boxes,dtype=torch.float32).reshape(-1,4),labels=torch.tensor(labels,dtype=torch.long))
    if cfg['task']=='is':target['masks']=torch.stack(masks) if masks else torch.zeros(0,nh,nw,dtype=torch.uint8)
    return x,target,dict(id=row['id'],image_id=row['image_id'],original_width=w,original_height=h)

def enrich_masks(rows,annotation):
    selected={r['image_id']:[] for r in rows};meta=json.loads(Path(annotation).read_text())
    for ann in meta['annotations']:
        if ann['image_id'] in selected:selected[ann['image_id']].append(ann)
    return [dict(r,task='is',instances=selected[r['image_id']]) for r in rows]

def select_cohorts(train,validation,cfg):
    selected=[];skipped=[]
    for row in ordered_subset(train,None,cfg['seed']):
        if len(selected)>=cfg['train_sources']+cfg['dev_sources']:break
        if cfg['task']=='ot':
            try:clip_indices(row,cfg['frames'],cfg['seed'])
            except (ValueError,OSError) as error:skipped.append(dict(source=row['id'],reason=str(error)));continue
        selected.append(row)
    if len(selected)<cfg['train_sources']+cfg['dev_sources']:raise ValueError('not enough independent usable TRAIN/dev sources')
    return selected[:cfg['train_sources']],selected[cfg['train_sources']:],ordered_subset(validation,cfg['validation_sources'],cfg['seed']),skipped

class VideoMaskCache(MaskCache):
    def identity(self,row,x):
        if row['task']!='ot':return super().identity(row,x)
        paths,indices,_,_=clip_indices(row,self.settings['clip_frames'],self.settings['clip_seed']);root=Path(row['path'])
        sources={paths[i].name:digest(paths[i]) for i in indices}
        for name in ('groundtruth.txt','cover.label'):
            if (root/name).is_file():sources[name]=digest(root/name)
        return dict(source=row['id'],source_files=sources,frame_indices=indices,pixels_sha256=tensor_digest(x),shape=list(x.shape),teacher=self.settings)

def prepare_masks(rows,cfg,out,budget):
    cache=VideoMaskCache(out/'sam_masks',cfg['sam2']);state=rng_state();teacher=None;records=[]
    try:
        if cfg['sam_arm']=='foreground':teacher=build_teacher(cfg['sam2'],out)
        for i,row in enumerate(rows):
            budget.require('offline_mask_preparation');x,_,_=load(row,cfg);values=[];parts=[]
            for t in range(x.shape[1]):
                if teacher is None:mask=torch.zeros(1,1,1,*x.shape[-2:]);part=dict(empty=True,coverage=0.,accepted=0)
                else:
                    rgb=(x[0,t].permute(1,2,0)*255).round().byte().numpy()
                    with torch.inference_mode():generated=teacher.generate(rgb)
                    mask,part=soft_mask(generated,x.shape[-2:],cfg['sam2'])
                values.append(mask);parts.append(part)
            cache.store(row,x,torch.cat(values,dim=1));records.append(dict(source=row['id'],pixels_sha256=tensor_digest(x),frames=parts))
            if i==0 or (i+1)%8==0:log_event(out,'tri_mask_cache',sources=i+1,total=len(rows),arm=cfg['sam_arm'])
    finally:
        del teacher;gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
        restore_rng(state)
    info=dict(sources=records,rendered_cohort_sha256=config_digest([(r['source'],r['pixels_sha256']) for r in records]))
    (out/'sam_cache.json').write_text(json.dumps(info,indent=2));return cache,info
