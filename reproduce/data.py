"""Original-task datasets, canonical labels, official splits and aligned GT geometry."""
from pathlib import Path
import csv
import hashlib
import json
import random
import re
import cv2
import numpy as np
from PIL import Image
import torch

def canonical(name):return re.sub(r'[^a-z0-9]+',' ',name.lower()).strip()

def verify_splits(train,validation):
    a={r['id'] for r in train};b={r['id'] for r in validation}
    if len(a)!=len(train) or len(b)!=len(validation) or a&b:
        raise ValueError('duplicate or overlapping source identities across splits')
    if not a or not b:raise ValueError('both original dataset splits must be nonempty')

def imagenet_records(root,split,class_to_idx=None):
    root=Path(root)
    folders=sorted(p for p in root.iterdir() if p.is_dir() and re.fullmatch(r'n\d{8}',p.name))
    if class_to_idx is None:
        if len(folders)!=1000:raise ValueError('ImageNet requires all1000 WNID folders or an explicit canonical map')
        class_to_idx={p.name:i for i,p in enumerate(folders)}
    result=[]
    for folder in folders:
        if folder.name not in class_to_idx:raise ValueError('ImageNet class is not in canonical map')
        for p in sorted(folder.iterdir()):
            if p.suffix.lower() not in ('.jpg','.jpeg','.png'):continue
            result.append(dict(id='imagenet/'+p.stem,task='ic',path=str(p),label=class_to_idx[folder.name],split=split))
    if not result:raise ValueError('no labeled ImageNet images')
    return result

def coco_records(images,annotations,split):
    meta=json.loads(Path(annotations).read_text())
    objects={}
    for a in meta['annotations']:
        if a.get('iscrowd',0):continue
        x,y,w,h=a['bbox']
        if w>0 and h>0:objects.setdefault(a['image_id'],[]).append([x,y,x+w,y+h,a['category_id']])
    rows=[dict(id='coco/'+str(im['id']),task='od',path=str(Path(images)/im['file_name']),
               image_id=im['id'],width=im['width'],height=im['height'],objects=objects.get(im['id'],[]),split=split)
          for im in meta['images']]
    if not rows or not Path(rows[0]['path']).is_file():raise ValueError('COCO annotation/image path mismatch')
    return rows

def kinetics_records(root,split):
    from torchvision.models.video import Swin3D_T_Weights
    mapping={canonical(n):i for i,n in enumerate(Swin3D_T_Weights.KINETICS400_V1.meta['categories'])}
    root=Path(root);candidates=[]
    if root.name==split:candidates=[root]
    else:candidates=[p for p in root.rglob(split) if p.is_dir()]
    if len(candidates)!=1:raise ValueError(f'need one declared K400 {split} folder; found{len(candidates)}')
    result=[];seen=set()
    for folder in sorted(candidates[0].iterdir()):
        name=canonical(folder.name)
        if not folder.is_dir() or name not in mapping:continue
        for p in sorted(folder.iterdir()):
            if p.suffix.lower() not in ('.mp4','.avi','.webm','.mkv'):continue
            match=re.match(r'^([A-Za-z0-9_-]{11})(?:$|_\d+)',p.stem)
            ident='kinetics/'+(match.group(1) if match else p.stem)
            if ident in seen:continue
            seen.add(ident)
            result.append(dict(id=ident,task='ar',path=str(p),label=mapping[name],split=split))
    if not result:raise ValueError('no canonically labeled K400 videos')
    return result

def kinetics_frame_records(root,annotation):
    """Join provider JPEG folders to the official K400 validation label table."""
    from torchvision.models.video import Swin3D_T_Weights
    mapping={canonical(n):i for i,n in enumerate(Swin3D_T_Weights.KINETICS400_V1.meta['categories'])}
    root=Path(root);available={p.name for p in root.iterdir()};result=[];seen=set()
    with Path(annotation).open(newline='',encoding='utf-8') as f:
        for row in csv.DictReader(f):
            ident=row['youtube_id'];label=canonical(row['label'])
            if row['split']!='val' or ident not in available:continue
            if not re.fullmatch(r'[A-Za-z0-9_-]{11}',ident) or label not in mapping:
                raise ValueError('invalid official K400 identity/class')
            if ident in seen:raise ValueError('duplicate K400 validation identity')
            seen.add(ident)
            result.append(dict(id='kinetics/'+ident,task='ar',path=str(root/ident),label=mapping[label],
                               split='val',source_format='provider_jpeg_frames',provider_cadence='unverified'))
    if not result:raise ValueError('no validation frame folders match official K400 annotations')
    return result

def got_records(root,split):
    if split=='test':raise ValueError('GOT10k hidden TEST requires server evaluation; local full-GT metrics are unavailable')
    root=Path(root)
    if (root/split).is_dir():root=root/split
    result=[]
    listing=root/'list.txt'
    if listing.is_file():
        names=[line.strip() for line in listing.read_text().splitlines() if line.strip()]
        if any(Path(name).name!=name or not name.lower().startswith('got-10k') for name in names):
            raise ValueError('invalid GOT sequence list')
        # Validate files only when selected: statting 9k cold-mounted directories can
        # take longer than a pilot. The official manifest establishes source names.
        return [dict(id='got10k/'+name,task='ot',path=str(root/name),split=split) for name in names]
    else:sequences=sorted(root.iterdir())
    for seq in sequences:
        if not seq.is_dir() or not seq.name.lower().startswith('got-10k'):continue
        gt=seq/'groundtruth.txt'
        if not gt.is_file():continue
        # Do not scan millions of images before training can start. load_sample validates
        # the requested source's exact frame/GT alignment before it is used.
        result.append(dict(id='got10k/'+seq.name,task='ot',path=str(seq),split=split))
    if not result:raise ValueError('no fully annotated GOT10k sequences')
    return result

def ordered_subset(rows,count,seed=20261023):
    rows=sorted(rows,key=lambda r:hashlib.sha256((str(seed)+'/'+r['id']).encode()).hexdigest())
    if count is None:return rows
    if count<1 or len(rows)<count:raise ValueError('not enough independent validation sources')
    return rows[:count]

def _rgb(path):
    with Image.open(path) as im:return im.convert('RGB').copy()

def _tensor(images):
    arr=np.stack([np.asarray(im) for im in images])
    return torch.from_numpy(np.ascontiguousarray(arr)).permute(0,3,1,2).float()[None]/255

def load_sample(record,cfg,training=False):
    task=record['task'];size=cfg['npp_size'];t=cfg['frames']
    meta=dict(id=record['id'],task=task,split=record['split'])
    if task=='od':
        im=_rgb(record['path']);w,h=im.size
        if (w,h)!=(record['width'],record['height']):raise ValueError('COCO dimensions differ from GT')
        scale=min(size/min(w,h),2*size/max(w,h));nw=max(2,round(w*scale));nh=max(2,round(h*scale))
        # Even dimensions for real YUV420 standard codecs; aspect ratio is preserved up to rounding.
        nw+=nw%2;nh+=nh%2
        image=im.resize((nw,nh),Image.Resampling.BILINEAR)
        boxes=[];labels=[]
        for x1,y1,x2,y2,label in record['objects']:
            boxes.append([x1*nw/w,y1*nh/h,x2*nw/w,y2*nh/h]);labels.append(label)
        target={'boxes':torch.tensor(boxes,dtype=torch.float32).reshape(-1,4),
                'labels':torch.tensor(labels,dtype=torch.long)}
        meta.update(image_id=record['image_id'],original_width=w,original_height=h)
        return _tensor([image]),target,meta
    if task=='ic':
        im=_rgb(record['path']);w,h=im.size;short=round(size*256/224)
        scale=short/min(w,h);nw=round(w*scale);nh=round(h*scale)
        im=im.resize((nw,nh),Image.Resampling.BILINEAR)
        xx=random.randrange(nw-size+1) if training else (nw-size)//2
        yy=random.randrange(nh-size+1) if training else (nh-size)//2
        im=im.crop((xx,yy,xx+size,yy+size))
        return _tensor([im]),torch.tensor([record['label']],dtype=torch.long),meta
    if task=='ar':
        if record.get('source_format')=='provider_jpeg_frames':
            paths=sorted(Path(record['path']).glob('img_*.jpg'));stride=2;needed=(t-1)*stride+1
            policy=cfg.get('ar_validation_short_clip','error')
            if policy not in ('error','loop'):raise ValueError('unknown AR validation short-clip policy')
            if not paths:raise ValueError('empty AR validation frame sequence')
            short=len(paths)<needed
            if short and (training or policy!='loop'):
                raise ValueError('validation frame sequence too short for declared AR sampling')
            start=0 if short else (random.randrange(len(paths)-needed+1) if training else (len(paths)-needed)//2)
            indices=[(start+i*stride)%len(paths) for i in range(t)]
            imgs=[]
            for index in indices:
                im=_rgb(paths[index]);w,h=im.size;scale=round(size*256/224)/min(w,h)
                nw=round(w*scale);nh=round(h*scale);im=im.resize((nw,nh),Image.Resampling.BILINEAR)
                xx=(nw-size)//2;yy=(nh-size)//2;imgs.append(im.crop((xx,yy,xx+size,yy+size)))
            meta.update(start_frame=start,frame_stride=stride,source_format='provider_jpeg_frames',provider_cadence='unverified',
                        frame_indices=indices,source_frames=len(paths),short_clip_policy=policy,short_clip_looped=short)
            return _tensor(imgs),torch.tensor([record['label']],dtype=torch.long),meta
        cap=cv2.VideoCapture(record['path']);imgs=[];stride=2
        try:
            total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));needed=(t-1)*stride+1
            if not cap.isOpened() or total<needed:raise ValueError('video too short for declared AR sampling')
            start=random.randrange(total-needed+1) if training else (total-needed)//2
            cap.set(cv2.CAP_PROP_POS_FRAMES,start)
            for index in range(needed):
                ok,frame=cap.read()
                if not ok:raise ValueError('undecodable AR frame')
                if index%stride:continue
                im=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB));w,h=im.size
                scale=round(size*256/224)/min(w,h);nw=round(w*scale);nh=round(h*scale)
                im=im.resize((nw,nh),Image.Resampling.BILINEAR)
                xx=(nw-size)//2;yy=(nh-size)//2;imgs.append(im.crop((xx,yy,xx+size,yy+size)))
        finally:cap.release()
        meta.update(start_frame=start,frame_stride=stride)
        return _tensor(imgs),torch.tensor([record['label']],dtype=torch.long),meta
    if task=='ot':
        seq=Path(record['path']);paths=sorted(seq.glob('*.jpg'))
        gt=np.loadtxt(seq/'groundtruth.txt',delimiter=',',dtype=np.float32).reshape(-1,4)
        if len(paths)!=len(gt):raise ValueError('tracking frame/annotation alignment failed')
        cover_path=seq/'cover.label'
        if cover_path.is_file():
            cover=np.loadtxt(cover_path,dtype=np.float32).reshape(-1)
            if len(cover)!=len(paths):raise ValueError('GOT visibility/frame count mismatch')
        else:
            if record['split']=='val':raise ValueError('GOT validation requires original cover.label visibility metadata')
            cover=np.ones(len(paths),dtype=np.float32)
        if t==0:
            if training:raise ValueError('training requires an explicit finite temporal crop')
            t=len(paths)
        if len(paths)<t:raise ValueError('tracking sequence shorter than requested clip')
        start=random.randrange(len(paths)-t+1) if training else 0
        if training:
            starts=np.arange(len(paths)-t+1);visible=cover>0
            counts=np.r_[0,np.cumsum(visible)]
            starts=starts[visible[starts] & ((counts[starts+t]-counts[starts+1])>0)]
            if not len(starts):raise ValueError('tracking crop needs visible reference and test frames')
            start=int(random.choice(starts.tolist()))
        ims=[];boxes=[]
        for i in range(start,start+t):
            im=_rgb(paths[i]);w,h=im.size;ims.append(im.resize((size,size),Image.Resampling.BILINEAR))
            boxes.append(gt[i]*[size/w,size/h,size/w,size/h])
        boxes=torch.tensor(np.asarray(boxes),dtype=torch.float32)
        visible=torch.from_numpy(cover[start:start+t]>0)
        if not torch.isfinite(boxes).all() or (boxes[visible,2:]<=0).any():raise ValueError('invalid visible tracking target')
        meta.update(start_frame=start,frame_stride=1)
        return _tensor(ims),{'boxes_xywh':boxes,'cover':torch.from_numpy(cover[start:start+t].copy())},meta
    raise ValueError('unknown original task')
