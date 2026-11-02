"""Real-codec task curves, whole-cohort AP and conservative four-point BD-rate."""
import contextlib
import io
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from scipy.interpolate import PchipInterpolator
from .data import load_sample
from .training import log_event,config_digest,records_digest
from .tasks import fingerprint
from .assets import digest

def bd_rate(anchor_rate,anchor_quality,rate,quality):
    try:
        ar,aq,r,q=[np.asarray(v,dtype=float) for v in (anchor_rate,anchor_quality,rate,quality)]
        if any(v.shape!=(4,) for v in (ar,aq,r,q)):raise ValueError('exactly four QP points required')
        if not all(np.isfinite(v).all() for v in (ar,aq,r,q)) or min(ar.min(),r.min())<=0:
            raise ValueError('finite quality and positive paid rates required')
        if not all((np.diff(v)<0).all() for v in (ar,aq,r,q)):
            raise ValueError('nonmonotone or tied four-QP operating curve')
        lo=max(aq.min(),q.min());hi=min(aq.max(),q.max())
        if hi<=lo:raise ValueError('no overlapping quality interval')
        aa=PchipInterpolator(aq[::-1],np.log(ar[::-1]));bb=PchipInterpolator(q[::-1],np.log(r[::-1]))
        delta=(bb.integrate(lo,hi)-aa.integrate(lo,hi))/(hi-lo)
        return dict(valid=True,percent=float(100*np.expm1(delta)),quality_interval=[float(lo),float(hi)],method='PCHIP log-rate integral')
    except ValueError as error:return dict(valid=False,percent=None,reason=str(error))

def coco_ap(annotation_path,image_ids,predictions):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    with contextlib.redirect_stdout(io.StringIO()):
        gt=COCO(str(annotation_path))
        if predictions:dt=gt.loadRes(predictions)
        else:
            dt=COCO();dt.dataset=dict(images=[im for im in gt.dataset['images'] if im['id'] in image_ids],
                                      annotations=[],categories=gt.dataset['categories']);dt.createIndex()
        ev=COCOeval(gt,dt,'bbox');ev.params.imgIds=sorted(image_ids);ev.evaluate();ev.accumulate();ev.summarize()
    return dict(ap=float(ev.stats[0]),ap50=float(ev.stats[1]),ap75=float(ev.stats[2]))

def tracking_metrics(prediction,ground_truth,cover=None,bound=None):
    a=np.asarray(prediction,dtype=float)[1:].copy();b=np.asarray(ground_truth,dtype=float)[1:].copy()
    if a.shape!=b.shape or a.ndim!=2 or a.shape[1]!=4:raise ValueError('tracking predictions/GT shape mismatch')
    if bound is not None:
        for boxes in (a,b):
            boxes[:,0]=np.clip(boxes[:,0],0,bound[0]);boxes[:,1]=np.clip(boxes[:,1],0,bound[1])
            boxes[:,2]=np.clip(boxes[:,2],0,bound[0]-boxes[:,0]);boxes[:,3]=np.clip(boxes[:,3],0,bound[1]-boxes[:,1])
    lo=np.maximum(a[:,:2],b[:,:2]);hi=np.minimum(a[:,:2]+a[:,2:],b[:,:2]+b[:,2:])
    inter=np.maximum(0,hi-lo).prod(axis=1);union=a[:,2:].prod(axis=1)+b[:,2:].prod(axis=1)-inter
    iou=np.divide(inter,union,out=np.zeros_like(inter),where=union>0)
    if cover is not None:
        cover=np.asarray(cover).reshape(-1)[1:]
        if len(cover)!=len(iou):raise ValueError('GOT visibility/prediction count mismatch')
        iou=iou[cover>0]
    if not len(iou):return dict(ao=None,sr50=None,sr75=None,frames=0)
    return dict(ao=float(iou.mean()),sr50=float((iou>.5).mean()),sr75=float((iou>.75).mean()),frames=len(iou))

def aggregate_tracking(rows):
    total=sum(r['frames'] for r in rows)
    if total<=0:raise ValueError('no visible evaluated tracking frames')
    result={k:sum(r[k]*r['frames'] for r in rows if r['frames'])/total for k in ('ao','sr50','sr75')}
    result['visible_frames']=total;return result

def processed_video(npp,x,qp,chunk,output_device=None):
    # Fixed-size temporal windows bound GPU memory on full GOT10k sequences.
    device=next(npp.parameters()).device
    return torch.cat([npp(x[:,i:i+chunk].to(device),qp).to(output_device or x.device)
                      for i in range(0,x.shape[1],chunk)],dim=1)

def evaluate(cfg,rows,npp,analyzer,codec,out,annotation_path=None,max_seconds=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);npp.eval()
    device=next(npp.parameters()).device;analyzer.model.to(device);analyzer.model.eval()
    curves=[];task=cfg['task'];started=time.monotonic()
    identity=dict(config_sha256=config_digest(cfg),sources=[r['id'] for r in rows],
                  records_sha256=records_digest(rows),annotation_sha256=digest(annotation_path) if annotation_path else None,
                  npp_sha256=fingerprint(npp),analyzer_sha256=analyzer.weight_digest)
    identity_path=out/'evaluation_identity.json'
    if identity_path.is_file() and json.loads(identity_path.read_text())!=identity:
        raise ValueError('validation cache belongs to a different checkpoint, configuration, analyzer or cohort')
    identity_path.write_text(json.dumps(identity,indent=2))
    scope='all_available_validation' if cfg['validation_sources'] is None else 'declared_validation_subset'
    def paused():
        result=dict(complete=False,task=task,curves=curves,scope=scope,stop_reason='evaluation_session_budget')
        (out/'metrics_partial.json').write_text(json.dumps(result,indent=2));return result
    for arm in ('codec_only','npp'):
        for cod in cfg['codecs']:
            for qp in cfg['qps']:
                cache=out/'cache'/arm/cod/str(qp);cache.mkdir(parents=True,exist_ok=True)
                done=cache/'curve.json'
                if done.is_file():
                    curves.append(json.loads(done.read_text()));continue
                points=[];predictions=[];image_ids=[];correct=0;correct5=0;perclass={};track=[]
                with torch.no_grad():
                    for index,record in enumerate(rows):
                        saved=cache/f'{index:08}.json'
                        if saved.is_file():
                            row=json.loads(saved.read_text())
                            if row['source']!=record['id']:raise ValueError('cached validation source/order differs')
                        else:
                            if max_seconds is not None and time.monotonic()-started>=max_seconds:return paused()
                            sample_cfg=dict(cfg)
                            if task=='ot' and cfg['mode']=='full':sample_cfg['frames']=0
                            x,target,meta=load_sample(record,sample_cfg,False)
                            # Keep complete GOT sequences and decoded frames on CPU. Only
                            # NPP windows and the tracker's own search crop occupy the GPU.
                            if task!='ot' or cfg['mode']!='full':x=x.to(device)
                            y=x if arm=='codec_only' else processed_video(npp,x,qp,cfg['frames'],output_device=x.device)
                            directory=out/'streams'/arm/cod/str(qp)/str(index)
                            actual=codec(y,cod,qp,directory);decoded=actual.reconstruction
                            row=dict(source=record['id'],codec=cod,qp=qp,arm=arm,bytes=actual.bytes,bpp=actual.bpp,
                                     pixels=int(x.shape[1]*x.shape[-2]*x.shape[-1]),stream_sha256=actual.sha256,
                                     source_mse=float((decoded-x).square().mean()))
                            if task=='ar':
                                row['sampling']={k:meta[k] for k in ('start_frame','frame_stride','frame_indices',
                                    'source_frames','short_clip_policy','short_clip_looped','provider_cadence') if k in meta}
                            if task in ('ar','ic'):
                                logits=analyzer.predict(decoded);label=int(target.item())
                                hit=int(logits.argmax(1).item()==label);hit5=int(label in logits.topk(5,1).indices[0].tolist())
                                row.update(label=label,predicted=int(logits.argmax(1).item()),correct=hit,correct5=hit5)
                            elif task=='od':
                                prediction=analyzer.predict(decoded);h,w=x.shape[-2:];boxes=[]
                                for box,score,label in zip(prediction['boxes'].cpu(),prediction['scores'].cpu(),prediction['labels'].cpu()):
                                    if float(score)<.001:continue
                                    x1,y1,x2,y2=map(float,box)
                                    boxes.append(dict(image_id=meta['image_id'],category_id=int(label),score=float(score),
                                        bbox=[x1*meta['original_width']/w,y1*meta['original_height']/h,
                                              (x2-x1)*meta['original_width']/w,(y2-y1)*meta['original_height']/h]))
                                row.update(image_id=meta['image_id'],predictions=boxes)
                            else:
                                with torch.enable_grad():prediction=analyzer.track(decoded,target['boxes_xywh'][0].tolist())
                                row['tracking']=tracking_metrics(prediction,target['boxes_xywh'].numpy(),
                                                                 target['cover'].numpy(),(x.shape[-1],x.shape[-2]))
                                (directory/'predicted_boxes.json').write_text(json.dumps(prediction.tolist()))
                            temporary=saved.with_suffix('.writing');temporary.write_text(json.dumps(row));temporary.replace(saved)
                        if task in ('ar','ic'):
                            correct+=row['correct'];correct5+=row['correct5'];perclass.setdefault(row['label'],[]).append(row['correct'])
                        elif task=='od':predictions.extend(row['predictions']);image_ids.append(row['image_id'])
                        else:track.append(row['tracking'])
                        points.append(row)
                        if index%20==0:log_event(out,'validation',task=task,arm=arm,codec=cod,qp=qp,source=index,total=len(rows))
                if task in ('ar','ic'):
                    metrics=dict(top1=correct/len(rows),top5=correct5/len(rows),mean_class_accuracy=float(np.mean([np.mean(v) for v in perclass.values()])))
                    quality=metrics['top1']
                elif task=='od':
                    if annotation_path is None:raise ValueError('official COCO GT required for reported AP')
                    metrics=coco_ap(annotation_path,image_ids,predictions);quality=metrics['ap']
                    (out/f'predictions_{arm}_{cod}_qp{qp}.json').write_text(json.dumps(predictions))
                else:
                    metrics=aggregate_tracking(track)
                    quality=metrics['ao'];metrics['protocol']='official GOT visibility mask, initial frame excluded, visible-frame weighting; '+('full sequences' if cfg['mode']=='full' else 'pilot clips')
                total_bytes=sum(p['bytes'] for p in points);pixels=sum(p['pixels'] for p in points)
                curve=dict(arm=arm,codec=cod,qp=qp,quality=quality,bpp=8*total_bytes/pixels,
                           mean_source_bpp=float(np.mean([p['bpp'] for p in points])),bytes=total_bytes,pixels=pixels,
                           metrics=metrics,sources=len(rows),scope=scope)
                curves.append(curve)
                done.write_text(json.dumps(curve,indent=2))
                (out/f'points_{arm}_{cod}_qp{qp}.json').write_text(json.dumps(points,indent=2))
                log_event(out,'validation_point',task=task,**curve)
    bd={}
    for cod in cfg['codecs']:
        a=sorted([r for r in curves if r['arm']=='codec_only' and r['codec']==cod],key=lambda r:r['qp'])
        b=sorted([r for r in curves if r['arm']=='npp' and r['codec']==cod],key=lambda r:r['qp'])
        bd[cod]=bd_rate([r['bpp'] for r in a],[r['quality'] for r in a],[r['bpp'] for r in b],[r['quality'] for r in b])
    result=dict(complete=True,task=task,curves=curves,bd_rate=bd,scope=scope,hidden_test_evaluated=False)
    (out/'metrics.json').write_text(json.dumps(result,indent=2))
    return result
