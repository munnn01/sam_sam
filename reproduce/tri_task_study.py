"""Three separate tasks using the unchanged NPP/real-codec/proxy/offline-mask topology."""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import random
import time

import torch
from torch.nn import functional as F

from .assets import ensure_bpg,ensure_tracking,records_for_task,dataset_mount,digest
from .codecs import RealCodec
from .data import ordered_subset
from .evaluation import bd_rate,tracking_metrics,aggregate_tracking,processed_video
from .image_study import load_proxy_input,canonicalize_proxy_masks,GaussianNPP
from .image_proxy_audit import restore_proxy_state
from .protocol import QPS,real_forward,task_lambda,proxy_lambda
from .proxy_recovery import Budget,BudgetExpired,model_digest
from .sam2_masks import mask_objective
from .tasks import freeze
from .training import log_event,rng_state,restore_rng,config_digest,records_digest
from .tri_task_data import config,load,enrich_masks,select_cohorts,prepare_masks,PUBLIC_PROXY
from .tri_task_models import PaperNPP,FrameProxy,CocoTask,TrackingTask

def fidelity_result(mses,threshold):
    if not mses or any(not math.isfinite(v) or v<0 for v in mses):raise ValueError('finite nonnegative reconstruction MSE required')
    mse=sum(mses)/len(mses);psnr=-10*math.log10(max(mse,1e-12))
    return dict(mse=mse,psnr_db=psnr,passed=psnr>=threshold,threshold_db=threshold,sources=len(mses))

def task_trained(state):return state.get('task_updates',0)>0

def task_gradient_norm(loss,npp):
    if not loss.requires_grad:raise RuntimeError('task gradient is detached')
    values=torch.autograd.grad(loss,list(npp.parameters()),allow_unused=True,retain_graph=True)
    values=[g for g in values if g is not None]
    if not values or not all(torch.isfinite(v).all() for v in values):raise RuntimeError('task gradient is missing/nonfinite')
    norm=float(torch.sqrt(sum(v.detach().square().sum() for v in values)))
    if norm<=0:raise RuntimeError('task gradient is zero')
    return norm

def validate_resume(saved,cfg,source):
    if saved.get('config_sha256')!=config_digest(cfg) or saved.get('config')!=cfg or saved.get('source_sha256')!=source:
        raise ValueError('resume source/config differs')
    if not saved.get('optimizer') or not saved.get('rng') or not saved.get('proxy'):raise ValueError('resume optimizer/RNG/proxy missing')

def continuation_state(old):
    state=dict(old);state['previous_session_stop_reason']=state.get('stop_reason');state['stop_reason']=None
    return state

def stage_fidelity_threshold(cfg,phase,state):
    return cfg['collapse_psnr'] if phase=='task' or task_trained(state) else cfg['gate_psnr']

def initialize_task(factory):
    state=rng_state()
    try:return factory()
    finally:restore_rng(state)

def continue_warmup(cfg,state,phase):
    return phase=='warmup' and state['warmup_updates']<cfg['warmup_steps'] and not state.get('warmup_converged',False)

def warmup_ready(cfg,state,gate):
    return state['warmup_updates']>=cfg['warmup_min_steps'] and gate['passed']

def finite_step(loss,model,optimizer,max_norm=10.):
    if not torch.isfinite(loss):raise RuntimeError('nonfinite objective')
    loss.backward();params=[p for p in model.parameters() if p.requires_grad]
    grads=[p.grad for p in params if p.grad is not None]
    if not grads or not all(torch.isfinite(v).all() for v in grads):raise RuntimeError('missing/nonfinite NPP gradient')
    norm=float(torch.nn.utils.clip_grad_norm_(params,max_norm,error_if_nonfinite=True));optimizer.step();return norm

def save(out,cfg,state,npp,proxy,optimizer,phase):
    data=dict(schema=2,config=cfg,config_sha256=config_digest(cfg),state=state,npp=npp.state_dict(),npp_adaptive=npp.adaptive,
        proxy=proxy.state_dict(),optimizer=optimizer.state_dict() if optimizer else None,optimizer_phase=phase,rng=rng_state(),
        source_sha256=os.environ.get('PAPER_SOURCE_SHA256'),npp_sha256=model_digest(npp))
    temp=out/'last.writing';torch.save(data,temp);os.replace(temp,out/'last.pt')

def check_fidelity(npp,samples,cfg,threshold,budget):
    training=npp.training;npp.eval();mses=[];saturated=[]
    try:
        with torch.no_grad():
            for x,_,_ in samples:
                budget.require('fidelity_probe');x=x.to('cuda');y=npp(x,35)
                mses.append(float((y-x).square().mean()));saturated.append(float(((y<=0)|(y>=1)).float().mean()))
    finally:npp.train(training)
    return dict(fidelity_result(mses,threshold),mean_clipped_pixel_fraction=sum(saturated)/len(saturated))

def freeze_npp_bn(npp):
    for layer in npp.modules():
        if isinstance(layer,torch.nn.modules.batchnorm._BatchNorm):layer.eval()

def calibrate_bn(npp,samples):
    npp.train()
    with torch.no_grad():
        for x,_,_ in samples:npp(x.to('cuda'),35)
    npp.eval()

def fit_video_proxy(proxy,samples,cfg,codec,out,budget):
    proxy.train().requires_grad_(True)
    main={q:torch.optim.Adam(proxy.bank.main_parameters(q),lr=1e-4) for q in QPS}
    aux={q:torch.optim.Adam(proxy.bank.aux_parameters(q),lr=1e-3) for q in QPS}
    head=torch.optim.Adam([proxy.log_scale,proxy.log_floor],lr=3e-3);updates=0
    for step in range(cfg['proxy_steps']):
        if budget.remaining()<cfg['evaluation_reserve']+900:break
        q=QPS[step%4];x=samples[(step//4)%len(samples)][0].to('cuda');h,w=x.shape[-2:];size=cfg['proxy_crop']
        yy=(h-size)//2;xx=(w-size)//2;x=x[...,yy:yy+size,xx:xx+size]
        actual=codec(x,cfg['codecs'][0],q);value=proxy(x,q)
        main[q].zero_grad(set_to_none=True);aux[q].zero_grad(set_to_none=True);head.zero_grad(set_to_none=True)
        loss=proxy_lambda(q)*(value.reconstruction-actual.reconstruction).square().mean()+(
            value.bpp.clamp_min(1e-8).log()-math.log(max(actual.bpp,1e-8))).square()+.001*value.raw_bpp
        if not torch.isfinite(loss):raise RuntimeError('nonfinite H264 proxy calibration')
        loss.backward();torch.nn.utils.clip_grad_norm_(proxy.parameters(),10.,error_if_nonfinite=True)
        main[q].step();head.step();auxloss=proxy.bank.aux_loss(q)
        if not torch.isfinite(auxloss):raise RuntimeError('nonfinite proxy auxiliary loss')
        auxloss.backward();aux[q].step();updates+=1
        if step%32==0:log_event(out,'tri_proxy_fit',step=updates,qp=q,real_bpp=actual.bpp,proxy_bpp=float(value.bpp.detach()),loss=float(loss.detach()))
    freeze(proxy);return dict(updates=updates,target=cfg['proxy_steps'],codec=cfg['codecs'][0],scope='short codec-specific reconstruction and rate refit; no convergence claim')

def proxy_probe(proxy,samples,cfg,codec,out,budget):
    proxy.eval();rows=[]
    with torch.no_grad():
        for x,_,meta in samples[:2]:
            x=x.to('cuda')
            for q in QPS:
                budget.require('proxy_pair_probe');p=proxy(x,q);a=codec(x,cfg['codecs'][0],q)
                rows.append(dict(source=meta['id'],qp=q,real_bpp=a.bpp,proxy_bpp=float(p.bpp),
                    reconstruction_mse=float((p.reconstruction-a.reconstruction).square().mean())))
    (out/'proxy_probe.json').write_text(json.dumps(rows,indent=2));return rows

def coco_metrics(annotation,ids,predictions,kind):
    import contextlib,io
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    with contextlib.redirect_stdout(io.StringIO()):
        gt=COCO(str(annotation))
        if predictions:dt=gt.loadRes(predictions)
        else:
            dt=COCO();dt.dataset=dict(images=[v for v in gt.dataset['images'] if v['id'] in ids],annotations=[],categories=gt.dataset['categories']);dt.createIndex()
        ev=COCOeval(gt,dt,kind);ev.params.imgIds=sorted(ids);ev.evaluate();ev.accumulate();ev.summarize()
    return dict(ap=float(ev.stats[0]),ap50=float(ev.stats[1]),ap75=float(ev.stats[2]),iou_type=kind)

def coco_predictions(prediction,meta,shape,task):
    rows=[];h,w=shape;ow=meta['original_width'];oh=meta['original_height']
    for index,(box,score,label) in enumerate(zip(prediction['boxes'].detach().cpu(),prediction['scores'].detach().cpu(),prediction['labels'].detach().cpu())):
        if float(score)<.001:continue
        x1,y1,x2,y2=map(float,box);row=dict(image_id=meta['image_id'],category_id=int(label),score=float(score))
        if task=='is':
            import numpy as np
            from pycocotools import mask as coco_mask
            binary=F.interpolate(prediction['masks'][index:index+1].detach().cpu(),(oh,ow),mode='bilinear',align_corners=False)[0,0]>.5
            rle=coco_mask.encode(np.asfortranarray(binary.numpy().astype('uint8')));rle['counts']=rle['counts'].decode('ascii');row['segmentation']=rle
        else:row['bbox']=[x1*ow/w,y1*oh/h,(x2-x1)*ow/w,(y2-y1)*oh/h]
        rows.append(row)
    return rows

def evaluate(cfg,rows,npp,task,codec,out,annotation,budget):
    out.mkdir(parents=True,exist_ok=True);npp.eval();identity=dict(config_sha256=config_digest(cfg),sources=[r['id'] for r in rows],
        records_sha256=records_digest(rows),npp_sha256=model_digest(npp),task_sha256=task.weight_digest,
        annotation_sha256=digest(annotation) if annotation else None)
    path=out/'evaluation_identity.json'
    if path.is_file() and json.loads(path.read_text())!=identity:raise ValueError('evaluation cache identity differs')
    path.write_text(json.dumps(identity,indent=2));curves=[];cod=cfg['codecs'][0]
    for q in QPS:
        # Pair each trained operating point with its codec baseline before moving to the next QP.
        for arm in ('npp','codec_only'):
            directory=out/'cache'/arm/str(q);directory.mkdir(parents=True,exist_ok=True);points=[]
            for index,row in enumerate(rows):
                target_file=directory/f'{index:05}.json'
                if target_file.is_file():point=json.loads(target_file.read_text())
                else:
                    if budget.remaining()<2:
                        result=dict(complete=False,curves=curves,stop_reason='evaluation_budget',scope='declared_validation_subset')
                        (out/'metrics_partial.json').write_text(json.dumps(result,indent=2));return result
                    settings=dict(cfg,frames=cfg['evaluation_frames']) if cfg['task']=='ot' else cfg
                    x,target,meta=load(row,settings);x=x.to('cuda')
                    with torch.no_grad():
                        y=processed_video(npp,x,q,cfg['frames']) if arm=='npp' else x;actual=codec(y,cod,q,out/'streams'/arm/str(q)/str(index))
                        point=dict(source=row['id'],bytes=actual.bytes,pixels=x.shape[1]*x.shape[-2]*x.shape[-1],
                            source_mse=float((actual.reconstruction-x).square().mean()),stream_sha256=actual.sha256)
                        if cfg['task']!='ot':point.update(image_id=meta['image_id'],predictions=coco_predictions(task.predict(actual.reconstruction),meta,x.shape[-2:],cfg['task']))
                    if cfg['task']=='ot':
                        with torch.enable_grad():pred=task.predict(actual.reconstruction,target['boxes_xywh'][0].tolist())
                        point['tracking']=tracking_metrics(pred,target['boxes_xywh'].numpy(),target['cover'].numpy(),(x.shape[-1],x.shape[-2]))
                    target_file.write_text(json.dumps(point))
                if point['source']!=row['id']:raise ValueError('cached source differs')
                points.append(point)
            if cfg['task']=='ot':metrics=aggregate_tracking([p['tracking'] for p in points]);quality=metrics['ao']
            else:
                metrics=coco_metrics(annotation,[p['image_id'] for p in points],[v for p in points for v in p['predictions']],
                    'segm' if cfg['task']=='is' else 'bbox');quality=metrics['ap']
            curve=dict(arm=arm,qp=q,codec=cod,quality=quality,metrics=metrics,sources=len(points),
                bytes=sum(p['bytes'] for p in points),pixels=sum(p['pixels'] for p in points))
            curve['bpp']=8*curve['bytes']/curve['pixels'];curves.append(curve)
            (directory/'curve.json').write_text(json.dumps(curve,indent=2));log_event(out,'tri_validation_point',**curve)
    a=sorted((r for r in curves if r['arm']=='codec_only'),key=lambda r:r['qp']);b=sorted((r for r in curves if r['arm']=='npp'),key=lambda r:r['qp'])
    result=dict(complete=True,curves=curves,bd_rate=bd_rate([r['bpp'] for r in a],[r['quality'] for r in a],
        [r['bpp'] for r in b],[r['quality'] for r in b]),scope='declared_validation_subset',hidden_test_evaluated=False)
    (out/'metrics.json').write_text(json.dumps(result,indent=2));return result

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--proxy',type=Path,required=True);parser.add_argument('--proxy-sha256',required=True)
    parser.add_argument('--inputs',type=Path,default=Path('/kaggle/input'));parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--resume',type=Path);parser.add_argument('--warm-start',type=Path);parser.add_argument('--export-warmup',type=Path)
    args=parser.parse_args();cfg=json.loads(args.config.read_text())
    if (args.warm_start or args.export_warmup) and cfg.get('study_profile')!='coco_codecs':raise ValueError('warm-start reserved for paired codec study')
    if args.resume and args.warm_start:raise ValueError('resume and warm-start are exclusive')
    if cfg!=config(cfg['task'],cfg['sam_arm']):raise ValueError('declared configuration differs')
    if args.proxy_sha256!=PUBLIC_PROXY['sha256']:raise ValueError('declared public proxy SHA differs')
    out=args.out;out.mkdir(parents=True,exist_ok=True);budget=Budget(cfg['max_session_seconds']);started=time.time()
    state=dict(warmup_updates=0,task_updates=0,proxy_updates=0,stop_reason=None);report=dict(task=cfg['task'],arm=cfg['sam_arm'],
        scientific_complete=False,execution_complete=False,state=state)
    npp=proxy=optimizer=None;phase='setup'
    try:
        if not torch.cuda.is_available():raise RuntimeError('CUDA required')
        import numpy as np
        random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed']);torch.set_num_threads(2)
        torch.backends.cudnn.benchmark=False
        codec_args=dict(preset=cfg['preset'],gop=cfg['gop'],fps=cfg['fps'])
        if cfg['codecs'][0]=='bpg':
            assets=ensure_bpg(Path('/kaggle/working/paper_assets/bpg'));codec_args.update(bpgenc=assets['bpgenc'],bpgdec=assets['bpgdec'])
        codec=RealCodec(**codec_args);original_train,original_val,inventory=records_for_task('ot' if cfg['task']=='ot' else 'od',args.inputs)
        train,dev,validation,skipped=select_cohorts(original_train,original_val,cfg);annotation=None
        if cfg['task']!='ot':
            root=dataset_mount(args.inputs,'awsaf49/coco-2017-dataset');paths=list(root.rglob('instances_val2017.json'))
            if len(paths)!=1:raise ValueError('one official COCO annotation required')
            annotation=paths[0]
            if cfg['task']=='is':
                train=enrich_masks(train,annotation.with_name('instances_train2017.json'));dev=enrich_masks(dev,annotation.with_name('instances_train2017.json'));validation=enrich_masks(validation,annotation)
        cfg['runtime_identity']=dict(train=records_digest(train),dev=records_digest(dev),validation=records_digest(validation),datasets=inventory['dataset_versions'])
        (out/'cohort.json').write_text(json.dumps(dict(train=[r['id'] for r in train],dev=[r['id'] for r in dev],validation=[r['id'] for r in validation],skipped=skipped,identity=cfg['runtime_identity']),indent=2))
        cache,mask_info=prepare_masks(train,cfg,out,budget);report['mask_cohort_sha256']=mask_info['rendered_cohort_sha256']
        samples=[load(row,cfg) for row in train];dev_samples=[load(row,cfg) for row in dev]
        from .image_proxy import ImageProxyBank
        saved=load_proxy_input(args.proxy,args.proxy_sha256,'od');bank=ImageProxyBank(saved['proxy_quality'],pretrained=False);restore_proxy_state(bank,saved['proxy'])
        report['proxy_origin']={k:v for k,v in saved.items() if k!='proxy'};del saved
        report['proxy_mask_canonicalization']=canonicalize_proxy_masks(bank);proxy=FrameProxy(bank).to('cuda')
        warm=None
        if args.warm_start:
            from .coco_codec_study import validate_warm_start
            warm=torch.load(args.warm_start,map_location='cpu',weights_only=False)
            validate_warm_start(warm,cfg,os.environ.get('PAPER_SOURCE_SHA256'));proxy.load_state_dict(warm['proxy'])
        elif cfg['proxy_steps']>0:
            report['proxy_fit']=fit_video_proxy(proxy,samples,cfg,codec,out,budget);state['proxy_updates']=report['proxy_fit']['updates']
        freeze(proxy);report['proxy_probe']=proxy_probe(proxy,dev_samples,cfg,codec,out,budget);report['frozen_proxy_sha256']=model_digest(proxy)
        if cfg.get('study_profile')=='coco_codecs' and not cfg.get('rate_profile'):
            from .coco_codec_study import direction_probe
            report['rate_direction']=direction_probe(proxy,dev_samples,cfg,codec,out,budget)
        torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed']);npp=PaperNPP().to('cuda')
        report['initial_npp_sha256']=model_digest(npp);optimizer=torch.optim.Adam(npp.parameters(),lr=cfg['lr']);phase='warmup'
        if warm is not None:
            npp.load_state_dict(warm['npp']);optimizer.load_state_dict(warm['optimizer']);state.update(warm['state']);restore_rng(warm['rng'])
            report['warm_start_sha256']=digest(args.warm_start)
            del warm
        if args.resume:
            old=torch.load(args.resume,map_location='cpu',weights_only=False)
            validate_resume(old,cfg,os.environ.get('PAPER_SOURCE_SHA256'))
            npp=PaperNPP(adaptive=old['npp_adaptive']).to('cuda');npp.load_state_dict(old['npp']);proxy.load_state_dict(old['proxy']);state.update(continuation_state(old['state']))
            optimizer=torch.optim.Adam(npp.parameters(),lr=cfg['lr']);optimizer.load_state_dict(old['optimizer']);phase=old['optimizer_phase'];restore_rng(old['rng'])
        log_event(out,'tri_ready',task=cfg['task'],arm=cfg['sam_arm'],train=len(train),dev=len(dev),validation=len(validation),source=os.environ.get('PAPER_SOURCE_SHA256'))
        npp.train();initial={k:v.detach().cpu().clone() for k,v in npp.named_parameters()}
        while continue_warmup(cfg,state,phase):
            if budget.remaining()<cfg['evaluation_reserve']+600:state['stop_reason']='warmup_budget';break
            x=samples[random.randrange(len(samples))][0].to('cuda');h,w=x.shape[-2:];scale=min(1,256/max(h,w))
            x=F.interpolate(x.flatten(0,1),size=(max(16,round(h*scale)),max(16,round(w*scale))),mode='bilinear',align_corners=False)[None]
            optimizer.zero_grad(set_to_none=True);y=npp(x,35);loss=(y-x).square().mean();norm=finite_step(loss,npp,optimizer);state['warmup_updates']+=1
            state['warmup_calibrated']=False;state.pop('warmup_fidelity',None)
            if state['warmup_updates']==1 or state['warmup_updates']%100==0:
                log_event(out,'tri_warmup',step=state['warmup_updates'],mse=float(loss.detach()),gradient_norm=norm,lr=optimizer.param_groups[0]['lr'])
            if state['warmup_updates']%cfg['checkpoint_every']==0:save(out,cfg,state,npp,proxy,optimizer,phase)
            if state['warmup_updates']>=cfg['warmup_min_steps'] and state['warmup_updates']%cfg['warmup_probe_every']==0:
                calibrate_bn(npp,samples[:32]);state['warmup_calibrated']=True
                probe=check_fidelity(npp,dev_samples,cfg,cfg['gate_psnr'],budget);state['warmup_fidelity']=probe
                state['warmup_converged']=warmup_ready(cfg,state,probe)
                log_event(out,'tri_warmup_dev',step=state['warmup_updates'],**probe);save(out,cfg,state,npp,proxy,optimizer,phase)
                if not state['warmup_converged']:npp.train()
        task_resume=phase=='task' or task_trained(state)
        if not task_resume and not state.get('warmup_calibrated'):
            calibrate_bn(npp,samples[:32]);state['warmup_calibrated']=True
        report['frozen_proxy_sha256']=model_digest(proxy)
        gate=check_fidelity(npp,dev_samples,cfg,stage_fidelity_threshold(cfg,phase,state),budget)
        if task_resume:
            report['fidelity_gate']=state.get('warmup_fidelity');report['task_resume_fidelity']=gate
        else:state['warmup_fidelity']=gate;report['fidelity_gate']=gate
        log_event(out,'tri_task_resume_fidelity' if task_resume else 'tri_fidelity_gate',**gate);save(out,cfg,state,npp,proxy,optimizer,phase)
        if not gate['passed'] and not task_resume:
            state['stop_reason']='fidelity_gate_failed';report['execution_complete']=True;return
        if not gate['passed']:state['stop_reason']='task_fidelity_collapse'
        if not task_resume and not warmup_ready(cfg,state,gate):
            report['execution_complete']=True;return
        if cfg.get('rate_profile'):
            from .coco_content_rate_study import calibrate_content,audit_content
            if not args.warm_start and not task_resume:
                report['content_calibration']=calibrate_content(proxy,npp,samples,cfg,codec,out,budget)
            audit=audit_content(proxy,npp,dev_samples,cfg,codec,out,budget,'initial')
            report['content_screen']=audit['gate'];report['frozen_proxy_sha256']=model_digest(proxy)
            if not audit['gate']['passed']:raise RuntimeError('content proxy failed held-out screening before task optimization')
        report['warmup_npp_sha256']=model_digest(npp)
        if args.export_warmup and not args.warm_start:
            args.export_warmup.parent.mkdir(parents=True,exist_ok=True)
            save(out,cfg,state,npp,proxy,optimizer,phase)
            import shutil
            shutil.copyfile(out/'last.pt',args.export_warmup)
        def make_task():
            if cfg['task']=='ot':
                tracker_assets=ensure_tracking(Path('/kaggle/working/paper_assets/tracking'));return TrackingTask(tracker_assets['checkpoint'])
            return CocoTask(cfg['task'],device='cuda')
        task=initialize_task(make_task)
        (out/'analyzer.json').write_text(json.dumps(task.receipt(),indent=2))
        if not npp.adaptive:
            npp=PaperNPP.from_fixed(npp);optimizer=torch.optim.Adam(npp.parameters(),lr=cfg['lr'])
        phase='task';npp.train();freeze_npp_bn(npp);proxy.bank.train();proxy.requires_grad_(False)
        while state['task_updates']<cfg['task_steps'] and state['stop_reason']!='task_fidelity_collapse':
            if budget.remaining()<cfg['evaluation_reserve']:state['stop_reason']='task_budget';break
            i=random.randrange(len(samples));x,target,meta=samples[i];mask=cache.load(train[i],x);x=x.to('cuda')
            q=cfg['qps'][state['task_updates']%4] if cfg.get('study_profile')=='coco_codecs' else random.choice(cfg['qps'])
            optimizer.zero_grad(set_to_none=True);y=npp(x,q);p=proxy(y,q);actual=codec(y,cfg['codecs'][0],q)
            decoded=real_forward(p.reconstruction,actual.reconstruction);rate=real_forward(p.bpp,p.bpp.new_tensor(actual.bpp));task_loss=task.loss(decoded,target)
            if state['task_updates']==0:
                report['task_npp_gradient_norm']=task_gradient_norm(task_loss,npp)
                log_event(out,'tri_task_gradient_passed',task=cfg['task'],norm=report['task_npp_gradient_norm'])
            loss,parts=mask_objective(cfg,rate,task_loss,x,y,q,mask)
            if cfg.get('rate_profile') and state['task_updates']%128<4:
                from .codec_content_rate import gradient_balance
                task_term=cfg['lambda_scale']*task_lambda(q)*task_loss
                info=gradient_balance(rate,task_term,loss-rate-task_term,npp.parameters())
                log_event(out,'content_component_gradients',step=state['task_updates']+1,qp=q,**info)
            norm=finite_step(loss,npp,optimizer);state['task_updates']+=1
            if state['task_updates']==1 or state['task_updates']%cfg['log_every']==0:
                log_event(out,'tri_task_train',step=state['task_updates'],qp=q,source=meta['id'],loss=float(loss.detach()),real_bpp=actual.bpp,
                    task_loss=float(task_loss.detach()),gradient_norm=norm,lr=optimizer.param_groups[0]['lr'],**{k:float(v.detach()) for k,v in parts.items()})
            if state['task_updates']%50==0:
                probe=check_fidelity(npp,dev_samples,cfg,cfg['collapse_psnr'],budget);report['last_task_fidelity']=probe;npp.train();freeze_npp_bn(npp)
                if not probe['passed']:state['stop_reason']='task_fidelity_collapse';break
            if state['task_updates']%cfg['checkpoint_every']==0:save(out,cfg,state,npp,proxy,optimizer,phase)
        if state['stop_reason'] is None:state['stop_reason']='task_schedule_finished'
        save(out,cfg,state,npp,proxy,optimizer,phase)
        if model_digest(proxy)!=report['frozen_proxy_sha256'] or model_digest(task.model)!=task.weight_digest:raise RuntimeError('frozen task/proxy changed')
        report['npp_sha256']=model_digest(npp);report['task_trained']=task_trained(state)
        report['theta_delta_norm']=float(torch.sqrt(sum((v.detach().cpu()-initial[k]).square().sum() for k,v in npp.named_parameters() if k in initial)))
        if cfg.get('rate_profile'):
            report['final_content_screen']=audit_content(proxy,npp,dev_samples,cfg,codec,out,budget,'final')['gate']
        report['metrics']=evaluate(cfg,validation,npp,task,codec,out/'validation',annotation,budget)
        if cfg.get('classical_sigmas') and cfg['sam_arm']=='control':
            from .sam_structured import SAMGaussianNPP
            report['classical_baselines']={}
            for sigma in cfg['classical_sigmas']:
                gaussian=SAMGaussianNPP(sigma).to('cuda')
                report['classical_baselines'][str(sigma)]=evaluate(cfg,validation,gaussian,task,codec,out/'classical'/str(sigma),annotation,budget)
        report['execution_complete']=True
    except BaseException as error:
        report['error']=f'{type(error).__name__}: {error}'
        if npp is not None and proxy is not None:save(out,cfg,state,npp,proxy,optimizer,phase)
        raise
    finally:
        report['wall_seconds']=time.time()-started;report['state']=state
        (out/'config.json').write_text(json.dumps(cfg,indent=2));(out/'study_run.json').write_text(json.dumps(report,indent=2))
        log_event(out,'tri_end',task=cfg['task'],execution_complete=report['execution_complete'],state=state,error=report.get('error'))

if __name__=='__main__':main()
