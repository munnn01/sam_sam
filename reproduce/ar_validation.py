"""Evaluate a completed, SHA-bound AR checkpoint without any optimization steps."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time
import traceback

import numpy as np
import torch

from .training import config_digest, log_event, runtime_identity


def verified_checkpoint(path, sha256, source_sha256):
    h=hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda:source.read(1024*1024),b''):h.update(block)
    if h.hexdigest()!=sha256:raise ValueError('checkpoint SHA differs')
    saved=torch.load(path,map_location='cpu',weights_only=False)
    if saved.get('source_manifest_sha256')!=source_sha256:raise ValueError('training source differs')
    cfg=saved['config'];state=saved['state']
    if config_digest(cfg)!=saved.get('config_sha256'):raise ValueError('checkpoint config digest differs')
    if (cfg['task']!='ar' or cfg['mode']!='full' or cfg['qps']!=[30,35,40,45]
        or cfg.get('identity_fallback') is not False
        or state['phase_steps'].get('prior')!=cfg['prior_steps']
        or state['phase_steps'].get('adaptive')!=cfg['adaptive_steps']
        or state.get('selected_step')!=cfg['adaptive_steps']):
        raise ValueError('AR training schedule is incomplete or differs')
    # Frozen older trainers save last.pt just before setting completed=True.
    # Exact phase counts and selected_step establish schedule completion here.
    return saved


def restore_evaluation_cache(source,destination):
    source=Path(source);destination=Path(destination)
    incoming=source/'evaluation_identity.json'
    if not incoming.is_file():raise ValueError('incoming evaluation cache identity is absent')
    identity=json.loads(incoming.read_text())
    if destination.is_dir() and any(destination.iterdir()):
        existing=destination/'evaluation_identity.json'
        if not existing.is_file() or json.loads(existing.read_text())!=identity:
            raise ValueError('destination evaluation cache identity differs or is absent')
    if source.resolve()!=destination.resolve():shutil.copytree(source,destination,dirs_exist_ok=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True,type=Path)
    parser.add_argument('--checkpoint-sha256',required=True)
    parser.add_argument('--training-source-sha256',required=True)
    parser.add_argument('--inputs',type=Path,default=Path('/kaggle/input'))
    parser.add_argument('--assets',type=Path,default=Path('/kaggle/working/paper_assets'))
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--cache-from',type=Path)
    args=parser.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=True)
    started=time.time();receipt=dict(task='ar',mode='full',evaluation_only=True,execution_complete=False,
                                   scientific_complete=False,started=started)
    try:
        if not torch.cuda.is_available():raise RuntimeError('CUDA GPU required')
        saved=verified_checkpoint(args.checkpoint,args.checkpoint_sha256,args.training_source_sha256)
        original=saved['config'];cfg=copy.deepcopy(original)
        cfg.update(ar_validation_short_clip='loop',evaluation_only=True,
                   training_config_sha256=saved['config_sha256'],
                   training_source_sha256=args.training_source_sha256,
                   training_checkpoint_sha256=args.checkpoint_sha256,
                   evaluation_source_sha256=os.environ.get('PAPER_SOURCE_SHA256'))
        cfg['fidelity_notes'].append('Evaluation repair: retain every available JPEG validation source; short clips loop frame indices at stride2. Long clips keep the original centered indices. Training sampling and weights are unchanged.')
        random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed'])
        torch.cuda.manual_seed_all(cfg['seed']);torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
        from .assets import records_for_task
        from .data import ordered_subset
        from .models import VideoNPP
        from .tasks import Analyzer,fingerprint
        from .codecs import RealCodec
        from .evaluation import evaluate
        rows,validation,inventory=records_for_task('ar',args.inputs,args.assets)
        selected=ordered_subset(validation,cfg['validation_sources'],cfg['seed'])
        analyzer=Analyzer('ar','cuda')
        if runtime_identity(rows,validation,inventory,analyzer.weight_digest)!=original['runtime_identity']:
            raise ValueError('evaluation data/analyzer differs from training checkpoint identity')
        npp=VideoNPP(cfg['video_width']).to('cuda');npp.load_state_dict(saved['npp'],strict=True);npp.eval()
        before=fingerprint(npp)
        checkpoint_out=out/'last.pt'
        if args.checkpoint.resolve()!=checkpoint_out.resolve():shutil.copy2(args.checkpoint,checkpoint_out)
        for name,value in [('config.json',cfg),('training_config.json',original),
                           ('data_inventory.json',inventory),('analyzer.json',analyzer.receipt())]:
            (out/name).write_text(json.dumps(value,indent=2))
        receipt.update(training_schedule_complete=True,trained_npp_steps=cfg['adaptive_steps'],
                       phase_steps=saved['state']['phase_steps'],training_checkpoint_sha256=args.checkpoint_sha256,
                       npp_sha256=before,validation_sources=len(selected),full_original_training_corpus=inventory['full_original_training_corpus'])
        log_event(out,'evaluation_start',task='ar',evaluation_only=True,training_schedule_complete=True,
                  trained_npp_steps=cfg['adaptive_steps'],training_source_sha256=args.training_source_sha256,
                  training_checkpoint_sha256=args.checkpoint_sha256,npp_sha256=before,
                  validation_sources=len(selected),qps=cfg['qps'],short_clip_policy='loop')
        if args.cache_from:
            source=args.cache_from/'validation'
            restore_evaluation_cache(source,out/'validation')
        codec=RealCodec(preset=cfg['preset'],gop=cfg['gop'],fps=cfg['fps'])
        metrics=evaluate(cfg,selected,npp,analyzer,codec,out/'validation',
                         max_seconds=max(0,cfg['max_session_seconds']-(time.time()-started)))
        if fingerprint(npp)!=before or fingerprint(analyzer.model)!=analyzer.weight_digest:
            raise RuntimeError('evaluation changed NPP or analyzer weights')
        receipt.update(metrics=metrics,execution_complete=True,needs_resume=not metrics['complete'],
                       frozen_analyzer_verified=True,npp_unchanged=True,
                       scientific_complete=metrics['complete'] and inventory['full_original_training_corpus']
                           and len(selected)==cfg['full_validation_sources'])
    except BaseException as error:
        receipt.update(error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc())
        raise
    finally:
        receipt['wall_seconds']=time.time()-started
        (out/'run.json').write_text(json.dumps(receipt,indent=2))
        log_event(out,'run_end',task='ar',execution_complete=receipt['execution_complete'],
                  scientific_complete=receipt['scientific_complete'],error=receipt.get('error'))


if __name__=='__main__':main()
