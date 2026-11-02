"""Three-arm original-NPP study with mode-power proxy and SAM-only instance edges."""
import argparse,gc,json,sys
from pathlib import Path
import torch
from . import tri_task_study as study
from .coco_content_rate_study import config as content_config,reconstruction_fit
from .coco_codec_study import paired_complete
from .codec_content_rate import NALCodec,ContentProxy
from .sam_structured import SAMTask,structured_objective,prepare_structured_masks

ARMS=('control','foreground','sam_boundary')
def config(task,arm,codec='h264'):
    if arm not in ARMS:raise ValueError('three declared SAM ablation arms required')
    cfg=content_config(task,'control' if arm=='control' else 'foreground',codec)
    cfg.update(rate_profile='nal_content_v2_power',sam_arm=arm,sam_guidance='structured_instance_maps_v1',
        classical_sigmas=[.5,.8],boundary_weight=2. if arm=='sam_boundary' else 0.)
    cfg['sam2'].update(gamma=0.,foreground_extra=0. if arm=='control' else 1.,
        structured_maps_schema='instance_core_edge_v1')
    cfg['fidelity_notes']=[s for s in cfg['fidelity_notes'] if 'TRAIN-GT' not in s and 'positive content scale fitted' not in s]
    cfg['fidelity_notes']+=['Original released NPP retained; no residual or identity initialization.',
        'Three arms isolate SAM interior then SAM instance boundary; no GT edge extraction.',
        'Separate rounded/relaxed positive-power rate heads with checkpoint-persistent mode flag.',
        'Gaussian0.5/0.8 baselines share64VAL sources and actual-codec AP protocol.']
    return cfg

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--proxy',required=True);parser.add_argument('--proxy-sha256',required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();initial=json.loads(args.config.read_text());task=initial['task'];codec_name=initial['codecs'][0]
    if initial!=config(task,'control',codec_name):raise ValueError('three-arm recipe differs')
    study.config=lambda task,arm:config(task,arm,codec_name)
    study.CocoTask=SAMTask;study.mask_objective=structured_objective;study.prepare_masks=prepare_structured_masks
    codecs=[]
    def make_codec(**kwargs):
        value=NALCodec(codec_name,**kwargs);codecs.append(value);return value
    study.RealCodec=make_codec;study.FrameProxy=lambda bank:ContentProxy(bank,codecs[-1])
    study.fit_video_proxy=reconstruction_fit
    args.out.mkdir(parents=True,exist_ok=True);warm=args.out/'shared_warmup.pt';reports={}
    for arm in ARMS:
        cfg=config(task,arm,codec_name);path=args.out/(arm+'_config.json');path.write_text(json.dumps(cfg,indent=2))
        sys.argv=['tri_task_study','--config',str(path),'--proxy',args.proxy,'--proxy-sha256',args.proxy_sha256,
            '--out',str(args.out/arm),'--export-warmup',str(warm)]
        if arm!='control':sys.argv+=['--warm-start',str(warm)]
        study.main();reports[arm]=json.loads((args.out/arm/'study_run.json').read_text())
        gc.collect();torch.cuda.empty_cache()
    for arm in ARMS[1:]:
        if reports[arm]['warmup_npp_sha256']!=reports['control']['warmup_npp_sha256'] or reports[arm]['frozen_proxy_sha256']!=reports['control']['frozen_proxy_sha256']:
            raise RuntimeError('three-arm warm-up/proxy differs')
    complete=all(paired_complete([reports['control'],reports[arm]],initial['task_steps']) for arm in ARMS[1:])
    result=dict(task=task,codec=codec_name,complete=complete,
        warmup_npp_sha256=reports['control']['warmup_npp_sha256'],
        arms={arm:dict(state=r['state'],metrics=r.get('metrics'),content_screen=r.get('content_screen')) for arm,r in reports.items()},
        classical_baselines=reports['control'].get('classical_baselines'),scope=initial['scope'])
    (args.out/'three_arm_results.json').write_text(json.dumps(result,indent=2))
    if not complete:raise RuntimeError('three-arm schedule/evaluation incomplete; inspect stage counters')

if __name__=='__main__':main()
