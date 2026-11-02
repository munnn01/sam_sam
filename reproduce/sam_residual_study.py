"""Three-arm Residual-NPP study with mode-power proxy, SAM instance boundaries, and bootstrap CI."""
import argparse
import gc
import json
import random
import sys
from pathlib import Path
import torch
from . import tri_task_study as study
from .coco_content_rate_study import config as content_config, reconstruction_fit
from .coco_codec_study import paired_complete
from .codec_content_rate import NALCodec, ContentProxy
from .evaluation import bd_rate
from .protocol import QPS
from .residual_npp import ResidualPaperNPP
from .sam_structured import SAMTask, structured_objective, prepare_structured_masks

ARMS = ('control', 'foreground', 'sam_boundary')


def config(task, arm, codec='h264'):
    if arm not in ARMS:
        raise ValueError('three declared SAM ablation arms required')
    cfg = content_config(task, 'control' if arm == 'control' else 'foreground', codec)
    cfg.update(
        npp_architecture='residual_tanh_v1',
        rate_profile='nal_content_v2_power',
        sam_arm=arm,
        sam_guidance='structured_instance_maps_v1',
        classical_sigmas=[0.5, 0.8],
        warmup_steps=1000,
        warmup_min_steps=500,
        warmup_probe_every=500,
        gate_psnr=32.0,
        collapse_psnr=24.0,
        boundary_weight=2.0 if arm == 'sam_boundary' else 0.0,
    )
    cfg['sam2'].update(
        gamma=0.0,
        foreground_extra=0.0 if arm == 'control' else 1.0,
        structured_maps_schema='instance_core_edge_v1',
    )
    cfg['fidelity_notes'] = [
        s for s in cfg['fidelity_notes']
        if 'TRAIN-GT' not in s and 'positive content scale fitted' not in s
    ]
    cfg['fidelity_notes'] += [
        'Residual-bounded NPP (y = clip(x + 0.25*tanh(delta))) with small non-zero output init; >45 dB initial PSNR.',
        'Stricter fidelity gates: 32 dB warm-up gate and 24 dB task collapse gate.',
        'Three arms isolate SAM interior then SAM instance boundary; no GT edge extraction.',
        'Separate rounded/relaxed positive-power rate heads with checkpoint-persistent mode flag.',
        'Gaussian 0.5/0.8 baselines share 64 VAL sources and actual-codec AP protocol.',
    ]
    return cfg


def paired_bd_rate_bootstrap(anchor_curves, candidate_curves, n_boot=500, seed=20261101):
    if set(anchor_curves) != set(QPS) or set(candidate_curves) != set(QPS):
        raise ValueError('all four QPs required for paired bootstrap')
    n = len(anchor_curves[QPS[0]])
    if n < 2:
        raise ValueError('at least two paired sources required')
    rng = random.Random(seed)
    draws = []
    for _ in range(n_boot):
        indices = [rng.randrange(n) for _ in range(n)]
        a_rates, a_quals, b_rates, b_quals = [], [], [], []
        for q in QPS:
            a_pts = [anchor_curves[q][i] for i in indices]
            b_pts = [candidate_curves[q][i] for i in indices]
            a_rates.append(8.0 * sum(p['bytes'] for p in a_pts) / sum(p['pixels'] for p in a_pts))
            b_rates.append(8.0 * sum(p['bytes'] for p in b_pts) / sum(p['pixels'] for p in b_pts))
            a_quals.append(sum(p['quality'] for p in a_pts) / len(a_pts))
            b_quals.append(sum(p['quality'] for p in b_pts) / len(b_pts))
        res = bd_rate(a_rates, a_quals, b_rates, b_quals)
        if res['valid']:
            draws.append(res['percent'])
    draws.sort()
    if not draws:
        return dict(valid_draws=0, total_draws=n_boot, mean_percent=None, ci95_low=None, ci95_high=None)
    lo_idx = int(0.025 * (len(draws) - 1))
    hi_idx = int(0.975 * (len(draws) - 1))
    return dict(
        valid_draws=len(draws),
        total_draws=n_boot,
        mean_percent=sum(draws) / len(draws),
        ci95_low=draws[lo_idx],
        ci95_high=draws[hi_idx],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--proxy', required=True)
    parser.add_argument('--proxy-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    initial = json.loads(args.config.read_text())
    task = initial['task']
    codec_name = initial['codecs'][0]
    if initial != config(task, 'control', codec_name):
        raise ValueError('three-arm residual recipe differs')
    old_npp = study.PaperNPP
    study.config = lambda task, arm: config(task, arm, codec_name)
    study.PaperNPP = ResidualPaperNPP
    study.CocoTask = SAMTask
    study.mask_objective = structured_objective
    study.prepare_masks = prepare_structured_masks
    codecs = []

    def make_codec(**kwargs):
        value = NALCodec(codec_name, **kwargs)
        codecs.append(value)
        return value

    study.RealCodec = make_codec
    study.FrameProxy = lambda bank: ContentProxy(bank, codecs[-1])
    study.fit_video_proxy = reconstruction_fit
    try:
        args.out.mkdir(parents=True, exist_ok=True)
        warm = args.out / 'shared_warmup.pt'
        reports = {}
        for arm in ARMS:
            cfg = config(task, arm, codec_name)
            path = args.out / (arm + '_config.json')
            path.write_text(json.dumps(cfg, indent=2))
            sys.argv = [
                'tri_task_study', '--config', str(path),
                '--proxy', args.proxy, '--proxy-sha256', args.proxy_sha256,
                '--out', str(args.out / arm), '--export-warmup', str(warm),
            ]
            if arm != 'control':
                sys.argv += ['--warm-start', str(warm)]
            study.main()
            reports[arm] = json.loads((args.out / arm / 'study_run.json').read_text())
            gc.collect()
            torch.cuda.empty_cache()
        for arm in ARMS[1:]:
            if (reports[arm]['warmup_npp_sha256'] != reports['control']['warmup_npp_sha256']
                    or reports[arm]['frozen_proxy_sha256'] != reports['control']['frozen_proxy_sha256']):
                raise RuntimeError('three-arm warm-up/proxy differs')
        complete = all(paired_complete([reports['control'], reports[arm]], initial['task_steps']) for arm in ARMS[1:])
        result = dict(
            task=task,
            codec=codec_name,
            npp_architecture='residual_tanh_v1',
            complete=complete,
            warmup_npp_sha256=reports['control']['warmup_npp_sha256'],
            arms={
                arm: dict(
                    state=r['state'],
                    metrics=r.get('metrics'),
                    content_screen=r.get('content_screen'),
                )
                for arm, r in reports.items()
            },
            classical_baselines=reports['control'].get('classical_baselines'),
            scope=initial['scope'],
        )
        (args.out / 'three_arm_results.json').write_text(json.dumps(result, indent=2))
        if not complete:
            raise RuntimeError('three-arm residual schedule/evaluation incomplete; inspect stage counters')
    finally:
        study.PaperNPP = old_npp


if __name__ == '__main__':
    main()
