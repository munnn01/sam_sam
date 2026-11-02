"""Published equations and explicit user deviations; no identity checkpoint fallback."""
import copy
import math

QPS=(30,35,40,45)
IMAGE_SOURCE='6a77c4633cd5bb6750c77ea9d213d38ffc88ee7a'
TRACKING_SOURCE='7eb9e74bd3d40e29dbcec444902237da13de247b'

def log_interpolate(x,xs,ys):
    if not math.isfinite(float(x)) or len(xs)!=len(ys) or len(xs)<2:
        raise ValueError('finite value and paired log-interpolation anchors required')
    i=next((i for i in range(len(xs)-1) if x<=xs[i+1]),len(xs)-2)
    a=(x-xs[i])/(xs[i+1]-xs[i])
    return 2**((1-a)*math.log2(ys[i])+a*math.log2(ys[i+1]))

def proxy_lambda(qp):
    # Image paper Table I, RGB in [0,1]. Out-of-grid QPs are an explicit user adaptation.
    return log_interpolate(qp,[28,31,34,37,41,44],[4096,2048,1024,512,256,128])

def task_lambda(qp):
    # Image paper IV-A reported task trade-off pairs, not proxy RD lambdas.
    return log_interpolate(qp,[28,31,34,37,41],[.5,1,2,4,8])

def real_forward(proxy,real):
    return proxy+(real-proxy).detach()

def validate_checkpoint(step,score):
    if type(step) is not int or step<=0 or not math.isfinite(float(score)):
        raise ValueError('only finite trained checkpoints can be selected')
    return step

def run_receipt(task,completed,target,reason):
    if task not in ('od','ic','ar','ot') or not 0<=completed<=target or target<=0:
        raise ValueError('invalid task or completed/target steps')
    return dict(task=task,completed_steps=completed,target_steps=target,stop_reason=reason,
                scientific_complete=(completed==target and reason=='finished'))

def recipe(task,mode='full'):
    if task not in ('od','ic','ar','ot') or mode not in ('full','pilot'):
        raise ValueError('original tasks od/ic/ar/ot; mode full/pilot required')
    image=task in ('od','ic')
    cfg=dict(task=task,paper='image' if image else 'video',qps=list(QPS),
        codecs=['bpg'] if image else ['h264','h265'],identity_fallback=False,
        seed=20261023,mode=mode,scope='paper_schedule_with_declared_deviations' if mode=='full' else 'pilot_not_full_reproduction',
        lr=1e-4,beta=.5,alpha=10.,rate_weight=.001,fixed_qp=35,
        proxy_steps=400000 if image else 0,fixed_steps=120000 if image else 0,
        adaptive_steps=100000,npp_size={'od':640,'ic':224,'ar':224,'ot':288}[task],
        frames=16 if not image else 1,batch_size=1,preset='medium',fps=25,
        gop=16,checkpoint_every=1000,log_every=20,max_session_seconds=36000,
        validation_sources=None,
        full_validation_sources={'od':5000,'ic':50000,'ar':20000,'ot':180}[task],
        video_width=16,temporal_kernel=3,block_size=8,motion_radius=4,
        prior_steps=512 if not image else 0,proxy_quality={'30':5,'35':4,'40':3,'45':2},
        source_pins={'image_npp':IMAGE_SOURCE,'pytracking':TRACKING_SOURCE},
        input_condition='released_lambda_over_8' if image else 'none_as_paper',
        checkpoint_selection='last_trained',use_amp=False,
        fidelity_notes=[
          'Four QPs30/35/40/45 explicitly requested by user.',
          'One representative published backbone family per original task; Torchvision image/AR adapters differ from original OpenMMLab implementations.',
          'Full mode evaluates all available original validation sources with resumable per-source caches; pilot mode is a declared subset.',
        ])
    if image:
        cfg['fidelity_notes'] += [
          'Official released image NPP used; source conditioning uses loss-lambda/8 rather than raw QP.',
          'Proxy is CompressAI Minnen2018 pretrained implementation, independently calibrated to actual BPG reconstructions.',
          '400k proxy updates are global across four QP-specific proxies; paper does not disambiguate per-proxy accounting.',
          'Image proxy/task lambda values at new QPs are log-linearly interpolated/extrapolated from published tables.',
        ]
    else:
        cfg['fidelity_notes'] += [
          'Video author code not verified available; channels/attention and block-search details reconstructed from under-specified figures.',
          'Temporal/spatial3D convolutions, sigmoid conditional channel attention, one residual block, width16 are explicit implementation choices.',
          'Block8 DCT, DC=.5 intra candidate,9 integer motion candidates and straight-through quantization are explicit virtual-codec choices.',
          'Factorized nonparametric CDF is fitted on TRAIN residuals then frozen; paper does not provide entropy-prior weights.',
          '100k NPP steps,16-frame crops and512 prior-fit steps are declared budgets; video paper does not state iteration count.',
          'Available Kinetics mount is a5percent subset; not full K400 paper training data.',
          'GOT10k validation metrics use ground truth; hidden TEST requires benchmark-server evaluation and is not claimed.',
        ]
    if mode=='pilot':
        cfg.update(proxy_steps=64 if image else 0,fixed_steps=64 if image else 0,
                   adaptive_steps=128,prior_steps=32 if not image else 0,
                   checkpoint_every=32,log_every=4,validation_sources=8,
                   npp_size={'od':320,'ic':224,'ar':224,'ot':288}[task],max_session_seconds=14400)
    return copy.deepcopy(cfg)
