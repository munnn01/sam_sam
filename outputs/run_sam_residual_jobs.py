"""Operate the four Residual-NPP + SAM-Boundary Kaggle workers in SAM_2."""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/sam_residual_20261101'
BASE.mkdir(parents=True, exist_ok=True)
sys.path[:0] = [str(ROOT), 'D:/STUDY/LAB/bao_1/shared_ar_od_proxy']

from ar_od_proxy.kaggle import pool_environment
from reproduce.codec_content_rate import fit_mode_power_scales, screen_gate
from reproduce.kaggle import source_files, validate_submission
from reproduce.sam_residual_cloud import prepare, verify
from reproduce.sam_residual_study import ARMS, config
from reproduce.tri_task_data import PUBLIC_PROXY

JOBS = [
    ('is', 'h264', 'thuha1205'),
    ('is', 'h265', 'htran123456'),
    ('od', 'h264', 'shungg05'),
    ('od', 'h265', 'dieulinhh'),
]


def cli(account, args, timeout=120):
    env, secret = pool_environment(ROOT.parent / 'pool.json', account)
    try:
        p = subprocess.run(
            [sys.executable, '-u', '-m', 'kaggle', *args],
            env=env, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=timeout,
        )
        return p.returncode, (p.stdout + p.stderr).replace(secret, '[REDACTED]')
    except subprocess.TimeoutExpired as e:
        def decode(v):
            return v.decode('utf-8', errors='replace') if isinstance(v, bytes) else (v or '')
        return -1, (decode(e.stdout) + decode(e.stderr)).replace(secret, '[REDACTED]')


def verify_core_and_offline_gates():
    hashes = source_files(ROOT)
    orig = json.loads(
        Path('D:/STUDY/LAB/bao_1/output/paper_reproduction_20261028/legacy_source_manifest.json').read_text()
    )['source_hashes']
    if not all(hashes.get(k) == v for k, v in orig.items()):
        raise ValueError('original 42 core files altered')
    source_sha = hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    (ROOT / 'SOURCE_MANIFEST.json').write_text(json.dumps(dict(source_sha256=source_sha, source_hashes=hashes), indent=2))

    cached_base = ROOT / 'outputs/content_rate_20261101/jobs'
    offline_gates = {}
    for key in ('is-h264', 'is-h265', 'od-h264', 'od-h265'):
        task, codec = key.split('-')
        cfg = config(task, 'control', codec)
        cal = json.loads((cached_base / key / 'downloads/paper_results/control/content_calibration.json').read_text())
        aud = json.loads((cached_base / key / 'downloads/paper_results/control/content_audit_initial.json').read_text())
        fits = fit_mode_power_scales(cal['rows'])
        recomputed = []
        for r in aud['rows']:
            idx = [30, 35, 40, 45].index(r['qp'])
            raw_relaxed = r['relaxed_content_bpp'] / cal['scales'][idx]
            s = fits['relaxed']['scales'][idx]
            p = fits['relaxed']['powers'][idx]
            new_content = s * (raw_relaxed ** p)
            recomputed.append(dict(r, relaxed_content_bpp=new_content, relaxed_bpp=new_content + r['real_overhead_bpp']))
        gate = screen_gate(recomputed, aud['direction'], cfg)
        if not gate['passed']:
            raise RuntimeError(f'offline mode-power gate failed for {key}')
        offline_gates[key] = gate
    (BASE / 'offline_gate_verification.json').write_text(json.dumps(offline_gates, indent=2))
    return source_sha, offline_gates


action = sys.argv[1]
if action == 'prepare':
    source_sha, offline_gates = verify_core_and_offline_gates()
    rows = []
    for task, cod, account in JOBS:
        key = f'{task}-{cod}'
        cfg = config(task, 'control', cod)
        payload_dir = BASE / 'jobs' / key / 'payload'
        if payload_dir.exists():
            shutil.rmtree(payload_dir)
        job = prepare(ROOT, payload_dir, account, f'sam-residual-{key}-20261101', cfg)
        rows.append(dict(key=key, handle=job['handle'], source_sha256=job['source_sha256']))
    (BASE / 'prepared_jobs.json').write_text(json.dumps(rows, indent=2))
    print(json.dumps(dict(source_sha256=source_sha, jobs=rows)))
elif action == 'quota':
    account = sys.argv[2]
    env, _ = pool_environment(ROOT.parent / 'pool.json', account)
    os.environ.update(env)
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    with api.build_kaggle_client() as client:
        response = client.kernels.kernels_api_client.get_accelerator_quota_statistics()
    q = response.gpu_quota
    used = q.time_used.total_seconds()
    reserved = q.time_reserved.total_seconds()
    total = q.total_time_allowed.total_seconds()
    receipt = dict(account=account, used=used, reserved=reserved, total=total, remaining=total - used - reserved, checked=time.time())
    (BASE / f'quota_{account}.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt))
elif action == 'push':
    key = sys.argv[2]
    folder = BASE / 'jobs' / key
    payload = folder / 'payload'
    meta, job = verify(payload, ROOT)
    account = job['handle'].split('/')[0]
    quota = json.loads((BASE / f'quota_{account}.json').read_text())
    if time.time() - quota['checked'] > 1800 or quota['remaining'] < 19200 or quota['reserved'] > 0:
        raise ValueError(f'fresh unreserved quota >=19200s required for {account}: {quota}')
    env, _ = pool_environment(ROOT.parent / 'pool.json', account)
    os.environ.update(env)
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetRequest
    api = KaggleApi()
    api.authenticate()
    with api.build_kaggle_client() as client:
        req = ApiGetDatasetRequest()
        req.owner_slug = PUBLIC_PROXY['ref'].split('/')[0]
        req.dataset_slug = PUBLIC_PROXY['ref'].split('/')[1]
        data = client.datasets.dataset_api_client.get_dataset(req)
    if data.is_private is not False or data.current_version_number != 1:
        raise ValueError('public proxy binding changed')
    code, output = cli(account, ['kernels', 'push', '-p', str(payload), '--accelerator', 'NvidiaTeslaT4', '--timeout', '19200'], 180)
    (folder / 'submission.txt').write_text(output, encoding='utf-8')
    if code:
        raise RuntimeError(output[-1800:])
    result = validate_submission(output, job['handle'])
    (folder / 'submission.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
elif action == 'verify':
    key = sys.argv[2]
    folder = BASE / 'jobs' / key
    payload = folder / 'payload'
    meta, job = verify(payload, ROOT)
    account = job['handle'].split('/')[0]
    dest = folder / 'server_snapshot'
    if dest.exists():
        shutil.rmtree(dest)
    code, output = cli(account, ['kernels', 'pull', job['handle'], '-p', str(dest), '-m'])
    if code:
        raise RuntimeError(output[-1500:])
    actual = json.loads((dest / 'kernel-metadata.json').read_text())
    files = list(dest.glob('*.ipynb'))
    if actual['id'] != job['handle'] or actual['is_private'] is not True or len(files) != 1:
        raise ValueError('remote source/privacy differs')
    local = ''.join(json.loads((payload / 'notebook.ipynb').read_text())['cells'][0]['source'])
    remote = ''.join(json.loads(files[0].read_text())['cells'][0]['source'])
    if local != remote:
        raise ValueError('remote notebook differs')
    result = dict(handle=job['handle'], source_sha256=job['source_sha256'], private_verified=True, notebook_verified=True, checked=time.time())
    (folder / 'server_verification.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
elif action == 'observe':
    key = sys.argv[2]
    folder = BASE / 'jobs' / key
    payload = folder / 'payload'
    meta, job = verify(payload, ROOT)
    account = job['handle'].split('/')[0]
    _, status = cli(account, ['kernels', 'status', job['handle']], 60)
    _, stream = cli(account, ['kernels', 'logs', job['handle'], '-f'], 25)
    (folder / 'stream.log').write_text(stream, encoding='utf-8')
    events = []
    for line in stream.splitlines():
        try:
            e = json.loads(line[line.index('{'):line.rindex('}') + 1])
            if isinstance(e, dict):
                events.append(e)
        except ValueError:
            pass
    result = dict(
        key=key,
        handle=job['handle'],
        status=status.strip(),
        checked=time.time(),
        latest=events[-1] if events else None,
        screens=[e for e in events if e.get('stage') == 'content_rate_audit'],
        gradients=[e for e in events if e.get('stage') == 'content_component_gradients'],
        terminal=[e for e in events if e.get('stage') in ('tri_end', 'cloud_end')],
    )
    (folder / 'launch.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
elif action == 'registry':
    rows = []
    for item in json.loads((BASE / 'prepared_jobs.json').read_text()):
        folder = BASE / 'jobs' / item['key']
        job = json.loads((folder / 'payload/job.json').read_text())
        r = dict(item, config=job['config'], input_spec=job['input_spec'])
        for name in ('submission', 'launch', 'server_verification'):
            p = folder / f'{name}.json'
            if p.is_file():
                r[name] = json.loads(p.read_text())
        m = re.search(r'KernelWorkerStatus\.([A-Z_]+)', r.get('launch', {}).get('status', ''))
        r['status'] = m.group(1) if m else 'SUBMITTED'
        rows.append(r)
    receipt = dict(
        study='sam_residual_three_arm_mode_power',
        local_root=str(ROOT),
        updated=time.time(),
        arms=list(ARMS),
        jobs=rows,
        scientific_complete=False,
        tracking='DEFERRED',
        worker_seconds_per_arm=6000,
        service_seconds_per_worker=19200,
        protocol='COCO independent one-frame H264/H265 streams; 4QP; 64VAL; ResidualPaperNPP (>45dB init, 32dB/24dB gates); shared warm-up/proxy/RNG; Gaussian 0.5/0.8 baselines',
    )
    (ROOT / 'SAM_RESIDUAL_RUNS_20261101.json').write_text(json.dumps(receipt, indent=2))
    (BASE / 'deployment_receipt.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps([dict(key=r['key'], handle=r['handle'], status=r['status'], latest=r.get('launch', {}).get('latest')) for r in rows]))
else:
    raise ValueError('unknown action')
