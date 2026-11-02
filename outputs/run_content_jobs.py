"""Operate the four public-input private Kaggle workers; all outputs stay in SAM."""
import json,os,sys,time,subprocess,re,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'outputs/content_rate_20261101';BASE.mkdir(parents=True,exist_ok=True)
sys.path[:0]=[str(ROOT),'D:/STUDY/LAB/bao_1/shared_ar_od_proxy']
from ar_od_proxy.kaggle import pool_environment
from reproduce.coco_content_rate_study import config
from reproduce.coco_content_rate_cloud import prepare,verify
from reproduce.tri_task_data import PUBLIC_PROXY
from reproduce.kaggle import validate_submission
JOBS=[('is','h264','tranthihongdieu'),('is','h265','dngbolm'),('od','h264','hoangminhhuy123'),('od','h265','ngynanhthuw')]
def cli(account,args,timeout=120):
    env,secret=pool_environment(ROOT.parent/'pool.json',account)
    try:
        p=subprocess.run([sys.executable,'-u','-m','kaggle',*args],env=env,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=timeout)
        return p.returncode,(p.stdout+p.stderr).replace(secret,'[REDACTED]')
    except subprocess.TimeoutExpired as e:
        def decode(v):return v.decode('utf-8',errors='replace') if isinstance(v,bytes) else (v or '')
        return -1,(decode(e.stdout)+decode(e.stderr)).replace(secret,'[REDACTED]')
action=sys.argv[1]
if action=='prepare':
    rows=[]
    for task,cod,account in JOBS:
        key=task+'-'+cod;cfg=config(task,'control',cod)
        job=prepare(ROOT,BASE/'jobs'/key/'payload',account,'sam-content-'+key+'-20261101',cfg)
        rows.append(dict(key=key,handle=job['handle'],source_sha256=job['source_sha256']))
    (BASE/'prepared_jobs.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows))
elif action=='quota':
    account=sys.argv[2];env,secret=pool_environment(ROOT.parent/'pool.json',account);os.environ.update(env)
    from kaggle.api.kaggle_api_extended import KaggleApi
    api=KaggleApi();api.authenticate()
    with api.build_kaggle_client() as client:response=client.kernels.kernels_api_client.get_accelerator_quota_statistics()
    q=response.gpu_quota
    used=q.time_used.total_seconds();reserved=q.time_reserved.total_seconds();total=q.total_time_allowed.total_seconds()
    receipt=dict(account=account,used=used,reserved=reserved,total=total,remaining=total-used-reserved,checked=time.time())
    (BASE/('quota_'+account+'.json')).write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
elif action=='registry':
    rows=[]
    for item in json.loads((BASE/'prepared_jobs.json').read_text()):
        folder=BASE/'jobs'/item['key'];job=json.loads((folder/'payload/job.json').read_text());r=dict(item,config=job['config'],input_spec=job['input_spec'])
        for name in ('submission','launch','server_verification'):
            p=folder/(name+'.json')
            if p.is_file():r[name]=json.loads(p.read_text())
        m=re.search(r'KernelWorkerStatus\.([A-Z_]+)',r.get('launch',{}).get('status',''))
        r['status']=m.group(1) if m else 'SUBMITTED';rows.append(r)
    receipt=dict(study='deployment_geometry_content_rate',local_root=str(ROOT),updated=time.time(),jobs=rows,
        scientific_complete=False,tracking='DEFERRED',worker_seconds_per_arm=6000,service_seconds_per_pair=13200,
        protocol='COCO independent one-frame H264/H265 streams; 4QP; 64VAL; shared warm-up/proxy/RNG',
        limitations=['Screening must pass before task updates; RUNNING is not a task result.','No identity fallback; no new BD-rate claim until real-codec curves complete.'])
    (ROOT/'CONTENT_RATE_RUNS_20261101.json').write_text(json.dumps(receipt,indent=2))
    (BASE/'deployment_receipt.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps([dict(key=r['key'],status=r['status'],latest=r.get('launch',{}).get('latest')) for r in rows]))
else:
    key=sys.argv[2];folder=BASE/'jobs'/key;payload=folder/'payload';meta,job=verify(payload);account=job['handle'].split('/')[0]
    if action=='push':
        quota=json.loads((BASE/('quota_'+account+'.json')).read_text())
        if time.time()-quota['checked']>1800 or quota['remaining']<13200:raise ValueError('fresh sufficient quota required')
        env,secret=pool_environment(ROOT.parent/'pool.json',account);os.environ.update(env)
        from kaggle.api.kaggle_api_extended import KaggleApi
        from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetRequest
        api=KaggleApi();api.authenticate()
        with api.build_kaggle_client() as client:
            req=ApiGetDatasetRequest();req.owner_slug=PUBLIC_PROXY['ref'].split('/')[0];req.dataset_slug=PUBLIC_PROXY['ref'].split('/')[1]
            data=client.datasets.dataset_api_client.get_dataset(req)
        if data.is_private is not False or data.current_version_number!=1:raise ValueError('public proxy binding changed')
        code,output=cli(account,['kernels','push','-p',str(payload),'--accelerator','NvidiaTeslaT4','--timeout','13200'],180)
        (folder/'submission.txt').write_text(output,encoding='utf-8')
        if code:raise RuntimeError(output[-1800:])
        result=validate_submission(output,job['handle']);(folder/'submission.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
    elif action=='observe':
        _,status=cli(account,['kernels','status',job['handle']],60);_,stream=cli(account,['kernels','logs',job['handle'],'-f'],25)
        (folder/'stream.log').write_text(stream,encoding='utf-8');events=[]
        for line in stream.splitlines():
            try:
                e=json.loads(line[line.index('{'):line.rindex('}')+1])
                if isinstance(e,dict):events.append(e)
            except ValueError:pass
        result=dict(key=key,handle=job['handle'],status=status.strip(),checked=time.time(),latest=events[-1] if events else None,
            screens=[e for e in events if e.get('stage')=='content_rate_audit'],
            gradients=[e for e in events if e.get('stage')=='content_component_gradients'],
            terminal=[e for e in events if e.get('stage') in ('tri_end','cloud_end')])
        (folder/'launch.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
    elif action=='verify':
        dest=folder/'server_snapshot';code,output=cli(account,['kernels','pull',job['handle'],'-p',str(dest),'-m'])
        if code:raise RuntimeError(output[-1500:])
        actual=json.loads((dest/'kernel-metadata.json').read_text());files=list(dest.glob('*.ipynb'))
        if actual['id']!=job['handle'] or actual['is_private'] is not True or len(files)!=1:raise ValueError('remote source/privacy differs')
        local=''.join(json.loads((payload/'notebook.ipynb').read_text())['cells'][0]['source'])
        remote=''.join(json.loads(files[0].read_text())['cells'][0]['source'])
        if local!=remote:raise ValueError('remote notebook differs')
        result=dict(handle=job['handle'],source_sha256=job['source_sha256'],private_verified=True,notebook_verified=True,checked=time.time())
        (folder/'server_verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
    elif action=='fetch':
        env,secret=pool_environment(ROOT.parent/'pool.json',account);os.environ.update(env)
        from kaggle.api.kaggle_api_extended import KaggleApi
        from kagglesdk.kernels.types.kernels_api_service import ApiDownloadKernelOutputRequest
        import requests
        api=KaggleApi();api.authenticate();downloaded=[];missing=[]
        names=['paired_results.json','cloud_status.json']+[arm+'/'+name for arm in ('control','foreground') for name in
            ('study_run.json','config.json','cohort.json','analyzer.json','content_calibration.json','content_audit_initial.json','content_audit_final.json','proxy_probe.json','validation/metrics.json','validation/metrics_partial.json')]
        with api.build_kaggle_client() as client:
            for name in names:
                req=ApiDownloadKernelOutputRequest();req.owner_slug=account;req.kernel_slug=job['handle'].split('/')[1];req.version_number=1;req.file_path='paper_results/'+name
                try:
                    response=client.kernels.kernels_api_client.download_kernel_output(req);json.loads(response.content)
                    p=folder/'downloads/paper_results'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(response.content);downloaded.append(name)
                except requests.HTTPError as e:
                    code=e.response.status_code if e.response is not None else None
                    if code==404:missing.append(name)
                    else:raise RuntimeError('report HTTP status '+str(code)) from None
        print(json.dumps(dict(key=key,downloaded=downloaded,missing=missing)))
    else:raise ValueError('unknown action')
