import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'outputs/content_rate_20261101'
result=json.loads((ROOT/'CONTENT_RATE_RESULTS_20261101.json').read_text())
registry=json.loads((ROOT/'CONTENT_RATE_RUNS_20261101.json').read_text())
registry['results_file']='paper_reproduction/CONTENT_RATE_RESULTS_20261101.json'
registry['screening_passed_jobs']=0;registry['task_trained_jobs']=0;registry['new_bd_rate_available']=False
registry['verification']=json.loads((BASE/'verification.json').read_text())
for row in registry['jobs']:
 r=result['jobs'][row['key']]
 row['result']={k:r[k] for k in ('status','task_updates','bd_rate','warmup_updates','warmup_psnr_db','gate')}
registry['offline_diagnosis']='Separate rounded/relaxed positive-power calibration fitted to TRAIN cached features meets rate thresholds on the same TRAIN-dev; no GPU task run or BD-rate for that candidate.'
(ROOT/'CONTENT_RATE_RUNS_20261101.json').write_text(json.dumps(registry,indent=2))
(BASE/'deployment_receipt.json').write_text(json.dumps(registry,indent=2))
files={'paper_reproduction/CONTENT_RATE_RUNS_20261101.json':ROOT/'CONTENT_RATE_RUNS_20261101.json',
 'paper_reproduction/CONTENT_RATE_RESULTS_20261101.json':ROOT/'CONTENT_RATE_RESULTS_20261101.json',
 'reports/CONTENT_RATE_RESULTS_20261101.md':BASE/'RESULTS.md'}
elements=[];proof={}
for name,path in files.items():
 raw=path.read_bytes();elements.append(dict(path=name,mode='100644',type='blob',content=raw.decode('utf-8')))
 proof[name]=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
print(json.dumps(dict(elements=elements,proof=proof),ensure_ascii=True))
