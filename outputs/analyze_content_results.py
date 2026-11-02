"""Read-only numerical audit of completed content-rate screening runs."""
import json,math,statistics,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'outputs/content_rate_20261101'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def quantile(v,q):
 v=sorted(v);p=(len(v)-1)*q;i=int(p);f=p-i
 return v[i]*(1-f)+v[min(i+1,len(v)-1)]*f
def error_summary(rows,predicted):
 total=[];content=[]
 for r,p in zip(rows,predicted):
  content.append(abs(p/r['real_content_bpp']-1))
  total.append(abs((p+r['real_overhead_bpp'])/r['real_bpp']-1))
 return dict(total_median=quantile(total,.5),total_p90=quantile(total,.9),
  content_median=quantile(content,.5),content_p90=quantile(content,.9))
def train_fit(rows,mode,power=False):
 fits={}
 for q in (30,35,40,45):
  selected=[r for r in rows if r['qp']==q and r['mode']==mode]
  xs=[math.log(r['raw_bpp']) for r in selected];ys=[math.log(r['content_bpp']) for r in selected]
  mx=statistics.mean(xs);my=statistics.mean(ys)
  b=1.
  if power:
   var=sum((x-mx)**2 for x in xs)
   b=min(4.,max(.25,sum((x-mx)*(y-my) for x,y in zip(xs,ys))/max(var,1e-12)))
  fits[str(q)]=dict(scale=math.exp(my-b*mx),power=b)
 return fits
jobs={}
for key in ('is-h264','is-h265','od-h264','od-h265'):
 folder=BASE/'jobs'/key;out=folder/'downloads/paper_results';control=out/'control'
 cloud=read(out/'cloud_status.json');run=read(control/'study_run.json');audit=read(control/'content_audit_initial.json')
 calibration=read(control/'content_calibration.json');cfg=read(control/'config.json');cohort=read(control/'cohort.json')
 binding=read(folder/'payload/job.json')
 assert cloud['source_sha256']==binding['source_sha256']
 assert cloud['worker_exit_code']==1 and not run['execution_complete'] and run['state']['task_updates']==0
 assert 'held-out screening' in run['error'] and not audit['gate']['passed']
 assert run['fidelity_gate']['passed']
 assert not set(calibration['sources'])&set(cohort['dev'])
 assert set(calibration['sources'])<=set(cohort['train'])
 assert {r['source'] for r in audit['rows']}==set(cohort['dev'])
 assert len(audit['rows'])==64 and len(cohort['validation'])==64
 assert not (control/'validation/metrics.json').exists()
 for r in audit['rows']:
  assert r['total_bytes']==r['content_bytes']+r['overhead_bytes']
  assert abs(r['real_bpp']-r['real_content_bpp']-r['real_overhead_bpp'])<1e-10
  assert abs(r['proxy_overhead_bpp']-r['real_overhead_bpp'])<1e-6
 observed=error_summary(audit['rows'],[r['relaxed_content_bpp'] for r in audit['rows']])
 assert abs(observed['content_p90']-audit['gate']['content_p90_relative_error'])<1e-9
 assert abs(observed['total_median']-audit['gate']['median_relative_error'])<1e-6
 candidates={}
 for power,name in ((False,'separate_mode_scale'),(True,'separate_mode_power')):
  fits={mode:train_fit(calibration['rows'],mode,power) for mode in ('rounded','relaxed')}
  predictions=[]
  for r in audit['rows']:
   q=str(r['qp']);raw=r['relaxed_content_bpp']/calibration['scales'][[30,35,40,45].index(r['qp'])]
   f=fits['relaxed'][q];predictions.append(f['scale']*raw**f['power'])
  errors=error_summary(audit['rows'],predictions)
  candidates[name]=dict(fit=fits,dev_error=errors,
   rate_error_thresholds_met=max(errors['total_median'],errors['content_median'])<=cfg['rate_median_error_max']
    and max(errors['total_p90'],errors['content_p90'])<=cfg['rate_p90_error_max'],
   scope='offline TRAIN-only fit on cached features, same TRAIN-dev audit; no new GPU/task run or BD-rate')
 per_qp={}
 for q in (30,35,40,45):
  rs=[r for r in audit['rows'] if r['qp']==q]
  per_qp[str(q)]=dict(error=error_summary(rs,[r['relaxed_content_bpp'] for r in rs]),
   relaxed_rounded_ratio_median=statistics.median(r['relaxed_content_bpp']/r['proxy_content_bpp'] for r in rs),
   min_ratio=min(r['relaxed_content_bpp']/r['proxy_content_bpp'] for r in rs),
   max_ratio=max(r['relaxed_content_bpp']/r['proxy_content_bpp'] for r in rs))
 jobs[key]=dict(handle=binding['handle'],status='ERROR_SCREENING_BLOCKED',task_updates=0,bd_rate=None,
  warmup_updates=run['state']['warmup_updates'],warmup_psnr_db=run['fidelity_gate']['psnr_db'],
  gate=audit['gate'],observed_errors=observed,per_qp=per_qp,offline_candidates=candidates,
  proxy_steps=run['state']['proxy_updates'],cloud_wall_seconds=cloud['wall_seconds'],
  fit_sources=len(calibration['sources']),dev_sources=len(cohort['dev']),
  paired_signs=dict(agree=sum(r['same_sign'] for r in audit['paired_changes'] if not r['near_tie']),
   eligible=sum(not r['near_tie'] for r in audit['paired_changes'])),
  report_hashes={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*.json')})
result=dict(client_date='2026-11-01',jobs=jobs,screening_passed_jobs=0,task_trained_jobs=0,new_bd_rate_available=False,
 source_sha256='aa741f13901d513ba4c1615d03580a55ec3be3703d4c6f263f832df95ec29cb2',
 findings=['All four jobs deliberately rejected the proxy before task optimization; no task gradients or validation BD-rate exist.',
 'Exact non-VCL separation verified in downloaded paired audit; fixed overhead is not fitted into content targets.',
 'The current single positive scale is fitted jointly to rounded and noise-relaxed likelihoods; these modes differ substantially.',
 'Offline candidate comparisons use TRAIN-fitted coefficients and cached TRAIN-dev features only; they are diagnostics, not new trained/GPU-verified models.'])
(BASE/'RESULTS.json').write_text(json.dumps(result,indent=2));(ROOT/'CONTENT_RATE_RESULTS_20261101.json').write_text(json.dumps(result,indent=2))
lines=['# Content-rate screening results: 2026-11-01','','All four workers stopped before task optimization. No new AP/BD-rate exists. Gate limits: total AND content median error<=35%, p90<=80%, finite direction alignment>=75%.','',
 '| Worker | Warm-up PSNR | Total median error | Content median error | Content p90 error | Direction | Task updates |',
 '|---|---:|---:|---:|---:|---:|---:|']
for key,j in jobs.items():
 g=j['gate'];lines.append(f"| [{key}](https://www.kaggle.com/code/{j['handle']}) | {j['warmup_psnr_db']:.2f} dB | {100*g['median_relative_error']:.2f}% | {100*g['content_median_relative_error']:.2f}% | {100*g['content_p90_relative_error']:.2f}% | {g['direction_alignment']*g['direction_untied']:.0f}/{g['direction_untied']} | 0 |")
lines+=['','## Cached-feature diagnosis','','A single scale fitted jointly to rounded and relaxed rates compromises between two different feature distributions. The relaxed path is the path supplying NPP gradients. Separate heads per quantization mode are an immediate candidate; a positive power mapping is a second candidate. Both remain unverified by a new GPU task run.','',
 '| Worker | Separate-scale content median / p90 | Rate gate met offline | Separate-power content median / p90 | Rate gate met offline |',
 '|---|---:|---|---:|---|']
for key,j in jobs.items():
 a=j['offline_candidates']['separate_mode_scale'];b=j['offline_candidates']['separate_mode_power']
 lines.append(f"| {key} | {100*a['dev_error']['content_median']:.2f}% / {100*a['dev_error']['content_p90']:.2f}% | {a['rate_error_thresholds_met']} | {100*b['dev_error']['content_median']:.2f}% / {100*b['dev_error']['content_p90']:.2f}% | {b['rate_error_thresholds_met']} |")
for key,j in jobs.items():
 lines+=['',f'## {key}: error by QP','','| QP | Content median / p90 | Median relaxed/rounded content ratio |','|---|---:|---:|']
 for q,v in j['per_qp'].items():lines.append(f"| {q} | {100*v['error']['content_median']:.2f}% / {100*v['error']['content_p90']:.2f}% | {v['relaxed_rounded_ratio_median']:.2f}x |")
lines+=['','## Checks','','Paid bytes/content/non-VCL accounting checked against downloaded audit. Training fit sources do not intersect TRAIN-dev. Screening statistics recomputed. No model/recipe change, gate relaxation or additional job submission during this check.','',
 'Candidate fitting uses the original16TRAIN sources and is assessed on the same8TRAIN-dev sources; using this assessment for choosing the next model is development tuning, not an independent final benchmark. COCO VAL was not evaluated because screening failed.']
(BASE/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({k:dict(gate=j['gate'],qp_modes={q:v['relaxed_rounded_ratio_median'] for q,v in j['per_qp'].items()},
 candidates={n:dict(errors=v['dev_error'],rate_gate=v['rate_error_thresholds_met']) for n,v in j['offline_candidates'].items()}) for k,j in jobs.items()}))
